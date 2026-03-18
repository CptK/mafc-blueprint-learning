from ezmm import Image, Video
from openai import OpenAI
import openai
import threading
import tiktoken
import numpy as np
import os
from typing import cast

from ezmm import MultimodalSequence
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from mafc.common.modeling.model import API, APIResponse, Model, Response
from mafc.common.modeling.message import Message
from mafc.common.modeling.prompt import Prompt
from mafc.common.modeling.utils import messages_with_videos_as_frames
from mafc.common.logger import logger

_tiktoken_lock = threading.Lock()
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


class OpenAIAPI(API):
    def __init__(self, model: str, context_window: int):
        self.model = model
        self.context_window = context_window
        # Prefer standard env var; fall back to lower-case or api_keys map
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. Set OPENAI_API_KEY or openai_api_key in the environment or config/.env."
            )
        self.client = OpenAI(api_key=api_key, timeout=300)

    def __call__(self, messages: list[Message], **kwargs) -> APIResponse:
        provider_messages = cast(
            list[ChatCompletionMessageParam],
            [
                {
                    "role": message.role.value,
                    "content": cast(
                        list[ChatCompletionContentPartParam],
                        format_input(message.content, context_window=self.context_window),
                    ),
                }
                for message in messages
            ],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=provider_messages,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            max_completion_tokens=kwargs.get("max_response_length", 2048),
        )

        content = response.choices[0].message.content
        if not content:
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None
            details = getattr(usage, "completion_tokens_details", None) if usage else None
            reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
            refusal = getattr(response.choices[0].message, "refusal", None)
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            logger.error(
                f"[OpenAI] Empty response from model '{self.model}': "
                f"finish_reason={finish_reason!r}, refusal={refusal!r}, "
                f"input_tokens={input_tokens}, output_tokens={output_tokens}, reasoning_tokens={reasoning_tokens}, max_response_length={kwargs.get('max_response_length', 2048)}"
            )
            content = "Failed to generate a response."
        return APIResponse(
            text=content,
            input_token_count=response.usage.prompt_tokens if response.usage else None,
            output_token_count=response.usage.completion_tokens if response.usage else None,
            total_token_count=response.usage.total_tokens if response.usage else None,
        )


class OpenAIModel(Model):

    def __init__(
        self,
        specifier: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        max_response_length: int = 2048,
        video_frames_to_sample: int = 5,
    ):
        super().__init__(
            specifier=specifier,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_response_length=max_response_length,
            video_frames_to_sample=video_frames_to_sample,
        )
        self.api = OpenAIAPI(model=self.model, context_window=self.context_window)

    def generate(self, messages: list[Message]) -> Response:
        try:
            api_response = self.api(
                messages_with_videos_as_frames(messages, self.video_frames_to_sample),
                temperature=self.temperature,
                top_p=self.top_p,
                max_response_length=self.max_response_length,
            )
        except openai.RateLimitError as e:
            logger.error(
                "Rate limit exceeded. Consider reducing the frequency of requests or upgrading your OpenAI plan."
            )
            raise e
        except openai.AuthenticationError as e:
            logger.error("Authentication failed. Check your OpenAI API key.")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while communicating with the OpenAI API: {e}")
            raise e

        return Response(
            text=api_response.text,
            input_token_count=api_response.input_token_count,
            output_token_count=api_response.output_token_count,
            total_token_count=api_response.total_token_count,
            total_cost=self.compute_cost(api_response),
        )


def count_tokens(prompt: MultimodalSequence | str) -> int:
    with _tiktoken_lock:
        n_text_tokens = len(_get_encoding().encode(str(prompt), disallowed_special=()))
    n_image_tokens = 0
    if isinstance(prompt, MultimodalSequence) and prompt.has_images():
        for image in prompt.images:
            n_image_tokens += count_image_tokens(image)
    return n_text_tokens + n_image_tokens


def count_image_tokens(image: Image) -> int:
    """See the formula here: https://openai.com/api/pricing/

    Returns an integer token count based on 512x512 tiling.
    """
    # Use integer math for tile counting and ensure integer return
    n_tiles = int(np.ceil(image.width / 512) * np.ceil(image.height / 512))
    return int(85 + 170 * n_tiles)


def format_input(content: MultimodalSequence, context_window: int) -> list[dict]:
    """Format one multimodal message content payload for the OpenAI chat API.

    - Truncates text to fit in the remaining token budget.
    - Includes whole images only when enough tokens remain.
    - Assumes videos have been converted to frames upstream.
    """
    content_formatted: list[dict] = []
    remaining = int(context_window)

    for block in content.to_list():
        # Stop immediately if no tokens remain
        if remaining <= 0:
            break

        if isinstance(block, str):
            with _tiktoken_lock:
                tokens = _get_encoding().encode(block, disallowed_special=())
                if len(tokens) > remaining:
                    tokens = tokens[:remaining]
                    block = _get_encoding().decode(tokens)
                    remaining = 0
                else:
                    remaining -= len(tokens)
            content_formatted.append({"type": "text", "text": block})

        elif isinstance(block, Image):
            image_token_count = int(count_image_tokens(block))
            if image_token_count > remaining:
                # Do not include partial images
                break
            image_encoded = block.get_base64_encoded()
            content_formatted.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_encoded}"}}
            )
            remaining -= image_token_count

        elif isinstance(block, Video):
            # At this point all videos should have been turned into frames
            pass

    return content_formatted


if __name__ == "__main__":
    from mafc.common.modeling.message import MessageRole

    model = OpenAIModel(specifier="OPENAI:gpt-5-mini-2025-08-07", temperature=1.0)
    response = model.generate(
        [Message(role=MessageRole.USER, content=Prompt(text="What is the capital of France?"))]
    )
    print(response)
