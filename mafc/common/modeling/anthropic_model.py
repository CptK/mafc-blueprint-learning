from ezmm import Image, Video
import anthropic
import tiktoken
import numpy as np
import os

from ezmm import MultimodalSequence
from mafc.common.modeling.model import API, APIResponse, Model, Response
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt
from mafc.common.modeling.utils import messages_with_videos_as_frames
from mafc.common.logger import logger

encoding = tiktoken.get_encoding("cl100k_base")


def _resolve_anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_api_key")


def count_image_tokens_estimate(image: Image) -> int:
    """Estimate image token cost similar to OpenAI's tiling heuristic.

    Anthropic's actual vision tokenization may differ; this is a conservative
    approximation to avoid egregious overflows when packing inputs.
    """
    n_tiles = int(np.ceil(image.width / 512) * np.ceil(image.height / 512))
    return int(85 + 170 * n_tiles)


def format_input(content: MultimodalSequence, context_window: int) -> list[dict]:
    """Format one message content payload for the Anthropic Messages API.

    - Truncates text to the remaining budget (approx via cl100k).
    - Includes images fully when they fit the estimated budget.
    - Assumes videos have been converted to frames upstream.
    """
    content_formatted: list[dict] = []
    remaining = int(context_window)

    for block in content.to_list():
        if remaining <= 0:
            break

        if isinstance(block, str):
            tokens = encoding.encode(block, disallowed_special=())
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
                block = encoding.decode(tokens)
                remaining = 0
            else:
                remaining -= len(tokens)
            content_formatted.append({"type": "text", "text": block})

        elif isinstance(block, Image):
            img_tokens = count_image_tokens_estimate(block)
            if img_tokens > remaining:
                break
            image_encoded = block.get_base64_encoded()
            content_formatted.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_encoded,
                    },
                }
            )
            remaining -= img_tokens

        elif isinstance(block, Video):
            # Expect videos to be converted to frames via Prompt.with_videos_as_frames
            pass

    return content_formatted


class AnthropicAPI(API):
    def __init__(self, model: str, context_window: int):
        self.model = model
        self.context_window = context_window
        api_key = _resolve_anthropic_key()
        if not api_key:
            raise ValueError(
                "Missing Anthropic API key. Set ANTHROPIC_API_KEY or anthropic_api_key in the environment or config/.env."
            )
        self.client = anthropic.Anthropic(api_key=api_key, timeout=300)

    def __call__(self, messages: list[Message], **kwargs) -> APIResponse:
        max_response_length = kwargs.get("max_response_length", 2048)
        # Subtract overhead for chat-template tokens and tokenizer mismatch.
        input_budget = self.context_window - max_response_length - 200
        system_parts = [
            str(message.content).strip() for message in messages if message.role == MessageRole.SYSTEM
        ]
        anthropic_messages = [
            {
                "role": message.role.value,
                "content": format_input(message.content, context_window=input_budget),
            }
            for message in messages
            if message.role != MessageRole.SYSTEM
        ]

        try:
            # Anthropic models do not allow specifying both temperature and top_p simultaneously.
            # Prefer temperature when both are provided; otherwise pass whichever is set.
            temp = kwargs.get("temperature")
            topp = kwargs.get("top_p")

            create_kwargs = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_response_length,
            }
            if temp is not None and topp is not None:
                logger.warning("Both temperature and top_p specified; using temperature for Anthropic.")
                topp = None
            if temp is not None:
                create_kwargs["temperature"] = temp
            elif topp is not None:
                create_kwargs["top_p"] = topp
            if system_parts:
                create_kwargs["system"] = "\n\n".join(part for part in system_parts if part)

            response = self.client.messages.create(**create_kwargs)
        except anthropic.RateLimitError as e:
            logger.error("Anthropic rate limit exceeded.")
            raise e
        except anthropic.AuthenticationError as e:
            logger.error("Anthropic authentication failed. Check your API key.")
            raise e
        except Exception as e:
            logger.error(
                f"An error occurred while communicating with the Anthropic API: {e}\n"
                f"Input: {[str(m.content) for m in messages]}"
            )
            raise e

        # Extract text parts from content blocks
        text_parts = []
        try:
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", ""))
        except Exception:
            # Fallback: try to coerce to string
            text_parts = [str(response)]

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        total_tokens = (
            (input_tokens or 0) + (output_tokens or 0)
            if (input_tokens is not None or output_tokens is not None)
            else None
        )

        return APIResponse(
            text=("\n".join(text_parts) if text_parts else "Failed to generate a response."),
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            total_token_count=total_tokens,
        )


class AnthropicModel(Model):
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
        self.api = AnthropicAPI(model=self.model, context_window=self.context_window)

    def _do_generate(self, messages: list[Message]) -> Response:
        try:
            api_response = self.api(
                messages_with_videos_as_frames(messages, self.video_frames_to_sample),
                temperature=self.temperature,
                top_p=self.top_p,
                max_response_length=self.max_response_length,
            )
        except Exception:
            # Errors already logged in API layer
            raise

        return Response(
            text=api_response.text,
            input_token_count=api_response.input_token_count,
            output_token_count=api_response.output_token_count,
            total_token_count=api_response.total_token_count,
            total_cost=self.compute_cost(api_response),
        )


if __name__ == "__main__":
    from mafc.common.modeling.message import MessageRole

    model = AnthropicModel(specifier="ANTHROPIC:claude-haiku-4-5-20251001", temperature=1.0)
    response = model.generate(
        [Message(role=MessageRole.USER, content=Prompt(text="What is the capital of France?"))]
    )
    print(response)
