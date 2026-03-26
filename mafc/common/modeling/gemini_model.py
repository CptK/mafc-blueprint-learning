import os
import base64
import numpy as np
import tiktoken
from typing import cast

from ezmm import Image, MultimodalSequence, Video
from google import genai
from google.genai.types import Blob, Content, ContentListUnionDict, GenerateContentConfig, Part

from mafc.common.modeling.model import API, APIResponse, Model, Response
from mafc.common.modeling.message import Message
from mafc.common.modeling.prompt import Prompt
from mafc.common.modeling.utils import messages_with_videos_as_frames
from mafc.common.logger import logger

encoding = tiktoken.get_encoding("cl100k_base")


def _resolve_gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("gemini_api_key")


def count_image_tokens_estimate(image: Image) -> int:
    # Heuristic similar to OpenAI tiling to avoid overflows
    n_tiles = int(np.ceil(image.width / 512) * np.ceil(image.height / 512))
    return int(85 + 170 * n_tiles)


def _image_bytes(block: Image) -> bytes:
    try:
        return base64.b64decode(block.get_base64_encoded())
    except Exception:
        raise ValueError("Unable to obtain image bytes for Gemini input.")


def format_input(content: MultimodalSequence, context_window: int) -> list[Part]:
    """Format one message content payload to Google GenAI parts.

    - Truncate text by token budget (approx cl100k).
    - Include images only if they fit the remaining budget.
    - Assume videos converted to frames upstream.
    """
    parts: list[Part] = []
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
            parts.append(Part(text=block))

        elif isinstance(block, Image):
            img_tokens = count_image_tokens_estimate(block)
            if img_tokens > remaining:
                break
            parts.append(Part(inline_data=Blob(mime_type="image/jpeg", data=_image_bytes(block))))
            remaining -= img_tokens

        elif isinstance(block, Video):
            # Expect videos converted to frames upstream
            pass

    return parts


class GeminiAPI(API):
    def __init__(self, model: str, context_window: int):
        self.model = model
        self.context_window = context_window
        api_key = _resolve_gemini_key()
        if not api_key:
            raise ValueError(
                "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment or config/.env."
            )
        self.client = genai.Client(api_key=api_key)

    def __call__(self, messages: list[Message], **kwargs) -> APIResponse:
        max_response_length = kwargs.get("max_response_length", 2048)
        # Subtract overhead for chat-template tokens and tokenizer mismatch.
        input_budget = self.context_window - max_response_length - 200
        system_parts = [
            str(message.content).strip() for message in messages if message.role.value == "system"
        ]
        contents = [
            Content(
                role=message.role.value,
                parts=format_input(message.content, context_window=input_budget),
            )
            for message in messages
            if message.role.value != "system"
        ]

        config = GenerateContentConfig(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            max_output_tokens=max_response_length,
            system_instruction="\n\n".join(part for part in system_parts if part) if system_parts else None,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=cast(ContentListUnionDict, contents),
                config=config,
            )
        except Exception as e:
            logger.error(
                f"An error occurred while communicating with the Gemini API: {e}\n"
                f"Input: {[str(m.content) for m in messages]}"
            )
            raise

        # Extract primary text, falling back to aggregating candidate parts
        text = getattr(response, "text", None)
        if not text:
            texts: list[str] = []
            try:
                for cand in getattr(response, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    if content:
                        for p in getattr(content, "parts", []) or []:
                            t = getattr(p, "text", None)
                            if isinstance(t, str) and t:
                                texts.append(t)
            except Exception:
                pass
            text = "\n".join(texts) if texts else "Failed to generate a response."

        # Usage accounting (fields vary; keep optional)
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        total_tokens = getattr(usage, "total_token_count", None) if usage else None

        return APIResponse(
            text=text,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            total_token_count=total_tokens,
        )


class GeminiModel(Model):
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
        self.api = GeminiAPI(model=self.model, context_window=self.context_window)

    def _do_generate(self, messages: list[Message]) -> Response:
        api_response = self.api(
            messages_with_videos_as_frames(messages, self.video_frames_to_sample),
            temperature=self.temperature,
            top_p=self.top_p,
            max_response_length=self.max_response_length,
        )

        return Response(
            text=api_response.text,
            input_token_count=api_response.input_token_count,
            output_token_count=api_response.output_token_count,
            total_token_count=api_response.total_token_count,
            total_cost=self.compute_cost(api_response),
        )


if __name__ == "__main__":
    from mafc.common.modeling.message import MessageRole

    model = GeminiModel(specifier="GOOGLE:gemini-3.1-flash-lite-preview")
    response = model.generate(
        [Message(role=MessageRole.USER, content=Prompt(text="What is the capital of France?"))]
    )
    print(response)
