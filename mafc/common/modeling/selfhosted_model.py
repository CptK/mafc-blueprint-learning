import httpx
import openai
from openai import OpenAI
import time
from typing import cast
from urllib.parse import urlparse

from ezmm import MultimodalSequence
from ezmm.common.items import Image
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from mafc.common.modeling.model import API, APIResponse, Model, Response
from mafc.common.modeling.message import Message
from mafc.common.modeling.utils import abbreviate, messages_with_videos_as_frames
from mafc.common.logger import logger
import config.globals as globals
from mafc.common.modeling.openai_model import count_image_tokens, format_input as _tiktoken_format_input

format_input = _tiktoken_format_input


def _resolve_selfhosted_url(model_name: str) -> str:
    """Return the vLLM endpoint URL for a given model name.

    Looks for a model-specific env var first (SELFHOSTED_URL_<NORMALIZED_NAME>),
    then falls back to the global selfhosted_url. This allows each model to be
    served by a separate vLLM instance (e.g. separate SLURM jobs) while keeping
    backwards compatibility when only one endpoint is configured.

    Example: model "Qwen/Qwen3.5-35B-A3B-FP8" → env var "selfhosted_url_qwen_qwen3_5_35b_a3b_fp8"
    """
    import re
    import os

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    url = os.environ.get(f"selfhosted_url_{normalized}") or globals.selfhosted_url
    if not url:
        raise ValueError(
            f"Missing self-hosted model URL for '{model_name}'. "
            f"Set selfhosted_url_{normalized} or selfhosted_url in the environment or config/.env."
        )
    return url


class SelfhostedAPI(API):
    def __init__(self, model: str, context_window: int):
        self.model = model
        self.context_window = context_window
        url = _resolve_selfhosted_url(model)
        self.client = OpenAI(base_url=url, api_key="none", timeout=600)
        # vLLM tokenize/detokenize endpoints live at the server root (no /v1 prefix)
        parsed = urlparse(url.rstrip("/"))
        server_root = f"{parsed.scheme}://{parsed.netloc}"
        self._tokenize_url = f"{server_root}/tokenize"
        self._detokenize_url = f"{server_root}/detokenize"
        self._http = httpx.Client(timeout=10.0)
        _ = self.client.chat  # warm up lazy openai submodule imports in this thread

    def _tokenize(self, text: str) -> list[int] | None:
        """Tokenize text via vLLM /tokenize. Returns token IDs, or None on failure."""
        try:
            resp = self._http.post(
                self._tokenize_url,
                json={"model": self.model, "prompt": text, "add_special_tokens": False},
            )
            resp.raise_for_status()
            return list(resp.json()["tokens"])
        except Exception as e:
            logger.debug(f"[Selfhosted] /tokenize failed: {e}")
            return None

    def _detokenize(self, tokens: list[int]) -> str | None:
        """Detokenize token IDs via vLLM /detokenize. Returns text, or None on failure."""
        try:
            resp = self._http.post(
                self._detokenize_url,
                json={"model": self.model, "tokens": tokens},
            )
            resp.raise_for_status()
            return str(resp.json()["prompt"])
        except Exception as e:
            logger.debug(f"[Selfhosted] /detokenize failed: {e}")
            return None

    def _format_input(self, content: MultimodalSequence, token_budget: int) -> list[dict]:
        """Format message content, truncating text via the model's own tokenizer.

        If /tokenize is unavailable, falls back to tiktoken with a 15% safety margin
        to compensate for cross-tokenizer drift.
        """
        content_formatted: list[dict] = []
        remaining = token_budget

        for block in content.to_list():
            if remaining <= 0:
                break

            if isinstance(block, str):
                tokens = self._tokenize(block)
                if tokens is None:
                    logger.warning(
                        "[Selfhosted] /tokenize unavailable; falling back to tiktoken with 15% margin."
                    )
                    return format_input(content, int(token_budget * 0.85))

                if len(tokens) <= remaining:
                    remaining -= len(tokens)
                    content_formatted.append({"type": "text", "text": block})
                else:
                    truncated_text = self._detokenize(tokens[:remaining])
                    if truncated_text is None:
                        # /detokenize unavailable: estimate truncation point from char/token ratio
                        chars_per_token = len(block) / len(tokens) if tokens else 4.0
                        truncated_text = block[: int(remaining * chars_per_token)]
                    content_formatted.append({"type": "text", "text": truncated_text})
                    remaining = 0

            elif isinstance(block, Image):
                image_tokens = int(count_image_tokens(block))
                if image_tokens > remaining:
                    break
                image_encoded = block.get_base64_encoded()
                content_formatted.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_encoded}"}}
                )
                remaining -= image_tokens

        return content_formatted

    def __call__(self, messages: list[Message], **kwargs) -> APIResponse:
        max_response_length = kwargs.get("max_response_length", 2048)
        # Small overhead for chat-template markers; text truncation uses the model's
        # own tokenizer via /tokenize so no cross-tokenizer drift margin is needed.
        input_budget = self.context_window - max_response_length - 500
        provider_messages = cast(
            list[ChatCompletionMessageParam],
            [
                {
                    "role": message.role.value,
                    "content": cast(
                        list[ChatCompletionContentPartParam],
                        self._format_input(message.content, token_budget=input_budget),
                    ),
                }
                for message in messages
            ],
        )

        create_kwargs: dict = dict(
            model=self.model,
            messages=provider_messages,
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            max_tokens=max_response_length,
        )
        if kwargs.get("presence_penalty"):
            create_kwargs["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("extra_body"):
            create_kwargs["extra_body"] = kwargs["extra_body"]

        for _attempt in range(3):
            try:
                response = self.client.chat.completions.create(**create_kwargs)
                break
            except openai.APIConnectionError as e:
                if _attempt == 2:
                    raise
                wait = 2**_attempt * 5  # 5s, 10s
                logger.warning(
                    f"[Selfhosted] Connection error (attempt {_attempt + 1}/3), retrying in {wait}s: {e}"
                )
                time.sleep(wait)

        content = response.choices[0].message.content
        if not content:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            logger.error(
                f"[Selfhosted] Empty response from model '{self.model}': "
                f"finish_reason={finish_reason!r}, max_response_length={kwargs.get('max_response_length', 2048)}"
            )
            content = f"Failed to generate a response (finish_reason={finish_reason!r})."
        return APIResponse(
            text=content,
            input_token_count=response.usage.prompt_tokens if response.usage else None,
            output_token_count=response.usage.completion_tokens if response.usage else None,
            total_token_count=response.usage.total_tokens if response.usage else None,
        )


class SelfhostedModel(Model):
    def __init__(
        self,
        specifier: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        max_response_length: int = 2048,
        video_frames_to_sample: int = 5,
        thinking: bool = True,
        presence_penalty: float = 0.0,
    ):
        super().__init__(
            specifier=specifier,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_response_length=max_response_length,
            video_frames_to_sample=video_frames_to_sample,
        )
        self.thinking = thinking
        self.presence_penalty = presence_penalty
        self.api = SelfhostedAPI(model=self.model, context_window=self.context_window)

    def _do_generate(self, messages: list[Message]) -> Response:
        extra_body: dict = {}
        if not self.thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.top_k != 50:  # only pass top_k when explicitly set (vLLM default differs)
            extra_body["top_k"] = self.top_k
        try:
            api_response = self.api(
                messages_with_videos_as_frames(messages, self.video_frames_to_sample),
                temperature=self.temperature,
                top_p=self.top_p,
                max_response_length=self.max_response_length,
                presence_penalty=self.presence_penalty,
                extra_body=extra_body or None,
            )
        except openai.RateLimitError as e:
            logger.error("Rate limit exceeded on self-hosted model.")
            raise e
        except openai.AuthenticationError as e:
            logger.error("Authentication failed for self-hosted model.")
            raise e
        except Exception as e:
            logger.error(
                f"An error occurred while communicating with the Self-Hosted API: {e}\n"
                f"Input: {[abbreviate(str(m.content)) for m in messages_with_videos_as_frames(messages, self.video_frames_to_sample)]}"
            )
            raise e

        return Response(
            text=api_response.text,
            input_token_count=api_response.input_token_count,
            output_token_count=api_response.output_token_count,
            total_token_count=api_response.total_token_count,
            total_cost=self.compute_cost(api_response),
        )


if __name__ == "__main__":
    # Quick test to verify connectivity to the self-hosted model
    api = SelfhostedAPI(model="Qwen/Qwen3.5-122B-A10B-FP8", context_window=4096)

    from ezmm import MultimodalSequence
    from mafc.common.modeling import Message, MessageRole

    messages = [Message(role=MessageRole.USER, content=MultimodalSequence("What is the capital of France?"))]

    response = api(messages)
    print("Response from self-hosted model:", response.text)
