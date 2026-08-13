from ezmm import Image, Video
import anthropic
import tiktoken
import numpy as np
import os

from ezmm import MultimodalSequence
from mafc.common.modeling.model import API, APIResponse, Model, Response
from mafc.common.modeling.message import Message, MessageRole
from mafc.common.modeling.prompt import Prompt
from mafc.common.modeling.utils import abbreviate, messages_with_videos_as_frames
from mafc.common.logger import logger

encoding = tiktoken.get_encoding("cl100k_base")

# Models that do not accept `temperature` or `top_p` sampling parameters.
# On these, sampling parameters were removed rather than deprecated: sending any
# of them returns a 400, so they must be dropped from the request entirely.
# Steer these models by prompting instead.
_NO_SAMPLING_PARAMS: frozenset[str] = frozenset(
    {
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
    }
)

# Models that take adaptive thinking, where the model decides how much to think per
# request instead of spending a fixed budget. This is the only supported form on
# 4.7 and later, which reject an explicit `budget_tokens`, and the recommended one
# on 4.6, where it also switches on interleaved thinking with no beta header.
# Anything not listed here predates adaptive thinking and needs a token budget.
_ADAPTIVE_THINKING: frozenset[str] = frozenset(
    {
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
    }
)

# For the legacy budget path only: the API floor on a thinking budget, and the share
# of the response allowance to hand it. The budget must stay strictly under
# `max_tokens`, which caps thinking and answer text together.
_MIN_THINKING_BUDGET = 1024
_THINKING_BUDGET_SHARE = 0.5

# Where to place the prompt-cache breakpoint. Caching is a prefix match, so a marker
# caches everything rendered before it (tools, then system, then messages) and the
# correct placement depends on which part of the request repeats verbatim:
#
#   "system" — the system prompt repeats while the user turn varies. The fact-check
#              planner: a stable blueprint prefix, fresh per-iteration state.
#   "prompt" — the entire request repeats. The judge sampling one claim n times.
#
# Placement is not a free choice. "prompt" on a caller whose user turn changes every
# call is worse than no caching, since each call writes an entry at 1.25x that
# nothing ever reads. "system" on a caller with a small system prompt is merely
# inert, because a prefix under the model's minimum never caches.
CACHE_SYSTEM = "system"
CACHE_PROMPT = "prompt"
_CACHE_PLACEMENTS = frozenset({CACHE_SYSTEM, CACHE_PROMPT})


def _resolve_anthropic_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_api_key")


# Anthropic rejects a request carrying many images if any of them exceeds this on
# either dimension ("max allowed size for many-image requests"). Judge prompts
# routinely carry a dozen or more evidence images, so every image is downscaled to
# fit rather than gambling on the per-request image count. Gemini has no such
# limit, which is why traces recorded on Gemini can only be replayed here after
# resizing.
_MAX_IMAGE_EDGE_PX = 2000


def _encode_image_within_limits(image: Image) -> str:
    """Base64-encode an image, downscaling so neither edge exceeds the API limit.

    Aspect ratio is preserved; images already within the limit are encoded
    untouched, so this is a no-op for the common case.
    """
    longest_edge = max(image.width, image.height)
    if longest_edge <= _MAX_IMAGE_EDGE_PX:
        return image.get_base64_encoded()

    scale = _MAX_IMAGE_EDGE_PX / longest_edge
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    logger.debug(
        f"[Anthropic] Downscaling {image.reference} from {image.width}x{image.height} "
        f"to {new_size[0]}x{new_size[1]} to satisfy the {_MAX_IMAGE_EDGE_PX}px limit."
    )
    pil_image = image.image
    from ezmm.common.items.image import to_base64  # local import: avoids a cycle at module load

    return to_base64(pil_image.resize(new_size))


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
            if block:  # skip empty strings — Anthropic rejects empty text blocks
                content_formatted.append({"type": "text", "text": block})

        elif isinstance(block, Image):
            img_tokens = count_image_tokens_estimate(block)
            if img_tokens > remaining:
                break
            image_encoded = _encode_image_within_limits(block)
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
        self._warned_sampling_conflict = False

    def _thinking_config(self, enabled: bool, max_response_length: int) -> dict | None:
        """Build the request's thinking block, or None to leave thinking off.

        Adaptive thinking is a bare type on current models. Older ones need an
        explicit budget that fits under `max_tokens`, since that ceiling covers
        thinking and answer text together; if no budget fits, thinking is skipped
        rather than sending a request the API would reject.
        """
        if not enabled:
            return None
        if self.model in _ADAPTIVE_THINKING:
            return {"type": "adaptive"}

        budget = max(int(max_response_length * _THINKING_BUDGET_SHARE), _MIN_THINKING_BUDGET)
        if budget >= max_response_length:
            logger.warning(
                f"[Anthropic] Thinking requested for {self.model} but max_response_length "
                f"({max_response_length}) leaves no room for the {_MIN_THINKING_BUDGET}-token "
                f"minimum budget; sending the request without thinking."
            )
            return None
        return {"type": "enabled", "budget_tokens": budget}

    def __call__(self, messages: list[Message], **kwargs) -> APIResponse:
        max_response_length = kwargs.get("max_response_length", 2048)
        # Subtract overhead for chat-template tokens and tokenizer mismatch.
        input_budget = self.context_window - max_response_length - 200
        system_parts = [
            str(message.content).strip() for message in messages if message.role == MessageRole.SYSTEM
        ]
        anthropic_messages = []
        for message in messages:
            if message.role == MessageRole.SYSTEM:
                continue
            content_blocks = format_input(message.content, context_window=input_budget)
            if not content_blocks:
                # Budget exhausted before this message — insert a minimal placeholder
                # so Anthropic doesn't reject an empty content array.
                content_blocks = [{"type": "text", "text": "[truncated]"}]
            anthropic_messages.append({"role": message.role.value, "content": content_blocks})

        cache = kwargs.get("cache")
        if cache == CACHE_PROMPT and anthropic_messages:
            # Marks the final content block, so the cached prefix spans the whole
            # request: system, then every message. Correct only when the caller
            # repeats the request verbatim.
            anthropic_messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        try:
            create_kwargs = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_response_length,
            }

            thinking_config = self._thinking_config(kwargs.get("thinking", False), max_response_length)
            if thinking_config is not None:
                create_kwargs["thinking"] = thinking_config
                effort = kwargs.get("effort")
                if effort:
                    # Depth/spend dial for adaptive thinking: low | medium | high | max.
                    # Nested under output_config, not top-level. Omitted means the API
                    # default, which is high.
                    create_kwargs["output_config"] = {"effort": effort}

            # An explicit thinking budget is incompatible with sampling parameters on
            # the older models that still require one, so the budget path drops them.
            sampling_allowed = self.model not in _NO_SAMPLING_PARAMS and not (
                thinking_config is not None and thinking_config.get("type") == "enabled"
            )
            if sampling_allowed:
                # Anthropic models do not allow specifying both temperature and top_p simultaneously.
                # Prefer temperature when both are provided; otherwise pass whichever is set.
                temp = kwargs.get("temperature")
                topp = kwargs.get("top_p")
                if temp is not None and topp is not None:
                    if not self._warned_sampling_conflict:
                        # Once per client: the caller's configuration is fixed for the
                        # run, so repeating this per request only buries other warnings.
                        logger.warning(
                            "Both temperature and top_p specified; using temperature for Anthropic."
                        )
                        self._warned_sampling_conflict = True
                    topp = None
                if temp is not None:
                    create_kwargs["temperature"] = temp
                elif topp is not None:
                    create_kwargs["top_p"] = topp
            if system_parts:
                system_text = "\n\n".join(part for part in system_parts if part)
                system_block: dict = {"type": "text", "text": system_text}
                if cache == CACHE_SYSTEM:
                    # Caches tools plus system. Prefixes below the model's minimum
                    # (512-4096 tokens depending on the model) silently do not cache at
                    # all, which shows up as cache_read_token_count staying 0.
                    system_block["cache_control"] = {"type": "ephemeral"}
                create_kwargs["system"] = [system_block]

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
                f"Input: {[abbreviate(str(m.content)) for m in messages]}"
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
        # Cached prompt tokens are reported apart from input_tokens, which holds only
        # the uncached remainder. Both are needed to price the call and to tell whether
        # caching is working at all: reads stuck at 0 mean the prefix is not matching.
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", None) if usage else None
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", None) if usage else None
        counted = [input_tokens, output_tokens, cache_write_tokens, cache_read_tokens]
        total_tokens = (
            sum(count or 0 for count in counted) if any(count is not None for count in counted) else None
        )

        return APIResponse(
            text=("\n".join(text_parts) if text_parts else "Failed to generate a response."),
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            total_token_count=total_tokens,
            cache_write_token_count=cache_write_tokens,
            cache_read_token_count=cache_read_tokens,
        )


class AnthropicModel(Model):
    def __init__(
        self,
        specifier: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int = 50,
        max_response_length: int = 2048,
        video_frames_to_sample: int = 5,
        thinking: bool = True,
        effort: str | None = None,
        cache: str | None = None,
        **kwargs,
    ):
        # Default to None rather than a concrete value so a caller's choice is
        # distinguishable from a value nobody picked. Anthropic rejects temperature and
        # top_p together, and forwarding both class defaults made every single request
        # look like a deliberate conflict. Only what the caller actually set is sent;
        # sending neither leaves the API on its own defaults.
        self.send_temperature = temperature is not None
        self.send_top_p = top_p is not None
        super().__init__(
            specifier=specifier,
            temperature=temperature if temperature is not None else 1.0,
            top_p=top_p if top_p is not None else 1.0,
            top_k=top_k,
            max_response_length=max_response_length,
            video_frames_to_sample=video_frames_to_sample,
        )
        # Off by default: caching pays only for callers that repeat a prefix verbatim,
        # and most here send a fresh prompt every time. See _CACHE_PLACEMENTS for which
        # placement suits which caller. A typo would silently disable caching, which is
        # the exact failure this whole path is trying to make visible, so reject it.
        if cache is not None and cache not in _CACHE_PLACEMENTS:
            raise ValueError(
                f"Unknown cache placement {cache!r}; expected one of {sorted(_CACHE_PLACEMENTS)} or None."
            )
        self.cache = cache
        self.thinking = thinking
        # None leaves effort unset, which the API reads as its own default (high).
        self.effort = effort
        self.api = AnthropicAPI(model=self.model, context_window=self.context_window)

    def _do_generate(self, messages: list[Message]) -> Response:
        try:
            api_response = self.api(
                messages_with_videos_as_frames(messages, self.video_frames_to_sample),
                temperature=self.temperature if self.send_temperature else None,
                top_p=self.top_p if self.send_top_p else None,
                max_response_length=self.max_response_length,
                cache=self.cache,
                thinking=self.thinking,
                effort=self.effort,
            )
        except Exception:
            # Errors already logged in API layer
            raise

        return Response(
            text=api_response.text,
            input_token_count=api_response.input_token_count,
            output_token_count=api_response.output_token_count,
            total_token_count=api_response.total_token_count,
            cache_write_token_count=api_response.cache_write_token_count,
            cache_read_token_count=api_response.cache_read_token_count,
            total_cost=self.compute_cost(api_response),
        )


if __name__ == "__main__":
    from mafc.common.modeling.message import MessageRole

    model = AnthropicModel(specifier="ANTHROPIC:claude-haiku-4-5-20251001", temperature=1.0)
    response = model.generate(
        [Message(role=MessageRole.USER, content=Prompt(text="What is the capital of France?"))]
    )
    print(response)
