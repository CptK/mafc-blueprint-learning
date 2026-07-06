import random
import time
from abc import ABC, abstractmethod
from pydantic import BaseModel

from mafc.common.logger import logger
from mafc.common.modeling.message import Message
from mafc.common.modeling.utils import (
    model_specifier_to_shorthand,
    get_model_context_window,
    get_model_api_pricing,
)

# Transient API failures that warrant a retry (matched case-insensitively against
# the exception text). Anything else — auth, validation, context overflow — fails fast.
_TRANSIENT_ERROR_MARKERS = (
    "503",
    "unavailable",
    "429",
    "resource_exhausted",
    "rate limit",
    "overloaded",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "recvmsg",
    "readerror",
    "remotedisconnected",
    "500 internal",
    "internal server error",
)

_MAX_GENERATE_ATTEMPTS = 3


def _is_transient_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


class Response(BaseModel):
    text: str
    input_token_count: int | None = None
    output_token_count: int | None = None
    total_token_count: int | None = None
    total_cost: float
    duration_ms: float | None = None


class APIResponse(BaseModel):
    text: str
    input_token_count: int | None = None
    output_token_count: int | None = None
    total_token_count: int | None = None


class API(ABC):
    @abstractmethod
    def __call__(self, messages: list[Message], **kwargs) -> APIResponse:
        """Send a role-annotated message list to the model and return the response."""
        pass


class Model(ABC):
    def __init__(
        self,
        specifier: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        max_response_length: int = 2048,
        video_frames_to_sample: int = 5,
    ):
        self.specifier = specifier
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_response_length = max_response_length
        self.video_frames_to_sample = video_frames_to_sample

        self.name, self.model = model_specifier_to_shorthand(specifier)
        self.context_window = get_model_context_window(self.name)
        self.input_token_cost, self.output_token_cost = get_model_api_pricing(self.name)

    @abstractmethod
    def _do_generate(self, messages: list[Message]) -> Response:
        """Send a role-annotated message list to the model and return the response."""
        pass

    def generate(self, messages: list[Message]) -> Response:
        """Timed wrapper around _do_generate; sets duration_ms on the returned Response.

        Transient API failures (503/429/timeouts/connection resets) are retried with
        exponential backoff; other exceptions propagate immediately.
        """
        t0 = time.monotonic()
        for attempt in range(1, _MAX_GENERATE_ATTEMPTS + 1):
            try:
                response = self._do_generate(messages)
                break
            except Exception as exc:
                if attempt >= _MAX_GENERATE_ATTEMPTS or not _is_transient_error(exc):
                    raise
                delay = 3.0 * (2 ** (attempt - 1)) + random.uniform(0.0, 1.0)
                logger.warning(
                    f"[{self.name}] Transient API error (attempt {attempt}/{_MAX_GENERATE_ATTEMPTS}), "
                    f"retrying in {delay:.1f}s: {type(exc).__name__}: {exc}"
                )
                time.sleep(delay)
        response.duration_ms = (time.monotonic() - t0) * 1000
        return response

    def compute_cost(self, api_response: APIResponse) -> float:
        total_cost = 0.0
        if api_response.input_token_count and api_response.output_token_count:
            total_cost = (api_response.input_token_count / 1_000_000) * self.input_token_cost + (
                api_response.output_token_count / 1_000_000
            ) * self.output_token_cost
        return total_cost
