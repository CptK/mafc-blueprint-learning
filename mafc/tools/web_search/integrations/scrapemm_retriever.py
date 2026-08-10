import asyncio
import atexit
import base64
import io
import logging
import threading
from collections.abc import Awaitable
from ezmm import MultimodalSequence
from typing import Any, cast
import pypdf

from mafc.common.logger import logger
from mafc.tools.web_search.integrations.integration import RetrievalIntegration

# Base64 encoding of the PDF magic bytes "%PDF"
_PDF_BASE64_PREFIX = "JVBER"
_PDF_RAW_PREFIX = "%PDF"


def _try_extract_pdf_text(content: str) -> str | None:
    """If content is a PDF (base64-encoded or raw binary), extract plain text.

    Returns the extracted text, or None if not a PDF or extraction fails.
    """
    try:
        if content.startswith(_PDF_BASE64_PREFIX):
            pdf_bytes = base64.b64decode(content)
        elif content.startswith(_PDF_RAW_PREFIX):
            pdf_bytes = content.encode("latin-1")
        else:
            return None
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p for p in pages if p.strip())
        return text or None
    except Exception:
        return None


def _decode_pdf_blocks(content: MultimodalSequence) -> MultimodalSequence:
    """Replace any PDF text blocks (base64 or raw) with extracted plain text.

    If a block is detected as a PDF but text extraction fails (e.g. scanned
    image-only PDF), it is dropped rather than passed as raw binary, which
    would be unusable and blow up downstream token budgets.
    """
    blocks = []
    for block in content.to_list():
        if isinstance(block, str) and (
            block.startswith(_PDF_BASE64_PREFIX) or block.startswith(_PDF_RAW_PREFIX)
        ):
            extracted = _try_extract_pdf_text(block)
            if extracted:
                blocks.append(extracted)
            # else: drop — never propagate raw binary
        else:
            blocks.append(block)
    return MultimodalSequence(*blocks)


_stdin_prompts_disabled = False


def _disable_stdin_prompts() -> None:
    """Make scrapeMM fail instead of prompting when it can't find Firecrawl.

    `locate_firecrawl` loops on a bare `input()` whenever no Firecrawl instance
    answers — which happens whenever the VPN is down, since the configured
    instance is university-internal. That `input()` runs on the shared event
    loop thread, so on a TTY it blocks every retrieval in the process and the
    `asyncio.wait_for` timeout below can never fire, because the loop that would
    run the timeout callback is itself stuck. Shadowing `input` in that module
    makes it raise EOFError instead, which scrapemm already handles as a failed
    method and falls through to the next one — the same behaviour we get for
    free when stdin isn't a TTY (pytest, piped output).
    """
    global _stdin_prompts_disabled
    if _stdin_prompts_disabled:
        return
    try:
        from scrapemm.integrations.firecrawl import firecrawl

        def _no_stdin(prompt: str = "") -> str:
            raise EOFError("scrapeMM asked for interactive input (is the VPN up?)")

        firecrawl.input = _no_stdin  # type: ignore[attr-defined]
        _stdin_prompts_disabled = True
    except Exception as e:  # upstream moved the prompt; not worth failing over
        logger.warning(f"[ScrapeMMRetriever] could not disable scrapeMM stdin prompts: {e}")


def _retrieve_url(url: str) -> Awaitable[Any]:
    from scrapemm import retrieve  # needs to be lazy import because of runner tests

    logging.getLogger("scrapeMM").setLevel(logging.WARNING)  # scrapemm resets its logger to DEBUG on import
    _disable_stdin_prompts()

    return cast(Awaitable[Any], retrieve(url, show_progress=False))


class ScrapeMMRetriever(RetrievalIntegration):
    """Integration for the ScrapMM API, which retrieves the contents of a webpage
    given its URL. It is used as a fallback when the Google Search API only
    returns the URL but not the content of a source."""

    domains = ["*"]  # can retrieve from any domain

    # One event loop shared across ALL instances and threads. Multiple independent
    # event loops doing concurrent SSL (aiohttp/Decodo) from separate threads
    # causes segfaults on macOS and Linux / Python 3.13. A single loop serialises
    # all async I/O into one thread, which is inherently thread-safe.
    _shared_loop: asyncio.AbstractEventLoop | None = None
    _shared_loop_thread: threading.Thread | None = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self, timeout_seconds: float = 30.0, n_workers: int = 8):
        super().__init__(n_workers=n_workers)
        self.timeout_seconds = timeout_seconds
        with ScrapeMMRetriever._class_lock:
            if ScrapeMMRetriever._shared_loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=loop.run_forever, daemon=True, name="scrapemm-event-loop")
                thread.start()
                ScrapeMMRetriever._shared_loop = loop
                ScrapeMMRetriever._shared_loop_thread = thread
                atexit.register(ScrapeMMRetriever._shutdown_shared_loop)
        self._loop = ScrapeMMRetriever._shared_loop
        self._loop_thread = ScrapeMMRetriever._shared_loop_thread

    @classmethod
    def _shutdown_shared_loop(cls) -> None:
        with cls._class_lock:
            loop = cls._shared_loop
            thread = cls._shared_loop_thread
            cls._shared_loop = None
            cls._shared_loop_thread = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5.0)

    def _run_retrieve(self, url: str) -> Any:
        coro = asyncio.wait_for(_retrieve_url(url), timeout=self.timeout_seconds)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=self.timeout_seconds + 1.0)
        except TimeoutError as e:
            # Both deadlines raise a bare TimeoutError whose message is the empty string,
            # yet they mean very different things -- so say which one fired. A future that
            # is done means wait_for expired inside the loop: this one URL is slow. A
            # future that is not means the loop never got round to running wait_for's
            # timeout at all, i.e. it is blocked rather than slow. Since every retrieval
            # shares this one loop, that second case stalls the whole process.
            if future.done():
                raise TimeoutError(f"no response within {self.timeout_seconds}s") from e
            raise TimeoutError(
                f"shared event loop unresponsive for {self.timeout_seconds + 1.0}s -- "
                f"retrieval is stalled process-wide, not just for this URL"
            ) from e

    def _retrieve(self, url: str) -> MultimodalSequence | None:
        try:
            result = self._run_retrieve(url)
            if result.successful:
                logger.debug(f"[ScrapeMMRetriever] ✅ Successfully retrieved content from {url} with ScrapMM")
                return _decode_pdf_blocks(MultimodalSequence(result.content))
            else:
                logger.warning(
                    f"[ScrapeMMRetriever] ⚠️ Failed to retrieve content from {url} with ScrapMM: {result.errors}"
                )
                return None

        except Exception as e:
            logger.error(
                f"[ScrapeMMRetriever] ❌ Failed to retrieve content from {url} with ScrapMM: "
                f"{type(e).__name__}: {e}"
            )
            return None
