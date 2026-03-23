import asyncio
import base64
import io
import logging
import threading
from collections.abc import Awaitable
from ezmm import MultimodalSequence
from scrapemm import retrieve
from typing import Any, cast
import pypdf

from mafc.common.logger import logger
from mafc.tools.web_search.integrations.integration import RetrievalIntegration

logging.getLogger("scrapeMM").setLevel(logging.WARNING)  # scrapemm resets its logger to DEBUG on import

# Base64 encoding of the PDF magic bytes "%PDF"
_PDF_BASE64_PREFIX = "JVBER"


def _try_extract_pdf_text(content: str) -> str | None:
    """If content is a base64-encoded PDF, decode it and extract plain text.

    Returns the extracted text, or None if the content is not a base64 PDF
    or extraction fails.
    """
    if not content.startswith(_PDF_BASE64_PREFIX):
        return None
    try:
        pdf_bytes = base64.b64decode(content)
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p for p in pages if p.strip())
        return text or None
    except Exception:
        return None


def _decode_pdf_blocks(content: MultimodalSequence) -> MultimodalSequence:
    """Replace any base64-encoded PDF text blocks with extracted plain text."""
    blocks = []
    for block in content.to_list():
        if isinstance(block, str):
            blocks.append(_try_extract_pdf_text(block) or block)
        else:
            blocks.append(block)
    return MultimodalSequence(*blocks)


def _retrieve_url(url: str) -> Awaitable[Any]:
    return cast(Awaitable[Any], retrieve(url, show_progress=False))


class ScrapeMMRetriever(RetrievalIntegration):
    """Integration for the ScrapMM API, which retrieves the contents of a webpage
    given its URL. It is used as a fallback when the Google Search API only
    returns the URL but not the content of a source."""

    domains = ["*"]  # can retrieve from any domain

    # One event loop shared across ALL instances and threads. Multiple independent
    # event loops doing concurrent SSL (aiohttp/Decodo) from separate threads
    # causes segfaults on macOS / Python 3.13. A single loop serialises all
    # async I/O into one thread, which is inherently thread-safe.
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
        self._loop = ScrapeMMRetriever._shared_loop
        self._loop_thread = ScrapeMMRetriever._shared_loop_thread

    def _run_retrieve(self, url: str) -> Any:
        coro = asyncio.wait_for(_retrieve_url(url), timeout=self.timeout_seconds)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.timeout_seconds + 1.0)

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
            logger.error(f"[ScrapeMMRetriever] ❌ Failed to retrieve content from {url} with ScrapMM: {e}")
            return None
