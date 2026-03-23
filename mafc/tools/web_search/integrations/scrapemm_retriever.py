import asyncio
import logging
import threading
from collections.abc import Awaitable
from ezmm import MultimodalSequence
from typing import Any, cast

from mafc.common.logger import logger
from mafc.tools.web_search.integrations.integration import RetrievalIntegration

logging.getLogger("scrapeMM").setLevel(logging.WARNING)
logging.getLogger("scrapeMM").propagate = False
logging.getLogger("firecrawl").setLevel(logging.WARNING)
logging.getLogger("firecrawl").propagate = False


def _retrieve_url(url: str) -> Awaitable[Any]:
    from scrapemm import retrieve  # lazy import to allow tests to pass without configuing secrets

    return cast(Awaitable[Any], retrieve(url, show_progress=False))


class ScrapeMMRetriever(RetrievalIntegration):
    """Integration for the ScrapMM API, which retrieves the contents of a webpage
    given its URL. It is used as a fallback when the Google Search API only
    returns the URL but not the content of a source."""

    domains = ["*"]  # can retrieve from any domain

    def __init__(self, timeout_seconds: float = 30.0, n_workers: int = 8):
        super().__init__(n_workers=n_workers)
        self.timeout_seconds = timeout_seconds
        # A single persistent event loop running in a background daemon thread.
        # Using one loop per instance avoids the semaphore leaks and segfaults
        # that occur when asyncio.run() repeatedly creates/destroys event loops
        # from many ThreadPoolExecutor threads simultaneously.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="scrapemm-event-loop"
        )
        self._loop_thread.start()

    def _run_retrieve(self, url: str) -> Any:
        coro = asyncio.wait_for(_retrieve_url(url), timeout=self.timeout_seconds)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.timeout_seconds + 1.0)

    def _retrieve(self, url: str) -> MultimodalSequence | None:
        try:
            result = self._run_retrieve(url)
            if result.successful:
                logger.debug(f"[ScrapeMMRetriever] ✅ Successfully retrieved content from {url} with ScrapMM")
                return MultimodalSequence(result.content)
            else:
                logger.warning(
                    f"[ScrapeMMRetriever] ⚠️ Failed to retrieve content from {url} with ScrapMM: {result.errors}"
                )
                return None

        except Exception as e:
            logger.error(f"[ScrapeMMRetriever] ❌ Failed to retrieve content from {url} with ScrapMM: {e}")
            return None
