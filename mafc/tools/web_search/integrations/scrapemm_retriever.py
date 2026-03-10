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

    async def _retrieve_with_timeout(self, coro: Awaitable[Any]) -> Any:
        return await asyncio.wait_for(coro, timeout=self.timeout_seconds)

    def _run_retrieve(self, url: str) -> Any:
        from scrapemm.common.scraping_response import ScrapingResponse

        coro = _retrieve_url(url)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._retrieve_with_timeout(coro))

        result: ScrapingResponse | None = None
        error: Exception | None = None

        def runner() -> None:
            nonlocal result, error
            try:
                result = asyncio.run(self._retrieve_with_timeout(coro))
            except Exception as exc:
                error = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds + 1.0)

        if thread.is_alive():
            raise TimeoutError(f"ScrapeMM retrieval timed out after {self.timeout_seconds} seconds")

        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("ScrapeMM retrieval finished without a result")
        return result

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
