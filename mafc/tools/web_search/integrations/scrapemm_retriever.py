from ezmm import MultimodalSequence
from scrapemm import retrieve
from scrapemm.common.scraping_response import ScrapingResponse
from typing import cast

from mafc.common.logger import logger
from mafc.tools.web_search.integrations.integration import RetrievalIntegration


class ScrapeMMRetriever(RetrievalIntegration):
    """Integration for the ScrapMM API, which retrieves the contents of a webpage
    given its URL. It is used as a fallback when the Google Search API only
    returns the URL but not the content of a source."""

    domains = ["*"]  # can retrieve from any domain

    def _retrieve(self, url: str) -> MultimodalSequence | None:
        try:
            result = cast(ScrapingResponse, retrieve(url, show_progress=False))
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
