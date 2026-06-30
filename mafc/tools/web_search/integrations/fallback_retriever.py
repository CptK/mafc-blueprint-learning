from ezmm import MultimodalSequence

from mafc.common.logger import logger
from mafc.tools.web_search.integrations.integration import RetrievalIntegration


class FallbackRetriever(RetrievalIntegration):
    """Tries an ordered list of retrievers and returns the first usable result.

    Each sub-retriever returns ``None`` when it cannot retrieve usable content, so
    the next one is tried. The intended ordering is cheap-first: a plain HTTP
    retriever, falling back to a heavier JS-rendering scraper for the pages the
    cheap one cannot handle (login/anti-bot walls, JS-only shells).
    """

    domains = ["*"]

    def __init__(self, retrievers: list[RetrievalIntegration], n_workers: int = 8):
        super().__init__(n_workers=n_workers)
        if not retrievers:
            raise ValueError("FallbackRetriever requires at least one retriever.")
        self.retrievers = retrievers

    def _retrieve(self, url: str) -> MultimodalSequence | None:
        for retriever in self.retrievers:
            try:
                result = retriever.retrieve(url)
            except Exception as e:  # noqa: BLE001 — never let one backend abort the chain
                logger.debug(f"[FallbackRetriever] {type(retriever).__name__} raised for {url}: {e}")
                result = None
            if result is not None:
                return result
        return None
