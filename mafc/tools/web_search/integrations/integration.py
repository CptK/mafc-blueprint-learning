from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from ezmm import MultimodalSequence
import threading

from mafc.common.logger import logger
from mafc.utils.parsing import get_domain


class RetrievalIntegration(ABC):
    """An integration (external API or similar) that is able to retrieve the contents
    for a given URL belonging to a specific domain. Maintains a cache."""

    domains: list[str]

    def __init__(self, n_workers: int = 8):
        self.cache: dict[str, MultimodalSequence | None] = {}
        self._cache_lock = threading.Lock()
        self.n_workers = n_workers

    def retrieve(self, url: str) -> MultimodalSequence | None:
        """Returns the contents at the URL."""
        domain = get_domain(url)
        if "*" not in self.domains and domain not in self.domains:
            logger.error(
                f"[{type(self).__name__}] Domain '{domain}' is not allowed. "
                f"Allowed domains: {self.domains}"
            )
            return None

        with self._cache_lock:
            if url in self.cache:
                return self.cache[url]

        result = self._retrieve(url)
        with self._cache_lock:
            # Keep first completed result if multiple threads race on same URL.
            if url not in self.cache:
                self.cache[url] = result
            return self.cache[url]

    def retrieve_batch(self, urls: list[str]) -> list[MultimodalSequence | None]:
        """Retrieves the contents for a batch of URLs in parallel."""
        if not urls:
            return []

        unique_urls = list(dict.fromkeys(urls))
        max_workers = min(self.n_workers, len(unique_urls))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            unique_results = list(pool.map(self.retrieve, unique_urls))

        by_url = dict(zip(unique_urls, unique_results))
        return [by_url[url] for url in urls]

    @abstractmethod
    def _retrieve(self, url: str) -> MultimodalSequence | None:
        pass
