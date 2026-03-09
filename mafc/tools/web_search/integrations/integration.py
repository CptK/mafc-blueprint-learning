from abc import ABC, abstractmethod
from ezmm import MultimodalSequence

from mafc.common.logger import logger
from mafc.utils.parsing import get_domain


class RetrievalIntegration(ABC):
    """An integration (external API or similar) that is able to retrieve the contents
    for a given URL belonging to a specific domain. Maintains a cache."""

    domains: list[str]

    def __init__(self):
        self.cache = {}

    def retrieve(self, url: str) -> MultimodalSequence | None:
        """Returns the contents at the URL."""
        domain = get_domain(url)
        if "*" not in self.domains and domain not in self.domains:
            logger.error(
                f"[{type(self).__name__}] Domain '{domain}' is not allowed. "
                f"Allowed domains: {self.domains}"
            )
            return None
        if url in self.cache:
            return self.cache[url]
        else:
            result = self._retrieve(url)
            self.cache[url] = result
            return result

    @abstractmethod
    def _retrieve(self, url: str) -> MultimodalSequence | None:
        pass
