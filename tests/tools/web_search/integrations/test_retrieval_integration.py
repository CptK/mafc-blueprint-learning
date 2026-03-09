from ezmm import MultimodalSequence

from mafc.tools.web_search.integrations.integration import RetrievalIntegration


class WildcardRetriever(RetrievalIntegration):
    domains = ["*"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def _retrieve(self, url: str) -> MultimodalSequence | None:
        self.calls += 1
        return MultimodalSequence(f"content:{url}")


class RestrictedRetriever(RetrievalIntegration):
    domains = ["example.com"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def _retrieve(self, url: str) -> MultimodalSequence | None:
        self.calls += 1
        return MultimodalSequence(f"content:{url}")


def test_retrieve_allows_wildcard_domains_and_uses_cache() -> None:
    retriever = WildcardRetriever()
    url = "https://news.ycombinator.com/item?id=1"

    out1 = retriever.retrieve(url)
    out2 = retriever.retrieve(url)

    assert out1 is not None
    assert str(out1) == f"content:{url}"
    assert out2 is not None
    assert str(out2) == f"content:{url}"
    assert retriever.calls == 1


def test_retrieve_rejects_domains_not_in_allowlist() -> None:
    retriever = RestrictedRetriever()
    out = retriever.retrieve("https://wikipedia.org/wiki/Test")
    assert out is None
    assert retriever.calls == 0


def test_retrieve_batch_preserves_order_and_deduplicates() -> None:
    retriever = WildcardRetriever()
    urls = [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/a",
    ]

    out = retriever.retrieve_batch(urls)

    assert [str(item) if item is not None else None for item in out] == [
        "content:https://example.com/a",
        "content:https://example.com/b",
        "content:https://example.com/a",
    ]
    assert retriever.calls == 2


def test_retrieve_batch_applies_domain_filtering() -> None:
    retriever = RestrictedRetriever()
    out = retriever.retrieve_batch(
        [
            "https://example.com/allowed",
            "https://wikipedia.org/wiki/Test",
        ]
    )

    assert [str(item) if item is not None else None for item in out] == [
        "content:https://example.com/allowed",
        None,
    ]
    assert retriever.calls == 1
