import asyncio

from mafc.tools.web_search.integrations.scrapemm_retriever import ScrapeMMRetriever


class FakeScrapingResponse:
    def __init__(self, successful: bool, content: str = "", errors=None):
        self.successful = successful
        self.content = content
        self.errors = [] if errors is None else errors


def test_scrapemm_retriever_success(monkeypatch) -> None:
    url = "https://example.com/article"
    retriever = ScrapeMMRetriever()
    calls = {"n": 0}

    async def fake_retrieve(given_url, show_progress=False):
        calls["n"] += 1
        assert given_url == url
        assert show_progress is False
        return FakeScrapingResponse(successful=True, content="hello world")

    monkeypatch.setattr("mafc.tools.web_search.integrations.scrapemm_retriever._retrieve_url", fake_retrieve)

    out = retriever.retrieve(url)
    assert out is not None
    assert str(out) == "hello world"
    assert calls["n"] == 1


def test_scrapemm_retriever_failure_response(monkeypatch) -> None:
    retriever = ScrapeMMRetriever()

    async def fake_retrieve(*args, **kwargs):
        return FakeScrapingResponse(successful=False, errors=["unreachable"])

    monkeypatch.setattr("mafc.tools.web_search.integrations.scrapemm_retriever._retrieve_url", fake_retrieve)

    out = retriever.retrieve("https://example.com/fail")
    assert out is None


def test_scrapemm_retriever_handles_exception(monkeypatch) -> None:
    retriever = ScrapeMMRetriever()

    async def fake_retrieve(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("mafc.tools.web_search.integrations.scrapemm_retriever._retrieve_url", fake_retrieve)

    out = retriever.retrieve("https://example.com/error")
    assert out is None


def test_scrapemm_retriever_uses_cache(monkeypatch) -> None:
    url = "https://example.com/cached"
    retriever = ScrapeMMRetriever()
    calls = {"n": 0}

    async def fake_retrieve(*args, **kwargs):
        calls["n"] += 1
        return FakeScrapingResponse(successful=True, content="cached-content")

    monkeypatch.setattr("mafc.tools.web_search.integrations.scrapemm_retriever._retrieve_url", fake_retrieve)

    out1 = retriever.retrieve(url)
    out2 = retriever.retrieve(url)
    assert out1 is not None and out2 is not None
    assert str(out1) == "cached-content"
    assert str(out2) == "cached-content"
    assert calls["n"] == 1


def test_scrapemm_retriever_times_out(monkeypatch) -> None:
    retriever = ScrapeMMRetriever(timeout_seconds=0.01)

    async def fake_retrieve(*args, **kwargs):
        await asyncio.sleep(0.1)
        return FakeScrapingResponse(successful=True, content="late")

    monkeypatch.setattr("mafc.tools.web_search.integrations.scrapemm_retriever._retrieve_url", fake_retrieve)

    out = retriever.retrieve("https://example.com/slow")
    assert out is None
