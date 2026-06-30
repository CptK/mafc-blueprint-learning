import pytest
from ezmm import MultimodalSequence

from mafc.tools.web_search.integrations.integration import RetrievalIntegration
from mafc.tools.web_search.integrations.fallback_retriever import FallbackRetriever


class _Stub(RetrievalIntegration):
    domains = ["*"]

    def __init__(self, result: str | None, raises: bool = False):
        super().__init__()
        self.result = result
        self.raises = raises
        self.calls = 0

    def _retrieve(self, url: str) -> MultimodalSequence | None:
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return MultimodalSequence(self.result) if self.result is not None else None


def test_returns_first_usable_and_skips_later() -> None:
    first = _Stub("from-first")
    second = _Stub("from-second")
    fb = FallbackRetriever([first, second])

    out = fb.retrieve("https://example.com/x")

    assert str(out) == "from-first"
    assert first.calls == 1
    assert second.calls == 0  # short-circuits once one succeeds


def test_falls_back_when_earlier_returns_none() -> None:
    first = _Stub(None)
    second = _Stub("from-second")
    fb = FallbackRetriever([first, second])

    out = fb.retrieve("https://example.com/x")

    assert str(out) == "from-second"
    assert first.calls == 1
    assert second.calls == 1


def test_exception_in_one_retriever_does_not_abort_chain() -> None:
    first = _Stub(None, raises=True)
    second = _Stub("from-second")
    fb = FallbackRetriever([first, second])

    out = fb.retrieve("https://example.com/x")

    assert str(out) == "from-second"


def test_returns_none_when_all_fail() -> None:
    fb = FallbackRetriever([_Stub(None), _Stub(None)])
    assert fb.retrieve("https://example.com/x") is None


def test_requires_at_least_one_retriever() -> None:
    with pytest.raises(ValueError):
        FallbackRetriever([])
