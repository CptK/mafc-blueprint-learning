import asyncio
import base64
from unittest.mock import MagicMock, patch

from mafc.tools.web_search.integrations.scrapemm_retriever import (
    ScrapeMMRetriever,
    _PDF_BASE64_PREFIX,
    _decode_pdf_blocks,
    _try_extract_pdf_text,
)


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


# --- _try_extract_pdf_text ---


def test_try_extract_pdf_text_returns_none_for_plain_text() -> None:
    assert _try_extract_pdf_text("hello world") is None


def test_try_extract_pdf_text_returns_none_for_empty() -> None:
    assert _try_extract_pdf_text("") is None


def _fake_pdf_b64() -> str:
    """Valid base64 string that starts with the PDF prefix (encodes b'%PDF-fake')."""
    return base64.b64encode(b"%PDF-fake").decode()


def test_try_extract_pdf_text_extracts_text() -> None:
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Extracted page text"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch(
        "mafc.tools.web_search.integrations.scrapemm_retriever.pypdf.PdfReader", return_value=fake_reader
    ):
        result = _try_extract_pdf_text(_fake_pdf_b64())

    assert result == "Extracted page text"


def test_try_extract_pdf_text_joins_multiple_pages() -> None:
    pages = [MagicMock(), MagicMock()]
    pages[0].extract_text.return_value = "Page one"
    pages[1].extract_text.return_value = "Page two"
    fake_reader = MagicMock()
    fake_reader.pages = pages

    with patch(
        "mafc.tools.web_search.integrations.scrapemm_retriever.pypdf.PdfReader", return_value=fake_reader
    ):
        result = _try_extract_pdf_text(_fake_pdf_b64())

    assert result == "Page one\n\nPage two"


def test_try_extract_pdf_text_returns_none_when_all_pages_empty() -> None:
    fake_page = MagicMock()
    fake_page.extract_text.return_value = ""
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch(
        "mafc.tools.web_search.integrations.scrapemm_retriever.pypdf.PdfReader", return_value=fake_reader
    ):
        result = _try_extract_pdf_text(_fake_pdf_b64())

    assert result is None


def test_try_extract_pdf_text_returns_none_on_invalid_base64() -> None:
    assert _try_extract_pdf_text(_PDF_BASE64_PREFIX + "!!!not_valid!!!") is None


def test_try_extract_pdf_text_returns_none_on_pypdf_error() -> None:
    with patch(
        "mafc.tools.web_search.integrations.scrapemm_retriever.pypdf.PdfReader",
        side_effect=Exception("corrupt PDF"),
    ):
        result = _try_extract_pdf_text(_fake_pdf_b64())

    assert result is None


def test_decode_pdf_blocks_replaces_pdf_text_block() -> None:
    from ezmm import MultimodalSequence

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "PDF page content"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    seq = MultimodalSequence(_fake_pdf_b64())
    with patch(
        "mafc.tools.web_search.integrations.scrapemm_retriever.pypdf.PdfReader", return_value=fake_reader
    ):
        result = _decode_pdf_blocks(seq)

    assert str(result) == "PDF page content"


def test_decode_pdf_blocks_leaves_plain_text_unchanged() -> None:
    from ezmm import MultimodalSequence

    seq = MultimodalSequence("just normal text")
    result = _decode_pdf_blocks(seq)
    assert str(result) == "just normal text"


def test_decode_pdf_blocks_preserves_images_and_non_pdf_text() -> None:
    from ezmm import Image, MultimodalSequence

    fake_image = MagicMock(spec=Image)
    seq = MultimodalSequence("some text", fake_image, "more text")
    result = _decode_pdf_blocks(seq)

    blocks = result.to_list()
    assert blocks[0] == "some text"
    assert blocks[1] is fake_image
    assert blocks[2] == "more text"


def test_scrapemm_retriever_extracts_pdf_content(monkeypatch) -> None:
    url = "https://example.com/document.pdf"
    retriever = ScrapeMMRetriever()
    fake_pdf_content = _fake_pdf_b64()

    async def fake_retrieve(given_url, show_progress=False):
        return FakeScrapingResponse(successful=True, content=fake_pdf_content)

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "PDF page content"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    monkeypatch.setattr("mafc.tools.web_search.integrations.scrapemm_retriever._retrieve_url", fake_retrieve)

    with patch(
        "mafc.tools.web_search.integrations.scrapemm_retriever.pypdf.PdfReader", return_value=fake_reader
    ):
        out = retriever.retrieve(url)

    assert out is not None
    assert str(out) == "PDF page content"
