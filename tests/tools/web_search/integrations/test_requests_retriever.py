from unittest.mock import patch

from mafc.tools.web_search.integrations.requests_retriever import RequestsRetriever


class _Resp:
    def __init__(self, status=200, text="", content=b"", content_type="text/html"):
        self.status_code = status
        self.text = text
        self.content = content or text.encode()
        self.headers = {"Content-Type": content_type}


def _get(resp):
    return patch("mafc.tools.web_search.integrations.requests_retriever.requests.get", return_value=resp)


def test_extracts_html_text() -> None:
    html = "<html><body><p>" + ("Real article content. " * 60) + "</p><script>junk()</script></body></html>"
    with _get(_Resp(text=html)):
        out = RequestsRetriever().retrieve("https://example.com/a")
    assert out is not None
    assert "Real article content." in str(out)
    assert "junk()" not in str(out)  # scripts stripped


def test_thin_page_returns_none() -> None:
    with _get(_Resp(text="<html><body>hi</body></html>")):
        out = RequestsRetriever(min_chars=500).retrieve("https://example.com/thin")
    assert out is None


def test_non_200_returns_none() -> None:
    with _get(_Resp(status=403, text="x" * 1000)):
        assert RequestsRetriever().retrieve("https://example.com/blocked") is None


def test_request_exception_returns_none() -> None:
    with patch(
        "mafc.tools.web_search.integrations.requests_retriever.requests.get",
        side_effect=RuntimeError("conn reset"),
    ):
        assert RequestsRetriever().retrieve("https://example.com/err") is None
