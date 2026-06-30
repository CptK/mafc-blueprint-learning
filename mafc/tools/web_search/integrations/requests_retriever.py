import io
import re

import pypdf
import requests
from bs4 import BeautifulSoup
from ezmm import MultimodalSequence

from mafc.common.logger import logger
from mafc.tools.web_search.integrations.integration import RetrievalIntegration

# A real browser User-Agent. Many sites return 403/anti-bot pages to the default
# python-requests UA but serve normally to a browser-looking one.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Discard non-script/style/chrome elements before extracting visible text.
_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]

# Below this many characters of extracted text the page is treated as a failure
# (e.g. a JS-only shell or a login/anti-bot wall) so the caller falls back to a
# heavier retriever that can render JS.
_MIN_CHARS = 500

# Guard against pathological pages blowing up the downstream token budget.
_MAX_CHARS = 50_000


def _extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_pdf_text(content: bytes) -> str | None:
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
        return text or None
    except Exception:
        return None


class RequestsRetriever(RetrievalIntegration):
    """Lightweight retriever that fetches a URL with a plain HTTP request and a
    browser User-Agent, then extracts readable text (HTML via BeautifulSoup, PDFs
    via pypdf).

    Cheap and fast, and succeeds on the majority of static news/government/article
    pages that the heavier scraper backend fails on (anti-bot/timeout). It cannot
    render JavaScript or pass login/anti-bot walls — those yield little text and
    return ``None`` so a fallback retriever can take over.
    """

    domains = ["*"]

    def __init__(self, timeout_seconds: float = 15.0, n_workers: int = 8, min_chars: int = _MIN_CHARS):
        super().__init__(n_workers=n_workers)
        self.timeout_seconds = timeout_seconds
        self.min_chars = min_chars

    def _retrieve(self, url: str) -> MultimodalSequence | None:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout_seconds)
        except Exception as e:
            logger.debug(f"[RequestsRetriever] request failed for {url}: {e}")
            return None

        if resp.status_code != 200:
            logger.debug(f"[RequestsRetriever] {resp.status_code} for {url}")
            return None

        content_type = resp.headers.get("Content-Type", "").lower()
        is_pdf = "application/pdf" in content_type or resp.content[:4] == b"%PDF"
        text = _extract_pdf_text(resp.content) if is_pdf else _extract_html_text(resp.text)

        if not text or len(text) < self.min_chars:
            logger.debug(f"[RequestsRetriever] too little content ({len(text or '')} chars) for {url}")
            return None

        logger.debug(f"[RequestsRetriever] ✅ retrieved {len(text)} chars from {url}")
        return MultimodalSequence(text[:_MAX_CHARS])
