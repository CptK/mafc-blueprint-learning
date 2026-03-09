from mafc.tools.web_search.common import Query
from mafc.tools.web_search.google_search import GoogleSearchPlatform


def test_call_api_uses_google_vision_for_image_queries(monkeypatch) -> None:
    query = Query(text="x", image=object())
    sentinel = object()

    monkeypatch.setattr("mafc.tools.web_search.google_search.google_vision_api.search", lambda q: sentinel)
    monkeypatch.setattr("mafc.tools.web_search.google_search.serper_api.search", lambda q: None)

    platform = GoogleSearchPlatform(enable_ris=True, activate_cache=False)

    assert platform._call_api(query) is sentinel


def test_call_api_uses_serper_for_text_queries_or_when_ris_disabled(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr("mafc.tools.web_search.google_search.serper_api.search", lambda q: sentinel)
    monkeypatch.setattr("mafc.tools.web_search.google_search.google_vision_api.search", lambda q: None)

    platform = GoogleSearchPlatform(enable_ris=True, activate_cache=False)
    assert platform._call_api(Query(text="hello")) is sentinel

    platform_disabled = GoogleSearchPlatform(enable_ris=False, activate_cache=False)
    assert platform_disabled._call_api(Query(text="x", image=object())) is sentinel
