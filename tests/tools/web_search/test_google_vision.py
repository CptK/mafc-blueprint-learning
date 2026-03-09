from types import SimpleNamespace

import pytest

google_vision = pytest.importorskip("mafc.tools.web_search.google_vision")
Query = google_vision.Query


def test_google_ris_results_string_and_repr() -> None:
    results = google_vision.GoogleRisResults(
        sources=[],
        query=Query(text="q", image=object()),
        entities={"Mountain": 0.91},
        best_guess_labels=["Alps"],
    )
    text = str(results)
    assert "Reverse Image Search Results" in text
    assert "Mountain (91 %)" in text
    assert "Alps" in text
    assert "n_entities=1" in repr(results)


def test_search_returns_empty_when_query_has_no_image() -> None:
    api = google_vision.GoogleVisionAPI()
    api.client = object()

    out = api.search(Query(text="x"))

    assert out.sources == []
    assert out.entities == {}
    assert out.best_guess_labels == []


def test_search_returns_empty_when_client_missing() -> None:
    api = google_vision.GoogleVisionAPI()
    api.client = None

    out = api.search(Query(text="x", image=object()))

    assert out.sources == []


def test_search_returns_empty_for_unsupported_media(monkeypatch) -> None:
    class FakeImage:
        pass

    class FakeVideo:
        pass

    monkeypatch.setattr("mafc.tools.web_search.google_vision.Image", FakeImage)
    monkeypatch.setattr("mafc.tools.web_search.google_vision.Video", FakeVideo)
    api = google_vision.GoogleVisionAPI()
    api.client = object()

    out = api.search(Query(text="x", image=object()))

    assert out.sources == []


def test_search_uses_parse_results_for_image(monkeypatch) -> None:
    class FakeImage:
        def get_base64_encoded(self):
            return b"img-bytes"

    class FakeVideo:
        pass

    class FakeClient:
        def web_detection(self, image):
            return SimpleNamespace(
                error=SimpleNamespace(message=""),
                web_detection=SimpleNamespace(
                    web_entities=[], best_guess_labels=[], pages_with_matching_images=[]
                ),
            )

    query = Query(text="x", image=FakeImage())
    api = google_vision.GoogleVisionAPI()
    api.client = FakeClient()

    monkeypatch.setattr("mafc.tools.web_search.google_vision.Image", FakeImage)
    monkeypatch.setattr("mafc.tools.web_search.google_vision.Video", FakeVideo)
    monkeypatch.setattr("mafc.tools.web_search.google_vision.vision.Image", lambda content: content)
    expected = google_vision.GoogleRisResults(
        sources=[],
        query=query,
        entities={"Sky": 0.5},
        best_guess_labels=["Clouds"],
    )
    monkeypatch.setattr("mafc.tools.web_search.google_vision._parse_results", lambda wd, q: expected)

    out = api.search(query)
    assert out is expected


def test_parse_results_and_filter_unique_pages() -> None:
    page1 = SimpleNamespace(url="https://www.example.com/a", page_title="A")
    page2 = SimpleNamespace(url="https://m.example.com/b", page_title="B")
    page3 = SimpleNamespace(url="https://another.org/c", page_title="C")
    web_detection = SimpleNamespace(
        web_entities=[
            SimpleNamespace(description="Lake", score=0.8),
            SimpleNamespace(description=None, score=0.1),
        ],
        best_guess_labels=[SimpleNamespace(label="Landscape"), SimpleNamespace(label=None)],
        pages_with_matching_images=[page1, page2, page3],
    )

    result = google_vision._parse_results(web_detection, Query(text="q", image=object()))

    assert result.entities == {"Lake": 0.8}
    assert result.best_guess_labels == ["Landscape"]
    assert [source.reference for source in result.sources] == [
        "https://www.example.com/a",
        "https://another.org/c",
    ]
    assert result.sources[0].title == "A"
