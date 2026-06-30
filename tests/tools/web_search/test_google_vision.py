from types import SimpleNamespace

import pytest

google_vision = pytest.importorskip("mafc.tools.web_search.google_vision")
Query = google_vision.Query


def test_google_ris_results_string_and_repr() -> None:
    results = google_vision.GoogleRisResults(
        sources=[],
        query=Query(text="q", media=object()),
        entities={"Mountain": 0.91},
        best_guess_labels=["Alps"],
    )
    text = str(results)
    assert "Reverse Image Search Results" in text
    assert "Mountain" in text
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

    out = api.search(Query(text="x", media=object()))

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

    out = api.search(Query(text="x", media=object()))

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

    query = Query(text="x", media=FakeImage())
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


def test_merge_ris_results_dedupes_sources_and_keeps_max_score() -> None:
    from mafc.tools.web_search.common import WebSource

    q = Query(text="q", media=object())
    r1 = google_vision.GoogleRisResults(
        sources=[WebSource(reference="https://a.com")],
        query=q,
        entities={"X": 0.5},
        best_guess_labels=["L1"],
    )
    r2 = google_vision.GoogleRisResults(
        sources=[WebSource(reference="https://a.com"), WebSource(reference="https://b.com")],
        query=q,
        entities={"X": 0.9, "Y": 0.3},
        best_guess_labels=["L1", "L2"],
    )

    merged = google_vision._merge_ris_results([r1, r2], q)

    assert [s.url for s in merged.sources] == ["https://a.com", "https://b.com"]
    assert merged.entities == {"X": 0.9, "Y": 0.3}  # max score kept, sorted desc
    assert merged.best_guess_labels == ["L1", "L2"]


def test_search_samples_multiple_frames_for_video(monkeypatch) -> None:
    class FakeImage:
        pass

    class FakeVideo:
        def __init__(self):
            self.requested = None

        def sample_frames(self, n, format="jpeg"):
            self.requested = n
            return [f"frame{i}".encode() for i in range(n)]

    calls = {"n": 0}

    class FakeClient:
        def web_detection(self, image):
            calls["n"] += 1
            return SimpleNamespace(
                error=SimpleNamespace(message=""),
                web_detection=SimpleNamespace(
                    web_entities=[], best_guess_labels=[], pages_with_matching_images=[]
                ),
            )

    monkeypatch.setattr("mafc.tools.web_search.google_vision.Image", FakeImage)
    monkeypatch.setattr("mafc.tools.web_search.google_vision.Video", FakeVideo)
    monkeypatch.setattr("mafc.tools.web_search.google_vision.vision.Image", lambda content: content)

    vid = FakeVideo()
    api = google_vision.GoogleVisionAPI()
    api.client = FakeClient()
    api.search(Query(text="x", media=vid))

    assert vid.requested == google_vision._VIDEO_RIS_FRAMES
    assert calls["n"] == google_vision._VIDEO_RIS_FRAMES  # one detection per keyframe


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

    result = google_vision._parse_results(web_detection, Query(text="q", media=object()))

    assert result.entities == {"Lake": 0.8}
    assert result.best_guess_labels == ["Landscape"]
    assert [source.reference for source in result.sources] == [
        "https://www.example.com/a",
        "https://another.org/c",
    ]
    assert result.sources[0].title == "A"
