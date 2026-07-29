from pathlib import Path

import numpy as np
import pytest

from mafc.tools.media.trufor import (
    DetectTruForManipulation,
    ManipulationDetectionResults,
    ScoreRecord,
    ScoreStore,
    TruFor,
    file_sha256,
)
from mafc.tools.media.trufor.inference import TruForPrediction


class _FakeEngine:
    """Stands in for TruForModel so the tests never load the checkpoint."""

    def __init__(self, score: float = 0.8):
        self.score = score
        self.calls: list[Path] = []

    def predict_image(self, path, return_maps: bool = False) -> TruForPrediction:
        self.calls.append(Path(path))
        return TruForPrediction(
            score=self.score,
            localization_map=np.zeros((2, 2), dtype=np.float32) if return_maps else None,
            confidence_map=np.ones((2, 2), dtype=np.float32) if return_maps else None,
            image_size=(2, 2),
        )

    def predict_array(self, rgb, return_maps: bool = False) -> TruForPrediction:
        return TruForPrediction(score=self.score, image_size=rgb.shape[:2])


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    from PIL import Image as PILImage

    path = tmp_path / "img.png"
    PILImage.new("RGB", (4, 4), color=(10, 20, 30)).save(path)
    return path


def _tool(tmp_path: Path, engine: _FakeEngine, **kwargs) -> TruFor:
    tool = TruFor(stores=[], cache_dir=tmp_path / "cache", **kwargs)
    tool._engine = engine  # type: ignore[assignment]
    return tool


# --- store -------------------------------------------------------------------


def test_store_roundtrip(tmp_path: Path) -> None:
    store = ScoreStore(tmp_path / "store")
    store.put("abc", ScoreRecord(score=0.42, source_name="x.jpg", image_size=[10, 20]))
    store.save()

    reloaded = ScoreStore(tmp_path / "store")
    record = reloaded.get("abc")
    assert record is not None
    assert record.score == pytest.approx(0.42)
    assert record.image_size == [10, 20]
    assert record.created  # timestamp filled in automatically


def test_store_missing_index_is_empty(tmp_path: Path) -> None:
    assert len(ScoreStore(tmp_path / "nope")) == 0


def test_store_survives_corrupt_index(tmp_path: Path) -> None:
    path = tmp_path / "store"
    path.mkdir()
    (path / "index.json").write_text("{not json")
    assert len(ScoreStore(path)) == 0


def test_store_maps_roundtrip(tmp_path: Path) -> None:
    store = ScoreStore(tmp_path / "store")
    loc = np.arange(6, dtype=np.float32).reshape(2, 3)
    store.save_maps("key", loc, None)

    maps = store.load_maps("key")
    assert maps is not None
    assert np.array_equal(maps["map"], loc)
    assert "conf" not in maps
    assert store.load_maps("absent") is None


def test_file_sha256_is_content_based(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same"), b.write_bytes(b"same")
    assert file_sha256(a) == file_sha256(b)


# --- scoring and caching -----------------------------------------------------


def test_score_image_computes_then_caches(tmp_path: Path, image_file: Path) -> None:
    engine = _FakeEngine(score=0.8)
    tool = _tool(tmp_path, engine)

    first = tool.score_image(image_file)
    assert first.score == pytest.approx(0.8)
    assert first.is_manipulated is True
    assert first.from_cache is False

    second = tool.score_image(image_file)
    assert second.from_cache is True
    assert second.score == pytest.approx(0.8)
    assert len(engine.calls) == 1, "second call must be served from the cache"


def test_cache_survives_new_tool_instance(tmp_path: Path, image_file: Path) -> None:
    engine = _FakeEngine(score=0.3)
    _tool(tmp_path, engine).score_image(image_file)

    fresh_engine = _FakeEngine(score=0.9)
    result = _tool(tmp_path, fresh_engine).score_image(image_file)

    assert result.score == pytest.approx(0.3), "should come from the store, not be recomputed"
    assert fresh_engine.calls == []


def test_cache_is_keyed_on_content_not_path(tmp_path: Path, image_file: Path) -> None:
    engine = _FakeEngine(score=0.55)
    tool = _tool(tmp_path, engine)
    tool.score_image(image_file)

    copy = tmp_path / "renamed.png"
    copy.write_bytes(image_file.read_bytes())

    assert tool.score_image(copy).from_cache is True
    assert len(engine.calls) == 1


def test_read_only_store_takes_precedence(tmp_path: Path, image_file: Path) -> None:
    precomputed = ScoreStore(tmp_path / "dataset")
    precomputed.put(file_sha256(image_file), ScoreRecord(score=0.11, source_name=image_file.name))
    precomputed.save()

    engine = _FakeEngine(score=0.99)
    tool = TruFor(stores=[tmp_path / "dataset"], cache_dir=tmp_path / "cache")
    tool._engine = engine  # type: ignore[assignment]

    result = tool.score_image(image_file)
    assert result.score == pytest.approx(0.11)
    assert engine.calls == []


def test_use_cache_false_writes_nothing(tmp_path: Path, image_file: Path) -> None:
    engine = _FakeEngine(score=0.7)
    tool = TruFor(stores=[], use_cache=False)
    tool._engine = engine  # type: ignore[assignment]

    tool.score_image(image_file)
    tool.score_image(image_file)

    assert tool.cache is None
    assert len(engine.calls) == 2


def test_threshold_controls_verdict(tmp_path: Path, image_file: Path) -> None:
    engine = _FakeEngine(score=0.6)
    assert _tool(tmp_path / "a", engine, threshold=0.5).score_image(image_file).is_manipulated is True
    assert (
        _tool(tmp_path / "b", _FakeEngine(0.6), threshold=0.7).score_image(image_file).is_manipulated is False
    )


# --- action / tool interface -------------------------------------------------


def test_perform_reports_missing_media(tmp_path: Path) -> None:
    tool = _tool(tmp_path, _FakeEngine())
    action = DetectTruForManipulation(media="<image:999999>")

    result = tool._perform(action)

    assert result.error is not None
    assert result.is_useful() is False
    assert "failed" in str(result).lower()


def test_perform_scores_registered_image(tmp_path: Path, image_file: Path) -> None:
    from ezmm.common.items import Image

    tool = _tool(tmp_path, _FakeEngine(score=0.25))
    item = Image(file_path=str(image_file))

    result = tool._perform(DetectTruForManipulation(media=item.reference))

    assert result.error is None
    assert result.score == pytest.approx(0.25)
    assert result.is_manipulated is False


def test_summary_mentions_score_and_caveat(tmp_path: Path) -> None:
    tool = _tool(tmp_path, _FakeEngine())
    manipulated = ManipulationDetectionResults(score=0.9, is_manipulated=True)
    clean = ManipulationDetectionResults(score=0.1, is_manipulated=False)

    summary = str(tool._summarize(manipulated))
    assert "0.90" in summary and "likely manipulated" in summary
    assert "AI-generated" in summary, "the caveat about generated images must survive"
    assert "no evidence of manipulation" in str(tool._summarize(clean))

    assert (
        tool._summarize(ManipulationDetectionResults(score=float("nan"), is_manipulated=False, error="x"))
        is None
    )


def test_result_str_is_llm_readable() -> None:
    text = str(ManipulationDetectionResults(score=0.734, is_manipulated=True))
    assert "0.734" in text and "signs of manipulation" in text

    video_text = str(
        ManipulationDetectionResults(score=0.5, is_manipulated=True, n_frames=5, aggregation="median")
    )
    assert "5 sampled video frames" in video_text
    assert "median over frames" in video_text, "the summary must not claim max when median was used"


# --- video aggregation -------------------------------------------------------


def test_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match="video_aggregation"):
        TruFor(stores=[], use_cache=False, video_aggregation="loudest")


@pytest.mark.parametrize(
    "aggregation,expected",
    [("max", 0.9), ("median", 0.4), ("mean", pytest.approx((0.9 + 0.4 + 0.2) / 3))],
)
def test_cached_frame_scores_are_reaggregated(tmp_path: Path, aggregation: str, expected) -> None:
    """Changing the aggregation must not require rescoring the video."""
    store_dir = tmp_path / "store"
    store = ScoreStore(store_dir)
    store.put(
        "video-sha",
        ScoreRecord(score=0.9, source_name="v.mp4", n_frames=3, frame_scores=[0.9, 0.4, 0.2]),
    )
    store.save()

    tool = TruFor(stores=[store_dir], use_cache=False, video_aggregation=aggregation)
    record = store.get("video-sha")
    assert record is not None
    result = tool._result_from_record(record, ScoreStore(store_dir, writable=False), "video-sha")

    assert result.score == expected
    assert result.aggregation == aggregation


def test_image_records_are_not_reaggregated(tmp_path: Path, image_file: Path) -> None:
    """Images have no frame scores, so the stored score is used verbatim."""
    store_dir = tmp_path / "store"
    store = ScoreStore(store_dir)
    store.put(file_sha256(image_file), ScoreRecord(score=0.77, source_name=image_file.name))
    store.save()

    tool = TruFor(stores=[store_dir], use_cache=False, video_aggregation="median")
    result = tool.score_image(image_file)

    assert result.score == pytest.approx(0.77)
    assert result.aggregation is None


# --- precompute --------------------------------------------------------------


def test_precompute_skips_already_scored(tmp_path: Path, monkeypatch) -> None:
    from mafc.tools.media.trufor import precompute as precompute_module
    from PIL import Image as PILImage

    media_dir = tmp_path / "images"
    media_dir.mkdir()
    for i in range(3):
        PILImage.new("RGB", (4, 4), color=(i, i, i)).save(media_dir / f"{i}.png")

    engine = _FakeEngine(score=0.5)
    monkeypatch.setattr(precompute_module, "TruForModel", lambda **kwargs: engine)

    store_dir = tmp_path / "store"
    precompute_module.precompute(media_dir, store_dir=store_dir)
    assert len(ScoreStore(store_dir)) == 3
    assert len(engine.calls) == 3

    precompute_module.precompute(media_dir, store_dir=store_dir)
    assert len(engine.calls) == 3, "already-scored files must be skipped"


def test_default_store_dir_sits_beside_the_dataset(tmp_path: Path) -> None:
    from mafc.tools.media.trufor.precompute import default_store_dir

    dataset = tmp_path / "veritas"
    (dataset / "images").mkdir(parents=True)
    assert default_store_dir(dataset / "images") == dataset / "trufor"


def test_precompute_continues_after_a_bad_file(tmp_path: Path, monkeypatch) -> None:
    from mafc.tools.media.trufor import precompute as precompute_module
    from PIL import Image as PILImage

    media_dir = tmp_path / "images"
    media_dir.mkdir()
    PILImage.new("RGB", (4, 4)).save(media_dir / "ok.png")
    (media_dir / "broken.png").write_bytes(b"not an image")

    class _PickyEngine(_FakeEngine):
        def predict_image(self, path, return_maps: bool = False):
            if Path(path).name == "broken.png":
                raise OSError("cannot identify image file")
            return super().predict_image(path, return_maps)

    monkeypatch.setattr(precompute_module, "TruForModel", lambda **kwargs: _PickyEngine())
    store = precompute_module.precompute(media_dir, store_dir=tmp_path / "store")

    assert len(store) == 1, "the good file must still be scored"
