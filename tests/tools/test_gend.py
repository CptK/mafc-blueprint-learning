"""GenD face-deepfake detection.

The fixtures are aligned face crops taken from the upstream GenD repository's
own sample data, with known labels — two synthetic, two real. They are fed
straight to the model, bypassing detect+align, because they are *already*
aligned; running the alignment a second time would warp them out of the
training distribution.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

import config.globals  # noqa: F401  # Triggers .env loading like normal app startup.
from ezmm.common.items import Image as EzImage
from mafc.tools.media.gend import DetectGenDDeepfake, GenDChecker, GenDDetector

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gend"

# Fixture -> is it synthetic. From upstream's directory layout: FF/DF and
# CDFv2/Celeb-synthesis are manipulated, FF/real and CDFv2/YouTube-real are not.
LABELLED_CROPS = [
    ("fake_ff_df.png", True),
    ("real_ff.png", False),
    ("fake_cdf.png", True),
    ("real_cdf.png", False),
]


@pytest.fixture(scope="module")
def detector() -> GenDDetector:
    return GenDDetector()


@pytest.mark.integration
def test_position_ids_are_not_corrupted(detector: GenDDetector) -> None:
    """transformers>=5 materialises `position_ids` from uninitialised memory when
    a model nests from_pretrained, which we repair on load. The corruption is
    dangerous precisely because in-range garbage does not raise — it silently
    gathers the wrong positional embeddings — so assert the buffer directly
    rather than relying on inference merely not crashing.
    """
    embeddings = detector.model.feature_extractor.vision_model.embeddings
    position_ids = embeddings.position_ids.flatten().tolist()
    assert position_ids == list(range(len(position_ids)))


@pytest.mark.integration
def test_separates_known_fakes_from_known_reals(detector: GenDDetector) -> None:
    """Both fake/real pairs must be ordered correctly. Asserting the ordering
    rather than an absolute threshold keeps this robust to checkpoint updates
    while still failing if the model is loaded wrong."""
    scores = {}
    for name, _ in LABELLED_CROPS:
        image = Image.open(FIXTURES / name).convert("RGB")
        with torch.no_grad():
            tensor = detector.model.feature_extractor.preprocess(image).unsqueeze(0)
            scores[name] = float(detector.model(tensor).softmax(dim=-1)[0, 1])

    assert scores["fake_ff_df.png"] > scores["real_ff.png"], scores
    assert scores["fake_cdf.png"] > scores["real_cdf.png"], scores


@pytest.mark.integration
def test_no_face_yields_no_score(detector: GenDDetector, tmp_path: Path) -> None:
    """An image with no face must return None, not 0.0. Collapsing the two would
    let 'nothing to judge' masquerade as 'judged, and it looked real'."""
    blank = tmp_path / "blank.png"
    Image.new("RGB", (512, 512), (128, 128, 128)).save(blank)

    prediction = detector.score_image(blank)

    assert prediction.n_faces == 0
    assert prediction.p_fake is None
    assert prediction.has_face is False


@pytest.mark.integration
def test_tool_reports_no_face_as_not_useful(tmp_path: Path) -> None:
    """The Tool wrapper must not present a faceless image as a real result."""
    blank = tmp_path / "blank.png"
    Image.new("RGB", (512, 512), (200, 180, 160)).save(blank)
    item = EzImage(file_path=str(blank))

    checker = GenDChecker(stores=[], use_cache=False)
    result = checker._perform(DetectGenDDeepfake(media=item.reference))

    assert result.error is None, str(result)
    assert result.p_fake is None
    assert result.is_deepfake is None
    assert result.is_useful() is False
    assert "no face" in str(result).lower()


@pytest.mark.integration
def test_tiny_faces_are_skipped_not_scored() -> None:
    """Faces below min_face_px must be reported as skipped, never scored.

    A 14px crop upsampled to 224 is mostly interpolation; a confident number off
    it is noise wearing the costume of evidence.
    """
    scene = Path(__file__).resolve().parents[1] / "assets" / "ai-generated-city-scene.jpeg"
    if not scene.is_file():
        pytest.skip("city scene asset not available")

    permissive = GenDDetector(min_face_px=0, max_faces=None).score_image(scene)
    guarded = GenDDetector(min_face_px=50, max_faces=None).score_image(scene)

    assert permissive.n_faces > guarded.n_faces, "the guard should drop the small faces"
    assert guarded.n_faces_skipped == permissive.n_faces - guarded.n_faces
    assert all(f.crop_px >= 50 for f in guarded.faces)


@pytest.mark.integration
def test_multi_face_verdict_does_not_hinge_on_the_largest(detector: GenDDetector) -> None:
    """On a crowd scene the top two faces can differ by a couple of pixels yet
    score at opposite ends, so the aggregate must not equal the largest face's
    score by construction."""
    scene = Path(__file__).resolve().parents[1] / "assets" / "ai-generated-city-scene.jpeg"
    if not scene.is_file():
        pytest.skip("city scene asset not available")

    prediction = detector.score_image(scene)
    if prediction.n_faces < 2:
        pytest.skip("scene did not yield multiple scoreable faces")

    largest = max(prediction.faces, key=lambda f: f.area)
    assert prediction.aggregation == "median"
    # The median of >=2 differing scores need not equal the largest face's score.
    assert prediction.p_fake == pytest.approx(
        sorted(f.p_fake for f in prediction.faces)[len(prediction.faces) // 2]
        if len(prediction.faces) % 2
        else sum(
            sorted(f.p_fake for f in prediction.faces)[
                len(prediction.faces) // 2 - 1 : len(prediction.faces) // 2 + 1
            ]
        )
        / 2
    )
    assert 0.0 <= prediction.p_fake <= 1.0
    assert largest in prediction.faces


@pytest.mark.integration
def test_full_pipeline_runs_on_a_real_photo(detector: GenDDetector) -> None:
    """End-to-end detect -> align -> score on an unaligned photograph, which is
    the path real dataset images take."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    candidates = sorted(assets.glob("*.jpeg")) + sorted(assets.glob("*.jpg"))
    if not candidates:
        pytest.skip("no image assets available")

    for path in candidates:
        prediction = detector.score_image(path)
        if prediction.has_face:
            assert 0.0 <= prediction.p_fake <= 1.0
            assert len(prediction.faces) == prediction.n_faces
            return
    pytest.skip("no asset image contained a detectable face")
