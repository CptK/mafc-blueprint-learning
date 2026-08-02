"""Oracle manipulation detector — the ceiling instrument.

The critical property is NEGATIVE: the oracle must expose the verdict and
nothing else. The labels were derived from the fact-checker's justification,
which states the whole answer, so any leak of that text turns a
perfect-detector ceiling into a perfect-researcher ceiling and silently
inflates the result.
"""

import json
from pathlib import Path

import pytest

import config.globals  # noqa: F401  # Triggers .env loading like normal app startup.
from ezmm.common.items import Image
from mafc.tools.media.oracle import CheckOracleManipulation, OracleManipulationChecker

LABELS = Path("data/veritas_2026_q1/media_integrity_labels.json")


@pytest.fixture(scope="module")
def labels() -> dict:
    if not LABELS.is_file():
        pytest.skip("integrity labels not built")
    return json.loads(LABELS.read_text())["labels"]


@pytest.fixture(scope="module")
def checker() -> OracleManipulationChecker:
    if not LABELS.is_file():
        pytest.skip("integrity labels not built")
    return OracleManipulationChecker(labels_path=LABELS)


def _result_for(checker: OracleManipulationChecker, media_id: str):
    path = Path(f"data/veritas_2026_q1/images/{media_id}.jpg")
    if not path.is_file():
        pytest.skip(f"image {media_id} not available")
    item = Image(file_path=str(path))
    return checker._perform(CheckOracleManipulation(media=item.reference))


@pytest.mark.integration
def test_never_leaks_the_justification_or_evidence(checker, labels) -> None:
    """No stored free text may appear in what the oracle hands the pipeline."""
    claims = json.loads(Path("data/veritas_2026_q1/claims.json").read_text())["claims"]
    justifications = {
        str(m["id"]): m["authenticity"]["justification"]
        for c in claims
        for m in c.get("media", [])
        if m.get("authenticity", {}).get("justification")
    }

    checked = 0
    for media_id, record in list(labels.items())[:40]:
        path = Path(f"data/veritas_2026_q1/images/{media_id}.jpg")
        if not path.is_file():
            continue
        rendered = str(_result_for(checker, media_id))
        checked += 1

        evidence = (record.get("evidence") or "").strip()
        if evidence:
            assert evidence not in rendered, f"evidence quote leaked for {media_id}"
        justification = justifications.get(media_id, "")
        if justification:
            # Any distinctive 6-word run from the justification appearing verbatim
            # would mean the fact-check's own wording reached the pipeline.
            words = justification.split()
            for i in range(0, max(1, len(words) - 6), 6):
                span = " ".join(words[i : i + 6])
                assert span not in rendered, f"justification text leaked for {media_id}: {span!r}"
    assert checked > 0, "no media available to check"


@pytest.mark.integration
def test_unknown_is_reported_as_inconclusive_not_authentic(checker, labels) -> None:
    """An unassessed item must never read as 'clean' — that would invent evidence."""
    unknown = [k for k, v in labels.items() if v["label"] == "unknown"]
    if not unknown:
        pytest.skip("no unknown-labelled media")

    for media_id in unknown:
        if not Path(f"data/veritas_2026_q1/images/{media_id}.jpg").is_file():
            continue
        result = _result_for(checker, media_id)
        assert result.is_useful() is False
        assert "inconclusive" in str(result).lower()
        assert "unaltered" not in str(result).lower()
        return
    pytest.skip("no unknown-labelled image on disk")


@pytest.mark.integration
def test_verdicts_match_the_labels(checker, labels) -> None:
    # The rendered wording differs from the label name on purpose: "authentic"
    # is stated as UNALTERED so it reads as a claim about the file, not about
    # the surrounding claim being true.
    expected_wording = {"manipulated": "MANIPULATED", "authentic": "UNALTERED"}
    for want, wording in expected_wording.items():
        ids = [k for k, v in labels.items() if v["label"] == want]
        for media_id in ids:
            if not Path(f"data/veritas_2026_q1/images/{media_id}.jpg").is_file():
                continue
            result = _result_for(checker, media_id)
            assert result.label == want
            assert result.is_useful() is True
            assert wording in str(result)
            break


@pytest.mark.integration
def test_unlabelled_media_is_not_invented(checker, tmp_path) -> None:
    """Media with no label must report 'not found', never a default verdict."""
    from PIL import Image as PILImage

    path = tmp_path / "999999999.jpg"
    PILImage.new("RGB", (64, 64), (10, 20, 30)).save(path)
    result = checker._perform(CheckOracleManipulation(media=Image(file_path=str(path)).reference))

    assert result.found is False
    assert result.label is None
    assert result.is_useful() is False
