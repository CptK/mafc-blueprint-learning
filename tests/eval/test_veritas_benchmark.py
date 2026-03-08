import json
from pathlib import Path
from typing import cast

import pytest

from mafc.eval.veritas import benchmark as veritas_benchmark_module
from mafc.eval.veritas.benchmark import VeriTaS, _classify_integrity_7
from mafc.eval.veritas.labels import Veritas3Label, Veritas7Label
from mafc.eval.veritas.types import ClaimEntry


def _write_claims_file(path: Path, claims: list[dict]) -> None:
    payload = {
        "metadata": {"total_claims": len(claims)},
        "claims": claims,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_integrity_score_is_coerced_to_float() -> None:
    bench = object.__new__(VeriTaS)
    claim_entry = cast(ClaimEntry, {"id": 1, "integrity": {"score": "0.83"}})
    score = bench._get_integrity_score(claim_entry)
    assert score == pytest.approx(0.83)


def test_integrity_score_invalid_returns_none(monkeypatch) -> None:
    bench = object.__new__(VeriTaS)
    warnings: list[str] = []
    monkeypatch.setattr("mafc.eval.veritas.benchmark.logger.warning", warnings.append)

    claim_entry = cast(ClaimEntry, {"id": 1, "integrity": {"score": {"bad": "value"}}})
    score = bench._get_integrity_score(claim_entry)

    assert score is None
    assert warnings
    assert "Invalid integrity score" in warnings[0]


def test_integrity_score_none_and_non_finite(monkeypatch) -> None:
    bench = object.__new__(VeriTaS)
    warnings: list[str] = []
    monkeypatch.setattr("mafc.eval.veritas.benchmark.logger.warning", warnings.append)

    assert bench._get_integrity_score(cast(ClaimEntry, {"id": 1, "integrity": None})) is None
    assert bench._get_integrity_score(cast(ClaimEntry, {"id": 2, "integrity": {"score": None}})) is None
    assert bench._get_integrity_score(cast(ClaimEntry, {"id": 3, "integrity": {"score": "nan"}})) is None

    assert any("Non-finite integrity score" in message for message in warnings)


def test_register_media_rejects_non_numeric_id() -> None:
    bench = object.__new__(VeriTaS)

    with pytest.raises(ValueError, match="Non-numeric video ID"):
        bench._register_media("<video:a8388> some claim", claim_id="75310")


def test_register_media_replaces_registered_refs(tmp_path, monkeypatch) -> None:
    bench = object.__new__(VeriTaS)
    bench.data_dir = tmp_path

    image_path = tmp_path / "images" / "123.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"img")

    class DummyImage:
        def __init__(self, path: Path):
            assert path == image_path
            self.reference = "<image:999>"

    monkeypatch.setattr("mafc.eval.veritas.benchmark.Image", DummyImage)

    out = bench._register_media("prefix <image:123> suffix", claim_id="1")
    assert out == "prefix <image:999> suffix"


def test_register_media_missing_file_logs_warning(tmp_path, monkeypatch) -> None:
    bench = object.__new__(VeriTaS)
    bench.data_dir = tmp_path
    warnings: list[str] = []
    monkeypatch.setattr("mafc.eval.veritas.benchmark.logger.warning", warnings.append)

    out = bench._register_media("<video:42> text", claim_id="42")
    assert out == "<video:42> text"
    assert len(warnings) >= 1


def test_register_media_registration_error_is_raised(tmp_path, monkeypatch) -> None:
    bench = object.__new__(VeriTaS)
    bench.data_dir = tmp_path

    video_path = tmp_path / "videos" / "77.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"vid")

    class BrokenVideo:
        def __init__(self, path: Path):
            raise RuntimeError("boom")

    monkeypatch.setattr("mafc.eval.veritas.benchmark.Video", BrokenVideo)

    with pytest.raises(RuntimeError, match="boom"):
        bench._register_media("<video:77> text", claim_id="77")


def test_classify_integrity_7_fallback_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        veritas_benchmark_module,
        "THRESHOLDS_7",
        [(0.0, Veritas7Label.UNKNOWN)],
    )
    assert _classify_integrity_7(1.0) == Veritas7Label.INTACT_CERTAIN


def test_init_rejects_invalid_label_scheme(tmp_path) -> None:
    claims_path = tmp_path / "claims.json"
    _write_claims_file(claims_path, [])
    with pytest.raises(ValueError, match="Unsupported label_scheme"):
        VeriTaS(data_path=str(claims_path), label_scheme=5)


def test_init_rejects_missing_claims_file(tmp_path) -> None:
    missing_dir = tmp_path / "missing_split"
    with pytest.raises(ValueError, match="Claims file not found"):
        VeriTaS(data_path=str(missing_dir), label_scheme=7)


def test_build_text_media_path_and_justification_helpers(tmp_path) -> None:
    bench = object.__new__(VeriTaS)
    bench.data_dir = tmp_path

    text = bench._build_claim_text_with_media(
        {
            "id": 1,
            "text": "claim",
            "media": [{"type": "image", "id": "12"}, {"type": "video", "id": "34"}],
        },
        claim_id="1",
    )
    assert text.startswith("<image:12> <video:34> ")

    no_media_text = bench._build_claim_text_with_media({"id": 2, "text": "plain", "media": []}, claim_id="2")
    assert no_media_text == "plain"

    assert bench._get_media_path("image", "12") == tmp_path / "images" / "12.jpg"
    assert bench._get_media_path("video", "34") == tmp_path / "videos" / "34.mp4"

    justification = bench._get_justification(
        {
            "integrity": {"score": 0.2},
            "veracity": 0.3,
            "context_coverage": {"score": 0.4},
            "media": [
                {
                    "id": "12",
                    "type": "image",
                    "authenticity": {"score": 0.8},
                    "contextualization": 0.7,
                }
            ],
        }
    )
    assert justification["integrity"] == 0.2
    assert justification["veracity"] == 0.3
    assert justification["context_coverage"] == 0.4
    assert justification["media_verdicts"][0]["authenticity"] == 0.8
    assert justification["media_verdicts"][0]["contextualization"] == 0.7


def test_load_data_skips_claims_with_invalid_media_registration(tmp_path, monkeypatch) -> None:
    claims_path = tmp_path / "claims.json"
    claims = [
        {"id": 1, "text": "ok", "integrity": {"score": 0.9}, "media": []},
        {"id": 2, "text": "bad", "integrity": {"score": -0.9}, "media": []},
    ]
    _write_claims_file(claims_path, claims)

    def fake_register_media(self, claim_text: str, claim_id: str, claim_entry=None) -> str:
        if claim_id == "2":
            raise ValueError("bad media id")
        return claim_text

    class DummyClaim:
        def __init__(self, text: str, date=None, id=None):
            self.text = text
            self.id = str(id)

        def __str__(self) -> str:
            return self.text

    monkeypatch.setattr("mafc.eval.veritas.benchmark.VeriTaS._register_media", fake_register_media)
    monkeypatch.setattr("mafc.eval.veritas.benchmark.Claim", DummyClaim)

    bench = VeriTaS(data_path=str(claims_path), variant="unit", label_scheme=7)

    assert len(bench.data) == 1
    assert bench.data[0].id == "1"
    assert bench.data[0].label == Veritas7Label.INTACT_CERTAIN


def test_load_data_handles_bad_date_and_claim_creation_failure(tmp_path, monkeypatch) -> None:
    claims_path = tmp_path / "claims.json"
    claims = [
        {"id": 1, "text": "ok", "integrity": None, "media": [], "date": "not-a-date"},
        {"id": 2, "text": "bad", "integrity": {"score": 0.9}, "media": []},
    ]
    _write_claims_file(claims_path, claims)

    monkeypatch.setattr(
        "mafc.eval.veritas.benchmark.VeriTaS._register_media",
        lambda self, text, claim_id, claim_entry=None: text,
    )

    class DummyClaim:
        def __init__(self, text: str, date=None, id=None):
            if str(id) == "2":
                raise ValueError("claim construction failed")
            self.id = str(id)

    monkeypatch.setattr("mafc.eval.veritas.benchmark.Claim", DummyClaim)

    bench = VeriTaS(data_path=str(claims_path), variant="unit", label_scheme=7)
    assert len(bench.data) == 1
    assert bench.data[0].id == "1"
    assert bench.data[0].label == Veritas7Label.UNKNOWN


def test_three_class_thresholds(tmp_path, monkeypatch) -> None:
    claims_path = tmp_path / "claims.json"
    claims = [
        {"id": 1, "text": "c1", "integrity": {"score": 0.33}, "media": []},
        {"id": 2, "text": "c2", "integrity": {"score": -0.33}, "media": []},
        {"id": 3, "text": "c3", "integrity": {"score": 0.0}, "media": []},
    ]
    _write_claims_file(claims_path, claims)

    monkeypatch.setattr(
        "mafc.eval.veritas.benchmark.VeriTaS._register_media",
        lambda self, text, claim_id, claim_entry=None: text,
    )

    class DummyClaim:
        def __init__(self, text: str, date=None, id=None):
            self.id = str(id)

    monkeypatch.setattr("mafc.eval.veritas.benchmark.Claim", DummyClaim)

    bench = VeriTaS(data_path=str(claims_path), variant="unit", label_scheme=3)

    labels = [item.label for item in bench.data]
    assert labels == [Veritas3Label.INTACT, Veritas3Label.COMPROMISED, Veritas3Label.UNKNOWN]
