import json
from pathlib import Path
from typing import cast

import pytest

from mafc.eval.veritas.benchmark import VeriTaS
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


def test_register_media_rejects_non_numeric_id() -> None:
    bench = object.__new__(VeriTaS)

    with pytest.raises(ValueError, match="Non-numeric video ID"):
        bench._register_media("<video:a8388> some claim", claim_id="75310")


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
