"""Tests for the feedback-loop report/flagging/sampling logic (no LLM calls)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "feedback_loop",
    Path(__file__).resolve().parent.parent.parent / "scripts" / "learning" / "feedback_loop.py",
)
fb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fb)


def _result(cid: str, bp: str, predicted: str, gt: float) -> dict:
    return {
        "claim_id": cid,
        "blueprint_name": bp,
        "predicted": predicted,
        "ground_truth": "x",
        "gt_integrity_score": gt,
    }


def _write_run(tmp_path: Path, name: str, results: list[dict], blueprint_dir: str = "bp") -> dict:
    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return {
        "round": name,
        "run_dir": str(run_dir),
        "sample_ids": [r["claim_id"] for r in results],
        "blueprint_dir": blueprint_dir,
    }


@pytest.fixture()
def state(tmp_path: Path) -> dict:
    # Bucket "sick": all hard flips (gt intact, predicted compromised certain).
    # Bucket "healthy": all exact hits.
    sick = [_result(f"s{i}", "sick", "compromised (certain)", 1.0) for i in range(20)]
    healthy = [_result(f"h{i}", "healthy", "intact (certain)", 1.0) for i in range(20)]
    run = _write_run(tmp_path, "screen-01", sick + healthy)
    routing = {f"s{i}": "sick" for i in range(40)} | {f"h{i}": "healthy" for i in range(40)}
    return {
        "runs": [run],
        "blueprint_dir": "bp",
        "routing": routing,
        "flagged": [],
        "blueprint_dirs": [],
    }


def test_report_flags_sick_bucket_only(state: dict) -> None:
    report = fb._build_report(state, flag_margin=0.08, min_n=15, min_flips=3)
    assert report["flagged"] == ["sick"]
    assert report["buckets"]["sick"]["flips"] == 20
    assert report["buckets"]["healthy"]["mse"] == 0.0
    assert not report["buckets"]["healthy"]["flagged"]
    assert report["buckets"]["sick"]["routed_share"] == 0.5


def test_report_respects_min_n(state: dict) -> None:
    # Truncate sick bucket below min_n — must not be flagged on 5 samples.
    results_path = Path(state["runs"][0]["run_dir"]) / "results.jsonl"
    lines = results_path.read_text().splitlines()
    keep = [line for line in lines if '"h' in line] + [line for line in lines if '"s' in line][:5]
    results_path.write_text("\n".join(keep) + "\n")
    report = fb._build_report(state, flag_margin=0.08, min_n=15, min_flips=3)
    assert report["flagged"] == []


def test_latest_result_wins_across_runs(state: dict, tmp_path: Path) -> None:
    # A later run re-judging s0 as correct must supersede the earlier flip.
    fixed = [_result("s0", "sick", "intact (certain)", 1.0)]
    state["runs"].append(_write_run(tmp_path, "confirm-02", fixed))
    results = fb._load_results(state, "bp")
    assert results["s0"]["predicted"] == "intact (certain)"


def test_load_results_filters_by_blueprint_dir(state: dict, tmp_path: Path) -> None:
    other = [_result("x0", "sick", "intact (certain)", 1.0)]
    state["runs"].append(_write_run(tmp_path, "validate-03", other, blueprint_dir="bp-fb1"))
    assert "x0" not in fb._load_results(state, "bp")
    assert "x0" in fb._load_results(state, "bp-fb1")


def test_pick_claims_excludes_already_run(state: dict) -> None:
    picks = fb._pick_claims(state, ["sick"], per_bucket=50, seed=1)
    # 40 routed to sick, 20 already run -> only s20..s39 remain.
    assert len(picks["sick"]) == 20
    assert all(int(cid[1:]) >= 20 for cid in picks["sick"])


def test_pick_claims_deterministic_seed(state: dict) -> None:
    a = fb._pick_claims(state, ["sick"], per_bucket=5, seed=7)
    b = fb._pick_claims(state, ["sick"], per_bucket=5, seed=7)
    assert a == b


def test_launch_run_writes_subset_config(state: dict, tmp_path: Path, monkeypatch) -> None:
    import yaml

    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.dump(
            {
                "benchmark": {"name": "veritas", "split": "2026_q1", "label_scheme": 7, "first_n": 5},
                "agents": {"fact_check": {"model": "m"}},
                "blueprints": {"selector_model": "m", "config_dir": "orig_dir"},
                "run": {"concurrency": 8},
            }
        )
    )
    state["base_config"] = str(base_config)
    state["training_dir"] = "data/train"
    state["routing_split"] = "2025_train"

    calls: list = []
    monkeypatch.setattr(
        fb.subprocess, "run", lambda *a, **kw: calls.append(a) or type("P", (), {"returncode": 0})()
    )

    workdir = tmp_path / "wd"
    run_dir = fb._launch_run(state, workdir, "screen-01", ["c1", "c2"], concurrency=2)

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg["benchmark"]["sample_ids"] == ["c1", "c2"]
    assert cfg["benchmark"]["data_path"] == "data/train"
    assert cfg["benchmark"]["split"] == "2025_train"
    assert "first_n" not in cfg["benchmark"]
    assert cfg["blueprints"]["config_dir"] == "bp"  # state's current dir, not the original
    assert cfg["run"]["concurrency"] == 2
    assert calls and "--resume" in calls[0][0]

    with pytest.raises(SystemExit):
        fb._launch_run(state, workdir, "screen-01", ["c3"], concurrency=None)
