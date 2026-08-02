#!/usr/bin/env python3
"""Joins ground-truth integrity labels with the three detectors' scores into one CSV.

    python scripts/ablations/detector_comparison/build_table.py --data-dir data/veritas_2026_q1

Inputs (all produced ahead of time, so this is a pure join):
    claims.json                    the fact-checks, for score and claim id
    media_integrity_labels.json    label_media_integrity.py
    trufor/index.json              mafc.tools.media.trufor.precompute
    sightengine/index.json         mafc.tools.media.sightengine.precompute
    c2pa/index.json                scan_c2pa.py

Images only: the Sightengine video path is a paid per-file call and has not been
run for this dataset.

TruFor and Sightengine are keyed by file sha256 and carry the original filename
in `source_name`; the join goes through that filename back to the media id.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from mafc.common.logger import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_media_integrity import load_image_media

COLUMNS = [
    "image_id",
    "claim_id",
    "file_name",
    # ground truth
    "gt_label",
    "manipulation_type",
    "detectable_in_principle",
    "misleading_but_authentic",
    "gt_authenticity_score",
    "gt_evidence",
    # detectors
    "trufor_score",
    "sightengine_ai_generated",
    "sightengine_deepfake",
    "sightengine_top_generator",
    "gend_p_fake",
    "gend_n_faces",
    "gend_status",
    "c2pa_present",
    "c2pa_valid",
    "c2pa_declares_ai",
    "c2pa_generator",
]


def _by_source_name(index_path: Path) -> dict[str, dict]:
    """Re-key a sha256-keyed store by its original filename."""
    if not index_path.is_file():
        logger.warning(f"[table] missing {index_path}; those columns will be blank")
        return {}
    records = json.loads(index_path.read_text()).get("records", {})
    out: dict[str, dict] = {}
    for record in records.values():
        name = record.get("source_name")
        if name:
            out[name] = record
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/veritas_2026_q1"))
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <data-dir>/manipulation_comparison.csv"
    )
    args = parser.parse_args()

    out_path = args.out or args.data_dir / "manipulation_comparison.csv"

    media = load_image_media(args.data_dir)
    labels = json.loads((args.data_dir / "media_integrity_labels.json").read_text())["labels"]
    trufor = _by_source_name(args.data_dir / "trufor" / "index.json")
    sighte = _by_source_name(args.data_dir / "sightengine" / "index.json")
    gend = _by_source_name(args.data_dir / "gend" / "index.json")
    c2pa_path = args.data_dir / "c2pa" / "index.json"
    c2pa = json.loads(c2pa_path.read_text()).get("records", {}) if c2pa_path.is_file() else {}

    rows = []
    for media_id, rec in sorted(media.items(), key=lambda kv: int(kv[0])):
        name = rec["file_name"]
        label = labels.get(media_id, {})
        tf = trufor.get(name, {})
        se = sighte.get(name, {})
        gd = gend.get(name, {})
        cp = c2pa.get(name, {})

        # GenD abstains rather than guessing, so record *why* it has no score:
        # a blank must not be read as "looked, found nothing suspicious".
        if not gd:
            gend_status = ""
        elif gd.get("p_fake") is not None:
            gend_status = "scored"
        elif gd.get("n_faces_skipped"):
            gend_status = "faces_too_small"
        else:
            gend_status = "no_face"
        rows.append(
            {
                "image_id": media_id,
                "claim_id": rec["claim_id"],
                "file_name": name,
                "gt_label": label.get("label", "unknown"),
                "manipulation_type": label.get("manipulation_type", "none"),
                "detectable_in_principle": label.get("detectable_in_principle", "n/a"),
                "misleading_but_authentic": label.get("misleading_but_authentic", ""),
                "gt_authenticity_score": rec["score"],
                "gt_evidence": label.get("evidence", ""),
                "trufor_score": tf.get("score", ""),
                "sightengine_ai_generated": se.get("ai_generated_score", ""),
                "sightengine_deepfake": se.get("deepfake_score", ""),
                "sightengine_top_generator": se.get("top_generator") or "",
                "gend_p_fake": "" if gd.get("p_fake") is None else gd["p_fake"],
                "gend_n_faces": gd.get("n_faces", ""),
                "gend_status": gend_status,
                "c2pa_present": cp.get("present", ""),
                "c2pa_valid": (cp.get("provenance") == "valid") if cp else "",
                "c2pa_declares_ai": "" if cp.get("declares_ai") is None else cp["declares_ai"],
                "c2pa_generator": cp.get("generator") or "",
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    scored = sum(1 for r in rows if r["trufor_score"] != "" and r["sightengine_ai_generated"] != "")
    logger.info(f"[table] wrote {len(rows)} rows to {out_path} ({scored} with both detector scores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
