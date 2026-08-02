#!/usr/bin/env python3
"""Reads the C2PA manifest of every image in a dataset and caches what it found.

    python scripts/ablations/detector_comparison/scan_c2pa.py --data-dir data/veritas_2026_q1

C2PA is not a manipulation detector and cannot be scored like one: a missing
manifest is the overwhelmingly common case and says nothing about whether a
file was altered. The only question it can answer on a dataset like this is
coverage - of the images we independently know to be AI-generated, how many
carry a manifest that declares it.

Reading is local and free, so this scans every image rather than just the
AI-generated subset; that gives the false-positive side (manifests on images
known to be authentic) for the same cost.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from c2pa import Reader

from mafc.common.logger import logger
from mafc.tools.media.c2pa_checker import AI_CATEGORIES, SOURCE_TYPES, _actions, _short

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def inspect(path: Path) -> dict:
    """Mirror of C2PAChecker._inspect, minus the item-registry plumbing."""
    record = {
        "present": False,
        "validation_state": None,
        "provenance": "absent",
        "declares_ai": None,
        "verdict": "unknown",
        "source_types": [],
        "generator": None,
        "signer": None,
        "error": None,
    }
    try:
        reader = Reader.try_create(str(path))
    except Exception as e:
        # A parse failure is not the same as an absent manifest; keep them apart.
        record["error"] = str(e)
        return record
    if reader is None:
        return record

    try:
        store = json.loads(reader.json())
        state = reader.get_validation_state()
        record["present"] = True
        record["validation_state"] = state
        record["provenance"] = "valid" if state == "Valid" else "invalid"

        manifests = store.get("manifests") or {}
        active_label = store.get("active_manifest")
        categories = set()
        for label, manifest in manifests.items():
            for action in _actions(manifest):
                key = _short(action.get("digitalSourceType"))
                if not key:
                    continue
                category, _ = SOURCE_TYPES.get(key, ("unrecognized", ""))
                categories.add(category)
                record["source_types"].append(
                    {"source_type": key, "category": category, "active": label == active_label}
                )

        if categories & AI_CATEGORIES:
            record["declares_ai"] = True
            record["verdict"] = "ai_partial" if categories == {"ai_partial"} else "ai_generated"
        elif categories & {"captured", "human_created"}:
            record["declares_ai"] = False
            record["verdict"] = "not_ai_declared"
        elif categories:
            record["verdict"] = sorted(categories)[0]

        active = manifests.get(active_label) or {}
        gens = active.get("claim_generator_info") or []
        record["generator"] = ", ".join(g.get("name", "") for g in gens if isinstance(g, dict)) or None
        sig = active.get("signature_info") or {}
        record["signer"] = sig.get("issuer") or sig.get("common_name")
    except Exception as e:
        record["error"] = str(e)
    finally:
        reader.close()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/veritas_2026_q1"))
    parser.add_argument("--out", type=Path, default=None, help="default: <data-dir>/c2pa/index.json")
    args = parser.parse_args()

    images_dir = args.data_dir / "images"
    out_path = args.out or args.data_dir / "c2pa" / "index.json"

    paths = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    logger.info(f"[c2pa] scanning {len(paths)} images in {images_dir}")

    records = {p.name: inspect(p) for p in paths}

    present = sum(r["present"] for r in records.values())
    errors = sum(r["error"] is not None for r in records.values())
    logger.info(f"[c2pa] {present}/{len(records)} carry a manifest; {errors} read errors")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"format": 1, "model": "c2pa", "records": records}, indent=1))
    logger.info(f"[c2pa] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
