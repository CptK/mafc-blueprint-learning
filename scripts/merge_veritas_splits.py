#!/usr/bin/env python3
"""Merge multiple veritas dataset splits into a single dataset."""

import argparse
import json
import shutil
import sys
from pathlib import Path


def merge_splits(source_dirs: list[Path], target_dir: Path) -> None:
    if target_dir.exists():
        print(f"Error: target directory already exists: {target_dir}", file=sys.stderr)
        sys.exit(1)

    target_dir.mkdir(parents=True)
    (target_dir / "images").mkdir()
    (target_dir / "videos").mkdir()

    all_claims: list[dict] = []
    seen_claim_ids: set[int] = set()
    seen_media_files: set[str] = set()
    total_meta = {"claim_counts": {"intact": 0, "nei": 0, "compromised": 0, "total": 0},
                  "media_counts": {"images": 0, "videos": 0},
                  "sources": []}

    for source_dir in source_dirs:
        if not source_dir.is_dir():
            print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
            sys.exit(1)

        claims_path = source_dir / "claims.json"
        if not claims_path.exists():
            print(f"Error: no claims.json in {source_dir}", file=sys.stderr)
            sys.exit(1)

        with open(claims_path) as f:
            data = json.load(f)

        claims = data.get("claims", [])
        duplicates = 0
        added = 0

        for claim in claims:
            cid = claim["id"]
            if cid in seen_claim_ids:
                duplicates += 1
                continue
            seen_claim_ids.add(cid)
            all_claims.append(claim)
            added += 1

            for media in claim.get("media", []):
                rel_path = media.get("file_path", "")
                if not rel_path or rel_path in seen_media_files:
                    continue
                seen_media_files.add(rel_path)
                src_file = source_dir / rel_path
                dst_file = target_dir / rel_path
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)

        meta_path = source_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for key in ("intact", "nei", "compromised", "total"):
                total_meta["claim_counts"][key] += meta.get("claim_counts", {}).get(key, 0)
            total_meta["media_counts"]["images"] += meta.get("media_counts", {}).get("images", 0)
            total_meta["media_counts"]["videos"] += meta.get("media_counts", {}).get("videos", 0)
            total_meta["sources"].append({
                "path": str(source_dir),
                "year": meta.get("year"),
                "quarter": meta.get("quarter"),
                "claims_added": added,
                "claims_skipped_duplicate": duplicates,
            })

        print(f"  {source_dir.name}: {added} claims added, {duplicates} duplicates skipped")

    with open(target_dir / "claims.json", "w") as f:
        json.dump({"claims": all_claims}, f, ensure_ascii=False, indent=2)

    # Recount media from actual claims to reflect de-duplication
    total_meta["claim_counts"]["total"] = len(all_claims)
    total_meta["media_counts"]["images"] = sum(
        1 for c in all_claims for m in c.get("media", []) if m.get("type") == "image"
    )
    total_meta["media_counts"]["videos"] = sum(
        1 for c in all_claims for m in c.get("media", []) if m.get("type") == "video"
    )

    with open(target_dir / "meta.json", "w") as f:
        json.dump(total_meta, f, ensure_ascii=False, indent=2)

    print(f"\nMerged {len(all_claims)} claims into {target_dir}")
    print(f"  Images: {total_meta['media_counts']['images']}")
    print(f"  Videos: {total_meta['media_counts']['videos']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple veritas dataset splits into one."
    )
    parser.add_argument("sources", nargs="+", type=Path, help="Source split directories")
    parser.add_argument("--target", "-t", type=Path, required=True, help="Target directory to create")
    args = parser.parse_args()

    print(f"Merging {len(args.sources)} splits into {args.target}...\n")
    merge_splits(args.sources, args.target)


if __name__ == "__main__":
    main()
