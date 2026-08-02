#!/usr/bin/env python3
"""Derives file-level integrity labels for a dataset's images from the
free-text authenticity justifications in claims.json.

    python scripts/ablations/detector_comparison/label_media_integrity.py --data-dir data/veritas_2026_q1

`authenticity.score` alone cannot serve as ground truth for a forensic
detector: it blends *direction* with *certainty*, and it scores misleadingness
rather than file integrity. A staged-but-unaltered recording sits at -0.42
despite being a pristine capture, and a fact-check that simply never examined
provenance lands near 0 — indistinguishable, by score, from a genuine
borderline case.

So the label comes from the justification text instead:

    manipulated  the file itself is synthetic or was altered after capture
    authentic    the file is an unaltered capture, even if staged or miscaptioned
    unknown      the fact-check did not assess provenance at all

`unknown` rows are excluded from detector scoring. The original score is kept
alongside the label so the disagreements stay auditable.

Results are cached per media id, so re-running costs nothing for rows already
labelled and an interrupted run resumes where it stopped.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import anthropic

import config.globals as globals_cfg  # noqa: F401  (loads config/.env)
from mafc.common.logger import logger

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 10
MAX_WORKERS = 8

# Primary manipulation types. Order matters only for documentation; the model
# picks exactly one, the most specific that applies.
MANIPULATION_TYPES = {
    "ai_generated": "the visual was synthesized wholesale by a generative model",
    "deepfake": "a real person's face or likeness was swapped or synthesized onto real footage",
    "splice_composite": "parts of two or more real images were combined into one",
    "fabricated_screenshot": "a forged rendering of a post, article, document, or UI that never existed",
    "graphic_edit": "a real base image altered by overlay, retouch, recolour, or misleading crop",
    "temporal_edit": "real footage selectively cut, slowed, or sped up",
    "other_manipulation": "altered after capture in a way none of the above describes",
    "none": "the file was not altered after capture",
}

# Which manipulation types a pixel-forensics detector could catch even in
# principle. Anything else scores the detector against a case it cannot see.
DETECTABILITY = {
    "ai_generated": "yes",
    "deepfake": "yes",
    "splice_composite": "yes",
    "graphic_edit": "yes",
    "fabricated_screenshot": "partial",  # fully rendered forgeries leave no splice
    "temporal_edit": "no",  # nothing survives in a single still frame
    "other_manipulation": "partial",
    "none": "n/a",
}

SYSTEM_PROMPT = f"""You classify fact-check findings about media files.

For each item you are given a fact-checker's justification about one media file's
authenticity. Decide what the justification says about THE FILE ITSELF, not about
whether the surrounding claim was misleading.

Return `label`:
  "manipulated" - the justification says the file is synthetic, or was altered
                  after capture (composited, retouched, re-rendered, cut, slowed).
  "authentic"   - the justification says the file is an unaltered capture. This
                  INCLUDES files that are real recordings of staged, orchestrated,
                  or re-enacted events, and real files used with a false caption,
                  wrong date, or wrong location. A pristine recording of a staged
                  event is authentic: staging is not manipulation.
  "unknown"     - the justification does not assess the file's provenance at all.
                  Typically it says the review focused on the factual accuracy of
                  the claim and offers no evidence about origin or integrity.
                  Choose this whenever provenance was simply never examined, even
                  if the justification leans positive or negative in tone.

Hedging is not the same as not looking. If the justification states a positive
finding about the file ("an unaltered screenshot of the genuine article", "no
evidence of manipulation or AI generation") and then adds a caveat about limited
forensic rigour, that is "authentic", not "unknown". Reserve "unknown" for
justifications that offer no finding about the file at all.

Return `manipulation_type`, exactly one of:
{chr(10).join(f'  "{k}" - {v}' for k, v in MANIPULATION_TYPES.items())}
Use "none" when label is "authentic" or "unknown".

Pick the MOST SPECIFIC type that applies, and prefer the type describing how the
visual content was altered over how the file was assembled:
  - Any synthesized or altered human likeness - swapped faces, changed lip
    movements, a person made to say something - is "deepfake", even when the
    result was also cut or re-timed, and even when the giveaway was the audio.
  - "temporal_edit" is only for footage whose ONLY alteration is selective
    cutting, reordering, slowing, or speeding up, with every frame left intact.
  - A forged post, article, headline card, or document is
    "fabricated_screenshot" even when a real template was used as the base.

Return `misleading_but_authentic`: true when the file is authentic but the
justification says it misleads anyway (staged event, false caption, wrong
date/place). False otherwise.

Return `evidence`: a short quote (<=15 words) from the justification that
carries the decision.

Respond with a JSON array, one object per input item, in the same order, each
with keys: id, label, manipulation_type, misleading_but_authentic, evidence.
Output the raw JSON array only."""


def load_image_media(data_dir: Path, media_types: tuple[str, ...] = ("image",)) -> dict[str, dict]:
    """media_id -> {file_name, score, justification} for every item with a justification.

    Defaults to images because the pixel detectors were only precomputed for
    those; pass ("image", "video") when labelling the whole dataset, e.g. for
    the oracle experiment.
    """
    claims = json.loads((data_dir / "claims.json").read_text())["claims"]
    out: dict[str, dict] = {}
    for claim in claims:
        for media in claim.get("media", []):
            if media.get("type") not in media_types:
                continue
            auth = media.get("authenticity")
            if not auth or not auth.get("justification"):
                continue
            out[str(media["id"])] = {
                "claim_id": claim["id"],
                "media_type": media["type"],
                "file_name": os.path.basename(media["file_path"]),
                "file_path": media["file_path"],
                "score": auth.get("score"),
                "justification": auth["justification"],
            }
    return out


def _parse_array(text: str) -> list[dict]:
    """Pull a JSON array out of a model response, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in response: {text[:200]}")
    return json.loads(text[start : end + 1])


def label_batch(client: anthropic.Anthropic, batch: list[tuple[str, dict]]) -> dict[str, dict]:
    payload = [{"id": mid, "justification": rec["justification"]} for mid, rec in batch]
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    parsed = _parse_array(response.content[0].text)

    expected = {mid for mid, _ in batch}
    out: dict[str, dict] = {}
    for entry in parsed:
        mid = str(entry.get("id"))
        if mid not in expected:
            logger.warning(f"[label] model returned unexpected id {mid!r}; dropping")
            continue
        mtype = entry.get("manipulation_type", "none")
        if mtype not in MANIPULATION_TYPES:
            logger.warning(f"[label] unknown manipulation_type {mtype!r} for {mid}; using other_manipulation")
            mtype = "other_manipulation"
        label = entry.get("label")
        if label not in {"manipulated", "authentic", "unknown"}:
            logger.warning(f"[label] unknown label {label!r} for {mid}; skipping")
            continue
        # An authentic or unassessed file has no manipulation type to report.
        if label != "manipulated":
            mtype = "none"
        out[mid] = {
            "label": label,
            "manipulation_type": mtype,
            "detectable_in_principle": DETECTABILITY[mtype],
            "misleading_but_authentic": bool(entry.get("misleading_but_authentic")),
            "evidence": (entry.get("evidence") or "")[:200],
        }

    missing = expected - set(out)
    if missing:
        logger.warning(f"[label] model omitted {len(missing)} of {len(batch)} items")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/veritas_2026_q1"))
    parser.add_argument(
        "--out", type=Path, default=None, help="cache file (default: <data-dir>/media_integrity_labels.json)"
    )
    parser.add_argument("--limit", type=int, default=None, help="label at most N new items (for a trial run)")
    parser.add_argument("--redo", action="store_true", help="ignore the cache and relabel everything")
    parser.add_argument(
        "--media-types",
        default="image",
        help="comma-separated: image, video, or both (the oracle experiment needs both)",
    )
    args = parser.parse_args()

    out_path = args.out or args.data_dir / "media_integrity_labels.json"
    media_types = tuple(t.strip() for t in args.media_types.split(",") if t.strip())
    media = load_image_media(args.data_dir, media_types)
    logger.info(
        f"[label] {len(media)} {'/'.join(media_types)} items with an authenticity justification in {args.data_dir}"
    )

    cache: dict[str, dict] = {}
    if out_path.is_file() and not args.redo:
        cache = json.loads(out_path.read_text()).get("labels", {})
        logger.info(f"[label] {len(cache)} already cached in {out_path}")

    todo = [(mid, rec) for mid, rec in media.items() if mid not in cache]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        logger.info("[label] nothing to do")
        return 0
    logger.info(f"[label] labelling {len(todo)} items with {MODEL}")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_api_key")
    if not api_key:
        logger.error("[label] no anthropic_api_key in the environment or config/.env")
        return 1
    client = anthropic.Anthropic(api_key=api_key)

    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    lock = threading.Lock()
    done = 0

    def run(batch):
        nonlocal done
        try:
            result = label_batch(client, batch)
        except Exception as e:
            logger.error(f"[label] batch failed: {e}")
            return
        with lock:
            cache.update(result)
            done += 1
            logger.info(f"[label] {done}/{len(batches)} batches ({len(cache)}/{len(media)} labelled)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(run, batches))

    out_path.write_text(
        json.dumps({"format": 1, "model": MODEL, "labels": cache}, indent=1, ensure_ascii=False)
    )
    logger.info(f"[label] wrote {len(cache)} labels to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
