"""
Extract country-of-origin labels for images in the VeriTaS dataset using an LLM,
then copy the confidently-labeled real photographs into geolocation_test/veritas_data/
in the same format as geolocation_test/self_collected_data/ (images/<id>.<ext> + labels.csv).

Usage:
    python scripts/ablations/geo_tools_analysis/collect_geolocation_data.py --limit 20 --out geolocation_test/veritas_data_sample
    python scripts/ablations/geo_tools_analysis/collect_geolocation_data.py --out geolocation_test/veritas_data
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "veritas_2025_with_fact_checks"
CLAIMS_PATH = DATA_DIR / "claims.json"

load_dotenv(REPO_ROOT / "config" / ".env")
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_api_key")

MODEL = "claude-sonnet-5"
BATCH_SIZE = 8
ARTICLE_TRUNCATE = 3000

SYSTEM_PROMPT = """You are doing STRICT EXTRACTION (not inference/guessing) to build a geolocation
dataset from fact-checked media claims. Each input is a single-image claim: the debunking
article's text plus the image's own authenticity/contextualization assessments, which are
ground-truth fact-check annotations (not your own judgment about the image).

For each image, determine, for the ACTUAL underlying photograph (not the false claim about it):

1. is_real_photo: true only if the image is a genuine, unaltered photograph of a real scene
   (not AI-generated/synthetic, not a screenshot/graphic/meme/text-card, not a movie/game still,
   not digitally composited or significantly manipulated).
2. is_geolocatable_scene: true only if the photo shows an identifiable real-world place/scene
   that a person could plausibly geolocate from visual content alone (outdoor scenes, streets,
   landmarks, nature, distinctive buildings/interiors). false for TV broadcast stills, news
   studio interviews/talking heads, video calls, screenshots of documents/charts/social posts,
   or other content whose visible background carries no location signal, EVEN IF the country of
   the broadcast/event is known.
3. country: the country where the photo was ACTUALLY taken, in English (e.g. "India", "Kenya",
   "United States"). Only fill this in if `evidence_quote` (see below) explicitly and
   unambiguously states it — do not paraphrase, combine multiple clues, or infer from language/
   outlet/indirect context. null otherwise.
4. evidence_quote: a VERBATIM quote (copied exactly, character-for-character, from the
   "Article excerpt" or the justification fields given to you) that explicitly states the
   country/location. Required whenever country is non-null; null if country is null. Do not
   fabricate or paraphrase this quote — it must be an exact substring of the provided text.
5. location_details: a short specific location within the country if stated in the same or
   nearby text (city/region/event name), else null.
6. confidence: "high" if the quote explicitly and decisively names the location, "medium" if
   it names it but with minor ambiguity. Do not output "low" — if you'd say low, set country
   to null instead.

Be conservative: only extract a country when text you can quote verbatim actually states it.
Many claims recontextualize an old/unrelated real photo (correct country may still be quotable)
or use AI-generated/edited images (is_real_photo=false, country=null).

Return ONLY a JSON array, one object per input image, in the same order as given, each with keys:
media_id, is_real_photo, is_geolocatable_scene, country, evidence_quote, location_details,
confidence. No prose, no markdown fences."""


def load_image_records():
    with open(CLAIMS_PATH) as f:
        claims = json.load(f)["claims"]

    records = []
    for c in claims:
        images = [m for m in c.get("media", []) if m.get("type") == "image"]
        if len(images) != 1:
            # Skip multi-image claims entirely: article_content is shared across all
            # images in a claim, so there's no safe way to attribute claim-level text
            # to a specific image when there's more than one.
            continue
        m = images[0]
        fp = DATA_DIR / m["file_path"]
        if not fp.exists():
            continue
        article = (c.get("article_content") or "")[:ARTICLE_TRUNCATE]
        records.append(
            {
                "media_id": m["id"],
                "file_path": str(fp),
                "claim_id": c["id"],
                "language": c.get("language"),
                "review_url": c.get("review_url"),
                "date": c.get("date"),
                "authenticity_justification": (m.get("authenticity") or {}).get("justification"),
                "contextualization_justification": (m.get("contextualization") or {}).get("justification"),
                "article_content": article,
            }
        )
    return records


def normalize_for_substring_check(text):
    return " ".join((text or "").split()).lower()


def quote_is_grounded(quote, *source_texts):
    if not quote:
        return False
    needle = normalize_for_substring_check(quote)
    if not needle:
        return False
    haystack = normalize_for_substring_check(" \n".join(t or "" for t in source_texts))
    return needle in haystack


def build_batch_prompt(batch):
    parts = []
    for r in batch:
        parts.append(
            f"--- IMAGE media_id={r['media_id']} ---\n"
            f"Fact-check source: {r['review_url']}\n"
            f"Language: {r['language']}  Date: {r['date']}\n"
            f"Authenticity assessment: {r['authenticity_justification']}\n"
            f"Contextualization assessment: {r['contextualization_justification']}\n"
            f"Article excerpt: {r['article_content']}\n"
        )
    return "\n".join(parts)


def call_batch(client, batch, max_retries=4):
    prompt = build_batch_prompt(batch)
    expected_ids = [r["media_id"] for r in batch]
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            text = "".join(text_blocks).strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[1] if "\n" in text else text
                if text.lower().startswith("json"):
                    text = text[4:]
            results = json.loads(text)
            result_ids = [r["media_id"] for r in results]
            if result_ids != expected_ids:
                by_id = {r["media_id"]: r for r in results}
                results = [
                    by_id.get(
                        mid,
                        {
                            "media_id": mid,
                            "is_real_photo": False,
                            "country": None,
                            "location_details": None,
                            "confidence": "low",
                        },
                    )
                    for mid in expected_ids
                ]
            return results
        except Exception as e:
            wait = 2**attempt
            print(f"  batch failed ({e}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    print(f"  giving up on batch {expected_ids}", file=sys.stderr)
    return [
        {
            "media_id": mid,
            "is_real_photo": False,
            "country": None,
            "location_details": None,
            "confidence": "low",
        }
        for mid in expected_ids
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process first N images (for testing)")
    ap.add_argument("--out", type=str, default="geolocation_test/veritas_data")
    ap.add_argument("--min-confidence", choices=["high", "medium"], default="medium")
    ap.add_argument(
        "--raw-out", type=str, default=None, help="path to dump raw per-image LLM results as JSON"
    )
    args = ap.parse_args()

    if not API_KEY:
        print("No ANTHROPIC_API_KEY / anthropic_api_key found in env or config/.env", file=sys.stderr)
        sys.exit(1)

    records = load_image_records()
    print(f"Loaded {len(records)} image records from claims.json")
    if args.limit:
        records = records[: args.limit]
        print(f"Limiting to first {len(records)} for this run")

    client = anthropic.Anthropic(api_key=API_KEY)
    by_media_id = {r["media_id"]: r for r in records}
    all_results = []

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        print(
            f"Processing batch {i // BATCH_SIZE + 1}/{(len(records) - 1) // BATCH_SIZE + 1} "
            f"(media_ids {[r['media_id'] for r in batch]})..."
        )
        results = call_batch(client, batch)
        all_results.extend(results)

    if args.raw_out:
        with open(args.raw_out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote raw results to {args.raw_out}")

    accepted = []
    rejected_ungrounded_quote = 0
    conf_rank = {"high": 2, "medium": 1, "low": 0}
    min_rank = conf_rank[args.min_confidence]
    for res in all_results:
        if not res.get("is_real_photo"):
            continue
        if not res.get("is_geolocatable_scene"):
            continue
        if not res.get("country"):
            continue
        if conf_rank.get(res.get("confidence"), 0) < min_rank:
            continue
        rec = by_media_id.get(res["media_id"])
        if not rec or not quote_is_grounded(
            res.get("evidence_quote"),
            rec["authenticity_justification"],
            rec["contextualization_justification"],
            rec["article_content"],
        ):
            rejected_ungrounded_quote += 1
            continue
        accepted.append(res)

    print(
        f"\n{len(accepted)} / {len(all_results)} images accepted (real photo + geolocatable scene "
        f"+ known country with a verbatim, grounded quote + confidence >= {args.min_confidence})"
    )
    if rejected_ungrounded_quote:
        print(
            f"  ({rejected_ungrounded_quote} additionally rejected: evidence_quote was not a "
            f"verbatim match of the source text)"
        )

    out_dir = REPO_ROOT / args.out
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for res in accepted:
        rec = by_media_id[res["media_id"]]
        src = Path(rec["file_path"])
        dst = images_dir / f"{res['media_id']}{src.suffix}"
        shutil.copy2(src, dst)
        rows.append((res["media_id"], res["country"], res.get("location_details") or ""))

    rows.sort(key=lambda r: r[0])
    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w") as f:
        f.write("ID,country,details\n")
        for mid, country, details in rows:
            details_escaped = details.replace('"', '""')
            if "," in details_escaped or '"' in details_escaped:
                details_escaped = f'"{details_escaped}"'
            f.write(f"{mid},{country},{details_escaped}\n")

    print(f"Wrote {len(rows)} images to {images_dir}")
    print(f"Wrote labels to {labels_path}")

    audit_path = out_dir / "audit.json"
    audit_rows = []
    for res in accepted:
        rec = by_media_id[res["media_id"]]
        audit_rows.append(
            {
                "media_id": res["media_id"],
                "country": res["country"],
                "location_details": res.get("location_details"),
                "confidence": res.get("confidence"),
                "evidence_quote": res.get("evidence_quote"),
                "review_url": rec["review_url"],
            }
        )
    audit_rows.sort(key=lambda r: r["media_id"])
    with open(audit_path, "w") as f:
        json.dump(audit_rows, f, indent=2)
    print(f"Wrote audit trail (evidence quotes + source URLs) to {audit_path}")


if __name__ == "__main__":
    main()
