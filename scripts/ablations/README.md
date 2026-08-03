# Ablations

Two separate questions about manipulation detection, kept apart because they are
answered on different units and can disagree:

| Directory | Question | Unit of analysis |
|---|---|---|
| `detector_comparison/` | Which detector best separates manipulated from authentic files? | one image |
| `manipulation_detection/` | Does giving the pipeline a detector change its verdict? | one claim |

A detector can win the first and be worthless in the second — that is the point
of running both. `detector_comparison/` says Sightengine is the only usable
signal (AUC 0.74 for the `ai_gen` head alone, 0.79 combined with `deepfake`, on
the detectable subset);
`manipulation_detection/` says wiring it in changes nothing end-to-end, and
neither would a *perfect* detector.

## Data flow

`detector_comparison/` produces the ground truth both directories rely on:

```
claims.json
    |
    |  label_media_integrity.py     LLM extraction over the fact-check
    v                               justifications (NOT authenticity.score)
media_integrity_labels.json ----------------------> read at runtime by the
    |                                               oracle detector tool
    |  scan_c2pa.py                 c2pa/index.json
    |  (trufor/sightengine/gend precompute modules produce their own stores)
    v
    |  build_table.py               joins labels + every detector's scores
    v
manipulation_comparison.csv
    |
    +--> detector_comparison/evaluate.py       AUC, per-type recall, C2PA coverage
    +--> manipulation_detection/score_ablation.py   (integrity split only)
```

So run `detector_comparison/` first: without `media_integrity_labels.json` the
oracle arm of the ablation cannot run at all.

## Typical order

```bash
# 1. ground truth + detector scores (once per dataset)
python scripts/ablations/detector_comparison/label_media_integrity.py --data-dir data/veritas_2026_q1
python scripts/ablations/detector_comparison/scan_c2pa.py            --data-dir data/veritas_2026_q1
python -m mafc.tools.media.trufor.precompute      data/veritas_2026_q1/images
python -m mafc.tools.media.sightengine.precompute data/veritas_2026_q1/images
python -m mafc.tools.media.gend.precompute        data/veritas_2026_q1/images
python scripts/ablations/detector_comparison/build_table.py          --data-dir data/veritas_2026_q1

# 2. detector-level comparison
python scripts/ablations/detector_comparison/evaluate.py             --data-dir data/veritas_2026_q1

# 3. end-to-end ablation: run the arms (see the reproduction notes), then
python scripts/ablations/manipulation_detection/score_ablation.py
```

All of these need `PYTHONPATH=.` — the scripts add the repo root themselves, but
the benchmark runner they sit alongside does not.
