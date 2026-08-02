# Experiment configs

Run with `python -m scripts.run_benchmark --config <path>` (learning configs use
`scripts/run_learning.py`).

```
baselines/    full-pipeline runs with no manipulation of the setup — the reference
              points everything else is compared against
ablations/    each subdirectory answers one "does X earn its keep?" question, and
              mirrors the analysis code under scripts/ablations/
learning/     blueprint-learning runs (different schema — no `benchmark:` block)
```

## Ablations

| Directory | Question | Configs |
|---|---|---|
| `manipulation_detection/` | Does a manipulation detector change the verdict? | `veritas_oracle_ceiling` (perfect detector — the ceiling), `veritas_sightengine_arm` (best real detector) |
| `routing/` | Does blueprint routing earn its keep? | `veritas_v4_llm_tiebreak` (probe vs LLM tie-break), `veritas_forced_media_control` and `veritas_forced_primary_control` (the two forced-blueprint controls) |
| `blueprints/` | Does a specific blueprint edit help? | `veritas_recontext_fix` (A/B of the reworded `recontextualized_media` blueprint) |

Scored by the matching directory under `scripts/ablations/`.

## Conventions

Each ablation config carries a header comment naming **the run it is paired
against** and **the single thing that differs from it**. That pairing is the
experiment: these configs are not standalone runs but one half of a comparison,
and a config whose baseline run has been deleted or renamed can no longer be
scored. Keep the header accurate when copying a config to make a new arm.

Claim subsets (`sample_ids`) are inlined so a config fully determines its run.
Where a subset is shared across experiments it lives with the dataset instead —
e.g. `data/veritas_2026_q1/media_only_277_ids.json`, the 277 image-only claims
with a ground-truth integrity label, used by both manipulation-detection arms and
by the re-judge.
