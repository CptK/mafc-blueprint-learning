# TruFor manipulation detection

Detects locally manipulated images — splicing, copy-move, inpainting — using
[TruFor](https://github.com/grip-unina/TruFor) (Guillaro et al., CVPR 2023).
The network outputs a detection score in `[0, 1]` (higher = more likely
manipulated) plus a per-pixel localization map and a confidence map.

The network code is vendored here (see *Provenance*), so **no Docker is
involved**: the model runs in-process on the Apple GPU via MPS, on CUDA, or on
CPU, whichever is available.

## Usage

### On the fly

```python
from ezmm.common.items import Image
from mafc.tools.media.trufor import TruFor, DetectImageManipulation

tool = TruFor()
image = Image(file_path="some/photo.jpg")
result = tool.perform(DetectImageManipulation(media=image.reference))
print(result)          # LLM-facing summary
print(result.raw.score)
```

Or without the action/tool machinery:

```python
tool.score_image("some/photo.jpg").score
```

Videos are handled by sampling `n_video_frames` frames (default 5) and combining
the frame scores with `video_aggregation` (`"max"` by default, or `"median"` /
`"mean"`). Frames are read as raw RGB arrays, so they are not re-compressed
before scoring.

**Treat video scores with suspicion.** TruFor is an image model trained on JPEG
artifacts; H.264 frames are out of distribution. On a spot check of six
untouched dataset videos the median frame score was 0.35, but `max` over just
three frames pushed three of the six above the 0.5 threshold — the more frames
you sample, the more the maximum drifts up. Per-frame scores are always stored,
so you can switch aggregation (or pick a video-specific threshold) without
rescoring anything.

### Precomputed scores for a dataset

Scoring is ~1 s per megapixel, so datasets are scored once up front:

```bash
python -m mafc.tools.media.trufor.precompute data/veritas_2026_q1/images
python -m mafc.tools.media.trufor.precompute data/veritas_2026_q1 --videos --frames 5
```

By default this writes to `<dataset>/trufor/`, i.e. beside the media, so the
scores travel with the dataset. Re-running skips whatever is already scored, so
an interrupted run simply resumes. Useful flags: `--store` (output dir),
`--maps` (also save localization maps — large), `--limit`, `--device`.

Point the tool at one or more stores:

```python
tool = TruFor(stores=["data/veritas_2026_q1/trufor"])
```

or set the `trufor_stores` env var (colon-separated) and let the default apply.
Lookup order is: read-only stores in the given order, then the writable cache
(`temp/trufor` by default). A miss is computed and written to the cache, so
nothing is ever scored twice.

Records are keyed by the file's **sha256**, not its path — a store stays valid
when files move, and an image shared by two datasets is scored once.

## Layout

| Path | What |
|---|---|
| `tool.py` | `TruFor` tool, `DetectImageManipulation` action, results |
| `inference.py` | model loading, device selection, `predict_image` / `predict_array` |
| `store.py` | the sha256-keyed score store |
| `precompute.py` | CLI for scoring a directory |
| `model/` | vendored TruFor network |
| `weights/` | checkpoint, downloaded on first use (gitignored, ~280 MB) |

## Weights

`inference.ensure_weights()` downloads `trufor.pth.tar` from
grip.unina.it on first use into `weights/`. Override with the
`trufor_weights_path` env var or `TruFor(weights_path=...)`.

## Interpreting the score

The paper thresholds at **0.5** (`DEFAULT_THRESHOLD`). Two limits worth
repeating to anyone consuming the output:

- TruFor detects *local edits*. A fully AI-generated image has no manipulated
  region to find and may well score low.
- Heavy re-compression, resizing and screenshotting — i.e. anything that has
  been through a social platform — both hides real edits and produces false
  alarms.

The tool's `_summarize()` already appends this caveat for the LLM.

## Provenance and a decoder caveat

`model/` is copied from TruFor's `test_docker/src/models` with three changes:

1. `timm.models.layers` imports → `model/layers.py` (the three helpers used are
   `DropPath`, `to_2tuple`, `trunc_normal_`), dropping the `timm` dependency.
2. `yacs` config → `model/config.py`, which inlines the fixed `trufor.yaml`.
3. `from models.DnCNN import ...` → a relative import.

The network itself is untouched; scores match the upstream pipeline to ~3e-7.

**Decoder caveat.** Upstream's Docker image gets Pillow from conda, linked
against IJG **libjpeg 9d**. Every pip-installed Pillow — including this repo's —
links **libjpeg-turbo**, which upsamples JPEG chroma differently: ~19% of
decoded pixels differ (by up to 24 levels), which TruFor amplifies into score
differences of up to ~0.13 on JPEG inputs. Scores here are therefore
self-consistent but can differ from numbers produced by upstream's container.
This matters only when comparing against published TruFor benchmark figures;
within this repo, precomputed and on-the-fly scores use the same decoder and
agree.

## Licence

TruFor is released for **nonprofit use only** — see `LICENSE.txt` (and
`LICENSE_CMX.txt` for the CMX components), both carried over from upstream.
