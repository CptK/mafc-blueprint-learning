# GenD — face deepfake detection

Wraps **GenD** (*Deepfake Detection that Generalizes Across Benchmarks*, WACV 2026,
[arXiv:2508.06248](https://arxiv.org/abs/2508.06248)) as a mafc tool.

Upstream: <https://github.com/yermandy/deepfake-detection> (MIT — see `LICENSE.txt`).

## What GenD does, and does not

GenD is a **face** deepfake detector. It answers one question — *is this face
swapped, reenacted, or synthetic?* It has nothing to say about spliced
backgrounds, forged screenshots, or generated images without people.

Where no face is found it returns `p_fake = None`, never `0.0`. Those are
different claims: one says "nothing to judge", the other says "judged, looks
real". `DeepfakeDetectionResults.is_useful()` is False in the no-face case so
the distinction survives into the agent layer.

### Two deliberate departures from upstream defaults

Upstream scores **only the largest face**, with no size floor. Both were changed
after a crowd scene (`tests/assets/ai-generated-city-scene.jpeg`) exposed them:

- **`max_faces=5`, `image_aggregation="median"`** (upstream: 1). In that scene
  the largest face (47px) scored 0.976 and the runner-up (45px) scored 0.193 —
  same image, opposite verdicts, decided by a two-pixel difference in size. The
  median across faces is stable where "largest" is a coin flip.
- **`min_face_px=50`** (upstream: none). Faces down to 14px were being scored at
  0.94 confidence. At that size the aligned crop is mostly upsampling
  artefacts, and GenD trains on far larger crops. Such faces are now counted in
  `n_faces_skipped` and excluded; if *every* face is too small the result is
  `p_fake=None` with a note, not a number.

Set `min_face_px=0, max_faces=1` to reproduce upstream behaviour exactly.

For reference, on VeriTaS 2026 Q1 images (n=120 sampled): 27% contain no
detectable face at all, and the median largest-face crop is 187px — so the size
floor rarely binds there. The city scene was an adversarial case, not typical.

## What was vendored

Only the inference path was copied; the training stack (lightning, wandb, peft,
timm, gradio, …) was not.

| file | origin | changes |
|---|---|---|
| `modeling_gend.py` | `src/hf/modeling_gend.py` | 3 × `COMPAT(transformers>=5)`, see below |
| `retinaface.py` | `src/retinaface.py` | `prepare_model` only — weight caching |
| `align.py` | `detector.py::align_face` | dropped the mask branch (training-data only) |

`inference.py`, `store.py`, `tool.py`, `precompute.py` are ours.

### transformers 5.x compatibility

Upstream pins `transformers==4.56.2`; this repo runs 5.x. Three changes were
needed, all marked `COMPAT` in the source. None alters the architecture or the
weights — verified via `output_loading_info`, which reports no missing,
unexpected, or mismatched keys.

1. The nested `CLIPModel.from_pretrained` is pinned to CPU, because 5.x refuses
   a nested `from_pretrained` under the outer meta-device context.
2. `post_init()` is called, which 5.x's loader now requires.
3. **`position_ids` buffers are rebuilt after loading.** This one matters: the
   buffer is *non-persistent* (absent from the checkpoint, meant to be
   recreated by `torch.arange`), and 5.x materialises it from uninitialised
   memory instead. The observed garbage was
   `[0, 0, 71, 2, 49527317536, ...]`.

   Out-of-range garbage raises `IndexError`. **In-range garbage does not** — it
   silently gathers the wrong positional embeddings and returns confident,
   wrong scores. That was observed: with a corrupt buffer the same fixture
   scored 0.454, and 0.324 once repaired. `test_gend.py` asserts the buffer
   directly rather than merely checking that inference runs.

## Weights

- **GenD checkpoint** — pulled from Hugging Face on first use and cached there.
  Default `yermandy/GenD_CLIP_L_14`; `GenD_PE_L` also needs `timm`.
- **RetinaFace (buffalo_l `det_10g.onnx`)** — ~17 MB, downloaded on first use to
  `weights/` beside this file, pinned by URL revision and checked against a
  SHA-256. Upstream fetched it via `os.system("wget …")` to a path relative to
  the CWD, which fails silently when `wget` is absent (stock macOS) and leaves a
  zero-byte file that only surfaces later as an opaque ONNX parse error.

## Usage

```python
from mafc.tools.media.gend import GenDChecker

checker = GenDChecker()
result = checker.score_image("path/to/image.jpg")
print(result)  # p_fake, n_faces, verdict — or "no face found"
```

Precompute a dataset (images only by default; videos are much slower):

```bash
python -m mafc.tools.media.gend.precompute data/veritas_2026_q1/images
python -m mafc.tools.media.gend.precompute data/veritas_2026_q1 --videos
```

Point the tool at precomputed stores with the `gend_stores` env var, same as
`trufor_stores` / `sightengine_stores`:

```bash
gend_stores=data/veritas_2026_q1/gend
```

To expose GenD to the media agent, add `GenDChecker` to the tool list in
`mafc/agents/media/agent.py`. It is deliberately not wired in by default.
