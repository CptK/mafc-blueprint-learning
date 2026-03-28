# mafc-blueprint-learning

This repo implements a multi-agent fact-checking system for multimodal claims (text, images, video). Claims are verified through a pipeline of delegated agents — web search, media analysis, synthesis — orchestrated by a blueprint: a declarative specification that defines the verification strategy for a given claim type. Four blueprints are included covering generic claims, dated events, quote attributions, and media-centric claims.

## Usage

### Running a benchmark

Benchmarks are configured via a YAML file (see `config/experiments/` for examples) and run with:

```bash
python -m scripts.run_benchmark --config config/experiments/veritas_baseline.yaml
```

Results and per-claim traces are written to `out/<run>/`.

### Geolocator server

Media geolocation uses a separate model server to avoid CUDA/fork issues when running claims in parallel. Start it before running a benchmark that requires geolocation:

```bash
python scripts/geolocator_server.py --model geolocal/StreetCLIP --port 5555 --workers 5
```

## Trace Viewer

The repo includes a web-based viewer for fact-check execution traces. It renders the full pipeline as an interactive graph: claim, blueprint selection, iterations, delegated tasks, web searches, evidence retrieval, synthesis, and verdict. A separate view covers the blueprints themselves.

### Hosted version

Sample traces can be explored without any local setup at **https://cptk.github.io/mafc-blueprint-learning/**. Media attachments (images/video) are not available in the hosted version.

### Local setup

1. Generate a trace:
   ```bash
   python -m scripts.run_first_veritas_sample --trace-dir traces
   ```
2. Serve the viewer:
   ```bash
   python -m scripts.serve_trace_viewer
   ```
3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) — sample traces load automatically, or upload any JSON file from `traces/`.
