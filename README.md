# mafc-blueprint-learning

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
