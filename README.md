# mafc-blueprint-learning

## Trace Viewer

A hosted version with sample traces is available at **https://cptk.github.io/mafc-blueprint-learning/** — no setup required. Note that media attachments (images/video) are not available in the hosted version.

### Local setup

The repo includes a small high-level web viewer for fact-check traces:

1. Generate a trace:
   ```bash
   python -m scripts.run_first_veritas_sample --trace-dir traces
   ```
2. Serve the viewer:
   ```bash
   python -m scripts.serve_trace_viewer
   ```
3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and load one of the JSON files from `traces/`.

The current viewer is intentionally high-level. It shows:
- claim
- selected blueprint
- iterations
- delegated tasks
- final result

It is designed to be extended later with finer-grained events.
