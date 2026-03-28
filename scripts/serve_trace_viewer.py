#!/usr/bin/env python3
"""Serve the local trace viewer as a small static web app."""

from __future__ import annotations

import argparse
import http.server
import mimetypes
import sqlite3
import urllib.parse
from pathlib import Path
import socketserver
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the fact-check trace viewer locally.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to item_registry.db (default: temp/item_registry.db in repo root).",
    )
    return parser.parse_args()


class TraceViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server with extra API endpoints for media and blueprints."""

    registry_path: Path | None = None
    viewer_dir: Path | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Serve from repo root so that relative paths like ../../config/blueprints/
        # and ../../traces/ (relative to tools/trace_viewer/) resolve correctly.
        repo_root = self.__class__.viewer_dir.parents[1] if self.__class__.viewer_dir else Path(".")
        super().__init__(*args, directory=str(repo_root), **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/tools/trace_viewer/")
            self.end_headers()
            return

        if path.startswith("/api/media/"):
            parts = path.strip("/").split("/")  # ['api', 'media', kind, id]
            if len(parts) == 4:
                kind = parts[2]
                try:
                    media_id = int(parts[3])
                except ValueError:
                    self.send_error(400, "Invalid media ID")
                    return
                self._serve_media(kind, media_id)
            else:
                self.send_error(404, "Not found")
            return

        super().do_GET()

    def _serve_media(self, kind: str, media_id: int) -> None:
        if kind not in ("image", "video", "audio"):
            self.send_error(400, "Invalid media kind")
            return

        db_path = self.registry_path
        if db_path is None or not db_path.exists():
            self.send_error(503, "Registry not available")
            return

        try:
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(f"SELECT path FROM {kind} WHERE id = ?", (media_id,)).fetchone()
            conn.close()
        except Exception as exc:
            self.send_error(500, f"Registry error: {exc}")
            return

        if not row:
            self.send_error(404, f"{kind} {media_id} not found in registry")
            return

        file_path = Path(row[0])
        if not file_path.exists():
            self.send_error(404, f"File not found: {file_path}")
            return

        mime, _ = mimetypes.guess_type(str(file_path))
        mime = mime or "application/octet-stream"
        data = file_path.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Only log non-media requests to avoid spam
        if args and "/api/media/" not in str(args[0]):
            super().log_message(format, *args)


def main() -> None:
    args = parse_args()
    viewer_dir = REPO_ROOT / "tools" / "trace_viewer"

    registry_path = (
        Path(args.registry).resolve() if args.registry else REPO_ROOT / "temp" / "item_registry.db"
    )

    TraceViewerHandler.viewer_dir = viewer_dir
    TraceViewerHandler.registry_path = registry_path

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), TraceViewerHandler) as httpd:
        print(f"Serving trace viewer at http://{args.host}:{args.port}/tools/trace_viewer/")
        print(f"Viewer files: {viewer_dir}")
        if registry_path.exists():
            print(f"Registry:     {registry_path}")
        else:
            print(f"Registry not found at {registry_path} — media will not be displayed")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
