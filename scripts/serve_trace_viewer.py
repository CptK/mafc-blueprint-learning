#!/usr/bin/env python3
"""Serve the local trace viewer as a small static web app."""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import sqlite3
import urllib.parse
from pathlib import Path
import socketserver

import yaml

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
    parser.add_argument(
        "--blueprints",
        default=None,
        help="Path to blueprints config directory (default: config/blueprints in repo root).",
    )
    return parser.parse_args()


class TraceViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server with extra API endpoints for media and blueprints."""

    registry_path: Path | None = None
    viewer_dir: Path | None = None
    blueprints_dir: Path | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(self.viewer_dir), **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

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

        if path == "/api/blueprints":
            self._serve_blueprints_list()
            return

        if path.startswith("/api/blueprints/"):
            name = path[len("/api/blueprints/"):]
            self._serve_blueprint_by_name(name)
            return

        if path in ("/blueprints", "/blueprints/"):
            self._serve_static_file(self.__class__.viewer_dir / "blueprints.html")
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

    def _serve_blueprints_list(self) -> None:
        bp_dir = self.__class__.blueprints_dir
        if not bp_dir or not bp_dir.is_dir():
            self._send_json([])
            return
        items = []
        for bp_path in sorted(bp_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(bp_path.read_text(encoding="utf-8"))
                items.append({"name": data.get("name", bp_path.stem), "description": data.get("description", "")})
            except Exception:
                pass
        self._send_json(items)

    def _serve_blueprint_by_name(self, name: str) -> None:
        bp_dir = self.__class__.blueprints_dir
        if not bp_dir or not bp_dir.is_dir():
            self.send_error(503, "Blueprints directory not available")
            return
        for bp_path in sorted(bp_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(bp_path.read_text(encoding="utf-8"))
                if data.get("name") == name:
                    self._send_json(data)
                    return
            except Exception:
                pass
        self.send_error(404, f"Blueprint '{name}' not found")

    def _send_json(self, data: object) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static_file(self, file_path: Path) -> None:
        if not file_path or not file_path.exists():
            self.send_error(404, "File not found")
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "text/html")
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

    blueprints_dir = (
        Path(args.blueprints).resolve() if args.blueprints else REPO_ROOT / "config" / "blueprints"
    )

    TraceViewerHandler.viewer_dir = viewer_dir
    TraceViewerHandler.registry_path = registry_path
    TraceViewerHandler.blueprints_dir = blueprints_dir

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), TraceViewerHandler) as httpd:
        print(f"Serving trace viewer at http://{args.host}:{args.port}")
        print(f"Viewer files: {viewer_dir}")
        if registry_path.exists():
            print(f"Registry:     {registry_path}")
        else:
            print(f"Registry not found at {registry_path} — media will not be displayed")
        if blueprints_dir.exists():
            print(f"Blueprints:   {blueprints_dir}")
        else:
            print(f"Blueprints not found at {blueprints_dir} — /blueprints will be empty")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
