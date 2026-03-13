#!/usr/bin/env python3
"""Serve the local trace viewer as a small static web app."""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path
import socketserver


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer_dir = Path(__file__).resolve().parents[1] / "tools" / "trace_viewer"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(viewer_dir))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving trace viewer at http://{args.host}:{args.port}")
        print(f"Viewer files: {viewer_dir}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
