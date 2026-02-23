"""Minimal HTTP server for the traqo trace viewer.

Usage:
    python -m traqo.ui [TRACES_DIR] [--port PORT]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from functools import lru_cache
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return list of parsed events."""
    events: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _trace_summary(path: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract summary info from a trace's events."""
    summary: dict[str, Any] = {
        "file": str(path.name),
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }

    for ev in events:
        if ev.get("type") == "trace_start":
            summary["ts"] = ev.get("ts")
            summary["input"] = ev.get("input")
            summary["tags"] = ev.get("tags", [])
            summary["thread_id"] = ev.get("thread_id")
            summary["metadata"] = ev.get("metadata")
        elif ev.get("type") == "trace_end":
            summary["duration_s"] = ev.get("duration_s")
            summary["stats"] = ev.get("stats", {})
            summary["output"] = ev.get("output")
            summary["children"] = ev.get("children")

    return summary


def _make_handler(traces_dir: Path, static_dir: Path):
    """Create a request handler class bound to the given directories."""

    class TraqoHandler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/traces":
                self._handle_traces_list()
            elif path == "/api/trace":
                qs = urllib.parse.parse_qs(parsed.query)
                file_param = qs.get("file", [None])[0]
                self._handle_trace_detail(file_param)
            elif path == "/" or path == "/index.html":
                self._serve_static("index.html")
            else:
                self.send_error(404, "Not found")

        def _json_response(self, data: Any, status: int = 200) -> None:
            body = json.dumps(data, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, filename: str) -> None:
            filepath = static_dir / filename
            if not filepath.is_file():
                self.send_error(404, "Not found")
                return
            body = filepath.read_bytes()
            self.send_response(200)
            content_type = "text/html" if filename.endswith(".html") else "application/octet-stream"
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_traces_list(self) -> None:
            jsonl_files = sorted(traces_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            summaries = []
            for f in jsonl_files:
                try:
                    events = _read_jsonl(f)
                    summary = _trace_summary(f, events)
                    summary["file"] = str(f.relative_to(traces_dir))
                    summaries.append(summary)
                except Exception:
                    continue
            self._json_response(summaries)

        def _handle_trace_detail(self, file_param: str | None) -> None:
            if not file_param:
                self._json_response({"error": "Missing ?file= parameter"}, 400)
                return

            target = (traces_dir / file_param).resolve()
            # Security: ensure the file is within traces_dir
            try:
                target.relative_to(traces_dir.resolve())
            except ValueError:
                self._json_response({"error": "Path traversal not allowed"}, 403)
                return

            if not target.is_file():
                self._json_response({"error": f"File not found: {file_param}"}, 404)
                return

            events = _read_jsonl(target)
            self._json_response({"file": file_param, "events": events})

        def log_message(self, format: str, *args: Any) -> None:
            # Quieter logging — only show errors
            if args and str(args[1]).startswith("4"):
                super().log_message(format, *args)

    return TraqoHandler


def serve(traces_dir: str | Path, port: int = 7600) -> None:
    """Start the traqo trace viewer server."""
    traces_path = Path(traces_dir).resolve()
    if not traces_path.is_dir():
        print(f"Error: {traces_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    static_dir = Path(__file__).parent / "static"
    handler = _make_handler(traces_path, static_dir)
    server = HTTPServer(("127.0.0.1", port), handler)

    print(f"traqo ui — serving traces from {traces_path}")
    print(f"Open http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="traqo trace viewer")
    parser.add_argument("traces_dir", nargs="?", default=".", help="Directory containing .jsonl trace files")
    parser.add_argument("--port", "-p", type=int, default=7600, help="Port to serve on (default: 7600)")
    args = parser.parse_args()
    serve(args.traces_dir, args.port)


if __name__ == "__main__":
    main()
