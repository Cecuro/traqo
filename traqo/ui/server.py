"""Minimal HTTP server for the traqo trace viewer.

Usage:
    python -m traqo.ui [TRACES_DIR] [--port PORT]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

MIME_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


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


def _read_first_last(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read only the first and last lines of a JSONL file for fast summary."""
    first = None
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if first is None:
                first = parsed
            last = parsed
    return first, last


def _trace_summary(path: Path, first: dict[str, Any] | None, last: dict[str, Any] | None) -> dict[str, Any]:
    """Extract summary info from a trace's first/last events."""
    summary: dict[str, Any] = {"file": str(path.name)}

    if first and first.get("type") == "trace_start":
        summary["ts"] = first.get("ts")
        summary["input"] = first.get("input")
        summary["tags"] = first.get("tags", [])
        summary["thread_id"] = first.get("thread_id")

    if last and last.get("type") == "trace_end":
        summary["duration_s"] = last.get("duration_s")
        summary["stats"] = last.get("stats", {})

    return summary


def _static_mtime(static_dir: Path) -> float:
    """Return the most recent mtime of any file in the static directory."""
    latest = 0.0
    for p in static_dir.rglob("*"):
        if p.is_file():
            latest = max(latest, p.stat().st_mtime)
    return latest


_RELOAD_SCRIPT = """
<script>
(function(){var mt=0;setInterval(function(){fetch('/api/mtime').then(r=>r.json()).then(d=>{if(mt&&d.mtime!==mt)location.reload();mt=d.mtime;}).catch(()=>{});},1000);})();
</script>
"""


def _make_handler(traces_dir: Path, static_dir: Path, *, dev: bool = False):
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
            elif path == "/api/mtime":
                self._json_response({"mtime": _static_mtime(static_dir)})
            elif path == "/":
                self._serve_static("index.html")
            else:
                self._serve_static(path)

        def _json_response(self, data: Any, status: int = 200) -> None:
            body = json.dumps(data, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, rel_path: str) -> None:
            clean = urllib.parse.unquote(rel_path).lstrip("/")
            if not clean:
                clean = "index.html"
            filepath = (static_dir / clean).resolve()

            # Security: ensure the file is within static_dir
            try:
                filepath.relative_to(static_dir.resolve())
            except ValueError:
                self.send_error(403, "Forbidden")
                return

            if not filepath.is_file():
                self.send_error(404, "Not found")
                return

            ext = filepath.suffix.lower()
            mime = MIME_TYPES.get(ext, "application/octet-stream")
            body = filepath.read_bytes()
            if dev and ext == ".html":
                body = body.replace(b"</body>", _RELOAD_SCRIPT.encode() + b"</body>")
            self.send_response(200)
            if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
                self.send_header("Content-Type", f"{mime}; charset=utf-8")
            else:
                self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_traces_list(self) -> None:
            jsonl_files = sorted(traces_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            summaries = []
            for f in jsonl_files:
                try:
                    first, last = _read_first_last(f)
                    summary = _trace_summary(f, first, last)
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


def serve(
    traces_dir: str | Path,
    port: int = 7600,
    *,
    dev: bool = False,
    static_dir: str | Path | None = None,
) -> None:
    """Start the traqo trace viewer server."""
    traces_path = Path(traces_dir).resolve()
    if not traces_path.is_dir():
        print(f"Error: {traces_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    static_path = Path(static_dir).resolve() if static_dir else Path(__file__).parent / "static"
    if not static_path.is_dir():
        print(f"Error: static dir {static_path} not found", file=sys.stderr)
        sys.exit(1)

    handler = _make_handler(traces_path, static_path, dev=dev)
    server = HTTPServer(("127.0.0.1", port), handler)

    print(f"traqo ui — serving traces from {traces_path}")
    if dev:
        print(f"Hot reload enabled — serving static files from {static_path}")
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
    parser.add_argument("--dev", action="store_true", help="Enable hot reload on static file changes")
    parser.add_argument("--static-dir", default=None, help="Override static file directory (for development)")
    args = parser.parse_args()
    serve(args.traces_dir, args.port, dev=args.dev, static_dir=args.static_dir)


if __name__ == "__main__":
    main()
