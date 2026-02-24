"""Minimal HTTP server for the traqo trace viewer.

Usage:
    python -m traqo.ui [TRACES_DIR_OR_URI] [--port PORT]

Supports local directories, S3, and GCS:
    traqo ui ./local/traces
    traqo ui s3://my-bucket/traces/
    traqo ui gs://my-bucket/traces/
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from traqo.ui.sources import TraceSource, parse_source

MIME_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


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


def _make_handler(source: TraceSource, static_dir: Path, *, dev: bool = False):
    """Create a request handler class bound to the given source."""

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
            summaries = source.list_traces()
            self._json_response([s.to_dict() for s in summaries])

        def _handle_trace_detail(self, file_param: str | None) -> None:
            if not file_param:
                self._json_response({"error": "Missing ?file= parameter"}, 400)
                return

            events = source.read_all(file_param)
            if not events:
                self._json_response({"error": f"File not found: {file_param}"}, 404)
                return

            self._json_response({"file": file_param, "events": events})

        def log_message(self, format: str, *args: Any) -> None:
            # Quieter logging — only show errors
            if args and str(args[1]).startswith("4"):
                super().log_message(format, *args)

    return TraqoHandler


def serve(
    source_uri: str | Path,
    port: int = 7600,
    *,
    dev: bool = False,
    static_dir: str | Path | None = None,
) -> None:
    """Start the traqo trace viewer server."""
    uri = str(source_uri)

    # For local paths, validate the directory exists
    if not uri.startswith("s3://") and not uri.startswith("gs://"):
        local_path = Path(uri).resolve()
        if not local_path.is_dir():
            print(f"Error: {local_path} is not a directory", file=sys.stderr)
            sys.exit(1)

    source = parse_source(uri)

    static_path = Path(static_dir).resolve() if static_dir else Path(__file__).parent / "static"
    if not static_path.is_dir():
        print(f"Error: static dir {static_path} not found", file=sys.stderr)
        sys.exit(1)

    handler = _make_handler(source, static_path, dev=dev)
    server = HTTPServer(("127.0.0.1", port), handler)

    print(f"traqo ui — serving traces from {uri}")
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
    parser.add_argument(
        "source",
        nargs="?",
        default=".",
        help="Trace source: local directory, s3://bucket/prefix, or gs://bucket/prefix",
    )
    parser.add_argument("--port", "-p", type=int, default=7600, help="Port to serve on (default: 7600)")
    parser.add_argument("--dev", action="store_true", help="Enable hot reload on static file changes")
    parser.add_argument("--static-dir", default=None, help="Override static file directory (for development)")
    args = parser.parse_args()
    serve(args.source, args.port, dev=args.dev, static_dir=args.static_dir)


if __name__ == "__main__":
    main()
