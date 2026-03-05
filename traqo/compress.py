"""Split and compress trace files for efficient storage and loading.

Splits a raw JSONL trace into two compressed files:
- Main file (.jsonl.gz): everything except large span_start inputs
- Content file (.content.jsonl.zst): externalized large inputs, loaded on demand

Large span_start inputs are replaced with a reference stub:
    {"_ref": "<span_id>", "_size": <byte_count>}
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 10240  # 10 KB


def split_and_compress(
    trace_path: Path,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> tuple[Path, Path | None]:
    """Split a raw JSONL trace into compressed main + optional content files.

    Reads the trace line by line. For ``span_start`` events where the
    serialized ``input`` exceeds *threshold* bytes, the input is moved to
    the content file and replaced with a ``{"_ref": span_id, "_size": N}``
    stub in the main file.

    Args:
        trace_path: Path to the raw ``.jsonl`` trace file.
        threshold: Byte-size threshold for externalizing inputs.

    Returns:
        ``(main_path, content_path_or_None)`` — content path is ``None``
        when no inputs exceeded the threshold.
    """
    import zstandard as zstd

    stem = trace_path.stem  # e.g. "trace_abc123" from "trace_abc123.jsonl"
    parent = trace_path.parent
    main_path = parent / f"{stem}.jsonl.gz"
    content_path = parent / f"{stem}.content.jsonl.zst"

    has_content = False
    cctx = zstd.ZstdCompressor(level=3)

    content_file = None
    ok = False
    try:
        with (
            open(trace_path, encoding="utf-8") as raw,
            gzip.open(main_path, "wt", encoding="utf-8", compresslevel=6) as main_out,
        ):
            for line in raw:
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    # Preserve malformed lines in main file as-is
                    main_out.write(stripped + "\n")
                    continue

                if (
                    event.get("type") == "span_start"
                    and "input" in event
                    and event.get("id")
                ):
                    input_json = json.dumps(event["input"], separators=(",", ":"))
                    if len(input_json.encode("utf-8")) > threshold:
                        # Externalize this input
                        if content_file is None:
                            content_file = cctx.stream_writer(
                                open(content_path, "wb")  # noqa: SIM115
                            )
                            has_content = True

                        content_entry = json.dumps(
                            {"span_id": event["id"], "input": event["input"]},
                            separators=(",", ":"),
                        )
                        content_file.write((content_entry + "\n").encode("utf-8"))

                        # Replace input with reference stub
                        event["input"] = {
                            "_ref": event["id"],
                            "_size": len(input_json.encode("utf-8")),
                        }

                main_out.write(json.dumps(event, separators=(",", ":")) + "\n")
        ok = True
    finally:
        if content_file is not None:
            content_file.close()
        if not ok:
            # Clean up partial files so the raw .jsonl remains the sole source
            main_path.unlink(missing_ok=True)
            content_path.unlink(missing_ok=True)

    if not has_content:
        content_path.unlink(missing_ok=True)
        return main_path, None

    return main_path, content_path


def read_content(content_path: Path, span_id: str) -> dict | None:
    """Stream-decompress a content file to find a specific span's input.

    Uses streaming decompression with small chunks to keep peak memory
    around ~1 MB, even for content files that decompress to hundreds of MB.

    Args:
        content_path: Path to a ``.content.jsonl.zst`` file.
        span_id: The span ID to look up.

    Returns:
        The span's input data (as a parsed dict/list/str), or ``None``
        if the span ID is not found.
    """
    import zstandard as zstd

    if not content_path.is_file():
        return None

    dctx = zstd.ZstdDecompressor()
    buffer = b""

    with open(content_path, "rb") as f:
        reader = dctx.stream_reader(f, read_size=65536)
        while True:
            chunk = reader.read(65536)
            if not chunk:
                break
            buffer += chunk

            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                if not line_bytes:
                    continue
                try:
                    entry = json.loads(line_bytes)
                except json.JSONDecodeError:
                    continue
                if entry.get("span_id") == span_id:
                    return entry.get("input")

    # Check remaining buffer (last line without trailing newline)
    if buffer.strip():
        try:
            entry = json.loads(buffer)
            if entry.get("span_id") == span_id:
                return entry.get("input")
        except json.JSONDecodeError:
            pass

    return None
