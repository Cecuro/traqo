"""Tests for traqo.compress — split-and-compress logic."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from traqo.compress import read_content, split_and_compress

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_trace(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _make_span_start(
    span_id: str, input_data: Any, kind: str = "llm"
) -> dict[str, Any]:
    return {
        "type": "span_start",
        "id": span_id,
        "name": f"span_{span_id}",
        "kind": kind,
        "ts": "2025-01-01T00:00:00Z",
        "input": input_data,
    }


def _make_span_end(span_id: str) -> dict[str, Any]:
    return {
        "type": "span_end",
        "id": span_id,
        "name": f"span_{span_id}",
        "ts": "2025-01-01T00:00:01Z",
        "duration_s": 1.0,
        "status": "ok",
    }


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# No large inputs — content file should not be created
# ---------------------------------------------------------------------------


class TestNoLargeInputs:
    def test_small_inputs_only(self, tmp_path: Path):
        trace = tmp_path / "small.jsonl"
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("s1", "small input"),
            _make_span_end("s1"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        main_path, content_path = split_and_compress(trace)

        assert main_path.exists()
        assert main_path.name == "small.jsonl.gz"
        assert content_path is None

        # Verify main file contains all events with original inputs
        main_events = _read_gzip_jsonl(main_path)
        assert len(main_events) == 4
        assert main_events[1]["input"] == "small input"

    def test_empty_trace(self, tmp_path: Path):
        trace = tmp_path / "empty.jsonl"
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            {"type": "trace_end", "ts": "2025-01-01T00:00:00Z"},
        ]
        _write_trace(trace, events)

        main_path, content_path = split_and_compress(trace)
        assert main_path.exists()
        assert content_path is None
        assert len(_read_gzip_jsonl(main_path)) == 2


# ---------------------------------------------------------------------------
# With large inputs — content file should be created
# ---------------------------------------------------------------------------


class TestWithLargeInputs:
    def test_large_input_externalized(self, tmp_path: Path):
        trace = tmp_path / "big.jsonl"
        large_input = {"data": "x" * 20000}  # Well over 10 KB
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("s1", large_input),
            _make_span_end("s1"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        main_path, content_path = split_and_compress(trace)

        assert main_path.exists()
        assert content_path is not None
        assert content_path.exists()
        assert content_path.name == "big.content.jsonl.zst"

        # Main file should have a reference stub instead of the large input
        main_events = _read_gzip_jsonl(main_path)
        span_start = main_events[1]
        assert span_start["input"]["_ref"] == "s1"
        assert span_start["input"]["_size"] > 10000

    def test_mixed_small_and_large_inputs(self, tmp_path: Path):
        trace = tmp_path / "mixed.jsonl"
        large_input = {"data": "y" * 20000}
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("s1", "small"),
            _make_span_end("s1"),
            _make_span_start("s2", large_input),
            _make_span_end("s2"),
            _make_span_start("s3", "also small"),
            _make_span_end("s3"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        main_path, content_path = split_and_compress(trace)

        assert content_path is not None
        main_events = _read_gzip_jsonl(main_path)

        # s1 and s3 should have inline inputs
        assert main_events[1]["input"] == "small"
        assert main_events[5]["input"] == "also small"

        # s2 should have a reference
        assert main_events[3]["input"]["_ref"] == "s2"

    def test_span_end_outputs_stay_inline(self, tmp_path: Path):
        trace = tmp_path / "outputs.jsonl"
        large_input = {"data": "z" * 20000}
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("s1", large_input),
            {
                "type": "span_end",
                "id": "s1",
                "name": "span_s1",
                "output": {"large_output": "w" * 20000},
                "status": "ok",
            },
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        main_path, _ = split_and_compress(trace)
        main_events = _read_gzip_jsonl(main_path)

        # span_end output stays inline regardless of size
        assert main_events[2]["output"]["large_output"] == "w" * 20000


# ---------------------------------------------------------------------------
# Threshold edge cases
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_exactly_at_threshold_stays_inline(self, tmp_path: Path):
        trace = tmp_path / "edge.jsonl"
        # Create input that serializes to exactly threshold bytes
        # With separators=(",", ":"), {"data":"x..."} = ~threshold
        threshold = 100
        # {"data":"xxx..."} serialized with compact separators
        # 10 bytes overhead, so need threshold - 10 chars of x
        input_data = {"d": "x" * (threshold - 10)}
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("s1", input_data),
            _make_span_end("s1"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        main_path, content_path = split_and_compress(trace, threshold=threshold)
        main_events = _read_gzip_jsonl(main_path)

        # Check: if serialized size <= threshold, stays inline
        serialized_size = len(
            json.dumps(input_data, separators=(",", ":")).encode("utf-8")
        )
        if serialized_size <= threshold:
            assert content_path is None
            assert "_ref" not in str(main_events[1].get("input", {}))
        else:
            assert content_path is not None

    def test_custom_threshold(self, tmp_path: Path):
        trace = tmp_path / "custom.jsonl"
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("s1", "x" * 500),
            _make_span_end("s1"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        # With threshold=100, this 500-byte input should be externalized
        main_path, content_path = split_and_compress(trace, threshold=100)
        assert content_path is not None

        main_events = _read_gzip_jsonl(main_path)
        assert main_events[1]["input"]["_ref"] == "s1"


# ---------------------------------------------------------------------------
# Malformed lines
# ---------------------------------------------------------------------------


class TestMalformedLines:
    def test_malformed_lines_preserved_in_main(self, tmp_path: Path):
        trace = tmp_path / "malformed.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        with open(trace, "w") as f:
            f.write(json.dumps({"type": "trace_start", "ts": "t"}) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"type": "trace_end", "ts": "t"}) + "\n")

        main_path, content_path = split_and_compress(trace)
        assert content_path is None

        # Main file should have 3 lines (malformed preserved as-is)
        main_events = []
        with gzip.open(main_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    main_events.append(line)
        assert len(main_events) == 3
        assert main_events[1] == "not valid json"


# ---------------------------------------------------------------------------
# Roundtrip: split then read_content
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_read_content_retrieves_externalized_input(self, tmp_path: Path):
        trace = tmp_path / "roundtrip.jsonl"
        large_input = {"messages": [{"role": "user", "content": "a" * 20000}]}
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("span_abc", large_input),
            _make_span_end("span_abc"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        _, content_path = split_and_compress(trace)
        assert content_path is not None

        # Read back the externalized content
        result = read_content(content_path, "span_abc")
        assert result is not None
        assert result == large_input

    def test_read_content_returns_none_for_missing_span(self, tmp_path: Path):
        trace = tmp_path / "missing.jsonl"
        large_input = {"data": "x" * 20000}
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("s1", large_input),
            _make_span_end("s1"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        _, content_path = split_and_compress(trace)
        assert content_path is not None

        result = read_content(content_path, "nonexistent")
        assert result is None

    def test_read_content_returns_none_for_missing_file(self, tmp_path: Path):
        result = read_content(tmp_path / "nonexistent.zst", "s1")
        assert result is None

    def test_multiple_externalized_inputs(self, tmp_path: Path):
        trace = tmp_path / "multi.jsonl"
        input1 = {"messages": [{"content": "a" * 20000}]}
        input2 = {"messages": [{"content": "b" * 20000}]}
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            _make_span_start("s1", input1),
            _make_span_end("s1"),
            _make_span_start("s2", input2),
            _make_span_end("s2"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        _, content_path = split_and_compress(trace)
        assert content_path is not None

        # Both should be retrievable
        assert read_content(content_path, "s1") == input1
        assert read_content(content_path, "s2") == input2


# ---------------------------------------------------------------------------
# span_start without id — should not crash
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_span_start_without_id_stays_inline(self, tmp_path: Path):
        trace = tmp_path / "noid.jsonl"
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            {"type": "span_start", "name": "no_id", "input": "x" * 20000},
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        _, content_path = split_and_compress(trace, threshold=100)
        # No id means it can't be externalized
        assert content_path is None

    def test_span_start_without_input(self, tmp_path: Path):
        trace = tmp_path / "noinput.jsonl"
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z"},
            {"type": "span_start", "id": "s1", "name": "test"},
            _make_span_end("s1"),
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        main_path, content_path = split_and_compress(trace)
        assert content_path is None
        assert len(_read_gzip_jsonl(main_path)) == 4

    def test_non_span_start_events_with_large_data(self, tmp_path: Path):
        """Only span_start inputs get externalized, not other event types."""
        trace = tmp_path / "event.jsonl"
        events = [
            {"type": "trace_start", "ts": "2025-01-01T00:00:00Z", "input": "x" * 20000},
            {"type": "trace_end", "ts": "2025-01-01T00:00:02Z"},
        ]
        _write_trace(trace, events)

        main_path, content_path = split_and_compress(trace, threshold=100)
        assert content_path is None

        # trace_start input should stay inline
        main_events = _read_gzip_jsonl(main_path)
        assert main_events[0]["input"] == "x" * 20000
