"""Tests for storage backend protocol and implementations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import traqo
from tests.conftest import read_events
from traqo import Backend, Tracer
from traqo.backend import (
    flush_backends,
    shutdown_backends,
)
from traqo.backends.local import LocalBackend

# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class RecordingBackend:
    """Fake backend that records all calls for assertions."""

    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.trace_complete_paths: list[Path] = []
        self.close_count = 0

    def on_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def on_trace_complete(self, trace_path: Path) -> None:
        self.trace_complete_paths.append(trace_path)
        return None

    def close(self) -> None:
        self.close_count += 1


class CrashingBackend:
    """Backend that raises on every call."""

    def on_event(self, event: dict[str, Any]) -> None:
        raise RuntimeError("on_event crash")

    def on_trace_complete(self, trace_path: Path) -> None:
        raise RuntimeError("on_trace_complete crash")

    def close(self) -> None:
        raise RuntimeError("close crash")


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestBackendProtocol:
    def test_recording_backend_satisfies_protocol(self):
        assert isinstance(RecordingBackend(), Backend)

    def test_crashing_backend_satisfies_protocol(self):
        assert isinstance(CrashingBackend(), Backend)

    def test_missing_method_fails_protocol(self):
        class Incomplete:
            def on_event(self, event):
                pass

        assert not isinstance(Incomplete(), Backend)

    def test_tracer_validates_backends_on_construction(self, trace_file):
        with pytest.raises(TypeError, match="missing required callable method"):
            Tracer(path=trace_file, backends=[{"not": "a backend"}])

    def test_tracer_validates_non_callable_attribute(self, trace_file):
        class BadBackend:
            on_event = "not callable"
            on_trace_complete = None
            close = 42

        with pytest.raises(TypeError, match="missing required callable method"):
            Tracer(path=trace_file, backends=[BadBackend()])


# ---------------------------------------------------------------------------
# Tracer integration
# ---------------------------------------------------------------------------


class TestTracerBackendIntegration:
    def test_backend_receives_all_events(self, trace_file):
        backend = RecordingBackend()
        with Tracer(path=trace_file, backends=[backend]) as tracer:
            tracer.log("hello")
            with tracer.span("step"):
                pass
        # trace_start + event + span_start + span_end + trace_end = 5
        assert len(backend.events) == 5
        types = [e["type"] for e in backend.events]
        assert types == [
            "trace_start",
            "event",
            "span_start",
            "span_end",
            "trace_end",
        ]

    def test_backend_receives_trace_complete(self, trace_file):
        backend = RecordingBackend()
        with Tracer(path=trace_file, backends=[backend]):
            pass
        assert len(backend.trace_complete_paths) == 1
        assert backend.trace_complete_paths[0] == trace_file

    def test_backend_close_not_called_from_exit(self, trace_file):
        """Backends are long-lived — close() is NOT called from __exit__."""
        backend = RecordingBackend()
        with Tracer(path=trace_file, backends=[backend]):
            pass
        assert backend.close_count == 0

    def test_multiple_backends(self, trace_file):
        b1 = RecordingBackend()
        b2 = RecordingBackend()
        with Tracer(path=trace_file, backends=[b1, b2]) as tracer:
            tracer.log("x")
        # trace_start + event + trace_end = 3
        assert len(b1.events) == 3
        assert len(b2.events) == 3
        assert len(b1.trace_complete_paths) == 1
        assert len(b2.trace_complete_paths) == 1

    def test_no_backends_is_default(self, trace_file):
        """Omitting backends= produces normal behavior."""
        with Tracer(path=trace_file) as tracer, tracer.span("step"):
            pass
        events = read_events(trace_file)
        assert events[0]["type"] == "trace_start"
        assert events[-1]["type"] == "trace_end"

    def test_empty_backends_list(self, trace_file):
        with Tracer(path=trace_file, backends=[]) as tracer:
            tracer.log("ok")
        events = read_events(trace_file)
        assert len(events) == 3


# ---------------------------------------------------------------------------
# Child tracer propagation
# ---------------------------------------------------------------------------


class TestChildBackendInheritance:
    def test_child_inherits_backends(self, tmp_path):
        backend = RecordingBackend()
        parent_path = tmp_path / "parent.jsonl"
        with Tracer(path=parent_path, backends=[backend]) as parent:
            child = parent.child("agent_a")
            with child:
                child.log("from_child")
        # Child should trigger on_trace_complete for a child file
        assert len(backend.trace_complete_paths) >= 1
        child_paths = [p for p in backend.trace_complete_paths if "agent_a_" in p.name]
        assert len(child_paths) == 1

    def test_child_events_reach_backend(self, tmp_path):
        backend = RecordingBackend()
        parent_path = tmp_path / "parent.jsonl"
        with Tracer(path=parent_path, backends=[backend]) as parent:
            child = parent.child("agent_a")
            with child:
                child.log("child_event")
        # Find the child's log event
        child_events = [
            e
            for e in backend.events
            if e.get("type") == "event" and e.get("name") == "child_event"
        ]
        assert len(child_events) == 1


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestBackendErrorResilience:
    def test_crashing_backend_doesnt_crash_tracer(self, trace_file):
        backend = CrashingBackend()
        with Tracer(path=trace_file, backends=[backend]) as tracer:
            tracer.log("still works")
        events = read_events(trace_file)
        assert len(events) == 3  # trace_start, event, trace_end

    def test_crashing_backend_logs_warning(self, trace_file, caplog):
        backend = CrashingBackend()
        with caplog.at_level(logging.WARNING):
            with Tracer(path=trace_file, backends=[backend]):
                pass
        assert "backend" in caplog.text.lower()

    def test_crash_doesnt_block_other_backends(self, trace_file):
        crashing = CrashingBackend()
        recording = RecordingBackend()
        with Tracer(path=trace_file, backends=[crashing, recording]) as tracer:
            tracer.log("ok")
        # recording should still get all events despite crashing going first
        assert len(recording.events) == 3
        assert len(recording.trace_complete_paths) == 1


# ---------------------------------------------------------------------------
# Disabled tracer
# ---------------------------------------------------------------------------


class TestDisabledTracerBackends:
    def test_disabled_tracer_no_backend_event_calls(self, trace_file):
        traqo.disable()
        backend = RecordingBackend()
        with Tracer(path=trace_file, backends=[backend]):
            pass
        assert len(backend.events) == 0

    def test_disabled_tracer_no_trace_complete_calls(self, trace_file):
        traqo.disable()
        backend = RecordingBackend()
        with Tracer(path=trace_file, backends=[backend]):
            pass
        assert len(backend.trace_complete_paths) == 0


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


class TestLocalBackend:
    def test_copies_file_to_directory(self, tmp_path):
        trace_file = tmp_path / "source" / "run.jsonl"
        target_dir = tmp_path / "collected"
        backend = LocalBackend(target_dir)

        with Tracer(path=trace_file, backends=[backend]) as tracer:
            tracer.log("hello")

        copied_files = list(target_dir.iterdir())
        assert len(copied_files) == 1
        assert copied_files[0].suffix == ".jsonl"
        assert "run" in copied_files[0].stem
        # Verify content matches
        original_events = read_events(trace_file)
        copied_events = read_events(copied_files[0])
        assert original_events == copied_events

    def test_organize_by_date(self, tmp_path):
        trace_file = tmp_path / "run.jsonl"
        target_dir = tmp_path / "collected"
        backend = LocalBackend(target_dir, organize_by_date=True)

        with Tracer(path=trace_file, backends=[backend]):
            pass

        # Should have a date subdirectory
        subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
        assert len(subdirs) == 1
        # Date format: YYYY-MM-DD
        assert len(subdirs[0].name) == 10
        assert subdirs[0].name.count("-") == 2

        files = list(subdirs[0].iterdir())
        assert len(files) == 1

    def test_collision_safe_filenames(self, tmp_path):
        trace_file = tmp_path / "run.jsonl"
        target_dir = tmp_path / "collected"
        backend = LocalBackend(target_dir)

        # Run twice with same source filename
        with Tracer(path=trace_file, backends=[backend]):
            pass
        with Tracer(path=trace_file, backends=[backend]):
            pass

        files = list(target_dir.iterdir())
        assert len(files) == 2
        # Both should have unique names
        assert files[0].name != files[1].name

    def test_creates_target_directory(self, tmp_path):
        target_dir = tmp_path / "deep" / "nested" / "dir"
        trace_file = tmp_path / "run.jsonl"
        backend = LocalBackend(target_dir)

        with Tracer(path=trace_file, backends=[backend]):
            pass

        assert target_dir.exists()
        assert len(list(target_dir.iterdir())) == 1

    def test_satisfies_protocol(self):
        assert isinstance(LocalBackend(Path("/tmp")), Backend)


# ---------------------------------------------------------------------------
# S3Backend (mocked)
# ---------------------------------------------------------------------------


class TestS3Backend:
    def test_upload_called_on_trace_complete(self, trace_file):
        mock_client = MagicMock()
        # Inline import to avoid ImportError when boto3 is not installed
        try:
            from traqo.backends.s3 import S3Backend
        except ImportError:
            pytest.skip("boto3 not installed")

        backend = S3Backend("my-bucket", prefix="traces/", boto3_client=mock_client)

        with Tracer(path=trace_file, backends=[backend]):
            pass

        flush_backends()
        mock_client.upload_file.assert_called_once_with(
            str(trace_file), "my-bucket", f"traces/{trace_file.name}"
        )

    def test_custom_key_fn(self, trace_file):
        mock_client = MagicMock()
        try:
            from traqo.backends.s3 import S3Backend
        except ImportError:
            pytest.skip("boto3 not installed")

        backend = S3Backend(
            "my-bucket",
            key_fn=lambda p: f"custom/{p.stem}_v2{p.suffix}",
            boto3_client=mock_client,
        )

        with Tracer(path=trace_file, backends=[backend]):
            pass

        flush_backends()
        mock_client.upload_file.assert_called_once_with(
            str(trace_file),
            "my-bucket",
            f"custom/{trace_file.stem}_v2{trace_file.suffix}",
        )

    def test_satisfies_protocol(self):
        try:
            from traqo.backends.s3 import S3Backend
        except ImportError:
            pytest.skip("boto3 not installed")
        assert isinstance(S3Backend("b", boto3_client=MagicMock()), Backend)


# ---------------------------------------------------------------------------
# GCSBackend (mocked)
# ---------------------------------------------------------------------------


class TestGCSBackend:
    def test_upload_called_on_trace_complete(self, trace_file):
        try:
            from traqo.backends.gcs import GCSBackend
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        backend = GCSBackend("my-bucket", prefix="traces/", gcs_client=mock_client)

        with Tracer(path=trace_file, backends=[backend]):
            pass

        flush_backends()
        mock_bucket.blob.assert_called_once_with(f"traces/{trace_file.name}")
        mock_blob.upload_from_filename.assert_called_once_with(
            str(trace_file), content_type="application/x-ndjson"
        )

    def test_custom_blob_name_fn(self, trace_file):
        try:
            from traqo.backends.gcs import GCSBackend
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        backend = GCSBackend(
            "my-bucket",
            blob_name_fn=lambda p: f"custom/{p.name}",
            gcs_client=mock_client,
        )

        with Tracer(path=trace_file, backends=[backend]):
            pass

        flush_backends()
        mock_bucket.blob.assert_called_once_with(f"custom/{trace_file.name}")

    def test_satisfies_protocol(self):
        try:
            from traqo.backends.gcs import GCSBackend
        except ImportError:
            pytest.skip("google-cloud-storage not installed")
        mock_client = MagicMock()
        mock_client.bucket.return_value = MagicMock()
        assert isinstance(GCSBackend("b", gcs_client=mock_client), Backend)


# ---------------------------------------------------------------------------
# Background executor utilities
# ---------------------------------------------------------------------------


class TestFlushAndShutdown:
    def test_flush_allows_new_work_after(self, trace_file):
        backend = RecordingBackend()
        with Tracer(path=trace_file, backends=[backend]):
            pass
        flush_backends()
        # Should be able to start a new trace after flush
        trace_file2 = trace_file.parent / "run2.jsonl"
        with Tracer(path=trace_file2, backends=[backend]):
            pass
        assert len(backend.trace_complete_paths) == 2

    def test_shutdown_then_new_trace_still_works(self, trace_file):
        """After shutdown, a new executor is created lazily."""
        shutdown_backends()
        backend = RecordingBackend()
        with Tracer(path=trace_file, backends=[backend]):
            pass
        # Events are delivered synchronously via on_event, so this works
        assert len(backend.events) == 2  # trace_start + trace_end
