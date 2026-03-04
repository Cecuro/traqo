"""Tests for traqo.ui.sources — LocalSource, S3Source, GCSSource, parse_source."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from traqo.ui.sources import (
    LocalSource,
    TraceSummary,
    _enrich_summary,
    parse_source,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_trace(path: Path, events: list[dict[str, Any]]) -> None:
    """Write a list of events as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _make_trace_events(
    *,
    input_val: Any = "hello",
    tags: list[str] | None = None,
    duration: float = 1.5,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create minimal trace_start + trace_end events."""
    return [
        {
            "type": "trace_start",
            "ts": "2025-01-01T00:00:00Z",
            "input": input_val,
            "tags": tags or [],
        },
        {
            "type": "trace_end",
            "duration_s": duration,
            "stats": stats or {"input_tokens": 10},
        },
    ]


# ---------------------------------------------------------------------------
# TraceSummary
# ---------------------------------------------------------------------------


class TestTraceSummary:
    def test_to_dict_minimal(self):
        s = TraceSummary(key="run.jsonl")
        d = s.to_dict()
        assert d == {"file": "run.jsonl"}

    def test_to_dict_full(self):
        s = TraceSummary(
            key="run.jsonl",
            file="run.jsonl",
            ts="2025-01-01T00:00:00Z",
            input="hi",
            tags=["prod"],
            thread_id="t1",
            duration_s=2.0,
            stats={"input_tokens": 10},
        )
        d = s.to_dict()
        assert d["file"] == "run.jsonl"
        assert d["ts"] == "2025-01-01T00:00:00Z"
        assert d["input"] == "hi"
        assert d["tags"] == ["prod"]
        assert d["thread_id"] == "t1"
        assert d["duration_s"] == 2.0
        assert d["stats"] == {"input_tokens": 10}

    def test_to_dict_omits_none_fields(self):
        s = TraceSummary(key="run.jsonl", file="run.jsonl")
        d = s.to_dict()
        assert "ts" not in d
        assert "duration_s" not in d
        assert "thread_id" not in d


class TestEnrichSummary:
    def test_enriches_from_trace_start_and_end(self):
        s = TraceSummary(key="run.jsonl")
        first = {"type": "trace_start", "ts": "2025-01-01", "input": "q", "tags": ["a"]}
        last = {"type": "trace_end", "duration_s": 3.0, "stats": {"x": 1}}
        _enrich_summary(s, first, last)
        assert s.ts == "2025-01-01"
        assert s.input == "q"
        assert s.tags == ["a"]
        assert s.duration_s == 3.0
        assert s.stats == {"x": 1}

    def test_enriches_with_none_events(self):
        s = TraceSummary(key="run.jsonl")
        _enrich_summary(s, None, None)
        assert s.ts is None
        assert s.duration_s is None


# ---------------------------------------------------------------------------
# LocalSource
# ---------------------------------------------------------------------------


class TestLocalSource:
    def test_list_traces(self, tmp_path: Path):
        _write_trace(tmp_path / "a.jsonl", _make_trace_events(input_val="first"))
        _write_trace(tmp_path / "b.jsonl", _make_trace_events(input_val="second"))

        source = LocalSource(tmp_path)
        traces = source.list_traces()

        assert len(traces) == 2
        keys = {t.key for t in traces}
        assert "a.jsonl" in keys
        assert "b.jsonl" in keys

    def test_list_traces_enriched(self, tmp_path: Path):
        _write_trace(
            tmp_path / "run.jsonl",
            _make_trace_events(input_val="test", tags=["prod"], duration=2.5),
        )
        source = LocalSource(tmp_path)
        traces = source.list_traces()
        assert len(traces) == 1
        t = traces[0]
        assert t.input == "test"
        assert t.tags == ["prod"]
        assert t.duration_s == 2.5
        assert t.ts == "2025-01-01T00:00:00Z"

    def test_list_traces_nested_dirs(self, tmp_path: Path):
        _write_trace(tmp_path / "sub" / "deep.jsonl", _make_trace_events())
        source = LocalSource(tmp_path)
        traces = source.list_traces()
        assert len(traces) == 1
        assert traces[0].key == "sub/deep.jsonl"

    def test_list_traces_skips_non_jsonl(self, tmp_path: Path):
        (tmp_path / "readme.txt").write_text("not a trace")
        _write_trace(tmp_path / "run.jsonl", _make_trace_events())
        source = LocalSource(tmp_path)
        assert len(source.list_traces()) == 1

    def test_read_all(self, tmp_path: Path):
        events = _make_trace_events()
        _write_trace(tmp_path / "run.jsonl", events)
        source = LocalSource(tmp_path)
        result = source.read_all("run.jsonl")
        assert len(result) == 2
        assert result[0]["type"] == "trace_start"
        assert result[1]["type"] == "trace_end"

    def test_read_all_missing_file(self, tmp_path: Path):
        source = LocalSource(tmp_path)
        assert source.read_all("nonexistent.jsonl") == []

    def test_read_all_path_traversal(self, tmp_path: Path):
        source = LocalSource(tmp_path)
        assert source.read_all("../../etc/passwd") == []

    def test_read_first_last(self, tmp_path: Path):
        _write_trace(tmp_path / "run.jsonl", _make_trace_events())
        source = LocalSource(tmp_path)
        first, last = source.read_first_last("run.jsonl")
        assert first is not None
        assert first["type"] == "trace_start"
        assert last is not None
        assert last["type"] == "trace_end"

    def test_read_first_last_missing_file(self, tmp_path: Path):
        source = LocalSource(tmp_path)
        first, last = source.read_first_last("missing.jsonl")
        assert first is None
        assert last is None

    def test_empty_directory(self, tmp_path: Path):
        source = LocalSource(tmp_path)
        assert source.list_traces() == []


# ---------------------------------------------------------------------------
# S3Source (mocked)
# ---------------------------------------------------------------------------


class TestS3Source:
    def _make_source(
        self, mock_client: MagicMock, bucket: str = "b", prefix: str = "traces/"
    ):
        from traqo.ui.sources import S3Source

        return S3Source(bucket, prefix, boto3_client=mock_client)

    def _make_paginator(self, objects: list[dict[str, Any]]) -> MagicMock:
        """Create a mock paginator returning a single page of objects."""
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": objects}]
        return mock_client

    def test_import_guard(self):
        try:
            from traqo.ui.sources import S3Source  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")

    def test_list_traces(self):
        try:
            from traqo.ui.sources import S3Source  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")

        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_client = self._make_paginator(
            [
                {"Key": "traces/run1.jsonl", "LastModified": now},
                {"Key": "traces/run2.jsonl", "LastModified": now},
                {"Key": "traces/readme.txt", "LastModified": now},
            ]
        )

        source = self._make_source(mock_client)
        traces = source.list_traces()

        assert len(traces) == 2
        keys = {t.key for t in traces}
        assert "run1.jsonl" in keys
        assert "run2.jsonl" in keys

    def test_list_traces_strips_prefix(self):
        try:
            from traqo.ui.sources import S3Source  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")

        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_client = self._make_paginator(
            [
                {"Key": "traces/sub/deep.jsonl", "LastModified": now},
            ]
        )
        source = self._make_source(mock_client)
        traces = source.list_traces()
        assert traces[0].key == "sub/deep.jsonl"

    def test_read_all_downloads_and_caches(self, tmp_path: Path):
        try:
            from traqo.ui.sources import S3Source  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")

        events = _make_trace_events()
        content = "\n".join(json.dumps(e) for e in events)

        mock_client = MagicMock()

        def fake_download(bucket, key, dest):
            Path(dest).write_text(content)

        mock_client.download_file.side_effect = fake_download

        source = self._make_source(mock_client)
        result = source.read_all("run.jsonl")

        assert len(result) == 2
        assert result[0]["type"] == "trace_start"
        mock_client.download_file.assert_called_once_with(
            "b", "traces/run.jsonl", str(source._cache_dir / "run.jsonl")
        )

    def test_read_all_uses_cache_on_second_call(self, tmp_path: Path):
        try:
            from traqo.ui.sources import S3Source  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")

        events = _make_trace_events()
        content = "\n".join(json.dumps(e) for e in events)

        mock_client = MagicMock()

        def fake_download(bucket, key, dest):
            Path(dest).write_text(content)

        mock_client.download_file.side_effect = fake_download

        source = self._make_source(mock_client)
        # Simulate cloud mtime in the past so cache is valid
        source._cloud_mtimes["run.jsonl"] = 0.0

        # First call downloads
        source.read_all("run.jsonl")
        assert mock_client.download_file.call_count == 1

        # Second call uses cache (file mtime >= cloud mtime of 0.0)
        source.read_all("run.jsonl")
        assert mock_client.download_file.call_count == 1

    def test_read_first_last_returns_none_before_download(self):
        try:
            from traqo.ui.sources import S3Source  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")

        mock_client = MagicMock()
        source = self._make_source(mock_client)
        first, last = source.read_first_last("run.jsonl")
        assert first is None
        assert last is None

    def test_read_first_last_works_after_download(self):
        try:
            from traqo.ui.sources import S3Source  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")

        events = _make_trace_events()
        content = "\n".join(json.dumps(e) for e in events)

        mock_client = MagicMock()

        def fake_download(bucket, key, dest):
            Path(dest).write_text(content)

        mock_client.download_file.side_effect = fake_download

        source = self._make_source(mock_client)
        source.read_all("run.jsonl")

        first, last = source.read_first_last("run.jsonl")
        assert first is not None
        assert first["type"] == "trace_start"
        assert last is not None
        assert last["type"] == "trace_end"

    def test_list_enriches_cached_files(self):
        try:
            from traqo.ui.sources import S3Source  # noqa: F401
        except ImportError:
            pytest.skip("boto3 not installed")

        events = _make_trace_events(input_val="cached", duration=5.0)
        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_client = self._make_paginator(
            [
                {"Key": "traces/run.jsonl", "LastModified": now},
            ]
        )

        def fake_download(bucket, key, dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text("\n".join(json.dumps(e) for e in events))

        mock_client.download_file.side_effect = fake_download

        source = self._make_source(mock_client)

        # Download to cache first
        source.read_all("run.jsonl")

        # Now list should pick up cached data
        mock_client.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "traces/run.jsonl", "LastModified": now}]}
        ]
        traces = source.list_traces()
        assert len(traces) == 1
        assert traces[0].input == "cached"
        assert traces[0].duration_s == 5.0


# ---------------------------------------------------------------------------
# GCSSource (mocked)
# ---------------------------------------------------------------------------


class TestGCSSource:
    def _make_source(
        self, mock_client: MagicMock, bucket: str = "b", prefix: str = "traces/"
    ):
        from traqo.ui.sources import GCSSource

        return GCSSource(bucket, prefix, gcs_client=mock_client)

    def _make_blob(self, name: str, updated: datetime | None = None) -> MagicMock:
        blob = MagicMock()
        blob.name = name
        blob.updated = updated
        return blob

    def test_import_guard(self):
        try:
            from traqo.ui.sources import GCSSource  # noqa: F401
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

    def test_list_traces(self):
        try:
            from traqo.ui.sources import GCSSource  # noqa: F401
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = [
            self._make_blob("traces/run1.jsonl", now),
            self._make_blob("traces/run2.jsonl", now),
            self._make_blob("traces/readme.txt", now),
        ]

        source = self._make_source(mock_client)
        traces = source.list_traces()

        assert len(traces) == 2
        keys = {t.key for t in traces}
        assert "run1.jsonl" in keys
        assert "run2.jsonl" in keys

    def test_list_traces_strips_prefix(self):
        try:
            from traqo.ui.sources import GCSSource  # noqa: F401
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

        now = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = [
            self._make_blob("traces/sub/deep.jsonl", now),
        ]

        source = self._make_source(mock_client)
        traces = source.list_traces()
        assert traces[0].key == "sub/deep.jsonl"

    def test_read_all_downloads_and_caches(self):
        try:
            from traqo.ui.sources import GCSSource  # noqa: F401
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

        events = _make_trace_events()
        content = "\n".join(json.dumps(e) for e in events)

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        def fake_download(dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text(content)

        mock_blob.download_to_filename.side_effect = fake_download

        source = self._make_source(mock_client)
        result = source.read_all("run.jsonl")

        assert len(result) == 2
        assert result[0]["type"] == "trace_start"
        mock_bucket.blob.assert_called_with("traces/run.jsonl")
        mock_blob.download_to_filename.assert_called_once()

    def test_read_all_uses_cache_on_second_call(self):
        try:
            from traqo.ui.sources import GCSSource  # noqa: F401
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

        events = _make_trace_events()
        content = "\n".join(json.dumps(e) for e in events)

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        def fake_download(dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text(content)

        mock_blob.download_to_filename.side_effect = fake_download

        source = self._make_source(mock_client)
        source._cloud_mtimes["run.jsonl"] = 0.0

        source.read_all("run.jsonl")
        assert mock_blob.download_to_filename.call_count == 1

        source.read_all("run.jsonl")
        assert mock_blob.download_to_filename.call_count == 1

    def test_read_first_last_returns_none_before_download(self):
        try:
            from traqo.ui.sources import GCSSource  # noqa: F401
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        source = self._make_source(mock_client)
        first, last = source.read_first_last("run.jsonl")
        assert first is None
        assert last is None

    def test_read_first_last_works_after_download(self):
        try:
            from traqo.ui.sources import GCSSource  # noqa: F401
        except ImportError:
            pytest.skip("google-cloud-storage not installed")

        events = _make_trace_events()
        content = "\n".join(json.dumps(e) for e in events)

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        def fake_download(dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text(content)

        mock_blob.download_to_filename.side_effect = fake_download

        source = self._make_source(mock_client)
        source.read_all("run.jsonl")

        first, last = source.read_first_last("run.jsonl")
        assert first is not None
        assert first["type"] == "trace_start"
        assert last is not None
        assert last["type"] == "trace_end"


# ---------------------------------------------------------------------------
# parse_source
# ---------------------------------------------------------------------------


class TestParseSource:
    def test_local_path(self, tmp_path: Path):
        source = parse_source(str(tmp_path))
        assert isinstance(source, LocalSource)

    def test_s3_uri(self):
        try:
            source = parse_source("s3://my-bucket/prefix/path/")
        except ImportError:
            pytest.skip("boto3 not installed")
        from traqo.ui.sources import S3Source

        assert isinstance(source, S3Source)
        assert source._bucket == "my-bucket"
        assert source._prefix == "prefix/path/"

    def test_gs_uri(self):
        try:
            source = parse_source("gs://my-bucket/traces/")
        except ImportError:
            pytest.skip("google-cloud-storage not installed")
        from traqo.ui.sources import GCSSource

        assert isinstance(source, GCSSource)
        assert source._bucket_name == "my-bucket"
        assert source._prefix == "traces/"

    def test_s3_uri_no_prefix(self):
        try:
            source = parse_source("s3://my-bucket")
        except ImportError:
            pytest.skip("boto3 not installed")
        from traqo.ui.sources import S3Source

        assert isinstance(source, S3Source)
        assert source._bucket == "my-bucket"
        assert source._prefix == ""

    def test_gs_uri_no_prefix(self):
        try:
            source = parse_source("gs://my-bucket")
        except ImportError:
            pytest.skip("google-cloud-storage not installed")
        from traqo.ui.sources import GCSSource

        assert isinstance(source, GCSSource)
        assert source._bucket_name == "my-bucket"
        assert source._prefix == ""

    def test_relative_local_path(self):
        source = parse_source(".")
        assert isinstance(source, LocalSource)


# ---------------------------------------------------------------------------
# _open_jsonl magic-bytes check
# ---------------------------------------------------------------------------


class TestOpenJsonl:
    """Test that _open_jsonl checks gzip magic bytes, not just extension."""

    def test_gz_extension_with_gzip_content(self, tmp_path: Path):
        """A real .gz file should be opened with gzip."""
        import gzip as gzip_mod

        from traqo.ui.sources import _open_jsonl

        path = tmp_path / "trace.jsonl.gz"
        with gzip_mod.open(path, "wt", encoding="utf-8") as f:
            f.write('{"type": "trace_start"}\n')

        with _open_jsonl(path) as f:
            line = f.readline().strip()
        assert '"trace_start"' in line

    def test_gz_extension_with_plain_content(self, tmp_path: Path):
        """A .gz file that's actually plain text (GCS transparent decompression)."""
        from traqo.ui.sources import _open_jsonl

        path = tmp_path / "trace.jsonl.gz"
        path.write_text('{"type": "trace_start"}\n')

        with _open_jsonl(path) as f:
            line = f.readline().strip()
        assert '"trace_start"' in line

    def test_plain_jsonl(self, tmp_path: Path):
        """A regular .jsonl file."""
        from traqo.ui.sources import _open_jsonl

        path = tmp_path / "trace.jsonl"
        path.write_text('{"type": "trace_start"}\n')

        with _open_jsonl(path) as f:
            line = f.readline().strip()
        assert '"trace_start"' in line


# ---------------------------------------------------------------------------
# _trace_stem and dedup helpers
# ---------------------------------------------------------------------------


class TestTraceStem:
    def test_jsonl_gz(self):
        from traqo.ui.sources import _trace_stem

        assert _trace_stem("foo.jsonl.gz") == "foo"

    def test_jsonl(self):
        from traqo.ui.sources import _trace_stem

        assert _trace_stem("foo.jsonl") == "foo"

    def test_nested_path(self):
        from traqo.ui.sources import _trace_stem

        assert _trace_stem("sub/dir/trace.jsonl.gz") == "sub/dir/trace"

    def test_other_extension(self):
        from traqo.ui.sources import _trace_stem

        assert _trace_stem("file.txt") == "file.txt"


class TestLocalSourceDedup:
    """LocalSource should deduplicate .jsonl and .jsonl.gz with same stem."""

    def test_prefers_gz_over_raw(self, tmp_path: Path):
        import gzip as gzip_mod

        events = _make_trace_events()

        # Write both raw and compressed
        raw = tmp_path / "trace.jsonl"
        _write_trace(raw, events)
        gz = tmp_path / "trace.jsonl.gz"
        with gzip_mod.open(gz, "wt", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        source = LocalSource(tmp_path)
        traces = source.list_traces()

        assert len(traces) == 1
        assert traces[0].key.endswith(".jsonl.gz")

    def test_excludes_content_files(self, tmp_path: Path):
        events = _make_trace_events()
        _write_trace(tmp_path / "trace.jsonl", events)
        # Create a content sidecar — should not appear in listing
        (tmp_path / "trace.content.jsonl.zst").write_bytes(b"fake")

        source = LocalSource(tmp_path)
        traces = source.list_traces()

        assert len(traces) == 1
        assert not any("content" in t.key for t in traces)


class TestLocalSourceReadContent:
    """LocalSource.read_content should find externalized span inputs."""

    def test_read_content_roundtrip(self, tmp_path: Path):
        from traqo.compress import split_and_compress

        large_input = {"messages": [{"content": "x" * 20000}]}
        events = [
            {"type": "trace_start", "ts": "t"},
            {
                "type": "span_start",
                "id": "s1",
                "name": "test",
                "kind": "llm",
                "ts": "t",
                "input": large_input,
            },
            {"type": "span_end", "id": "s1", "name": "test", "status": "ok"},
            {"type": "trace_end", "ts": "t"},
        ]
        _write_trace(tmp_path / "trace.jsonl", events)
        main_path, content_path = split_and_compress(tmp_path / "trace.jsonl")
        assert content_path is not None

        source = LocalSource(tmp_path)
        result = source.read_content(main_path.name, "s1")
        assert result == large_input

    def test_read_content_returns_none_for_no_content_file(self, tmp_path: Path):
        events = _make_trace_events()
        _write_trace(tmp_path / "trace.jsonl", events)

        source = LocalSource(tmp_path)
        result = source.read_content("trace.jsonl", "nonexistent")
        assert result is None
