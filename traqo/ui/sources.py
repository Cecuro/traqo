"""Trace sources for the UI server.

Abstracts reading traces from local directories, S3, or GCS.
Cloud sources use a local temp-dir cache: listing is instant from the API,
full file content is downloaded on first access and served from cache thereafter.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class TraceSummary:
    """Lightweight summary for the trace list view."""

    key: str  # relative path / object key
    last_modified: float | None = None  # epoch seconds
    file: str = ""  # display name
    ts: str | None = None
    input: Any = None
    tags: list[str] = field(default_factory=list)
    thread_id: str | None = None
    duration_s: float | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"file": self.file or self.key}
        if self.ts is not None:
            d["ts"] = self.ts
        if self.input is not None:
            d["input"] = self.input
        if self.tags:
            d["tags"] = self.tags
        if self.thread_id is not None:
            d["thread_id"] = self.thread_id
        if self.duration_s is not None:
            d["duration_s"] = self.duration_s
        if self.stats:
            d["stats"] = self.stats
        return d


class TraceSource(Protocol):
    """Protocol for reading traces from any storage backend."""

    def list_traces(self) -> list[TraceSummary]: ...

    def read_first_last(
        self, key: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]: ...

    def read_all(self, key: str) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Helpers (moved from server.py)
# ---------------------------------------------------------------------------


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


def _read_first_last_lines(
    path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
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


def _enrich_summary(
    summary: TraceSummary, first: dict[str, Any] | None, last: dict[str, Any] | None
) -> None:
    """Populate summary fields from trace_start / trace_end events."""
    if first and first.get("type") == "trace_start":
        summary.ts = first.get("ts")
        summary.input = first.get("input")
        summary.tags = first.get("tags", [])
        summary.thread_id = first.get("thread_id")
    if last and last.get("type") == "trace_end":
        summary.duration_s = last.get("duration_s")
        summary.stats = last.get("stats", {})


# ---------------------------------------------------------------------------
# LocalSource
# ---------------------------------------------------------------------------


class LocalSource:
    """Read traces from a local directory."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()

    def list_traces(self) -> list[TraceSummary]:
        jsonl_files = sorted(
            self._path.rglob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        summaries: list[TraceSummary] = []
        for f in jsonl_files:
            try:
                key = str(f.relative_to(self._path))
                first, last = _read_first_last_lines(f)
                summary = TraceSummary(
                    key=key, file=key, last_modified=f.stat().st_mtime
                )
                _enrich_summary(summary, first, last)
                summaries.append(summary)
            except Exception:
                continue
        return summaries

    def read_first_last(
        self, key: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        target = (self._path / key).resolve()
        if not self._is_safe_path(target):
            return None, None
        if not target.is_file():
            return None, None
        return _read_first_last_lines(target)

    def read_all(self, key: str) -> list[dict[str, Any]]:
        target = (self._path / key).resolve()
        if not self._is_safe_path(target):
            return []
        if not target.is_file():
            return []
        return _read_jsonl(target)

    def _is_safe_path(self, resolved: Path) -> bool:
        try:
            resolved.relative_to(self._path)
            return True
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# _CachedCloudSource — shared caching logic for S3 and GCS
# ---------------------------------------------------------------------------


class _CachedCloudSource:
    """Base for cloud sources with local temp-dir caching.

    Subclasses implement ``_iter_blobs()`` and ``_download()``.
    """

    def __init__(self, cache_prefix: str, label: str) -> None:
        self._cache_dir = Path(tempfile.mkdtemp(prefix=f"traqo-{cache_prefix}-"))
        self._label = label

    # -- abstract (override in subclasses) ----------------------------------

    def _iter_blobs(self) -> Iterator[tuple[str, float]]:
        """Yield ``(relative_key, mtime_epoch)`` for each ``.jsonl`` object."""
        raise NotImplementedError

    def _download(self, key: str, dest: Path) -> None:
        """Download a single object to *dest*."""
        raise NotImplementedError

    # -- shared logic -------------------------------------------------------

    def list_traces(self) -> list[TraceSummary]:
        summaries: list[TraceSummary] = []
        for rel, mtime in self._iter_blobs():
            summary = TraceSummary(key=rel, file=rel, last_modified=mtime)
            cached = self._cache_dir / rel
            if cached.is_file():
                try:
                    first, last = _read_first_last_lines(cached)
                    _enrich_summary(summary, first, last)
                except Exception:
                    pass
            summaries.append(summary)
        summaries.sort(key=lambda s: s.last_modified or 0, reverse=True)
        logger.info("listed %d traces from %s", len(summaries), self._label)
        return summaries

    def read_first_last(
        self, key: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        cached = self._cache_dir / key
        if cached.is_file():
            return _read_first_last_lines(cached)
        return None, None

    def read_all(self, key: str) -> list[dict[str, Any]]:
        cached = self._cache_dir / key
        if not cached.is_file():
            self._download(key, cached)
        return _read_jsonl(cached)


# ---------------------------------------------------------------------------
# S3Source
# ---------------------------------------------------------------------------


class S3Source(_CachedCloudSource):
    """Read traces from an S3 bucket with local temp-dir cache."""

    def __init__(
        self, bucket: str, prefix: str = "", *, boto3_client: Any = None
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix
        if boto3_client is not None:
            self._client = boto3_client
        else:
            try:
                import boto3
            except ImportError as err:
                raise ImportError(
                    "boto3 is not installed. Install with: pip install traqo[s3]"
                ) from err
            self._client = boto3.client("s3")
        super().__init__("s3", f"s3://{bucket}/{prefix}")

    def _iter_blobs(self) -> Iterator[tuple[str, float]]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if not key.endswith(".jsonl"):
                    continue
                rel = key[len(self._prefix) :] if self._prefix else key
                yield rel, obj["LastModified"].timestamp()

    def _download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3_key = f"{self._prefix}{key}"
        logger.info("downloading s3://%s/%s", self._bucket, s3_key)
        self._client.download_file(self._bucket, s3_key, str(dest))


# ---------------------------------------------------------------------------
# GCSSource
# ---------------------------------------------------------------------------


class GCSSource(_CachedCloudSource):
    """Read traces from a GCS bucket with local temp-dir cache."""

    def __init__(
        self, bucket: str, prefix: str = "", *, gcs_client: Any = None
    ) -> None:
        self._bucket_name = bucket
        self._prefix = prefix
        if gcs_client is not None:
            client = gcs_client
        else:
            try:
                from google.cloud import storage as _gcs_storage
            except ImportError as err:
                raise ImportError(
                    "google-cloud-storage is not installed. "
                    "Install with: pip install traqo[gcs]"
                ) from err
            client = _gcs_storage.Client()
        self._gcs_bucket = client.bucket(bucket)
        super().__init__("gcs", f"gs://{bucket}/{prefix}")

    def _iter_blobs(self) -> Iterator[tuple[str, float]]:
        for blob in self._gcs_bucket.list_blobs(prefix=self._prefix or None):
            name: str = blob.name
            if not name.endswith(".jsonl"):
                continue
            rel = name[len(self._prefix) :] if self._prefix else name
            yield rel, blob.updated.timestamp() if blob.updated else 0.0

    def _download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob_name = f"{self._prefix}{key}"
        logger.info("downloading gs://%s/%s", self._bucket_name, blob_name)
        blob = self._gcs_bucket.blob(blob_name)
        blob.download_to_filename(str(dest))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def parse_source(uri: str) -> TraceSource:
    """Parse a URI into a TraceSource.

    Supported formats:
        s3://bucket/prefix/     -> S3Source
        gs://bucket/prefix/     -> GCSSource
        /local/path             -> LocalSource
    """
    if uri.startswith("s3://"):
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3Source(bucket, prefix)

    if uri.startswith("gs://"):
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return GCSSource(bucket, prefix)

    return LocalSource(Path(uri))
