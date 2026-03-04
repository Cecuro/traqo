"""Trace sources for the UI server.

Abstracts reading traces from local directories, S3, or GCS.
Cloud sources use a local temp-dir cache: listing is instant from the API,
full file content is downloaded on first access and served from cache thereafter.
"""

from __future__ import annotations

import gzip
import json
import logging
import tempfile
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

    def read_content(self, key: str, span_id: str) -> Any | None: ...


# ---------------------------------------------------------------------------
# Helpers (moved from server.py)
# ---------------------------------------------------------------------------


def _open_jsonl(path: Path):
    """Open a JSONL file, transparently handling .gz compression."""
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return list of parsed events."""
    events: list[dict[str, Any]] = []
    with _open_jsonl(path) as f:
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
    with _open_jsonl(path) as f:
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


def _content_key(key: str) -> str | None:
    """Derive the content file key from a main trace key.

    ``"trace.jsonl.gz"`` -> ``"trace.content.jsonl.zst"``
    ``"trace.jsonl"`` -> ``"trace.content.jsonl.zst"``
    """
    if key.endswith(".jsonl.gz"):
        return key[: -len(".jsonl.gz")] + ".content.jsonl.zst"
    if key.endswith(".jsonl"):
        return key[: -len(".jsonl")] + ".content.jsonl.zst"
    return None


def _is_trace_file(name: str) -> bool:
    """Return True if the filename is a main trace file (not a content file)."""
    if name.endswith(".content.jsonl.zst"):
        return False
    return name.endswith(".jsonl") or name.endswith(".jsonl.gz")


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
        # Collect both *.jsonl and *.jsonl.gz, excluding *.content.jsonl.zst
        trace_files: list[Path] = []
        for f in self._path.rglob("*"):
            if f.is_file() and _is_trace_file(f.name):
                trace_files.append(f)
        trace_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Deduplicate: if both .jsonl and .jsonl.gz exist, prefer .jsonl.gz
        seen_stems: set[str] = set()
        summaries: list[TraceSummary] = []
        for f in trace_files:
            key = str(f.relative_to(self._path))
            # Normalize stem for dedup: strip .jsonl.gz or .jsonl
            if f.name.endswith(".jsonl.gz"):
                stem = f.name[: -len(".jsonl.gz")]
            else:
                stem = f.name[: -len(".jsonl")]
            rel_stem = str(f.parent.relative_to(self._path) / stem)
            if rel_stem in seen_stems:
                continue
            seen_stems.add(rel_stem)
            try:
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

    def read_content(self, key: str, span_id: str) -> Any | None:
        content_rel = _content_key(key)
        if content_rel is None:
            return None
        target = (self._path / content_rel).resolve()
        if not self._is_safe_path(target) or not target.is_file():
            return None
        from traqo.compress import read_content

        return read_content(target, span_id)

    def _is_safe_path(self, resolved: Path) -> bool:
        try:
            resolved.relative_to(self._path)
            return True
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# S3Source
# ---------------------------------------------------------------------------


class S3Source:
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
        self._cache_dir = Path(tempfile.mkdtemp(prefix="traqo-s3-"))
        # key -> last_modified epoch from cloud listing
        self._cloud_mtimes: dict[str, float] = {}

    def list_traces(self) -> list[TraceSummary]:
        summaries: list[TraceSummary] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                obj_key: str = obj["Key"]
                rel = obj_key[len(self._prefix) :] if self._prefix else obj_key
                if not _is_trace_file(rel):
                    continue
                mtime = obj["LastModified"].timestamp()
                self._cloud_mtimes[rel] = mtime

                summary = TraceSummary(key=rel, file=rel, last_modified=mtime)

                # Enrich from cache if available
                cached = self._cache_dir / rel
                if cached.is_file():
                    try:
                        first, last = _read_first_last_lines(cached)
                        _enrich_summary(summary, first, last)
                    except Exception:
                        pass

                summaries.append(summary)

        # Sort by last_modified descending
        summaries.sort(key=lambda s: s.last_modified or 0, reverse=True)
        logger.info(
            "listed %d traces from s3://%s/%s",
            len(summaries),
            self._bucket,
            self._prefix,
        )
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
        need_download = True

        if cached.is_file():
            cloud_mtime = self._cloud_mtimes.get(key)
            if cloud_mtime is not None and cached.stat().st_mtime >= cloud_mtime:
                need_download = False

        if need_download:
            self._download(key, cached)

        return _read_jsonl(cached)

    def read_content(self, key: str, span_id: str) -> Any | None:
        content_rel = _content_key(key)
        if content_rel is None:
            return None
        cached = self._cache_dir / content_rel
        if not cached.is_file():
            try:
                self._download(content_rel, cached)
            except Exception:
                logger.warning(
                    "traqo: failed to download content file %s",
                    content_rel,
                    exc_info=True,
                )
                return None
        from traqo.compress import read_content

        return read_content(cached, span_id)

    def _download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3_key = f"{self._prefix}{key}"
        logger.info("downloading s3://%s/%s", self._bucket, s3_key)
        self._client.download_file(self._bucket, s3_key, str(dest))


# ---------------------------------------------------------------------------
# GCSSource
# ---------------------------------------------------------------------------


class GCSSource:
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
        self._bucket = client.bucket(bucket)
        self._cache_dir = Path(tempfile.mkdtemp(prefix="traqo-gcs-"))
        self._cloud_mtimes: dict[str, float] = {}

    def list_traces(self) -> list[TraceSummary]:
        summaries: list[TraceSummary] = []
        blobs = self._bucket.list_blobs(prefix=self._prefix or None)
        for blob in blobs:
            name: str = blob.name
            rel = name[len(self._prefix) :] if self._prefix else name
            if not _is_trace_file(rel):
                continue
            mtime = blob.updated.timestamp() if blob.updated else 0.0
            self._cloud_mtimes[rel] = mtime

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
        logger.info(
            "listed %d traces from gs://%s/%s",
            len(summaries),
            self._bucket_name,
            self._prefix,
        )
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
        need_download = True

        if cached.is_file():
            cloud_mtime = self._cloud_mtimes.get(key)
            if cloud_mtime is not None and cached.stat().st_mtime >= cloud_mtime:
                need_download = False

        if need_download:
            self._download(key, cached)

        return _read_jsonl(cached)

    def read_content(self, key: str, span_id: str) -> Any | None:
        content_rel = _content_key(key)
        if content_rel is None:
            return None
        cached = self._cache_dir / content_rel
        if not cached.is_file():
            try:
                self._download(content_rel, cached)
            except Exception:
                logger.warning(
                    "traqo: failed to download content file %s",
                    content_rel,
                    exc_info=True,
                )
                return None
        from traqo.compress import read_content

        return read_content(cached, span_id)

    def _download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob_name = f"{self._prefix}{key}"
        logger.info("downloading gs://%s/%s", self._bucket_name, blob_name)
        blob = self._bucket.blob(blob_name)
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
