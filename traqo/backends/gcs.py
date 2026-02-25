"""Google Cloud Storage backend."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from google.cloud import storage as gcs_storage
except ImportError as err:
    raise ImportError(
        "google-cloud-storage is not installed. Install with: pip install traqo[gcs]"
    ) from err

from traqo.backend import submit_background

logger = logging.getLogger(__name__)


class GCSBackend:
    """Upload completed trace files to Google Cloud Storage.

    Args:
        bucket: GCS bucket name.
        prefix: Blob name prefix (e.g. ``"traces/production/"``).
            Defaults to ``""``.
        blob_name_fn: Optional callable ``(Path) -> str`` that returns the
            blob name for a given trace file.  Overrides *prefix*-based
            name generation.
        gcs_client: Optional pre-configured ``google.cloud.storage.Client``.
            If not provided, a new client is created with Application Default
            Credentials.
        upload_kwargs: Extra keyword arguments forwarded to
            ``blob.upload_from_filename()``.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        blob_name_fn: Callable[[Path], str] | None = None,
        gcs_client: Any = None,
        upload_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._bucket_name = bucket
        self._prefix = prefix
        self._blob_name_fn = blob_name_fn
        client = gcs_client or gcs_storage.Client()
        self._bucket = client.bucket(bucket)
        self._upload_kwargs = upload_kwargs or {}

    def on_event(self, event: dict[str, Any]) -> None:
        pass

    def on_trace_complete(self, trace_path: Path) -> None:
        submit_background(self._upload, trace_path)

    def close(self) -> None:
        pass

    def _make_blob_name(self, trace_path: Path) -> str:
        if self._blob_name_fn is not None:
            return self._blob_name_fn(trace_path)
        return f"{self._prefix}{trace_path.name}"

    def _upload(self, trace_path: Path) -> None:
        blob_name = self._make_blob_name(trace_path)
        blob = self._bucket.blob(blob_name)
        blob.upload_from_filename(
            str(trace_path),
            content_type="application/x-ndjson",
            **self._upload_kwargs,
        )
        logger.debug(
            "traqo: uploaded %s to gs://%s/%s",
            trace_path,
            self._bucket_name,
            blob_name,
        )
