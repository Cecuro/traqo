"""S3-compatible storage backend."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import boto3
except ImportError as err:
    raise ImportError(
        "boto3 is not installed. Install with: pip install traqo[s3]"
    ) from err

from traqo.backend import submit_background

logger = logging.getLogger(__name__)


class S3Backend:
    """Upload completed trace files to S3-compatible storage.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix (e.g. ``"traces/production/"``).  Defaults to ``""``.
        key_fn: Optional callable ``(Path) -> str`` that returns the S3 key
            for a given trace file.  Overrides *prefix*-based key generation.
        boto3_client: Optional pre-configured boto3 S3 client.  If not
            provided, a new client is created with default credentials.
        upload_kwargs: Extra keyword arguments forwarded to
            ``s3.upload_file()`` (e.g. ``ExtraArgs`` for server-side
            encryption).
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        key_fn: Callable[[Path], str] | None = None,
        boto3_client: Any = None,
        upload_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix
        self._key_fn = key_fn
        self._client = boto3_client or boto3.client("s3")
        self._upload_kwargs = upload_kwargs or {}

    def on_event(self, event: dict[str, Any]) -> None:
        pass

    def on_trace_complete(self, trace_path: Path) -> None:
        submit_background(self._upload, trace_path)

    def close(self) -> None:
        pass

    def _make_key(self, trace_path: Path) -> str:
        if self._key_fn is not None:
            return self._key_fn(trace_path)
        return f"{self._prefix}{trace_path.name}"

    def _upload(self, trace_path: Path) -> None:
        key = self._make_key(trace_path)
        self._client.upload_file(
            str(trace_path),
            self._bucket,
            key,
            **self._upload_kwargs,
        )
        logger.debug(
            "traqo: uploaded %s to s3://%s/%s",
            trace_path,
            self._bucket,
            key,
        )
