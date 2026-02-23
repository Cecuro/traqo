"""Local filesystem storage backend."""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalBackend:
    """Copy completed trace files to a centralized local directory.

    This is a zero-dependency backend useful for collecting traces from
    scattered locations into a single directory for analysis.

    Args:
        directory: Target directory to copy trace files into.
        organize_by_date: If ``True``, create subdirectories by date
            (``YYYY-MM-DD``).  Defaults to ``False``.
    """

    def __init__(
        self,
        directory: Path,
        *,
        organize_by_date: bool = False,
    ) -> None:
        self._directory = Path(directory)
        self._organize_by_date = organize_by_date

    def on_event(self, event: dict[str, Any]) -> None:  # noqa: ARG002
        pass

    def on_trace_complete(self, trace_path: Path) -> None:
        try:
            self._copy(trace_path)
        except Exception:
            logger.warning(
                "traqo: LocalBackend failed to copy %s", trace_path, exc_info=True
            )

    def close(self) -> None:
        pass

    def _copy(self, trace_path: Path) -> None:
        target_dir = self._directory
        if self._organize_by_date:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            target_dir = target_dir / date_str

        target_dir.mkdir(parents=True, exist_ok=True)

        # Collision-safe filename: stem_YYYYMMDD_HHMMSS_hex8.suffix
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        dest_name = f"{trace_path.stem}_{ts}_{short_id}{trace_path.suffix}"

        dest = target_dir / dest_name
        shutil.copy2(trace_path, dest)
        logger.debug("traqo: copied %s to %s", trace_path, dest)
