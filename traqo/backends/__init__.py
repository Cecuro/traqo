"""Storage backends for traqo.

Concrete backends are imported from their own modules to avoid pulling
in optional dependencies at package level::

    from traqo.backends.s3 import S3Backend
    from traqo.backends.gcs import GCSBackend
    from traqo.backends.local import LocalBackend
"""

from __future__ import annotations

from traqo.backend import Backend

__all__ = ["Backend"]
