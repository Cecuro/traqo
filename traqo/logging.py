"""Logging configuration for traqo.

Library users can opt in to traqo log output:

    from traqo.logging import setup_logging
    setup_logging()           # WARNING level (backend failures, write errors)
    setup_logging(verbose=True)  # INFO level (cloud downloads, listing, etc.)

The ``traqo ui`` CLI calls this automatically.
"""

from __future__ import annotations

import logging
import os

_configured = False


def setup_logging(*, verbose: bool = False) -> None:
    """Configure the ``traqo`` logger hierarchy.

    Attaches a stderr handler with a clean format to the ``"traqo"`` logger.
    Safe to call multiple times — only configures once.

    Args:
        verbose: If True, set level to INFO. Otherwise WARNING.
            Overridden by the ``TRAQO_LOG_LEVEL`` environment variable
            (e.g. ``DEBUG``, ``INFO``, ``WARNING``).
    """
    global _configured
    if _configured:
        return
    _configured = True

    logger = logging.getLogger("traqo")

    env_level = os.environ.get("TRAQO_LOG_LEVEL", "").strip().upper()
    if env_level and hasattr(logging, env_level):
        level = getattr(logging, env_level)
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("traqo: %(message)s"))
    logger.addHandler(handler)
