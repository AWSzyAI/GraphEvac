"""
Lightweight logging setup for GraphEvac.

Provides a shared logger named "GraphEvac" that writes to console by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_LOGGER_NAME = "GraphEvac"
_LOG_FORMAT = "%(message)s"


def setup_logging(debug: bool = False) -> logging.Logger:
    """Return the project logger configured for console output."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        return logger

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    fmt = logging.Formatter(_LOG_FORMAT)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.debug("Logger initialized. Level=%s", logging.getLevelName(level))
    return logger


def get_logger() -> logging.Logger:
    """Return the shared project logger (call setup_logging() first)."""
    return logging.getLogger(_LOGGER_NAME)


def add_file_handler(path: Path | str, level: Optional[int] = None) -> None:
    """Add an additional file handler (used for per-run output dir)."""
    logger = get_logger()
    log_path = str(path)
    extra = getattr(logger, "_extra_file_handlers", set())
    if log_path in extra:
        return
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(_LOG_FORMAT)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level or logger.level)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    extra.add(log_path)
    logger._extra_file_handlers = extra
    logger.debug("Added extra log handler @ %s", log_path)
