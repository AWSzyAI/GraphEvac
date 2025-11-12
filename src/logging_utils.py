"""
Lightweight logging setup for GraphEvac.

Provides a shared logger named "GraphEvac" that logs to log/run.log. Use get_logger() from any module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


_LOGGER_NAME = "GraphEvac"
_LOG_FORMAT = "%(message)s"


def setup_logging(debug: bool = False, logfile: Optional[str] = None) -> logging.Logger:
    """Configure and return the project logger.

    - Creates log directory if needed.
    - Logs to log/run.log by default.
    - Sets level to DEBUG when debug=True else INFO.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        # Already configured; still allow level toggling
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        return logger

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Ensure log directory exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = logfile or os.path.join(log_dir, "run.log")

    fmt = logging.Formatter(_LOG_FORMAT)
    # File handler
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # Fallback to console-only if file not writable
        pass

    logger.debug("Logger initialized. Level=%s, file=%s", logging.getLevelName(level), log_path)
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
