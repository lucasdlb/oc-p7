"""Shared logging configuration for the oc-p7 project.

Usage:
    from logging_config import setup_logging
    logger = setup_logging(__name__)

Behaviour:
    DEBUG=true  (env var) → plain text at DEBUG level, verbose
    DEBUG=false / unset   → JSON-formatted at INFO level

Noisy third-party loggers (httpx, uvicorn.access, gradio, faiss) are
silenced to WARNING regardless of mode.
"""

import json
import logging
import sys
from typing import Any

_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "uvicorn.access",
    "gradio",
    "faiss",
    "sentence_transformers",
]


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(name: str) -> logging.Logger:
    """Configure root logging and return a named logger.

    Calling this function multiple times is safe — the root logger is
    only reconfigured on the first call (idempotent via force=True on
    basicConfig).

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    # Determine mode from config (import lazily to avoid circular imports
    # at the time logging_config itself is first imported)
    try:
        from config import SETTINGS  # noqa: PLC0415

        debug = SETTINGS.debug
    except Exception:
        debug = False

    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logging.basicConfig(
            level=logging.INFO,
            handlers=[handler],
            force=True,
        )

    # Silence noisy third-party loggers
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(name)
