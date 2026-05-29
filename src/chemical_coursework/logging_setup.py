"""Общий конфиг логгера проекта
По дефолту INFO. Формат: [HH:MM:SS] LEV chemcw.module | сообщение.
"""

from __future__ import annotations

import logging
import os
import sys
import time

_CONFIGURED = False


class _DurationFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._t0 = time.time()

    """в каждую запись elapsed - в сек"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.elapsed = time.time() - self._t0
        return True


def setup_logging(level: str | int | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [+%(elapsed)6.1fs] %(levelname)-5s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(_DurationFilter())
    root = logging.getLogger("chemcw")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    if not name.startswith("chemcw"):
        name = f"chemcw.{name}"
    return logging.getLogger(name)


def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "wtf???"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"
