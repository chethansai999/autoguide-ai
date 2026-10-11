"""Logging setup and a tiny stage timer used for latency breakdowns."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class StageTimer:
    """Collects per-stage wall-clock timings in milliseconds.

    >>> t = StageTimer()
    >>> with t.stage("retrieval"):
    ...     pass
    >>> "retrieval" in t.timings
    True
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round((time.perf_counter() - start) * 1000, 1)

    def total(self) -> float:
        self.timings["total"] = round((time.perf_counter() - self._start) * 1000, 1)
        return self.timings["total"]
