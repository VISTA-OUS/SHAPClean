"""Component 1 - measuring what a clustering run costs in wall-clock time and memory.

Peak memory has to include child processes: HDBSCAN's parallel core-distance step runs in worker
processes (measured on this project: 13 processes holding 1.92 GiB), so this process's own RSS would
under-count by more than the fit itself uses. Free system memory is tracked alongside because a
shortfall shows up as paging - the run slows to a crawl rather than failing - which is exactly how the
earlier six-hour full-dataset stall looked.

Under SLURM the job is capped by `--mem`, not by what the node happens to have free, so the ceiling is
read from `SLURM_MEM_PER_NODE` when present; exceeding it gets the job killed rather than paged.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import psutil


class PeakMemorySampler:
    """Tracks peak RSS of this process plus its children, and minimum free system memory, while active.

    HDBSCAN's parallel core-distance step can run in worker processes, so this process's RSS alone would
    under-count; the minimum free system memory is recorded too because paging, not a crash, is how a
    memory shortfall turns into a multi-hour stall.

    Args:
        interval_seconds: Sampling period.
    """

    def __init__(self, interval_seconds: float) -> None:
        self._interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss_bytes = 0
        self.min_available_bytes = psutil.virtual_memory().available

    def _current_rss(self) -> int:
        """RSS of this process and all live children, in bytes.

        Returns:
            Byte count.
        """
        total = self._process.memory_info().rss
        for child in self._process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        return total

    def _run(self) -> None:
        """Sampling loop."""
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._current_rss())
            self.min_available_bytes = min(self.min_available_bytes, psutil.virtual_memory().available)
            self._stop.wait(self._interval_seconds)

    def __enter__(self) -> "PeakMemorySampler":
        self.peak_rss_bytes = self._current_rss()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._current_rss())


def slurm_memory_ceiling_bytes() -> Optional[float]:
    """Memory granted to this job by SLURM `--mem`, if running under SLURM.

    `psutil.virtual_memory()` reports the whole node; on a shared node that can far exceed what the job
    may use, and exceeding `--mem` gets the job killed rather than paged.

    Returns:
        Byte count, or None outside SLURM (or when `--mem` was not used).
    """
    megabytes = os.environ.get("SLURM_MEM_PER_NODE")
    return float(megabytes) * 2 ** 20 if megabytes else None


def own_tree_rss_bytes() -> float:
    """RSS of this process plus its children.

    Returns:
        Byte count.
    """
    process = psutil.Process()
    return float(process.memory_info().rss + sum(child.memory_info().rss for child in process.children(recursive=True)))
