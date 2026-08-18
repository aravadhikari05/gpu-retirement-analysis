"""Shared timed-region context for the benchmark workloads.

The power monitor has to integrate over exactly the measured work, not over all
of a workload's run(). On gpt2-xl the cephfs model load is roughly 60 s against a
35 s measurement, and warmup is a further full-length region; folding either into
the energy integral overstates energy per unit of work with no visible symptom,
because runtime_seconds already excludes them. The two windows would silently
disagree.

RunContext.timed_region() is the seam that closes that gap. A workload wraps its
measured loop in `with ctx.timed_region(device):`, which:

  1. synchronises the device on entry and again on exit before the clock stops,
     so the region times kernel execution rather than kernel launches, and
  2. marks the region boundaries on the power monitor, so the monitor scopes both
     the trapezoidal integral and the NVML hardware energy counter to the same
     window.

Boundary marking lives on the monitor rather than in the runner because the
monitor owns both the sampled trace and the NVML handle. Scoping the integral in
the runner while the counter stayed whole-run would compare two different windows
and break the counter-vs-integral cross-check (measured agreement 0.59% on the
1080 Ti). See measurement/power_monitor.py.

When ctx carries no monitor (a --no-power run, or a standalone CLI invocation of a
benchmark), timing still happens and the marks are no-ops.
"""

import contextlib
import logging
import time

import torch

logger = logging.getLogger(__name__)


def sync_device(device: str) -> None:
    """Waits for the device to actually finish outstanding work.

    Without this the surrounding timer measures kernel launch time rather than
    execution time, which is the single most common benchmarking error. Shared so
    all three workloads sync identically.

    Args:
      device: Torch device string, for example "cuda", "cuda:0", "mps" or "cpu".
    """
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(d))
    elif d.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


class RunContext:
    """Carries the timed region and, when present, the power monitor to scope.

    The monitor is duck-typed: anything exposing mark_region_start() and
    mark_region_end() works, which keeps benchmarks free of a hard import of
    measurement/power_monitor.py.
    """

    def __init__(self, monitor=None):
        """
        Args:
          monitor: Optional power monitor to notify at the region boundaries.
            None on --no-power runs and standalone CLI invocations.
        """
        self._monitor = monitor
        self.region_start_s: float | None = None
        self.region_stop_s: float | None = None

    @contextlib.contextmanager
    def timed_region(self, device: str):
        """Times and marks the measured region.

        Synchronises on entry, records the start, then on exit synchronises
        before recording the stop, so the interval covers completed execution.
        The power monitor, if any, is marked at the same two instants so its
        energy figures cover this interval and nothing else.

        Args:
          device: Torch device string, synced at both boundaries.
        """
        sync_device(device)
        if self._monitor is not None:
            self._monitor.mark_region_start()
        self.region_start_s = time.perf_counter()
        try:
            yield
        finally:
            # Sync before the clock stops: the timer must wait for the kernels,
            # not for their launch. The monitor mark follows the sync for the
            # same reason, so the counter read covers completed work.
            sync_device(device)
            self.region_stop_s = time.perf_counter()
            if self._monitor is not None:
                self._monitor.mark_region_end()

    @property
    def region_runtime_s(self) -> float:
        """Wall-clock seconds of the timed region.

        Raises:
          RuntimeError: if read before a timed_region has completed.
        """
        if self.region_start_s is None or self.region_stop_s is None:
            raise RuntimeError(
                "region_runtime_s read before timed_region completed; a workload "
                "must run its measured loop inside `with ctx.timed_region(device):`"
            )
        return self.region_stop_s - self.region_start_s
