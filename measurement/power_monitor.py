"""Phase 4: GPU power measurement via NVML. Owner: Veda.

Moved here from the repo root on 2026-08-18. The root copy was the real
implementation while this path was a one-line stub, so the Dockerfile's
`COPY measurement/` shipped the stub. Three changes were made to the moved
code, per the contract recorded in CLAUDE.md:

  1. `readings` is retained rather than discarded, so a per-run power trace CSV
     can be written. The trace is also the only way to detect the cached
     reading problem (Yang et al., 2024) after the fact.
  2. A failed `nvmlDeviceGetPowerUsage` logs and skips that sample instead of
     killing the sampling thread. Previously a single raise ended monitoring
     while the benchmark carried on and reported success.
  3. `PowerResult.as_dict()`, so the CSV writer never reaches into attributes.

Samples GPU power on a background thread during the timed region and integrates
power over time into joules. Also reads NVML's hardware energy counter
(Volta and later) as an independent cross-check. NVML is imported lazily so
this module still imports on a Mac; it only needs NVIDIA hardware to monitor.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Below this, pynvml may return cached readings and the integral is not
# trustworthy. Cited to Yang et al. (2024). The monitor does not enforce it,
# it reports it, so a short run is excluded with a reason rather than deleted.
MIN_TRUSTWORTHY_DURATION_S = 30.0


def integrate_energy(samples: list[tuple[float, float]]) -> float:
    """Trapezoidal integral of power over time.

    Args:
      samples: list of (timestamp_s, power_w).

    Returns:
      Energy in joules. Zero if there are fewer than two samples.
    """
    if len(samples) < 2:
        return 0.0
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += (p0 + p1) / 2.0 * (t1 - t0)
    return energy


def _interp_power(samples: list[tuple[float, float]], t: float) -> float:
    """Linear interpolation of power at time t. Samples must be time-sorted.

    Clamps to the nearest endpoint for a t outside the sampled range, which
    holds the boundary power flat rather than extrapolating a trend.
    """
    for (ta, pa), (tb, pb) in zip(samples, samples[1:]):
        if ta <= t <= tb:
            if tb == ta:
                return pa
            return pa + (t - ta) / (tb - ta) * (pb - pa)
    if t <= samples[0][0]:
        return samples[0][1]
    return samples[-1][1]


def clip_samples(
    samples: list[tuple[float, float]], t0: float, t1: float
) -> list[tuple[float, float]]:
    """Restricts a sample series to [t0, t1] with interpolated endpoints.

    The interpolated endpoints are what make energy cover exactly the timed
    region rather than the interval between the nearest enclosing samples. With a
    constant 200 W trace over 35 s this returns endpoints at 200 W and the
    integral is exactly 7000 J.
    """
    if len(samples) < 2 or t1 <= t0:
        return []
    interior = [(t, p) for t, p in samples if t0 < t < t1]
    return (
        [(t0, _interp_power(samples, t0))]
        + interior
        + [(t1, _interp_power(samples, t1))]
    )


class PowerResult:
    """Summary of one monitored region.

    energy_j is the trapezoidal integral of the sampled power. energy_j_counter
    is NVML's own hardware energy counter over the same window, and is None on
    architectures that do not implement it (Pascal, so the GTX 1080 Ti has no
    cross-check). Record that asymmetry per run rather than averaging over it.
    """

    def __init__(
        self,
        energy_j: float,
        avg_power_w: float,
        peak_power_w: float,
        min_power_w: float,
        n_samples: int,
        duration_s: float,
        energy_j_counter: float | None,
        readings: list[dict],
        n_failed_samples: int = 0,
        power_window: str = "full_run",
    ):
        self.energy_j = energy_j
        self.avg_power_w = avg_power_w
        self.peak_power_w = peak_power_w
        self.min_power_w = min_power_w
        self.n_samples = n_samples
        self.duration_s = duration_s
        self.energy_j_counter = energy_j_counter
        self.readings = readings
        self.n_failed_samples = n_failed_samples
        # "region" when scoped to a timed_region the workload marked, "full_run"
        # when it covers the whole start()..stop() window (a --no-power path, or a
        # workload that marked no region). Recorded so an aggregate can never
        # silently average region energy against whole-run energy.
        self.power_window = power_window

    @property
    def below_floor(self) -> bool:
        """True if the sampled window is too short for the integral to be trusted."""
        return self.duration_s < MIN_TRUSTWORTHY_DURATION_S

    def as_dict(self) -> dict:
        """Scalar summary, without the sample trace."""
        return {
            "energy_j": self.energy_j,
            "energy_j_counter": self.energy_j_counter,
            "avg_power_w": self.avg_power_w,
            "peak_power_w": self.peak_power_w,
            "min_power_w": self.min_power_w,
            "n_power_samples": self.n_samples,
            "n_failed_power_samples": self.n_failed_samples,
            "power_duration_s": self.duration_s,
            "below_30s_floor": self.below_floor,
            "power_window": self.power_window,
        }


class PowerMonitor:
    """Samples GPU power on a daemon thread between start() and stop()."""

    def __init__(self, device_index: int = 0, interval: float = 0.2):
        """
        Args:
          device_index: NVML device index to sample.
          interval: Seconds between samples. 0.2 gives roughly 175 samples over
            a 35 s run, which is adequate; see docs/tasks/phase3-workload-sizing.md.
        """
        self.device_index = device_index
        self.interval = interval
        self._samples: list[tuple[float, float]] = []
        self._failed_samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._handle = None
        self._start_energy_mj: int | None = None
        # Region boundaries, set by mark_region_start/end when a workload runs its
        # timed_region. Left None on a run that marks no region, which falls back
        # to the whole start()..stop() window.
        self._region_start_t: float | None = None
        self._region_stop_t: float | None = None
        self._region_start_energy_mj: int | None = None
        self._region_stop_energy_mj: int | None = None

    def start(self) -> None:
        """Initialises NVML and starts the sampling thread."""
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        try:
            self._start_energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(
                self._handle
            )
        except pynvml.NVMLError:
            # Not implemented before Volta. The trapezoidal integral is then the
            # only energy figure available for this card.
            self._start_energy_mj = None
            logger.info(
                "NVML total energy counter unavailable on device %d. "
                "energy_j_counter will be null and the integral has no cross-check.",
                self.device_index,
            )

        self._samples.clear()
        self._failed_samples = 0
        self._region_start_t = None
        self._region_stop_t = None
        self._region_start_energy_mj = None
        self._region_stop_energy_mj = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _read_energy_mj(self) -> int | None:
        """Reads the NVML energy counter, or None when it is unavailable.

        Availability is fixed at start(); a card without the counter returns None
        here too, so the region delta stays None rather than raising.
        """
        if self._pynvml is None or self._handle is None:
            return None
        try:
            return self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except Exception:
            return None

    def mark_region_start(self) -> None:
        """Marks the start of the timed region. Called by RunContext on entry.

        Snapshots both the wall clock and the hardware energy counter so the
        reported energy covers the region and not the model load or warmup that
        preceded it.
        """
        self._region_start_t = time.perf_counter()
        self._region_start_energy_mj = self._read_energy_mj()

    def mark_region_end(self) -> None:
        """Marks the end of the timed region. Called by RunContext on exit."""
        self._region_stop_t = time.perf_counter()
        self._region_stop_energy_mj = self._read_energy_mj()

    def _sample_loop(self) -> None:
        p = self._pynvml
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                mw = p.nvmlDeviceGetPowerUsage(self._handle)
            except Exception as exc:
                # Skip the sample, keep the thread. A monitor that dies mid-run
                # leaves the benchmark reporting success on a truncated trace.
                self._failed_samples += 1
                if self._failed_samples == 1:
                    logger.warning("Power sample failed, skipping: %s", exc)
            else:
                self._samples.append((t, mw / 1000.0))
            self._stop.wait(self.interval)

    def stop(self) -> PowerResult:
        """Stops sampling and returns the summary."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

        p, h = self._pynvml, self._handle
        # Whole-window counter delta, used only when no region was marked. A
        # marked region uses the boundary snapshots instead.
        whole_end_mj = None
        if self._start_energy_mj is not None:
            try:
                whole_end_mj = p.nvmlDeviceGetTotalEnergyConsumption(h)
            except p.NVMLError as exc:
                logger.warning("NVML energy counter read failed at stop: %s", exc)
        p.nvmlShutdown()

        if self._failed_samples:
            logger.warning(
                "%d power samples failed and were skipped", self._failed_samples
            )

        region_marked = (
            self._region_start_t is not None and self._region_stop_t is not None
        )
        if region_marked:
            # Scope every energy figure to exactly the timed region. The integral
            # is clipped with interpolated endpoints; the counter delta uses the
            # snapshots taken at the same two instants, so both cover one window.
            power_window = "region"
            t0, t1 = self._region_start_t, self._region_stop_t
            window = clip_samples(self._samples, t0, t1)
            energy_j = integrate_energy(window)
            duration = t1 - t0
            powers = [pw for pw_t, pw in self._samples if t0 <= pw_t <= t1]
            if (
                self._region_start_energy_mj is not None
                and self._region_stop_energy_mj is not None
            ):
                energy_counter = (
                    self._region_stop_energy_mj - self._region_start_energy_mj
                ) / 1000.0
            else:
                energy_counter = None
        else:
            # No region marked: fall back to the whole start()..stop() window.
            power_window = "full_run"
            energy_j = integrate_energy(self._samples)
            duration = (
                (self._samples[-1][0] - self._samples[0][0])
                if len(self._samples) > 1
                else 0.0
            )
            powers = [pw for _, pw in self._samples]
            if self._start_energy_mj is not None and whole_end_mj is not None:
                energy_counter = (whole_end_mj - self._start_energy_mj) / 1000.0
            else:
                energy_counter = None

        result = PowerResult(
            energy_j=energy_j,
            avg_power_w=(sum(powers) / len(powers)) if powers else 0.0,
            peak_power_w=max(powers) if powers else 0.0,
            min_power_w=min(powers) if powers else 0.0,
            n_samples=len(powers),
            duration_s=duration,
            energy_j_counter=energy_counter,
            # Full trace regardless of window: the per-sample record is the only
            # way to detect cached readings (Yang et al., 2024) across load,
            # warmup and the region. power_window states which slice energy covers.
            readings=[{"timestamp": t, "power_w": pw} for t, pw in self._samples],
            n_failed_samples=self._failed_samples,
            power_window=power_window,
        )

        if result.below_floor:
            logger.warning(
                "Sampled window %.2f s is below the %.0f s floor. Exclude this "
                "run with a reason rather than reporting its energy.",
                result.duration_s,
                MIN_TRUSTWORTHY_DURATION_S,
            )
        return result
