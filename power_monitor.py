#!/usr/bin/env python3
"""Phase 4 - GPU Power Measurement (pynvml). Owner: Veda.
Samples GPU power on a background thread during the timed region and integrates
power-over-time into joules. Also reads NVML's hardware energy counter (Volta+)
as an independent cross-check. pynvml is imported lazily so this file still
imports on a Mac; it only needs NVIDIA hardware when you actually monitor."""

import threading
import time


def integrate_energy(samples):
    """Trapezoidal integral of power over time.
    samples: list of (timestamp_s, power_w). Returns joules."""
    if len(samples) < 2:
        return 0.0
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += (p0 + p1) / 2.0 * (t1 - t0)
    return energy


class PowerResult:
    def __init__(self, energy_j, avg_power_w, peak_power_w, min_power_w,
                 n_samples, duration_s, energy_j_counter):
        self.energy_j = energy_j
        self.avg_power_w = avg_power_w
        self.peak_power_w = peak_power_w
        self.min_power_w = min_power_w
        self.n_samples = n_samples
        self.duration_s = duration_s
        self.energy_j_counter = energy_j_counter


class PowerMonitor:
    def __init__(self, device_index=0, interval=0.2):
        self.device_index = device_index
        self.interval = interval
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._pynvml = None
        self._handle = None
        self._start_energy_mj = None

    def start(self):
        import pynvml
        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        try:
            self._start_energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except pynvml.NVMLError:
            self._start_energy_mj = None
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        p = self._pynvml
        while not self._stop.is_set():
            t = time.perf_counter()
            mw = p.nvmlDeviceGetPowerUsage(self._handle)
            self._samples.append((t, mw / 1000.0))
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        p, h = self._pynvml, self._handle
        energy_counter = None
        if self._start_energy_mj is not None:
            try:
                end_mj = p.nvmlDeviceGetTotalEnergyConsumption(h)
                energy_counter = (end_mj - self._start_energy_mj) / 1000.0
            except p.NVMLError:
                energy_counter = None
        p.nvmlShutdown()
        powers = [pw for _, pw in self._samples]
        energy_j = integrate_energy(self._samples)
        duration = (self._samples[-1][0] - self._samples[0][0]) if len(self._samples) > 1 else 0.0
        return PowerResult(
            energy_j=energy_j,
            avg_power_w=(sum(powers) / len(powers)) if powers else 0.0,
            peak_power_w=max(powers) if powers else 0.0,
            min_power_w=min(powers) if powers else 0.0,
            n_samples=len(powers),
            duration_s=duration,
            energy_j_counter=energy_counter,
        )
