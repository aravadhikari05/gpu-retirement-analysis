#!/usr/bin/env python3
"""Phase 2 orchestrator: runs one workload, measures it, writes one record.

The division of labour is the whole point of this file.

The runner owns everything that must be identical across workloads: hardware
capture, precision read-back, power measurement, the record schema, and output.
A workload owns only its work. It warms up, runs its timed region inside
`ctx.timed_region()`, and returns a WorkloadResult. It captures no hardware,
starts no monitor, and writes no files.

That split exists because the three workloads previously had three incompatible
shapes, which meant three datasets that could not be compared even once power
measurement worked.

Two schema decisions are settled and encoded here:

1. Integration is the primary energy method for every card, uniformly, even on
   cards where NVML's hardware energy counter is available. The counter is
   recorded as an independent cross-check, never as the headline number. The
   reason is that the paper's central claim compares old hardware against new,
   and nvmlDeviceGetTotalEnergyConsumption is Volta and newer. Using the
   counter where it exists would mean the GTX 1080 Ti and the L40S were
   measured with different instruments, which confounds the instrument with the
   result. Measuring everything the same way costs nothing and removes that.
   The residual measured on the cards that do have a counter then becomes an
   empirical error bar for the trapezoid method, which can be carried onto the
   cards that do not.

2. work_hash means two different things and the record says which, in
   work_hash_kind. "output" means the hash covers real workload output and
   proves identical work was done. That holds for greedy LLM decoding because
   argmax over a discrete vocabulary quantises away low-bit float differences.
   It does not transfer to ResNet loss or matmul output, which are continuous
   floats and will differ across architectures by design. Those workloads emit
   "config", meaning work is fixed structurally by a fixed iteration count
   rather than verified by output identity. Without this field a reader sees
   three populated work_hash values and assumes all three carry the guarantee
   that only one of them does.
"""

import argparse
import datetime as dt
import importlib
import json
import logging
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Bump when the meaning of an existing field changes, not when a field is added.
# Analysis code reads this to know whether it can compare two records directly.
SCHEMA_VERSION = 1

# Below this the power sensor may return stale readings, so a run shorter than
# this is recorded and excluded rather than deleted, per the repo convention.
MIN_TIMED_REGION_S = 30.0

# Fraction of the timed region the power samples must span for the run to count.
MIN_POWER_COVERAGE = 0.98

# Lazy import paths. Importing all three eagerly would make a matmul run depend
# on transformers, which is exactly the coupling the single-image layout is
# meant to avoid paying for at runtime.
WORKLOADS = {
    "llm_inference": "benchmarks.llm_inference",
    "resnet_train": "benchmarks.resnet_train",
    "matmul": "benchmarks.matmul",
}


@dataclass
class WorkloadResult:
    """What a workload returns. Everything else is the runner's job."""

    runtime_seconds: float
    work_hash: str
    work_hash_kind: str  # "output" or "config"
    work_hash_encoding: str
    # Flat, workload-specific fields. Namespaced into the record under
    # workload_metrics so adding one never shifts a shared column.
    metrics: dict = field(default_factory=dict)
    # Bulky diagnostics that would bloat the record: generated_text, token_ids,
    # per-iteration timings. Written to a sidecar file instead.
    sidecar: Optional[dict] = None


class RunContext:
    """Handed to the workload. Its only real job is marking the timed region.

    The power monitor has to wrap exactly the timed region, but the timed region
    lives inside the workload. On gpt2-xl, model load from cephfs is roughly 60
    seconds against a 35 second measurement, so sampling across the whole call
    would swamp the quantity being measured with load and warmup.
    """

    def __init__(self, device: str, monitor: Optional[Any] = None,
                 sync: Optional[Callable[[], None]] = None):
        self.device = device
        self._monitor = monitor
        self._sync = sync or (lambda: None)
        self.region_start: Optional[float] = None
        self.region_end: Optional[float] = None
        self.notes: dict = {}
        self._entered = False

    @contextmanager
    def timed_region(self):
        """Wraps the measured work. Starts power sampling, times, syncs, stops.

        The device sync happens here, on exit, before the clock stops. Leaving
        it to each workload means one of them eventually forgets it and times
        kernel launches instead of kernel execution.
        """
        if self._entered:
            raise RuntimeError("timed_region entered twice in one run")
        self._entered = True

        if self._monitor is not None:
            self._monitor.start()
        self.region_start = time.perf_counter()
        try:
            yield self
        finally:
            self._sync()
            self.region_end = time.perf_counter()
            if self._monitor is not None:
                self._monitor.stop()

    def note(self, key: str, value: Any) -> None:
        """Workload-specific metadata that is not a measured metric."""
        self.notes[key] = value

    @property
    def region_seconds(self) -> float:
        if self.region_start is None or self.region_end is None:
            return 0.0
        return self.region_end - self.region_start


def clip_to_region(samples, t_start, t_end):
    """Restricts samples to exactly the timed region.

    Two corrections in one pass, both of which change the joule figure:

    Trailing. The sampling thread can take a sample after the region ends,
    because the region closes on the device sync while the thread is still in
    its wait. Integrating that sample attributes energy to the region that was
    spent outside it.

    Leading. Sampling starts a moment after the region opens, so the interval
    between region start and first sample is otherwise unattributed.

    Inside the sampled span the boundary value is linearly interpolated. Outside
    it the nearest sample is extended flat, which is the only defensible guess
    when there is no reading. The correction is small at a 200 ms interval, but
    it is systematic and always in the same direction, so it survives averaging
    over a sweep rather than cancelling out.
    """
    if len(samples) < 2 or t_start is None or t_end is None:
        return list(samples)

    def value_at(t):
        if t <= samples[0][0]:
            return samples[0][1]
        if t >= samples[-1][0]:
            return samples[-1][1]
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            if t0 <= t <= t1:
                span = t1 - t0
                if span <= 0:
                    return p0
                return p0 + (p1 - p0) * (t - t0) / span
        return samples[-1][1]

    inner = [(t, w) for t, w in samples if t_start < t < t_end]
    return [(t_start, value_at(t_start))] + inner + [(t_end, value_at(t_end))]


def integrate_energy(samples):
    """Trapezoidal integral of power over time, in joules.

    samples: list of (perf_counter_seconds, watts). Pass the output of
    clip_to_region to get the integral over exactly the timed region.
    """
    if len(samples) < 2:
        return 0.0
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += (p0 + p1) / 2.0 * (t1 - t0)
    return energy


def power_fields(samples, t_start, t_end, counter_j, counter_error):
    """Builds every energy and power field in the record.

    Deliberately lives here rather than in power_monitor.py. Edge correction
    needs the timed-region boundaries, and only the runner knows those. Keeping
    integration here also means the monitor never computes a duration at all,
    which is what caused the sample-window-as-duration bug in the first place.
    """
    region_s = (t_end - t_start) if (t_start is not None and t_end is not None) else 0.0
    out = {
        "schema_energy_note": "energy_j is the primary figure and is always integration based",
        "energy_j": None,
        "energy_j_integrated": None,
        "energy_j_window": None,
        "energy_j_counter": counter_j,
        "energy_primary_method": "integration",
        "energy_methods_available": [],
        "energy_counter_unavailable_reason": counter_error or None,
        "energy_method_residual_pct": None,
        "avg_power_w": None,
        "peak_power_w": None,
        "min_power_w": None,
        "n_power_samples": len(samples),
        "power_sample_interval_actual_s": None,
        "runtime_seconds": region_s,
        "power_window_seconds": None,
        "power_window_lead_s": None,
        "power_window_lag_s": None,
        "power_coverage_fraction": None,
    }

    if len(samples) >= 2:
        out["energy_methods_available"].append("integration")
        watts = [w for _, w in samples]
        # Primary figure: the integral over exactly the timed region.
        clipped = integrate_energy(clip_to_region(samples, t_start, t_end))
        out["energy_j_integrated"] = clipped
        # Diagnostic: the raw first-sample-to-last-sample integral, kept so the
        # correction stays auditable instead of baked in silently.
        out["energy_j_window"] = integrate_energy(samples)
        out["energy_j"] = clipped
        out["avg_power_w"] = sum(watts) / len(watts)
        out["peak_power_w"] = max(watts)
        out["min_power_w"] = min(watts)

        window = samples[-1][0] - samples[0][0]
        out["power_window_seconds"] = window
        gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
        gaps.sort()
        out["power_sample_interval_actual_s"] = gaps[len(gaps) // 2]
        if t_start is not None:
            out["power_window_lead_s"] = samples[0][0] - t_start
        if t_end is not None:
            out["power_window_lag_s"] = t_end - samples[-1][0]
        if region_s > 0:
            out["power_coverage_fraction"] = window / region_s

    if counter_j is not None:
        out["energy_methods_available"].append("nvml_counter")
        # Residual against the primary, not a replacement for it. This is the
        # number that becomes the error bar for cards with no counter.
        if out["energy_j_integrated"]:
            out["energy_method_residual_pct"] = (
                (counter_j - out["energy_j_integrated"]) / counter_j * 100.0
            )

    return out


def observed_hardware(device: str) -> dict:
    """Hardware as observed at runtime from inside the pod.

    CLAUDE.md requires this observed at runtime and never joined against the
    stored census, because node labelling drifts: it moved by one node within an
    hour between two captures. gpu_uuid additionally catches two runs landing on
    different physical cards on the same node, which has already happened in the
    Phase 3 smoke records.

    This duplicates the equivalent helper in benchmarks/llm_inference.py for
    now. That copy goes away when llm_inference is converted to the ctx
    interface, since hardware capture becomes the runner's job for every
    workload. Tracked rather than hidden.
    """
    observed = {
        "node_name": os.environ.get("NODE_NAME", ""),
        "pod_name": os.environ.get("POD_NAME", ""),
        "k8s_job_name": os.environ.get("JOB_NAME", ""),
        "image_id": os.environ.get("IMAGE_ID", ""),
        "gpu_model_observed": "",
        "gpu_model_torch": "",
        "gpu_uuid": "",
        "driver_version": "",
        "sm_capability": "",
        "cuda_version": "",
        "hardware_source": "",
        "nvml_error": "",
    }

    try:
        import torch
    except ImportError:
        observed["nvml_error"] = "torch not importable"
        return observed

    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return observed

    index = int(device.split(":")[1]) if ":" in device else 0
    observed["gpu_model_torch"] = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    observed["sm_capability"] = f"sm_{major}{minor}"
    observed["cuda_version"] = torch.version.cuda or ""

    # pynvml and nvidia-ml-py both install a module named pynvml and disagree on
    # some symbols when both are present. driver_version is required metadata,
    # so a failure here falls through to nvidia-smi rather than leaving it blank,
    # and the record says which source answered.
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            observed["gpu_model_observed"] = _decode(pynvml.nvmlDeviceGetName(handle))
            observed["gpu_uuid"] = _decode(pynvml.nvmlDeviceGetUUID(handle))
            observed["driver_version"] = _decode(pynvml.nvmlSystemGetDriverVersion())
            observed["hardware_source"] = "pynvml"
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        observed["nvml_error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("NVML unavailable (%s), falling back to nvidia-smi", exc)
        observed.update(_nvidia_smi_hardware(index))

    return observed


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _nvidia_smi_hardware(index: int) -> dict:
    fields = {}
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--id={index}",
             "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        parts = [p.strip() for p in completed.stdout.strip().split(",")]
        if len(parts) >= 3:
            fields.update({
                "gpu_model_observed": parts[0],
                "driver_version": parts[1],
                "gpu_uuid": parts[2],
                "hardware_source": "nvidia-smi",
            })
        else:
            fields["nvml_error"] = f"unparsed nvidia-smi output: {completed.stdout!r}"
    except Exception as exc:
        fields["nvml_error"] = f"pynvml and nvidia-smi both failed: {exc}"
        logger.error("nvidia-smi fallback also failed: %s", exc)
    return fields


def precision_state() -> dict:
    """Reads back the precision flags actually in effect.

    Read back rather than requested. torch 2.9 deprecates allow_tf32 in favour
    of fp32_precision, and transformers 5.x silently ignores the old torch_dtype
    kwarg, so a record built from what was asked for can document a precision
    the run did not use. Captured after the workload has configured itself.
    """
    state = {
        "allow_tf32_matmul": None,
        "allow_tf32_cudnn": None,
        "matmul_fp32_precision": "",
        "cudnn_conv_fp32_precision": "",
        "allow_fp16_reduced_precision_reduction": None,
        "allow_bf16_reduced_precision_reduction": None,
        "deterministic_algorithms": None,
    }
    try:
        import torch
    except ImportError:
        return state

    cuda_b, cudnn_b = torch.backends.cuda, torch.backends.cudnn
    state["allow_tf32_matmul"] = getattr(cuda_b.matmul, "allow_tf32", None)
    state["allow_tf32_cudnn"] = getattr(cudnn_b, "allow_tf32", None)
    state["matmul_fp32_precision"] = str(getattr(cuda_b.matmul, "fp32_precision", ""))
    state["cudnn_conv_fp32_precision"] = str(
        getattr(getattr(cudnn_b, "conv", None), "fp32_precision", "")
    )
    state["allow_fp16_reduced_precision_reduction"] = getattr(
        cuda_b.matmul, "allow_fp16_reduced_precision_reduction", None)
    state["allow_bf16_reduced_precision_reduction"] = getattr(
        cuda_b.matmul, "allow_bf16_reduced_precision_reduction", None)
    state["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    return state


def library_versions() -> dict:
    """Versions of anything that can change a result, recorded per run."""
    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "transformers", "numpy"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:
            pass
    return versions


def build_monitor(device: str, interval: float):
    """Returns a power monitor, or None with a reason.

    measurement/power_monitor.py is still a one-line stub. An implementation
    exists at the repo root but has not landed here, and it is Veda's file. The
    runner therefore treats the monitor as optional: without it a run still
    produces runtime, work_hash and full hardware provenance, and the energy
    fields come back null with a stated reason rather than silently zero.
    """
    if not device.startswith("cuda"):
        return None, "power measurement requires a cuda device"
    try:
        from measurement.power_monitor import PowerMonitor
    except ImportError as exc:
        return None, f"measurement.power_monitor.PowerMonitor unavailable: {exc}"
    index = int(device.split(":")[1]) if ":" in device else 0
    return PowerMonitor(device_index=index, interval=interval), None


def _monitor_readout(monitor):
    """Pulls samples and counter energy off the monitor.

    The contract the monitor must satisfy: expose `samples` as a list of
    (perf_counter_seconds, watts), and `energy_j_counter` as joules from
    nvmlDeviceGetTotalEnergyConsumption or None. It must not compute a duration.
    """
    if monitor is None:
        return [], None, "no power monitor"
    samples = list(getattr(monitor, "samples", []) or [])
    counter = getattr(monitor, "energy_j_counter", None)
    reason = None
    if counter is None:
        # Volta and newer only. The GTX 1080 Ti is sm_61, so the counter is
        # absent on exactly the card the retirement argument depends on. That is
        # why integration is primary everywhere rather than best-available.
        reason = "nvmlDeviceGetTotalEnergyConsumption unsupported on this device"
    return samples, counter, reason


def run_workload(name: str, device: str, params: dict, sample_interval: float) -> dict:
    """Runs one workload end to end and returns the full record."""
    if name not in WORKLOADS:
        raise SystemExit(f"unknown workload {name!r}, expected one of {sorted(WORKLOADS)}")

    module = importlib.import_module(WORKLOADS[name])
    if not hasattr(module, "run"):
        raise SystemExit(f"{WORKLOADS[name]} has no run(), so it does not yet conform")

    monitor, monitor_reason = build_monitor(device, sample_interval)
    if monitor_reason:
        logger.warning("running without power measurement: %s", monitor_reason)

    sync = _make_sync(device)
    ctx = RunContext(device=device, monitor=monitor, sync=sync)

    started_utc = dt.datetime.now(dt.timezone.utc)
    result = module.run(ctx, device=device, **params)
    if not isinstance(result, WorkloadResult):
        raise SystemExit(f"{name}.run returned {type(result).__name__}, expected WorkloadResult")

    samples, counter_j, counter_reason = _monitor_readout(monitor)
    energy = power_fields(samples, ctx.region_start, ctx.region_end, counter_j,
                          monitor_reason or counter_reason)

    # The workload's own perf_counter reading and the context's should agree.
    # A disagreement means the workload timed something other than its region.
    if abs(result.runtime_seconds - ctx.region_seconds) > 0.05:
        logger.warning(
            "workload reported %.4f s but timed region was %.4f s",
            result.runtime_seconds, ctx.region_seconds)

    # Rebase sample timestamps to the region start before they leave this
    # function. perf_counter values are only meaningful inside the process that
    # produced them, so an absolute figure like 1031183.34 is unplottable later.
    # Relative seconds also make the lead and lag gaps visible directly: a
    # negative first timestamp means sampling began before the region opened.
    if ctx.region_start is not None:
        samples = [(t - ctx.region_start, w) for t, w in samples]

    hardware = observed_hardware(device)
    record = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": name,
        "run_id": _run_id(started_utc, hardware),
        "run_utc": started_utc.isoformat(timespec="seconds"),
        "device_requested": device,
        "work_hash": result.work_hash,
        "work_hash_kind": result.work_hash_kind,
        "work_hash_encoding": result.work_hash_encoding,
        "workload_params": params,
        "workload_metrics": result.metrics,
        "workload_notes": ctx.notes,
        "library_versions": library_versions(),
        "power_sample_interval_requested_s": sample_interval,
    }
    record.update(hardware)
    record.update(precision_state())
    record.update(energy)
    record.update(_exclusion(record))
    return record, result.sidecar, samples


def _make_sync(device: str):
    def sync():
        try:
            import torch
        except ImportError:
            return
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
    return sync


def _exclusion(record: dict) -> dict:
    """Flags a run rather than deleting it, per the repo convention.

    Two independent reasons, both of which the sizing work established: a timed
    region under the floor risks stale power readings, and thin sample coverage
    means the energy figure does not span the work it is attributed to.
    """
    reasons = []
    runtime = record.get("runtime_seconds") or 0.0
    if runtime < MIN_TIMED_REGION_S:
        reasons.append(
            f"timed region {runtime:.1f}s under the {MIN_TIMED_REGION_S:.0f}s floor")
    coverage = record.get("power_coverage_fraction")
    if coverage is not None and coverage < MIN_POWER_COVERAGE:
        reasons.append(f"power samples cover only {coverage:.1%} of the timed region")
    return {"excluded": bool(reasons), "exclusion_reason": "; ".join(reasons)}


def _run_id(started: dt.datetime, hardware: dict) -> str:
    """Matches the existing data/raw/llm_smoke naming: <utc>-<gpu slug>."""
    stamp = started.strftime("%Y%m%dt%H%M%Sz").lower()
    model = hardware.get("gpu_model_observed") or hardware.get("gpu_model_torch") or "cpu"
    slug = "".join(c for c in model.lower().replace(" ", "") if c.isalnum())
    return f"{stamp}-{slug}"


def write_outputs(record: dict, sidecar, samples, out_dir: str) -> str:
    """Record, power samples, and bulky diagnostics, as three separate files.

    Samples go to a sidecar rather than inline. At 200 ms over a 110 second run
    that is roughly 550 pairs, about 9 KB, or 1.3 MB across a 150 run sweep, so
    storage is not the reason. Keeping them out of the record is what lets the
    record stay loadable as a flat table.

    Sample t_s is seconds from the start of the timed region, not perf_counter.
    """
    os.makedirs(out_dir, exist_ok=True)
    run_id = record["run_id"]

    record_path = os.path.join(out_dir, f"{run_id}.json")
    with open(record_path, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    if samples:
        with open(os.path.join(out_dir, f"{run_id}.power.jsonl"), "w") as handle:
            for t, w in samples:
                handle.write(json.dumps({"t_s": round(t, 6), "power_w": w}) + "\n")

    if sidecar:
        with open(os.path.join(out_dir, f"{run_id}.workload.json"), "w") as handle:
            json.dump(sidecar, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")

    return record_path


def _parse_params(pairs) -> dict:
    """--param key=value, typed by JSON where possible so ints stay ints."""
    params = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Run one benchmark workload with power measurement.")
    parser.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="/results")
    parser.add_argument("--sample-interval", type=float, default=0.2,
                        help="power sampling interval in seconds")
    parser.add_argument("--param", action="append", metavar="KEY=VALUE",
                        help="workload parameter, repeatable")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    record, sidecar, samples = run_workload(
        args.workload, args.device, _parse_params(args.param), args.sample_interval)
    path = write_outputs(record, sidecar, samples, args.out_dir)

    logger.info("wrote %s", path)
    logger.info("runtime %.3f s, energy %s J via %s",
                record["runtime_seconds"], record["energy_j"],
                record["energy_primary_method"])
    if record["excluded"]:
        logger.warning("run marked excluded: %s", record["exclusion_reason"])


if __name__ == "__main__":
    main()
