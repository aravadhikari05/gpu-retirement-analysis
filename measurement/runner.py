"""Orchestrator: runs a benchmark with power monitoring and owns all writes.

This is the container entrypoint. Benchmarks return a dict and touch no files;
this module wraps the timed region with measurement/power_monitor.py and writes
the results.

Output layout under --out-dir:

  runs.jsonl                               appended, one line per repetition
  <benchmark>/<run_id>_power.csv           the power sample trace

Raw JSONL, not CSV. The three workloads have different natural fields
(n and iters against batches against max_new_tokens), which in a single CSV goes
either sparse or needs a sidecar that every analysis query then has to join
back. JSONL holds them inline, and lists such as the per-batch loss sequence
work without inventing a third file. This follows the repo convention that
data/raw/ holds raw telemetry and data/processed/ holds derived summary tables:
analysis/summarize_runs.py turns this into the Phase 8 CSV. A column nobody
thought of is then a re-derive rather than a re-run of the sweep.

Two loops, deliberately named apart. `repeat_index` is the outer loop, this
module's --repeats, which exists for statistical spread. `inner_iters` is the
workload's own loop inside the timed region, which exists to clear the 30 s
floor. Energy per unit of work is energy_j / inner_iters, so conflating them
silently scales every energy figure by the wrong factor.

Grain is one row per repetition, because that is both the unit of exclusion
(a run below the floor, or one that crashed) and the unit of independence for
a standard deviation. Anything finer lives inline as a list, or beside it in
the power trace.

Durability. Each repetition is written as it completes, not batched to the end.
That replaces the old "keep runs under 5 minutes" rule, which conflicted with
the chosen sweep design: the oldest cards run roughly 300 s of timed region plus
equal warmup, and the sizing sweep runs as one pod looping over all points
because the 7.71 GB image pull dominates otherwise. A preemption now costs one
repetition rather than the whole pod.
"""

import argparse
import csv
import datetime
import importlib
import json
import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)

BENCHMARKS = {
    "resnet_train": "benchmarks.resnet_train",
    "llm_inference": "benchmarks.llm_inference",
    "matmul": "benchmarks.matmul",
}


def _observed_hardware() -> dict:
    """Reads GPU identity, driver and node name from inside the pod.

    The census is a point-in-time snapshot and node labelling drifts, so what a
    run actually used has to be observed at runtime rather than joined against a
    stored table. NVML is tried first and nvidia-smi is the fallback; which one
    answered is recorded rather than left implicit.

    gpu_uuid is not decoration. It is the only way to tell five repetitions on
    one physical card from five repetitions across five cards, and CLAUDE.md
    names that distinction as an open question: both L4 runs on 2026-08-11 used
    GPU-e82f7d3b, which is knowable only because llm_inference.py recorded it.
    Without this field a standard deviation over repetitions silently overstates
    how much of the fleet was sampled.
    """
    fields = {
        "gpu_model_observed": "",
        "gpu_uuid": "",
        "driver_version": "",
        "hardware_source": "",
        "node_name": os.environ.get("NODE_NAME", ""),
        "image_ref": os.environ.get("IMAGE_REF", ""),
        "git_commit": os.environ.get("GIT_COMMIT", ""),
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        fields["gpu_uuid"] = uuid.decode() if isinstance(uuid, bytes) else uuid
        # nvmlSystemGetDriverVersion, not nvmlDeviceGetDriverVersion. The latter
        # does not exist; calling it raises AttributeError, which this function
        # catches, so the only symptom was hardware_source silently reading
        # "nvidia-smi" on every run. Found by preflight on 2026-08-18.
        driver = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        fields["gpu_model_observed"] = (
            name.decode() if isinstance(name, bytes) else name
        )
        fields["driver_version"] = (
            driver.decode() if isinstance(driver, bytes) else driver
        )
        fields["hardware_source"] = "pynvml"
        return fields
    except Exception as exc:
        logger.warning("NVML query failed, falling back to nvidia-smi: %s", exc)

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        name, driver = (part.strip() for part in out.split(",", 1))
        fields["gpu_model_observed"] = name
        fields["driver_version"] = driver
        fields["hardware_source"] = "nvidia-smi"
    except Exception as exc:
        # Left blank and recorded, never guessed.
        fields["hardware_source"] = f"failed: {exc}"
        logger.error("nvidia-smi fallback also failed: %s", exc)
    return fields


def _parse_set(pairs: list[str]) -> dict:
    """Parses repeated --set key=value into benchmark kwargs.

    Values are parsed as JSON when possible so ints, floats and bools survive,
    falling back to the literal string.
    """
    kwargs = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            kwargs[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            kwargs[key.strip()] = raw
    return kwargs


def _write_power_trace(path: str, readings: list[dict]) -> None:
    """Writes the raw power sample trace."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "power_w"])
        writer.writeheader()
        writer.writerows(readings)


def _append_jsonl(path: str, record: dict) -> None:
    """Appends one repetition as a single JSON line.

    Flushed and fsynced so a preemption between repetitions cannot lose the
    record that was just reported as complete. Append-only: never rewrite an
    earlier line, so a partially written sweep is still valid data.
    """
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_once(
    benchmark: str,
    repeat_index: int,
    out_dir: str,
    kwargs: dict,
    monitor_gpu: bool = True,
) -> dict:
    """Runs one repetition with power monitoring and writes its results."""
    module = importlib.import_module(BENCHMARKS[benchmark])

    from benchmarks._context import RunContext

    monitor = None
    if monitor_gpu:
        from measurement.power_monitor import PowerMonitor

        monitor = PowerMonitor()
        monitor.start()

    # The context carries the monitor so the workload's timed_region marks the
    # exact window energy is integrated over. Without this the monitor covers the
    # whole run(), including model load and warmup, while runtime_seconds covers
    # only the region: the two windows disagree and energy per unit of work is
    # overstated with no visible symptom. See benchmarks/_context.py.
    ctx = RunContext(monitor)

    failure = None
    record = {}
    try:
        record = module.run(ctx=ctx, **kwargs)
    except Exception as exc:
        # Kept with an explicit exclusion reason rather than vanishing, per the
        # repo convention. The row is written below and then the exception is
        # re-raised, so the pod still fails loudly.
        failure = f"{type(exc).__name__}: {exc}"
        logger.exception("Benchmark %s repeat %d raised", benchmark, repeat_index)
    finally:
        power = monitor.stop() if monitor is not None else None

    hardware = _observed_hardware() if monitor_gpu else {}
    run_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    # PID disambiguates two runner processes writing in the same second. Without
    # it a collision silently overwrites the earlier sidecar.
    run_id = (
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        f"-{benchmark}-r{repeat_index}-p{os.getpid()}"
    )

    power_fields = power.as_dict() if power is not None else {}

    # Run identity and provenance first, then the benchmark's own fields. The
    # benchmark cannot overwrite provenance, and workload-specific keys sit
    # inline rather than in a sidecar that analysis would have to join back.
    row = {
        "run_utc": run_utc,
        "run_id": run_id,
        "benchmark": benchmark,
        "repeat_index": repeat_index,
        "config_id": record.get("config_id", ""),
        "inner_iters": record.get("inner_iters", ""),
        "runtime_s": record.get("runtime_seconds", ""),
        "work_hash": record.get("work_hash", ""),
        "exclusion_reason": "",
        **power_fields,
        **{k: v for k, v in record.items() if k not in ("runtime_seconds",)},
        **hardware,
    }

    if not row["config_id"]:
        # Without it, analysis can only group by workload name, which would
        # average 32-token and 960-token runs into one meaningless number.
        logger.warning(
            "%s returned no config_id. Runs cannot be grouped by configuration.",
            benchmark,
        )

    # A run below the floor is kept with an explicit exclusion reason rather
    # than deleted, per the repo convention. A crashed run is kept the same way.
    if failure is not None:
        row["exclusion_reason"] = f"benchmark raised: {failure}"
    elif power_fields.get("below_30s_floor"):
        row["exclusion_reason"] = (
            f"power sample window {power_fields.get('power_duration_s', 0):.2f}s "
            "below the 30s floor, energy not trustworthy"
        )

    bench_dir = os.path.join(out_dir, benchmark)
    os.makedirs(bench_dir, exist_ok=True)

    if power is not None and power.readings:
        trace_path = os.path.join(bench_dir, f"{run_id}_power.csv")
        _write_power_trace(trace_path, power.readings)
        # Recorded so the trace can be found from the row without guessing.
        row["power_trace_path"] = os.path.relpath(trace_path, out_dir)

    _append_jsonl(os.path.join(out_dir, "runs.jsonl"), row)

    logger.info(
        "%s repeat %d: runtime=%.4fs energy=%.1fJ avg=%.1fW samples=%d%s",
        benchmark,
        repeat_index,
        record.get("runtime_seconds", 0.0),
        power_fields.get("energy_j", 0.0),
        power_fields.get("avg_power_w", 0.0),
        power_fields.get("n_power_samples", 0),
        " EXCLUDED" if row["exclusion_reason"] else "",
    )

    if failure is not None:
        # The row and sidecar are on the PVC now, so the failure is recorded.
        # Re-raise so the pod exits non-zero rather than reporting success.
        raise RuntimeError(
            f"benchmark {benchmark} repeat {repeat_index} failed: {failure}"
        )

    return row


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Benchmark and power monitor orchestrator"
    )
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repetitions. 5 rather than 3: failed runs are kept with an "
        "exclusion reason, so effective n falls below nominal n.",
    )
    parser.add_argument("--out-dir", default="/results")
    parser.add_argument(
        "--no-power",
        action="store_true",
        help="Skip power monitoring. For CPU smoke tests only, never for measured runs.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Benchmark keyword argument, repeatable. Example: --set precision=fp32",
    )
    args = parser.parse_args()

    kwargs = _parse_set(args.set)
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(
        "benchmark=%s repeats=%d out_dir=%s power=%s kwargs=%s",
        args.benchmark,
        args.repeats,
        args.out_dir,
        not args.no_power,
        kwargs,
    )

    for i in range(1, args.repeats + 1):
        run_once(
            benchmark=args.benchmark,
            repeat_index=i,
            out_dir=args.out_dir,
            kwargs=kwargs,
            monitor_gpu=not args.no_power,
        )


if __name__ == "__main__":
    main()
