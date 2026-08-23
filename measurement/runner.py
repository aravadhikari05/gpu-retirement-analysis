"""Orchestrator: runs a benchmark with power monitoring and owns all writes.

This is the container entrypoint. Benchmarks return a benchmarks._result
WorkloadResult and touch no files; this module wraps the timed region with
measurement/power_monitor.py, flattens the result with .to_row() and writes it.

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

Idle power is measured twice per pod, before the first repetition, and stamped
onto every row. It is the second term of real annual energy
(`energy_per_job * jobs + idle_watts * idle_hours`) and nothing else in the
project measures it; see measure_idle() and docs/tasks/phase8-break-even-inputs.md.

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

from benchmarks._result import WorkloadResult
from measurement.power_monitor import MIN_TRUSTWORTHY_DURATION_S

logger = logging.getLogger(__name__)

BENCHMARKS = {
    "resnet_train": "benchmarks.resnet_train",
    "llm_inference": "benchmarks.llm_inference",
    "matmul": "benchmarks.matmul",
}


# Idle sampling defaults. 60 s rather than 30 because the window has to clear the
# 30 s floor from Yang et al. (2024) with margin, not sit on it, and because
# k8s/arav-preflight-job.yaml was already raised from 20 s to 60 s for the same
# reason. Two windows per pod at this length cost 2 minutes against a 12 to 15
# GPU-hour sweep.
DEFAULT_IDLE_SECONDS = 60.0

# One entry per column suffix each idle window contributes. Each one forbids a
# specific wrong answer, which is the rule the rest of the schema was built by:
#
#   avg_w       the quantity the carbon model multiplies by idle hours
#   min_w       the card's floor, which a co-tenant's job cannot raise
#   peak_w      proves the window was actually idle. On a shared Nautilus node
#               another tenant's work inflates avg_w with no other symptom, and
#               peak far above avg is what exposes it
#   duration_s  proves the 30 s floor was cleared for this particular window
#   n_samples   proves the figure came from a real trace and not one reading
_IDLE_SUFFIXES = ("avg_w", "min_w", "peak_w", "duration_s", "n_samples")

# Named apart because they are different physical quantities, not two samples of
# one. See measure_idle().
IDLE_WINDOWS = ("idle_pre_context", "idle_post_context")


def _blank_idle_fields(skip_reason: str) -> dict:
    """Every idle column, empty, with the reason it was not measured.

    Emitted rather than omitted so a null in the summary table always has a
    stated cause. A missing key and an unmeasurable card look identical in JSONL.
    """
    fields = {f"{w}_{s}": "" for w in IDLE_WINDOWS for s in _IDLE_SUFFIXES}
    fields["idle_skip_reason"] = skip_reason
    return fields


def _sample_idle_window(prefix: str, duration_s: float) -> dict:
    """Samples power for duration_s with no work running.

    Reuses measurement/power_monitor.py rather than sampling separately: it owns
    the NVML handle, the failed-sample accounting and the trace. No region is
    marked, so the summary covers the whole start()..stop() window, which is
    exactly what is wanted here.

    Args:
      prefix: Column prefix, one of IDLE_WINDOWS.
      duration_s: Seconds to sample.

    Returns:
      dict of prefixed scalar columns.
    """
    from measurement.power_monitor import PowerMonitor

    monitor = PowerMonitor()
    monitor.start()
    time.sleep(duration_s)
    result = monitor.stop()

    if result.duration_s < MIN_TRUSTWORTHY_DURATION_S:
        # Logged, not raised. Idle power is small and flat, which is the regime
        # where a cached reading (Yang et al., 2024) is hardest to notice: a
        # stale value looks exactly like a plausible idle trace. The row records
        # duration_s so the figure can be excluded later on the same rule the
        # benchmark rows use.
        logger.warning(
            "%s window was %.2f s, below the %.0f s floor. Treat its power as "
            "untrustworthy.",
            prefix,
            result.duration_s,
            MIN_TRUSTWORTHY_DURATION_S,
        )

    logger.info(
        "%s: avg=%.2fW min=%.2fW peak=%.2fW over %.1fs (%d samples)",
        prefix,
        result.avg_power_w,
        result.min_power_w,
        result.peak_power_w,
        result.duration_s,
        result.n_samples,
    )
    return {
        f"{prefix}_avg_w": result.avg_power_w,
        f"{prefix}_min_w": result.min_power_w,
        f"{prefix}_peak_w": result.peak_power_w,
        f"{prefix}_duration_s": result.duration_s,
        f"{prefix}_n_samples": result.n_samples,
    }


def measure_idle(duration_s: float = DEFAULT_IDLE_SECONDS) -> dict:
    """Measures idle GPU power twice, once per pod, before any benchmark work.

    Why this exists. Every benchmark records energy inside its timed region, so
    every figure the sweep produces is energy while working. The project premise
    is the opposite case: a card that sits idle most of the time never pays back
    a replacement, and real annual energy is
    `energy_per_job * jobs + idle_watts * idle_hours`. Only the first term is
    otherwise measured. See docs/tasks/phase8-break-even-inputs.md, Gap 1. A
    1080 Ti was observed drawing 55.03 W while effectively idle against a 300 W
    limit, so the term is not small.

    Two windows, because idle before and after CUDA context creation are
    different quantities and the paper has to say which one it reports:

      idle_pre_context   no CUDA context exists in this process. The card's
                         floor as the cluster sees it between jobs.
      idle_post_context  a primary CUDA context exists and the allocator is
                         initialised, but no model is loaded and no kernel is
                         running. This is the NRP case the premise is about: a
                         pod holding a GPU it is not using.

    Neither window includes the benchmark's model load, so neither includes the
    power cost of refreshing resident weights. On gpt2-xl that is 6.43 GB of
    VRAM held for the life of the run, so the recorded idle understates what an
    allocated-but-quiescent inference pod actually draws. State that as a scope
    boundary rather than treating idle_post_context as a full accounting.

    Once per pod, not once per repetition, and cold rather than hot. Per
    repetition would multiply the cost by --repeats for a quantity that does not
    vary per repetition, and durability forbids the obvious alternative of a
    post-run window: rows are written as each repetition completes, so a figure
    only known at the end cannot appear on rows already flushed. A post-run
    window would also not be comparable across cards. The design fixes the work
    and lets the time vary, so a slow card runs hotter for longer before its
    idle reading than a fast one, and hot idle differs from cold idle through
    leakage and fan draw. Cold idle is the same measurement on every card.
    Drift across a pod's life is therefore not captured; drift across the fleet
    is, because the sweep runs many pods per model and each contributes one
    fresh reading.

    Args:
      duration_s: Seconds to sample per window.

    Returns:
      dict of idle columns, stamped unchanged onto every row this pod writes.
      On any failure, blank columns and a populated idle_skip_reason.
    """
    fields = _blank_idle_fields("")

    try:
        logger.info("Idle window 1 of 2, before any CUDA context, %.0fs", duration_s)
        fields.update(_sample_idle_window("idle_pre_context", duration_s))

        # Create the context explicitly here rather than letting the first
        # benchmark create it, so the second window measures a known state
        # instead of whatever the workload happened to have done first.
        import torch

        if not torch.cuda.is_available():
            fields["idle_skip_reason"] = (
                "post-context window skipped: torch reports no CUDA device"
            )
            return fields
        torch.cuda.init()
        # A real allocation, because context creation alone does not initialise
        # the caching allocator, and a pod holding a GPU has both.
        torch.zeros(1, device="cuda:0")
        torch.cuda.synchronize()

        logger.info("Idle window 2 of 2, CUDA context live, %.0fs", duration_s)
        fields.update(_sample_idle_window("idle_post_context", duration_s))
    except Exception as exc:
        # Never fatal. A missing idle figure is a gap in the carbon model; a
        # crash here would cost the whole pod's benchmark time.
        fields["idle_skip_reason"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Idle measurement failed, continuing without it: %s", exc)

    return fields


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
    idle_fields: dict | None = None,
) -> dict:
    """Runs one repetition with power monitoring and writes its results.

    Args:
      benchmark: Key into BENCHMARKS.
      repeat_index: 1-based index of this repetition within the pod.
      out_dir: Directory holding runs.jsonl and the per-benchmark trace files.
      kwargs: Keyword arguments forwarded to the workload's run().
      monitor_gpu: False disables power monitoring, for CPU smoke tests only.
      idle_fields: Columns from measure_idle(), measured once for the pod and
        stamped unchanged onto every one of its rows. Idle is a property of the
        card and node, not of a repetition, but it is repeated per row so a
        reader of runs.jsonl never has to join against a second source.

    Returns:
      The row that was appended to runs.jsonl.
    """
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
        result = module.run(ctx=ctx, **kwargs)
        if not isinstance(result, WorkloadResult):
            # An authoring error, not a run failure, but treated as one so the
            # power trace and a row with a reason still reach the PVC before the
            # pod exits non-zero. The contract exists precisely so a payload
            # missing a required field cannot reach analysis as a silent null.
            raise TypeError(
                f"{benchmark}.run() returned {type(result).__name__}, expected a "
                "benchmarks._result.WorkloadResult"
            )
        record = result.to_row()
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
        **(idle_fields or _blank_idle_fields("idle not measured for this run")),
        **power_fields,
        # runtime_seconds is dropped rather than carried twice: the column is
        # named runtime_s in every row already written and in
        # analysis/summarize_runs.py, and two names for one number is the drift
        # WorkloadResult exists to stop.
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
        "--idle-seconds",
        type=float,
        default=DEFAULT_IDLE_SECONDS,
        help="Seconds per idle sampling window, two windows once per pod. Must "
        "clear the 30 s floor to be trustworthy.",
    )
    parser.add_argument(
        "--no-idle",
        action="store_true",
        help="Skip idle power sampling. For CPU smoke tests, and for a re-run "
        "of a card whose idle figure is already recorded.",
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

    # Before the first repetition, so the card is cold and no model is resident.
    # Deliberately not repeated per repetition; see measure_idle().
    if args.no_idle:
        idle_fields = _blank_idle_fields("--no-idle")
    elif args.no_power:
        idle_fields = _blank_idle_fields("--no-power, no NVML to sample with")
    else:
        idle_fields = measure_idle(args.idle_seconds)

    for i in range(1, args.repeats + 1):
        run_once(
            benchmark=args.benchmark,
            repeat_index=i,
            out_dir=args.out_dir,
            kwargs=kwargs,
            monitor_gpu=not args.no_power,
            idle_fields=idle_fields,
        )


if __name__ == "__main__":
    main()
