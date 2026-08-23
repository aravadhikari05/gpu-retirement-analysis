"""Derives the Phase 8 summary tables from raw run records.

Reads the append-only JSONL written by measurement/runner.py and emits two CSVs
under data/processed/. The raw file stays the source of truth: this step is
re-runnable, so a column nobody thought of is a re-derive rather than a re-run
of the sweep.

Two outputs:

  runs_flat.csv      one row per repetition, the shared columns only
  energy_by_gpu.csv  one row per (config_id, gpu_model), the Phase 8 input

The aggregate is the whole point of the schema, so the validity rules live here
rather than being left to whoever writes the notebook:

  - rows with an exclusion_reason never enter an aggregate
  - a group whose rows disagree on work_hash is refused, not averaged, because
    those runs did not do the same work
  - n_physical_gpus is reported next to n_runs, because five repetitions on one
    card is not five samples of a GPU model
  - idle power is averaged over distinct observations, not over rows, because it
    is measured once per pod and copied onto every row of that pod
"""

import argparse
import csv
import json
import logging
import os
import statistics

logger = logging.getLogger(__name__)

# The runner writes into a per-run directory alongside the power traces, not to
# the top of data/raw/. This default pointed at data/raw/runs.jsonl, which has
# never existed, so anyone running the module without --runs got a bare
# FileNotFoundError rather than a table.
DEFAULT_RUNS = "data/raw/runs/runs.jsonl"

# Shared columns, present for every workload. Workload-specific fields stay in
# the JSONL and are deliberately not flattened here; add them to a purpose-built
# table instead of widening this one until it goes sparse.
FLAT_FIELDS = [
    "run_utc",
    "run_id",
    "benchmark",
    "workload",
    "config_id",
    "repeat_index",
    "inner_iters",
    "runtime_s",
    "energy_j",
    "energy_j_counter",
    "avg_power_w",
    "peak_power_w",
    "min_power_w",
    "n_power_samples",
    "n_failed_power_samples",
    "power_duration_s",
    "below_30s_floor",
    # Idle power, measured once per pod and repeated on every row of that pod.
    # Not sparse and not workload-specific, so it belongs in the flat table:
    # real annual energy is energy_per_job * jobs + idle_watts * idle_hours, and
    # nothing else in the pipeline carries the second term. peak far above avg
    # means a co-tenant was running and the window was not idle.
    "idle_pre_context_avg_w",
    "idle_pre_context_min_w",
    "idle_pre_context_peak_w",
    "idle_pre_context_duration_s",
    "idle_pre_context_n_samples",
    "idle_post_context_avg_w",
    "idle_post_context_min_w",
    "idle_post_context_peak_w",
    "idle_post_context_duration_s",
    "idle_post_context_n_samples",
    "idle_skip_reason",
    "work_hash",
    # What the hash covers. "output" means the run was bit-identical, "config"
    # means only that the same work was requested. Without it a reader takes the
    # LLM's guarantee to be true of matmul and resnet as well.
    "work_hash_kind",
    # Region energy and whole-run energy are different quantities and must never
    # be averaged together.
    "power_window",
    "gpu_model_observed",
    "gpu_uuid",
    "node_name",
    "driver_version",
    "hardware_source",
    "precision",
    "allow_tf32_matmul",
    "allow_tf32_cudnn",
    "image_ref",
    "git_commit",
    "power_trace_path",
    "exclusion_reason",
]


def load_runs(path: str) -> list[dict]:
    """Reads JSONL, skipping blank lines. A truncated final line is fatal.

    A partially written last line means the pod died mid-write. That is worth
    surfacing rather than silently dropping, because the run it describes may
    have completed and consumed GPU time.
    """
    runs = []
    with open(path) as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{lineno} is not valid JSON, likely a truncated "
                    f"write from an interrupted pod: {exc}"
                ) from exc
    return runs


def write_flat(runs: list[dict], path: str) -> None:
    """Writes one row per repetition, shared columns only."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow({k: run.get(k, "") for k in FLAT_FIELDS})


def _distinct_numeric(members: list[dict], field: str) -> list[float]:
    """Distinct numeric values of a field across a group, sorted.

    Used for fields recorded once per pod and copied onto every row of that pod,
    where a plain mean over rows would weight a pod by its repetition count.

    Args:
      members: Run records in one aggregation group.
      field: Column name to collect.

    Returns:
      Sorted list of the distinct numeric values present.
    """
    values = {m[field] for m in members if isinstance(m.get(field), (int, float))}
    return sorted(values)


def aggregate(runs: list[dict]) -> list[dict]:
    """Groups valid runs by (config_id, gpu_model) and summarises energy."""
    groups: dict[tuple, list[dict]] = {}
    excluded = 0
    for run in runs:
        if run.get("exclusion_reason"):
            excluded += 1
            continue
        key = (run.get("config_id", ""), run.get("gpu_model_observed", ""))
        groups.setdefault(key, []).append(run)

    if excluded:
        logger.info("%d run(s) excluded by exclusion_reason", excluded)

    rows = []
    for (config_id, gpu_model), members in sorted(groups.items()):
        hashes = {m.get("work_hash", "") for m in members}
        if len(hashes) > 1:
            # Refused rather than averaged. Differing hashes mean these runs did
            # not do the same work, so their mean is not a measurement of
            # anything. Surfaced loudly; the group is dropped from the output.
            logger.error(
                "REFUSED %s on %s: %d distinct work_hash values across %d runs. "
                "These did not do the same work and will not be aggregated.",
                config_id,
                gpu_model,
                len(hashes),
                len(members),
            )
            continue

        energies = [
            m["energy_j"]
            for m in members
            if isinstance(m.get("energy_j"), (int, float))
        ]
        runtimes = [
            m["runtime_s"]
            for m in members
            if isinstance(m.get("runtime_s"), (int, float))
        ]
        inner = {m.get("inner_iters") for m in members if m.get("inner_iters")}

        # Idle is measured once per pod and copied onto every row of that pod,
        # so averaging over rows would weight a pod by how many repetitions it
        # ran. Distinct values are averaged instead: two pods producing the same
        # float to full precision does not happen in practice, so the count of
        # distinct values is the count of independent idle observations, which
        # is reported beside the mean for the same reason n_physical_gpus is.
        idle_pre = _distinct_numeric(members, "idle_pre_context_avg_w")
        idle_post = _distinct_numeric(members, "idle_post_context_avg_w")

        row = {
            "config_id": config_id,
            "gpu_model": gpu_model,
            "benchmark": members[0].get("benchmark", ""),
            "n_runs": len(members),
            # Five repetitions on one card is not five samples of a GPU model.
            # Reported next to n_runs so nobody reads the standard deviation as
            # fleet variation when it is one card's run-to-run noise.
            "n_physical_gpus": len(
                {m.get("gpu_uuid", "") for m in members if m.get("gpu_uuid")}
            ),
            "work_hash": members[0].get("work_hash", ""),
            "precision": members[0].get("precision", ""),
            "work_hash_kind": members[0].get("work_hash_kind", ""),
            "inner_iters": sorted(inner)[0] if len(inner) == 1 else "MIXED",
            "energy_j_mean": statistics.fmean(energies) if energies else "",
            "energy_j_stdev": statistics.stdev(energies) if len(energies) > 1 else "",
            "runtime_s_mean": statistics.fmean(runtimes) if runtimes else "",
            "runtime_s_stdev": statistics.stdev(runtimes) if len(runtimes) > 1 else "",
            "idle_pre_context_avg_w_mean": (
                statistics.fmean(idle_pre) if idle_pre else ""
            ),
            "idle_post_context_avg_w_mean": (
                statistics.fmean(idle_post) if idle_post else ""
            ),
            "n_idle_observations": len(idle_post or idle_pre),
        }

        # The Phase 8 input: energy for one unit of work, comparable across
        # cards only because config_id and work_hash are held equal above.
        if energies and len(inner) == 1:
            unit = sorted(inner)[0]
            row["energy_j_per_inner_iter"] = statistics.fmean(energies) / unit
        else:
            row["energy_j_per_inner_iter"] = ""

        rows.append(row)
    return rows


def write_aggregate(rows: list[dict], path: str) -> None:
    """Writes the per (config, GPU) summary."""
    fields = [
        "config_id",
        "benchmark",
        "gpu_model",
        "n_runs",
        "n_physical_gpus",
        "inner_iters",
        "precision",
        "energy_j_mean",
        "energy_j_stdev",
        "energy_j_per_inner_iter",
        "work_hash_kind",
        "runtime_s_mean",
        "runtime_s_stdev",
        "idle_pre_context_avg_w_mean",
        "idle_post_context_avg_w_mean",
        "n_idle_observations",
        "work_hash",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Derive summary tables from runs.jsonl"
    )
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--flat-out", default="data/processed/runs_flat.csv")
    parser.add_argument("--agg-out", default="data/processed/energy_by_gpu.csv")
    args = parser.parse_args()

    runs = load_runs(args.runs)
    logger.info("read %d run record(s) from %s", len(runs), args.runs)

    write_flat(runs, args.flat_out)
    rows = aggregate(runs)
    write_aggregate(rows, args.agg_out)

    logger.info("wrote %s and %s (%d group(s))", args.flat_out, args.agg_out, len(rows))
    for row in rows:
        if row["n_runs"] > 1 and row["n_physical_gpus"] == 1:
            logger.warning(
                "%s on %s: %d runs but only 1 physical GPU. The stdev is "
                "run-to-run noise on one card, not variation across the model.",
                row["config_id"],
                row["gpu_model"],
                row["n_runs"],
            )


if __name__ == "__main__":
    main()
