"""Selects the analysable fleet rows from the raw run records.

`runs.jsonl` is append-only and holds every run the project has ever recorded,
across four commits and three sizing generations. Most downstream work wants
one specific slice of it: the runs that did the current, agreed-upon amount of
work and are trustworthy enough to aggregate. Rebuilding that slice by hand
each time invites two mistakes that look identical in a table, pooling runs of
different sizes and silently including an excluded run.

This module is the one definition of that slice, so the filter is stated once
and re-derived rather than snapshotted. Adding rows to the raw file and running
this again is the supported workflow; editing the output by hand is not.

Selection, all three conditions required:

  1. config_id is in KEEP_CONFIGS, which pins the workload sizing. Two runs with
     different iteration counts are not the same experiment, and config_id is
     what records that.
  2. exclusion_reason is empty. Failed runs are kept in the raw file with a
     reason rather than deleted, so they have to be filtered here instead.
  3. below_30s_floor is false. Below the floor the power figure is not
     trustworthy (Yang et al., 2024), which is the whole reason the flag exists.

Nothing is deleted from the raw file. The excluded runs are evidence about what
the old sizing did and why it was changed, and `data/raw/` is the source of
truth.

All three workloads are in scope. They are not directly comparable to each
other: matmul and resnet hash their inputs and the shape of the work, while the
LLM hashes its generated tokens, so only the LLM's work_hash proves the output
was bit-identical. Compare within a workload, and read work_hash_kind before
reading work_hash.

Two outputs, same rows, because the consumers differ:

  <name>.jsonl  full records including list-valued fields such as the per-batch
                loss sequence, for anything that needs them
  <name>.csv    scalar columns only, for a spreadsheet or a quick plot
"""

import argparse
import csv
import json
import logging
import os

logger = logging.getLogger(__name__)

# The sizing generation currently in use, measured 2026-08-23. Both were raised
# so the fastest reachable card still clears the 30 s floor. Changing a sizing
# constant changes config_id by design, so an old row can never be pooled with a
# new one: that is the schema working, not a migration to perform. Add the new
# config_id here rather than loosening the match to a workload name.
KEEP_CONFIGS = (
    "matmul|n8192|fp32|i2000|s20260818",
    "resnet50|cifar10|fp32|b32|n1000|s20260818",
    # The LLM carries i8 in its config_id, so a 1 iteration run can never pool
    # with an 8 iteration one. Its work_hash is output-kind while the other two
    # are config-kind, which is why work_hash_kind travels with every row rather
    # than being inferred from the workload name.
    "gpt2-xl|15ea56dee5df|fp32|b1|n960|i8|p72ef35ff2d6d",
)

DEFAULT_RUNS = "data/raw/runs/runs.jsonl"
DEFAULT_OUT_PREFIX = "data/processed/fleet_runs"


def load_runs(path: str) -> list[dict]:
    """Reads JSONL, skipping blank lines. A truncated final line is fatal.

    Args:
        path: Path to the append-only run record file.

    Returns:
        One dict per run record, in file order.

    Raises:
        RuntimeError: If a line is not valid JSON, which means a pod died
            mid-write. The run it describes may have completed and consumed GPU
            time, so it is surfaced rather than silently dropped.
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


def select(runs: list[dict], keep_configs: tuple[str, ...]) -> list[dict]:
    """Applies the three selection conditions documented in the module docstring.

    Args:
        runs: All run records, as read from the raw file.
        keep_configs: config_id values whose sizing is the current one.

    Returns:
        The subset that is safe to aggregate, in file order.
    """
    kept = []
    for run in runs:
        if run.get("config_id") not in keep_configs:
            continue
        if run.get("exclusion_reason"):
            continue
        # Compared against False rather than used for truthiness, so that a row
        # missing the field entirely is dropped rather than silently kept.
        if run.get("below_30s_floor") is not False:
            continue
        kept.append(run)
    return kept


def scalar_fields(runs: list[dict]) -> list[str]:
    """Collects column names in first-seen order, skipping list and dict values.

    The CSV is the convenience output. Fields such as the per-batch loss
    sequence stay in the JSONL rather than being flattened into a cell that
    nothing can parse back.

    Args:
        runs: The selected run records.

    Returns:
        Column names, in the order first encountered.
    """
    fields: list[str] = []
    for run in runs:
        for key, value in run.items():
            if key not in fields and not isinstance(value, (list, dict)):
                fields.append(key)
    return fields


def write_jsonl(runs: list[dict], path: str) -> None:
    """Writes one JSON record per line, preserving every field.

    Args:
        runs: The selected run records.
        path: Destination path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        for run in runs:
            handle.write(json.dumps(run) + "\n")


def write_csv(runs: list[dict], path: str) -> None:
    """Writes the scalar columns only.

    Args:
        runs: The selected run records.
        path: Destination path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=scalar_fields(runs), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(runs)


def summarise(runs: list[dict]) -> dict[tuple[str, str], int]:
    """Counts selected rows per workload and observed GPU model.

    Args:
        runs: The selected run records.

    Returns:
        Row counts keyed by (workload, gpu_model_observed).
    """
    counts: dict[tuple[str, str], int] = {}
    for run in runs:
        key = (run.get("workload", "?"), run.get("gpu_model_observed", "?"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Select the analysable fleet rows from runs.jsonl"
    )
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="Output path without extension. Writes .jsonl and .csv alongside.",
    )
    args = parser.parse_args()

    runs = load_runs(args.runs)
    kept = select(runs, KEEP_CONFIGS)
    logger.info("read %d record(s), selected %d", len(runs), len(kept))

    if not kept:
        # An empty result is almost always a stale KEEP_CONFIGS after a sizing
        # change, not an empty input file. Say so rather than writing an empty
        # table that a later step reads as "no runs happened".
        raise SystemExit(
            f"no rows matched {KEEP_CONFIGS}. If the sizing constants changed, "
            f"add the new config_id to KEEP_CONFIGS in {__name__}."
        )

    jsonl_path = f"{args.out_prefix}.jsonl"
    csv_path = f"{args.out_prefix}.csv"
    write_jsonl(kept, jsonl_path)
    write_csv(kept, csv_path)
    logger.info("wrote %s and %s", jsonl_path, csv_path)

    for (workload, gpu_model), count in sorted(summarise(kept).items()):
        logger.info("  %-13s %-26s %d", workload, gpu_model, count)

    # Repetitions on one physical card are not samples of a GPU model. The
    # aggregate step warns about this too, but the warning belongs here as well
    # because this file is what gets handed to a plot or a spreadsheet, where
    # nothing else carries the caveat.
    # Keyed by workload as well as model. A model can have two physical cards
    # across the file while still having only one within a given workload, which
    # is exactly the A4000's situation: its LLM rows span two cards and its
    # matmul and resnet rows do not. Checking per model alone would clear the
    # A4000 and hide that.
    by_model: dict[tuple[str, str], set] = {}
    for run in kept:
        key = (run.get("workload", "?"), run.get("gpu_model_observed", "?"))
        by_model.setdefault(key, set()).add(run.get("gpu_uuid"))
    for (workload, gpu_model), uuids in sorted(by_model.items()):
        if len(uuids) == 1:
            logger.warning(
                "%s on %s: all selected rows are one physical card. Spread "
                "here is run-to-run noise, not card-to-card variation.",
                workload,
                gpu_model,
            )
        else:
            logger.info(
                "%s on %s: %d physical cards, so between-card spread is "
                "measurable here.",
                workload,
                gpu_model,
                len(uuids),
            )


if __name__ == "__main__":
    main()
