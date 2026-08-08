"""Compares llm_inference run records and reports work_hash agreement.

This is the check that decides whether the fixed-work premise holds. Two runs
are comparable only when their config_id matches; within a config, work_hash
must be identical or the runs did different computations and the energy numbers
are not comparable.

When hashes disagree, the divergence index is reported from the token ID
sidecars rather than the pair being written off as invalid. Greedy decoding is
an argmax over logits and floating point addition is not associative, so a
near-tie that flips on one architecture explains a late divergence, while an
immediate divergence points at something else entirely, such as a different
dtype or TF32 silently enabled.

Usage:
  python3 k8s/compare_llm_runs.py data/raw/llm_smoke/*.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_record(path: str) -> dict:
    with open(path) as handle:
        record = json.load(handle)
    record["_path"] = path
    return record


def load_tokens(record: dict) -> list:
    """Loads the token ID sidecar next to a record, if it is there."""
    sidecar = record.get("token_ids_path")
    candidates = []
    if sidecar:
        candidates.append(sidecar)
        candidates.append(
            os.path.join(os.path.dirname(record["_path"]), os.path.basename(sidecar))
        )
    candidates.append(f"{os.path.splitext(record['_path'])[0]}.tokens.json")

    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate) as handle:
                tokens = json.load(handle).get("token_ids")
            if tokens and isinstance(tokens[0], list):
                return tokens[0]
            return tokens or []
    return []


def divergence_index(left: list, right: list):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def describe(record: dict) -> str:
    gpu = record.get("gpu_model_observed") or record.get("gpu_model_torch") or "unknown"
    return (
        f"{os.path.basename(record['_path'])}\n"
        f"      gpu        {gpu}\n"
        f"      driver     {record.get('driver_version') or 'unrecorded'}\n"
        f"      node       {record.get('node_name') or 'unrecorded'}\n"
        f"      dtype      {record.get('resolved_dtype')}"
        f"  tf32 matmul/cudnn {record.get('allow_tf32_matmul')}"
        f"/{record.get('allow_tf32_cudnn')}"
        f"  sm {record.get('sm_capability')}\n"
        f"      runtime    {record.get('runtime_seconds')} s\n"
        f"      work_hash  {record.get('work_hash')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare llm_inference run records")
    parser.add_argument("records", nargs="+", help="Paths to run record JSON files")
    args = parser.parse_args()

    # A bare *.json glob also matches the token sidecars written alongside each
    # record. They carry config_id and work_hash, so left in they would be
    # counted as extra runs of the same config.
    paths, skipped = [], []
    for path in args.records:
        (skipped if path.endswith(".tokens.json") else paths).append(path)
    if skipped:
        print(f"skipping {len(skipped)} token sidecar file(s)", file=sys.stderr)
    if not paths:
        print("no run records given", file=sys.stderr)
        return 2

    records = [load_record(p) for p in paths]

    by_config = defaultdict(list)
    for record in records:
        by_config[record.get("config_id", "<missing config_id>")].append(record)

    exit_code = 0

    for config_id, group in sorted(by_config.items()):
        print(f"\nconfig_id: {config_id}")
        print(f"  {len(group)} run(s)")
        for record in group:
            print(f"    - {describe(record)}")

        hashes = {r.get("work_hash") for r in group}
        if len(group) < 2:
            print("  RESULT: only one run, nothing to compare")
            continue

        if len(hashes) == 1:
            print(f"  RESULT: MATCH, all {len(group)} runs share work_hash")
            continue

        exit_code = 1
        print(f"  RESULT: MISMATCH, {len(hashes)} distinct work_hash values")

        base = group[0]
        base_tokens = load_tokens(base)
        for other in group[1:]:
            if other.get("work_hash") == base.get("work_hash"):
                continue
            other_tokens = load_tokens(other)
            if not base_tokens or not other_tokens:
                print(
                    f"    {os.path.basename(base['_path'])} vs "
                    f"{os.path.basename(other['_path'])}: token sidecar missing, "
                    "cannot locate divergence"
                )
                continue
            index = divergence_index(base_tokens, other_tokens)
            print(
                f"    {os.path.basename(base['_path'])} vs "
                f"{os.path.basename(other['_path'])}: first divergence at "
                f"generated token index {index} of {len(base_tokens)}"
            )
            if index is not None:
                lo = max(0, index - 3)
                hi = index + 4
                print(f"      a[{lo}:{hi}] = {base_tokens[lo:hi]}")
                print(f"      b[{lo}:{hi}] = {other_tokens[lo:hi]}")

    if len(by_config) > 1:
        print(
            f"\nnote: {len(by_config)} distinct config_id values are present. "
            "Runs are only comparable within a config_id."
        )

    print()
    if exit_code:
        print(
            "At least one config has divergent work_hash values. Do not proceed to "
            "full length runs. A genuine cross-architecture divergence is a finding "
            "about the method, not something to adjust away."
        )
    else:
        print("All configs with more than one run agree on work_hash.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
