# Runner output pulled off the results PVC

Source: PVC `matmul-results` (rook-cephfs, RWX), namespace `cmpm118`, mounted at
`/results`. Copied with `kubectl cp` on 2026-08-23T06:50Z from the CPU pod
`arav-cifar-prep-2n8pv`, which had the PVC mounted for CIFAR-10 staging. md5
verified equal to the PVC copy for all three files at copy time:

| File | md5 |
|---|---|
| `runs.jsonl` (7 rows, 2026-08-23T07:55Z) | `dfa0a0c6e86554deae12ef048f52ac38` |
| `runs.jsonl` (1 row, first pull) | `f6fe0f37f9bfe4f4954e20b8740ccb9e` |
| `matmul/20260819T001335Z-matmul-r1-p1_power.csv` | `3e68dd3c153efd1526053592c4964cdf` |
| `legacy_matmul.csv` | `0337b45ef94e129ee8ec12d8887bd1c3` |

The PVC copies are left in place. This is a second copy, not a move.

## runs.jsonl

Seven rows as of 2026-08-23: matmul on four cards (L4, GTX 1080 Ti, RTX 2080 Ti,
RTX 3090) and resnet50 on three (1080 Ti, 2080 Ti, 3090). The cross-card tables,
the sizing constants derived from them and the per-card energy-counter behaviour
are written up in `paper/methods-notes.md`; this file records provenance only.

**Four of the seven rows carry an `exclusion_reason`**: all three resnet runs and
the 3090 matmul run fell below the 30 s floor. Their `energy_j` is recorded and
not trustworthy. Any aggregation of this file must honour `exclusion_reason`.

Every row except the L4 one carries `image_ref` and `git_commit`. The L4 row does
not; see below.

### The original row

Veda's matmul run on an L4 (`nautilus-it-gpu03.fullerton.edu`,
`GPU-8d960d0e`, driver 595.71.05), 2026-08-19T00:13:35Z. This is the **first
benchmark ever run through `measurement/runner.py` with power attached on a
GPU**, and the first `energy_j` for any workload in this project.

Checks against the row, recomputed from the file rather than quoted:

- `power_window` is `region`, so the timed-region scoping took effect on
  hardware. It had only ever run on CPU with synthetic traces.
- `energy_j` 3200.677 J against `energy_j_counter` 3194.358 J, a difference of
  +6.319 J or **0.198%**. The region-scoped integral and the region-scoped NVML
  counter delta agree, which is the cross-check the scoping change had to pass.
  Note the sign flipped relative to the whole-run 1080 Ti comparison, where the
  counter read higher; here the integral does. Both are sub-1% and the endpoint
  interpolation explains either direction.
- `power_duration_s` 44.5704 s against `runtime_s` 44.5643 s, 6.1 ms apart. The
  power window and the timed region are the same window.
- `n_failed_power_samples` 0, `hardware_source` `pynvml`, `gpu_uuid` present,
  `below_30s_floor` false at 44.6 s.
- Energy per unit of work: 3200.677 J / 500 `inner_iters` = 6.4014 J per iter.

## What this row is missing

- **`image_ref` and `git_commit` are empty strings.** `runner.py` reads both
  from env and leaves them blank without erroring, and the pod that produced
  this row did not set them. The run therefore cannot be tied to an image
  digest or a commit. Nothing about the measurement is wrong, but provenance is
  absent, and per the Output contract table that is one of the wrong answers
  the schema exists to forbid. Set both on every subsequent run.
- **`work_hash_kind` is absent.** Expected: the enforced `WorkloadResult` is
  decided but not built. This row's hash is the config-kind one
  (`work_hash_covers` says "inputs and work shape, not the product"), not the
  LLM's output-identity kind.

## Trace against summary, deliberately different

`matmul/20260819T001335Z-matmul-r1-p1_power.csv` holds **238 rows spanning
47.56 s**, while the row reports `n_power_samples` 223 over 44.57 s. That is the
intended design, not a discrepancy: the full `start()..stop()` trace is written
for cached-reading diagnosis (Yang et al., 2024), and only the summary is
clipped to the marked region.

Incidental: the full trace bottoms at **17.158 W** on the L4, outside the
region. That is not an idle-power measurement (it is whatever the card was doing
between `start()` and the region mark) but it is the first L4 number in that
neighbourhood, against 55.03 W observed idle on the 1080 Ti. See the idle-power
item in `docs/tasks/phase8-break-even-inputs.md`.

## legacy_matmul.csv

Pre-runner output, 2026-08-14, one row, RTX 2080 Ti, from the flat-CSV path that
the Output contract replaced with JSONL. `energy_j` is empty: no power was
attached. Kept as history. Do not aggregate it with `runs.jsonl`; the schemas
are different and this row has no energy, no `work_hash` and no `config_id`.
