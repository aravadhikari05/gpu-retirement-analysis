# CLAUDE.md Consolidation Plan

**Goal:** Fold the spec content from `claudeback.md` into `CLAUDE.md`, so there is
one instruction file instead of two, without losing the empirical findings that
`CLAUDE.md` already carries and without re-asserting specs that measurement has
since contradicted.

**Sources:**
- `CLAUDE.md` - current, empirical. Hard rules, version traps, kaniko, results.
- `claudeback.md` - Arav's spec draft. Conventions, component specs, citations.
- `README.md` - authoritative project structure and flow.
- `docs/phases.md` - authoritative phase numbering (1 through 10).
- `docs/tasks/phase3-llm-inference.md` - authoritative LLM workload spec.

**Principle:** `CLAUDE.md` states rules and findings. It points at the other docs
for anything they already own, rather than copying them. Three copies of the repo
tree would drift within a week; one copy plus two pointers will not.

**Deferred, not in this plan:** the Dockerfile base image question, the CSV
schema contract, and the run-duration window. Each needs a decision that changes
code, not just docs. See "Deferred decisions" at the bottom.

---

### Task 1: Add a project-structure pointer, delete the competing tree

**Goal:** One tree, in `README.md`. `CLAUDE.md` refers to it.

**Files:**
- Modify: `CLAUDE.md`

**Steps:**
- [ ] Add a short "Project structure" section near the top of `CLAUDE.md` that
      names `README.md` as the authoritative tree and flow, in one or two lines.
- [ ] Do not copy the tree into `CLAUDE.md`.
- [ ] Note in that section that `claudeback.md`'s tree used the old project name
      `gpu-carbon-breakeven/` and listed files that were never created
      (`pyproject.toml`, `docs/papers.md`, `docs/architecture.md`,
      `docs/tutorials.md`, `results/figures/`, `k8s/pvc.yaml`). That tree is
      superseded, so nobody reintroduces it from an old copy.

**Verify:** `CLAUDE.md` contains no directory tree. `README.md` unchanged.

---

### Task 2: Adopt `phases.md` numbering throughout

**Goal:** Kill the off-by-one between the two docs.

**Files:**
- Modify: `CLAUDE.md`

**Context:** `CLAUDE.md` currently calls the census "Phase 0". `docs/phases.md`
calls it "Phase 1: Inventory the Fleet". Every phase reference in `CLAUDE.md` is
one lower than the shared plan. `docs/tasks/phase0-census.md` keeps its filename
(renaming it breaks links and git history for no gain), but its content is
Phase 1 work.

**Steps:**
- [ ] Renumber every phase reference in `CLAUDE.md` to match `docs/phases.md`.
- [ ] Add one line naming `docs/phases.md` as the numbering authority.
- [ ] Add a parenthetical where `docs/tasks/phase0-census.md` is referenced,
      noting the filename says phase0 but the work is Phase 1.
- [ ] Leave the "Phase 4 measures it" reference in Established results alone;
      power measurement is Phase 4 under both schemes.

**Verify:** grep `CLAUDE.md` for "Phase" and check each hit against
`docs/phases.md`. Census reads Phase 1, container Phase 2, workloads Phase 3,
power Phase 4, storage Phase 5, sweep Phase 6, embodied Phase 7, model Phase 8,
sensitivity Phase 9, writeup Phase 10.

---

### Task 3: Replace the "Current phase" section with actual state

**Goal:** The section currently says "Phase 0: read-only GPU fleet census. Do not
launch GPU workloads yet." That is false. GPU workloads ran on 2026-08-11 across
three cards.

**Files:**
- Modify: `CLAUDE.md`

**State to record, verified against the repo on 2026-08-17:**

| Phase | Status | Evidence |
|---|---|---|
| 1 Census | Done | `data/processed/census_fleet.csv`, `census_nodes.csv`, `k8s/inventory.sh`, `k8s/summarize_census.py` |
| 2 Container | Built, contents disputed | `Dockerfile` at `4ec4b70` copies only `matmul_benchmark.py` and `power_monitor.py`; ResNet and LLM benchmarks are not in the image |
| 3 Workloads | 1 of 3 measured | LLM done (`data/raw/llm_smoke/`, 7 runs, 3 GPU models); ResNet script written, never run; matmul script written but orphaned at repo root |
| 4 Power | Written, unwired | Real `power_monitor.py` at repo root; `measurement/power_monitor.py` still a one-line stub; `measurement/runner.py` still a one-line stub; nothing imports the real one |
| 5 Storage | Ad hoc | Three PVC yamls (`results-pvc.yaml`, `k8s/sample_pvc.yaml`, `k8s/aidan-llm-models-pvc.yaml`), no canonical one |
| 6-10 | Not started | `analysis/` is `.gitkeep` only |

**Steps:**
- [ ] Replace the "Current phase" section with the table above, or a prose
      equivalent, dated 2026-08-17.
- [ ] Delete the "Do not launch GPU workloads yet" restriction outright, rather
      than carrying the existing "no longer applies to that task" exception.
- [ ] Name the two live blockers: the ResNet and LLM benchmarks are not in the
      container image, and `measurement/runner.py` does not exist, so no
      benchmark can currently be run with power measurement attached.
- [ ] Flag the duplicated files as needing resolution, without resolving them
      here: `power_monitor.py` against `measurement/power_monitor.py`,
      `matmul_benchmark.py` against `benchmarks/matmul.py`.

**Verify:** Every claim in the section traces to a file or a commit. No status
asserted that was not checked.

---

### Task 4: Point the LLM workload spec at its task doc

**Goal:** `claudeback.md` specifies gpt2 124M, 500 tokens, prompt "The future of
sustainable computing depends on". That spec is dead and must not be carried
forward.

**Why it is dead:** measurement superseded it on three counts. The measured
configuration is gpt2-xl at revision `15ea56dee5df`, 960 new tokens, a 60-token
prompt. And the degeneration finding says gpt2 at 500 tokens would score far
below the 0.562 `distinct_token_ratio` already seen at 32 tokens, meaning the run
would be predominantly a KV-cache read loop rather than inference.

**Files:**
- Modify: `CLAUDE.md`

**Steps:**
- [ ] Do not copy `claudeback.md`'s LLM spec block into `CLAUDE.md`.
- [ ] Add a line naming `docs/tasks/phase3-llm-inference.md` as the authoritative
      spec for that workload.
- [ ] Lift the seven correctness requirements from that task doc into `CLAUDE.md`
      as cross-workload rules, since they are not LLM-specific: greedy or
      otherwise deterministic decoding, fixed iteration count, explicit TF32
      control, pre-staged inputs with pinned versions and no network in the timed
      region, warmup excluded, `torch.cuda.synchronize()` on both sides of the
      timer, and a `work_hash` proving two runs did the same work.
- [ ] State the generalisation explicitly: every workload needs a `work_hash`
      equivalent, not just the LLM one. Neither `resnet_train.py` nor
      `matmul_benchmark.py` currently emits one.

**Verify:** `CLAUDE.md` mentions no token count and no prompt string for the LLM
workload. Both live in the task doc.

---

### Task 5: Add coding conventions

**Goal:** `CLAUDE.md` currently has no style guidance at all.

**Files:**
- Modify: `CLAUDE.md`

**Steps:**
- [ ] Add a "Coding conventions" section carrying, from `claudeback.md`: ruff
      check and ruff format on all Python; Google Python Style Guide; Google
      Shell Style Guide; type hints on all function signatures; Google-style
      docstrings on public functions; no hardcoded paths, use module constants or
      argparse; CSV columns lowercase_snake_case; every script runnable as
      `python -m benchmarks.resnet_train`; `if __name__ == "__main__":` in every
      script; `logging` module rather than print, except for deliberate CLI
      output.
- [ ] Add the tech stack line: Python 3.10+, Bash, PyTorch, pynvml, pandas,
      matplotlib.
- [ ] Note that `benchmarks/resnet_train.py` already conforms and is the
      reference example, while `matmul_benchmark.py` does not (no type hints, no
      docstrings on functions, prints rather than logs).

**Verify:** Run `ruff check .` and record the actual result in the commit
message. Do not claim conformance without running it.

---

### Task 6: Strengthen the shared-repo rule with the README's procedure

**Goal:** `CLAUDE.md` says "Shared repo. Read existing files before modifying."
`README.md` adds the missing procedure.

**Files:**
- Modify: `CLAUDE.md`

**Steps:**
- [ ] Extend the existing hard rule with: a missing file or directory may mean a
      teammate has work in progress elsewhere, not that a step is unstarted.
      Check `git log`, `git blame`, and live Nautilus pod and PVC state before
      concluding something does not exist.
- [ ] Add the concrete instance now on record: Veda's matmul benchmark and power
      monitor existed as real code at the repo root for four days while the
      package paths stayed one-line stubs, so reading only `benchmarks/matmul.py`
      would have concluded the work was unstarted.

**Verify:** The rule names a checkable procedure, not just an instruction to be
careful.

---

### Task 7: Reconcile the PowerMonitor contract

**Goal:** Settle one interface for `measurement/power_monitor.py`. Decision made:
keep the implementation's hardware cross-check, add back the raw trace the spec
required, pick one naming scheme.

**The conflict:**

| `claudeback.md` spec | Shipped `power_monitor.py` |
|---|---|
| returns `dict` | returns `PowerResult` object |
| `total_energy_joules` | `energy_j` |
| `avg_power_watts` | `avg_power_w` |
| `max_power_watts` | `peak_power_w` |
| `min_power_watts` | `min_power_w` |
| `duration_seconds` | `duration_s` |
| `num_samples` | `n_samples` |
| `readings: list[dict]` | absent |
| absent | `energy_j_counter` |
| `sample_interval_ms: int = 200` | `interval: float = 0.2` |

**Resolution to record in `CLAUDE.md`:**
- [ ] Keep `energy_j_counter`. It reads NVML's hardware energy counter
      (`nvmlDeviceGetTotalEnergyConsumption`, Volta and later) and is an
      independent check on the trapezoidal integral. The spec never asked for it
      and it is strictly better than not having it. It is `None` on Pascal, so
      the GTX 1080 Ti will not have it and the integral is the only number there.
      That asymmetry has to be recorded per run, not silently averaged over.
- [ ] Restore `readings`, the timestamped sample list. `claudeback.md` requires a
      per-run power trace CSV at `/results/{gpu}/{benchmark}/run_{N}_power.csv`,
      and the shipped class discards the samples on `stop()`, so that file cannot
      be written. The trace is also the only way to detect the cached-reading
      problem from Yang et al. (2024) after the fact.
- [ ] Adopt the shipped short names (`energy_j`, `avg_power_w`, ...). They are
      already written and already used; renaming them buys nothing.
- [ ] Keep `PowerResult` rather than a bare dict, but require it to expose an
      `as_dict()` so the CSV writer does not reach into attributes.
- [ ] Keep the constructor as `interval` in seconds, matching the shipped code.
- [ ] Record the error-handling rule from the spec, which the implementation is
      missing: if `nvmlDeviceGetPowerUsage` raises, log a warning and skip that
      sample rather than killing the thread. A benchmark that loses its power
      thread mid-run currently reports success with a truncated trace.

**Verify:** The section states one contract. Note in the commit message that
`measurement/power_monitor.py` is still a stub and does not yet implement it;
this task writes the contract down, it does not build it.

---

### Task 8: Add embodied carbon sources and grid intensity presets

**Goal:** Phase 7 and Phase 8 inputs, absent from `CLAUDE.md`.

**Files:**
- Modify: `CLAUDE.md`

**Steps:**
- [ ] Add embodied carbon sources: the ACT model (Gupta et al., 2022), vendor
      product carbon footprint reports (Dell, HP, Lenovo, whole-system, worked
      backwards to a GPU contribution), and die sizes from techpowerup and
      anandtech die shots. Expected range 50 to 400 kg CO2e per GPU depending on
      generation, die size, and memory.
- [ ] Add the grid intensity presets in kg CO2 per kWh: CAISO approx 0.200, US
      national average approx 0.390, ERCOT approx 0.400, PJM approx 0.550.
- [ ] Mark all of the above **unsourced pending citation**, in the same style
      `paper/methods-notes.md` used for the bandwidth figures. They came from a
      spec draft with no citation attached, and the existing hard rule says
      estimated values must be labelled as estimated.
- [ ] Restate the existing hard rule in place: ranges, not point values.

**Verify:** No number in the section is presented as verified. Each carries its
status.

---

### Task 9: Add the break-even model target

**Goal:** Freeze the Phase 8 interface now, so Phase 6 measurement outputs are
shaped to feed it.

**Files:**
- Modify: `CLAUDE.md`

**Steps:**
- [ ] Record the core inequality: replacement is worthwhile when
      `embodied_new < (energy_per_job_old - energy_per_job_new) * expected_jobs *
      grid_intensity`.
- [ ] Record the three planned `analysis/carbon_model.py` entry points:
      `break_even_jobs`, `break_even_hours_per_year`, `payback_curve`.
- [ ] Note that `analysis/` is empty and these are targets, not code.

**Verify:** The section is marked as planned, not as existing.

---

### Task 10: Add citations and remaining pitfalls

**Goal:** `CLAUDE.md` leans on Yang et al. for the 30 second floor without ever
citing it.

**Files:**
- Modify: `CLAUDE.md`

**Steps:**
- [ ] Add the six related-work entries with what each one feeds: Gupta et al.
      (2022) embodied methodology; Yang et al. (2024) power sensor accuracy and
      the 30 s floor; Uwizeyimana and Jerger (2025) carbon-aware replacement
      theory, the research question; Li et al. (2023) HPC carbon footprint
      context; Nguyen et al. (2025) T4 against RTX6000 Ada, closest prior work
      and modelled rather than measured; Fadel Argerich et al. (2026) Watt Counts
      across 10 GPUs, similar method without replacement analysis.
- [ ] Add the pitfalls from `claudeback.md` that `CLAUDE.md` lacks: pynvml
      requires paired `nvmlInit()` and `nvmlShutdown()`; old consumer GPUs may
      quantise power readings in coarse steps such as 25 W, which must be logged
      per model and reported; CIFAR-10 and model weights must be pre-staged or
      cached on the PVC rather than downloaded inside a timed region.
- [ ] Do not re-add pitfalls `CLAUDE.md` already covers in sharper, measured
      form: CUDA sync, GPU warmup, and the model download rule are all already
      handled by the Phase 3 correctness requirements from Task 4.

**Verify:** No pitfall appears twice in the merged file.

---

### Task 11: Carry the implementation order across as-is

**Goal:** Decision made: keep it, unannotated.

**Files:**
- Modify: `CLAUDE.md`

**Steps:**
- [ ] Copy the ten-step implementation order from `claudeback.md` verbatim.
- [ ] Add one line noting it describes intended build order, and that the current
      state table in Task 3 is the authority on what is actually done. The two
      disagree: the order puts `power_monitor.py` at step 2 and `runner.py` at
      step 4, both still stubs, while `llm_inference.py` at step 7 is complete
      and measured.

**Verify:** The order is present and the disagreement with reality is noted
without editing the order itself.

---

### Task 12: Retire `claudeback.md`

**Goal:** Do not leave two instruction files, without destroying the draft.

**Files:**
- Move: `claudeback.md` to `docs/claudeback-original.md`

**Steps:**
- [x] Confirm every section of `claudeback.md` is either merged, deliberately
      dropped with the reason recorded, or deferred below.
- [x] Move rather than delete. The file was never committed, so deleting it
      would leave no history to recover from.
- [x] Point at it from `CLAUDE.md` as superseded, with its stale tree named so
      nobody reintroduces it.

**Verify:** `git status` shows the file at its new path and `CLAUDE.md`
modified.

---

## Decisions taken

**A. Dockerfile base image and cu121 (was item 11). DECIDED, NOT IMPLEMENTED.**
Revert to the pre-2026-08-13 Dockerfile: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
with the cu121 index URL and `COPY benchmarks/ measurement/`, plus Veda's
`nvidia-ml-py`, plus `transformers` installed from `requirements.txt` so the
`5.15.0` pin reaches the image. Held for a second commit because it reverts a
teammate's committed work and triggers a 22 minute CI build that moves
`:latest`.

Evidence that settled it: the verified `work_hash` results are dated
2026-08-11 and Veda's Dockerfile change is 2026-08-13, so every measured result
in `paper/methods-notes.md` came from the old image. The revert is a restore.

Two things must land in the same commit or the image regresses:
- Veda's `matmul_benchmark.py` and `power_monitor.py` move into the package
  paths, otherwise the revert drops her code out of the image.
- A real `measurement/runner.py`, otherwise the entrypoint
  `python3 -m measurement.runner` points at a one-line stub and the image does
  nothing when run.

Veda's own workflow is unaffected either way. `test-pod.yaml` pulls
`docker.io/vedajanga/matmul-bench:latest` and mounts her own `matmul-results`
PVC, neither of which this repo builds.

**B. CSV schema contract (was item 13). PARTLY DECIDED.**
`measurement/runner.py` owns all result writes; benchmarks return a dict and
touch no files. `resnet_train.py` already does this. `matmul_benchmark.py` must
stop writing its own CSV.

The exact column set is still open and needs Veda and Aidan. Recorded in
`CLAUDE.md` with the likely shape (shared CSV plus per-workload JSON sidecar)
and the constraint that it must satisfy the runtime provenance rule.

**C. Run duration window (was item 15). DECIDED.**
- No upper bound on run length. The 5 minute ceiling is withdrawn; it conflicts
  with the oldest cards' ~300 s timed region and with the single-pod sweep loop.
- Durability replaces it: `runner.py` writes each repetition to the PVC as it
  completes.
- Warmup stays full length until the sizing sweep measures it.
- 5 repetitions, not 3.

The "floor binds on the fastest card" question was not open. It was already
settled by measurement in `docs/tasks/phase3-workload-sizing.md` lines 197-202.
The 20260804 census independently confirms Aidan's assumption that the fastest
reachable card is 4090 or L40S class: every faster card in the fleet reports
`openly_schedulable_with_gpu = 0`.
