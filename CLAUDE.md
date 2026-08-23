# GPU Carbon Payback on NRP Nautilus

Undergraduate research, UC Santa Cruz. Aidan Nguyen, adviser: the prof.
Collaborators: Arav Adhikari, Veda Satvika.

## Where things are written down

This file holds rules, decisions, and measured findings. It does not duplicate
what another document already owns.

- `README.md` is the authoritative project structure and flow. Do not copy the
  tree into this file or into task docs; one copy, three pointers.
- `TODO.md` is the authoritative **pickup list**: what to do next, in order,
  with what blocks what. Written for someone with none of this conversation.
  The to-do list further down this file is the **historical record** of how
  items were closed and why; it is not where you look for the next task.
  When something is finished, close it here with its evidence and remove it
  from `TODO.md`.
- `docs/phases.md` is the authoritative phase numbering, 1 through 10.
- `docs/tasks/*.md` own per-workload specs and per-phase scoping.
  `phase3-workload-sizing.md` owns sweep sizing;
  `phase6-fleet-selection.md` owns which GPU models the sweep runs on and the
  same-model variance study;
  `overnight-fleet-run.md` is a self-contained runbook for the first fleet pass,
  written so an agent with none of the surrounding conversation can execute it;
  `phase8-break-even-inputs.md` owns what the carbon model needs and what is
  not being measured for it.
- `paper/methods-notes.md` owns measured facts destined for the paper. Put them
  there, not only into task docs.
- `docs/claudeback-original.md` is a superseded spec draft, kept for history.
  Its repo tree used the old project name `gpu-carbon-breakeven/` and listed
  files that were never created (`pyproject.toml`, `docs/papers.md`,
  `docs/architecture.md`, `docs/tutorials.md`, `results/figures/`,
  `k8s/pvc.yaml`). Do not reintroduce that tree.

## Hard rules

- Never fabricate numbers. No invented GPU counts, power readings, benchmark
  results, or carbon figures. Label estimated values as estimated.
- Never claim a command succeeded without seeing its actual output.
- No em dashes in any output, code comments included.
- This repo is public. Never commit credentials, kubeconfigs, S3 keys, or tokens.
- Preserve uncertainty. Ranges, not point values, for carbon estimates.
- **Cluster hygiene outranks finishing the task.** In order: do no damage, do
  not waste shared resources, then complete the work. An unfinished run costs a
  night; a GPU held by a forgotten pod costs another team their work.
  - Every pod you create, you delete. Before stopping for any reason, run
    `kubectl -n cmpm118 get pods,jobs` and confirm nothing prefixed with your
    name is left.
  - **Never delete another person's resources.** `cmpm118` is shared outside
    this project. Anything without an owner prefix is not yours, including pods
    that have sat in `Error` for days.
  - Every GPU job carries `activeDeadlineSeconds` and `ttlSecondsAfterFinished`.
    The deadline is what releases the card when the client dies, the VPN drops
    or an agent is killed.
  - Ask the cluster before probing it. `k8s/nrp_availability.py` answers what is
    free at zero cost. Launching a pod per GPU model to find out is a last
    resort, not a habit.
  - Never add a toleration for `nautilus.io/reservation=*`. Those taints fence
    other institutions' hardware.
- Shared repo. Read existing files before modifying them. Many files under
  benchmarks/, measurement/, and analysis/ are one-line stubs, not empty.
  A missing file or directory may mean a teammate has work in progress
  elsewhere, not that a step is unstarted. Check `git log`, `git blame`, and
  live Nautilus pod and PVC state before concluding something does not exist.
  On record: Veda's matmul benchmark and power monitor existed as real code at
  the repo root for four days while `benchmarks/matmul.py` and
  `measurement/power_monitor.py` stayed one-line stubs. Reading only the package
  paths would have concluded the work was unstarted.

## Cluster environment

- kubectl context: `nautilus`, namespace: `cmpm118` (shared with other students)
- Requires UCSC VPN
- User-level access only. No admin, no node access, no scheduler changes.
- Prefix every Kubernetes resource name with **your own name**: `aidan-`,
  `arav-`, `veda-`. `cmpm118` is shared with students outside this project, so
  the prefix is what keeps one person's cleanup from deleting another's run.
  This rule previously said to prefix everything with `aidan`, which was written
  when he was the only person deploying.
- Registry: `gitlab-registry.nrp-nautilus.io/aidan/aidan`, built by GitLab CI
  on the `gitlab` remote. GitHub Actions does not build the pod image.
- Fleet reachability matters as much as fleet size. As of the 20260804 census,
  every A100 variant, the H100, H200, RTX A6000, A40 and GH200 report
  `openly_schedulable_with_gpu = 0`. They are in the census and cannot be
  landed on. The fastest card actually reachable is 4090 or L40S class, which
  is what workload sizing is built against.
- **Free is not the same as reachable, and the census cannot see either.** Our
  RBAC forbids listing pods cluster-wide and reading individual Node objects, and
  a Node carries capacity, never allocation. NRP publishes what is free at
  `guest.ListNodeInfo` on <https://portal.nrp.ai/rpc>, unauthenticated;
  `k8s/nrp_availability.py` reads it and reports free-by-model. Plan against its
  `gpu_free_open` column, and confirm with a probe pod before committing a run:
  on 2026-08-23 the feed agreed with 8 of 10 placement probes.

  As of that snapshot the cards both reachable and free are the **1080 Ti (22),
  A4000 (20), 2080 Ti (14) and 3090 (5)**. The L4, L40S and 4090 are all zero.
  All 96 L4s sit behind `nautilus.io/reservation=csuf:NoSchedule`, including
  `nautilus-it-gpu03.fullerton.edu`, the node that produced both L4 measurements
  this project owns. **Do not add tolerations for another institution's
  reservation taint.** Ask NRP or CSUF if L4 access is needed.
- **The sweep targets the 1080 Ti, 2080 Ti, 3090 and A4000.** Decided
  2026-08-23 on reachability grounds, not scientific ones: the L4, L40S and 4090
  all report zero GPUs free on nodes this namespace can schedule. Three separate
  mechanisms are involved and they have different remedies. Namespace quota bans
  A100, H100, H200 and GH200 outright (`requests.nvidia.com/a100: 0/0` and the
  same for the rest). Reservation taints fence the L4 behind CSU Fullerton.
  Ordinary contention accounts for the 4090 and L40S, so those may be usable
  opportunistically but cannot be planned around.

  This changes what the paper claims. The project was framed as replacing an
  ageing card with a modern datacenter GPU; what is measurable is a consumer and
  workstation line from 2017 to 2021. Still a real retirement question, plausibly
  the one an academic cluster actually faces, but a different one.
  **Confirm the framing with the prof before spending the sweep on it.**
  Reasoning, evidence and the fallback for opportunistic runs are in
  `docs/tasks/phase6-fleet-selection.md`.
- Plan the sweep against `allocatable_gpu_sum_swg`, not `allocatable_gpu_sum`.
  The new column (Aidan, `32a1ee5`, 2026-08-17) sums allocatable GPUs on openly
  schedulable nodes only; the old one counts reserved and tainted nodes too. The
  A10 reports 269 allocatable against 141 reachable, the L40 67 against 23. Both
  columns stay, since the gap is itself the coverage-risk number.

## Coding conventions

- Python 3.10+ and Bash. PyTorch, pynvml, pandas, matplotlib.
- All Python passes `ruff check` and `ruff format`.
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  and [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html).
- Type hints on all function signatures. Google-style docstrings on public
  functions.
- No hardcoded paths. Module-level constants or argparse.
- CSV columns are lowercase_snake_case.
- Every script runnable as a module: `python -m benchmarks.resnet_train`.
- `if __name__ == "__main__":` in every script.
- `logging`, not print, except for deliberate CLI output.

`benchmarks/resnet_train.py` is the reference example. It also shows the right
shape for a benchmark: it returns a dict and writes no files.

Enforcement is weaker than it looks. Checked 2026-08-18: `ruff check` passes on
`benchmarks/` and `measurement/`, but there is no `pyproject.toml` or
`ruff.toml`, so it runs on ruff defaults (E4, E7, E9, F) and enforces none of
the type hint, docstring or naming rules above. Adding a ruff config that
actually checks them is unclaimed work.

`ruff format --check` is clean across `benchmarks/` and `measurement/` except
`llm_inference.py`, which is left unformatted deliberately: its output is
already committed under `data/raw/llm_smoke/` and reformatting it is a
coordinated change, not a drive-by. `k8s/summarize_census.py` is also
unformatted.

## Data conventions

- `data/raw/` holds raw telemetry and is committed, except node snapshots
  matching `nodes_*.json` which are gitignored for size
- `data/processed/` holds committed summary tables
- UTC timestamps everywhere. Structured output only: JSONL, CSV, or SQLite.
- Record raw telemetry before computing summaries
- Keep failed runs with an explicit exclusion reason rather than deleting them
- The fleet census is a point-in-time snapshot, not a constant. Node labeling
  drifted by one node within an hour between two captures. Phase 3 benchmark
  runs must record the GPU model, node name, and driver version observed at
  runtime from inside the pod, not by joining against a stored census. The
  census decides which models are worth targeting; it is not the source of
  truth for what a given run actually used.

## Working style

Before writing substantial code: restate the task, inspect relevant files,
propose a plan, name the assumptions, wait for approval on architectural
changes. Then implement in small reviewable steps, showing exact commands and
expected output.

## Benchmark correctness requirements

These originate in `docs/tasks/phase3-llm-inference.md` but are not
LLM-specific. Every workload obeys them.

1. Deterministic execution. For decoding that means greedy only,
   `do_sample=False`, `num_beams=1`. Sampling makes different cards produce
   different work.
2. Fixed iteration count, not a fixed stopping condition. For decoding,
   `min_new_tokens == max_new_tokens`, otherwise EOS can stop one card early.
3. Explicit TF32 control. Ampere and later can silently route FP32 matmuls
   through tensor cores; a GTX 1080 Ti cannot. Left to PyTorch defaults, two
   cards compute different arithmetic and the comparison is meaningless. Set
   `torch.backends.cuda.matmul.allow_tf32` and `cudnn.allow_tf32` explicitly.
4. Inputs pre-staged before the timed region, with pinned versions and
   revisions. No network access during measurement.
5. Warmup iterations before timing, excluded from the measurement.
6. `torch.cuda.synchronize()` before starting and before stopping the timer.
7. A `work_hash`: a SHA-256 over the workload output, proving two runs did the
   same work. Compare within a precision mode, not across.

All three workloads emit a `work_hash` as of 2026-08-18, but they do not all
prove the same thing, and the difference matters.

`llm_inference.py` hashes the generated token IDs, so it proves the output was
bit-identical. That works because greedy decoding is an argmax and the
per-token result either matches exactly or diverges visibly.

`resnet_train.py` and `matmul.py` hash the **inputs and the shape of the work**
(seed, dataset indices or input matrices, iteration counts, precision), not the
result. Training and repeated matmul are long chains of float reductions, and
floating point addition is not associative, so results are not bit-identical
across architectures and hashing them would fail for reasons unrelated to
whether the same work was done. Given identical inputs and a fixed iteration
count, the FLOP count and operation sequence are identical on every card, which
is the fixed-work premise the project needs. Both record a result checksum or
loss sequence alongside for divergence diagnosis, never asserted equal.

State that distinction in the methods section. A reviewer will otherwise read
one `work_hash` column and assume all three carry the LLM's guarantee.

The LLM workload's own parameters (model, revision, token counts, prompt) live
in `docs/tasks/phase3-llm-inference.md`, not here. The 500-token gpt2
configuration from the old spec draft is dead: the measured configuration is
gpt2-xl at revision `15ea56dee5df` with 960 new tokens, and the degeneracy
finding below rules out the old one independently.

## Measurement contract

Implemented 2026-08-18 in `measurement/power_monitor.py`, reconciled from the
spec draft and Veda's working implementation. Naming follows the working code;
the spec draft's longer names (`total_energy_joules`, `avg_power_watts`) are not
used.

`PowerMonitor(device_index=0, interval=0.2)`, interval in seconds.
`start()` begins a daemon sampling thread, `stop()` returns a `PowerResult`.

`PowerResult` carries `energy_j`, `avg_power_w`, `peak_power_w`, `min_power_w`,
`n_samples`, `duration_s`, `energy_j_counter`, `readings`, `power_window`, and
exposes `as_dict()` so the CSV writer never reaches into attributes.
`PowerMonitor` also exposes `mark_region_start()` / `mark_region_end()`, called
by `RunContext` to scope energy to the timed region; see the subsection below.

- Keep `energy_j_counter`. It reads NVML's hardware energy counter
  (`nvmlDeviceGetTotalEnergyConsumption`) and is an independent check on the
  trapezoidal integral.

  **Measured 2026-08-18: it works on the GTX 1080 Ti.** This was expected to be
  Volta and later, leaving Pascal with no cross-check. It is not. On driver
  580.159.04 the counter returned 6156.768 J against an integral of 6120.726 J
  over the same window, agreeing to **0.59%**. A second run agreed to 0.88%. In
  both the counter reads slightly higher, which is the expected direction since
  the integral cannot cover the gap between the last sample and `stop()`.

  Do not assume availability by architecture. `PowerMonitor` probes for it and
  records `None` when it is genuinely absent, which is still the right design;
  the assumption about which cards have it was simply wrong.

- The NVML call for the driver is `nvmlSystemGetDriverVersion`.
  `nvmlDeviceGetDriverVersion` does not exist. Calling it raises
  `AttributeError`, which both `preflight.py` and `runner.py` caught, so the
  only symptom was `hardware_source` quietly reading `nvidia-smi` on every run
  instead of `pynvml`. Fixed 2026-08-18.
- Restore `readings`, the timestamped sample list. The working implementation
  discards samples on `stop()`, which makes the per-run power trace CSV
  impossible to write. The trace is also the only way to detect the cached
  reading problem (Yang et al., 2024) after the fact.
- If `nvmlDeviceGetPowerUsage` raises, log a warning and skip that sample. Do
  not let the thread die. A benchmark that loses its power thread mid-run would
  otherwise report success on a truncated trace. The skipped count is recorded
  as `n_failed_power_samples`.

**Verified on one card, 2026-08-18.** GTX 1080 Ti, `k8s-gpu-2.ucsc.edu`:
`nvidia-ml-py` is the resolved NVML provider, the energy counter works, and
sampling is sane. 132 samples over 26.5 s spanning 55.03 W to 237.67 W, against
an nvidia-smi power limit of 300 W. Note the whole window is under load by
design, so 55.03 W is the lowest sample of a loaded trace and **not** an idle
figure; it was misread as one for several days. See the idle subsection below. `n_failed_power_samples` was 0.

Power granularity on this card is **0.001 W**, with 130 distinct values across
132 samples. The pitfall below about coarse 25 W quantisation does not apply to
the 1080 Ti. It may still apply to older cards; `preflight.py` reports
`min_observed_step_w` per model, so run it on each before trusting small energy
differences.

**The counter-versus-integral agreement is per-card, and 0.59% is not
representative.** Measured 2026-08-23 across four cards: 1080 Ti -0.001% on a
57.9 s matmul, L4 +0.198%, RTX 3090 -1.39%, and **RTX 2080 Ti +6.94% on matmul
and +6.57% on resnet**. The 2080 Ti's bias is systematic, appearing on a flat
40.8 s region well above the floor, so it is not a duration or signal-shape
effect. It is not coarse power quantisation either: every card measured reports
at 0.001 to 0.004 W granularity.

**The RTX 3090 shows what looks like the cached-reading signature.** Only 58
distinct power values across 145 samples on matmul, and 27 across 79 on resnet,
against 211 of 224 on the 2080 Ti. Treat 3090 power as unverified. Detail in
`paper/methods-notes.md`.

**But the distinct-value ratio is not a proxy for accuracy, measured 2026-08-23,
and this reading of it was probably wrong.** The fleet pass produced the exact
counter-example:

| Card | distinct values per sample | longest identical run | integral against counter |
|---|---|---|---|
| RTX A4000 | 0.28 to 0.35 | 5 | **+0.02%** |
| RTX 2080 Ti | 0.90 to 0.99 | 2 | **+6%** |

The card with the worse signature than the 3090's is accurate to two decimal
places against its own hardware counter, and the card with pristine fresh
readings is the biased one. A low distinct-value ratio is at least as likely to
mean the card's power is genuinely stable, or quantised at a coarser effective
resolution, as that readings are stale.

**The test that settles it is counter against integral, not distinct-value
counting.** A stale reading repeated across a window makes the trapezoidal
integral disagree with the hardware accumulator; if the two agree to 0.02%, the
samples were not stale no matter how few distinct values they took. Apply that
test before condemning any card's power data, and say so in the methods section
rather than repeating the distinct-value heuristic as though it were diagnostic.

This does not clear the 3090, which has never been checked against its counter
the way the A4000 now has. It does mean the 3090 was set aside on weak evidence
and the check is cheap when a card can next be obtained.

**Confirmed across 50 fleet rows, 2026-08-23, and the 2080 Ti's bias is now
beyond doubt.** Per-card and per-workload, integral against counter:

| Card | matmul | resnet | llm |
|---|---|---|---|
| GTX 1080 Ti | -0.08% to +0.02% | **-0.74% to -7.72%** | -0.75% to -0.69% |
| RTX 2080 Ti | +5.61% to +5.78% | +5.87% to +6.27% | +6.57% to +6.74% |
| RTX A4000 | +0.02% to +0.08% | +0.01% to +0.05% | +0.02% to +0.08% |

The 2080 Ti's gap now has five independent confirmations across three workloads
and a 60 s preflight window, clustered tightly at 5.6 to 6.9%.

**Characterised 2026-08-23. It is a model-level trait, not a defective card.**
The 16 fleet rows come from **two different physical 2080 Tis on two different
nodes**, `GPU-1b9d9ca0` on `k8s-haosu-18` at +5.889% over matmul and resnet, and
`GPU-91cd016c` on `k8s-haosu-11` at +6.639% over llm. One bad unit is ruled out.
Note the two cards ran different workloads, so card and workload are confounded
and the 0.75 point difference between them cannot be attributed to either.

What it is not: a sampling artifact. The 2080 Ti's traces are the healthiest in
the fleet, 90 to 99% distinct values with a longest identical run of 2 and no
gaps. The samples are fresh and plentiful and simply read about 6% high.

Multiplicative or additive is **not decidable from this data**. A 6.17% scale
error and a +10.1 W offset fit almost equally well, mean residual 0.60 W against
0.70 W, because all 16 rows sit in a narrow 161 to 170 W band. Separating them
needs measurements at very different power levels on the same card. The cheapest
route is to record an energy-counter delta across the idle windows, which sit
near 20 W: `measure_idle()` already runs `PowerMonitor` for 60 s and the counter
is already read, so this is a few lines and it would settle the question.

The likeliest explanation is that the instantaneous power sensor is calibrated
high on Turing while the energy counter uses a different accumulation path.
NVIDIA documents NVML power-reading accuracy at roughly plus or minus 5%, so 6%
is at the edge of spec rather than a fault.

Weak hint, two points and worth no more than a hypothesis: within the one card
that ran two workloads, the gap grows with trace variability, matmul at
cv 0.084 giving +5.61% and resnet at cv 0.193 giving +6.27%. That would suit a
sensor that over-reads transients.

**Diagnosed 2026-08-23: the 1080 Ti resnet gap is sampling aliasing, not a card
defect.** `PowerMonitor` samples every 0.2 s. resnet on the 1080 Ti runs 1000
batches in about 202 s, which is **0.201 s per batch**. Sampler period and
workload period coincide, so every sample lands at nearly the same phase of each
batch and the mean reflects that phase rather than the cycle average. It does not
wash out over the run.

Evidence, in order of strength:

1. **It is the only combination where the periods match.** Across all nine
   workload-and-card pairs the per-iteration period divided by the 0.2 s interval
   is 1.005 for 1080 Ti resnet and either at or below 0.81, or at or above 117,
   for every other pair. One ratio near unity, one anomaly, same row.
2. **The per-repetition scatter predicts itself.** The deviation tracks how many
   beat cycles the sampler completes per run, that is how thoroughly it walks
   through phase: 43 beats gives -0.74%, 17 beats gives -7.72%. Pearson r is
   +0.873 over the six repetitions. This is also why the first repetition looked
   clean, its runtime was 197.3 s against 202 to 203 s for the rest.
3. **Everything else is excluded.** No cached readings, 975 or more distinct
   values per 1000 samples with a longest identical run of 2. No lost samples,
   zero gaps above 1 s at a mean interval of 0.206 s. The counter is stable at
   205.2 to 206.1 W across repetitions 2 to 5 while the sampled mean climbs from
   189.8 to 198.9 W.

**The counter is right and the integral is biased here.** A hardware accumulator
cannot alias.

**Decided 2026-08-23 anyway: report the integral throughout, and document the
deviations.** Switching one card-and-workload pair to the counter would mix two
instruments inside one table, which is exactly what a reviewer should object to,
and adopting the counter everywhere would force a bet on the 2080 Ti, where
nobody knows which of its two figures is correct. One method applied uniformly,
with the known deviations quantified, is the defensible position.

It is also the conservative one, which is what settles it. The integral
**understates the energy saving in all six replacement pairs**, because the two
biases compound: aliasing makes the old card look cheaper than it is, and the
2080 Ti's bias makes a new card look costlier. Delta energy comes out low by
0.1% (matmul, A4000) to 16.8% (resnet, 2080 Ti), so every break-even threshold
derived from these rows is too pessimistic rather than too optimistic. A paper
that understates its own result is making the safe error.

`energy_j_counter` must therefore stay in the rows and in the derived tables. It
is the evidence for the claim, and "conservative by 0.1 to 16.8%" is only
checkable because both figures exist side by side on every row.

**0.2 s is a bad default interval** and the roundness is what makes it dangerous:
it sits close to a plausible per-batch time for image training on mid-range
cards, which is precisely where it collides. A less round value, or a dithered
interval, would avoid locking to any workload period. Changing it means changing
`measurement/power_monitor.py` and therefore the image digest, which separates
any re-run from these 50 rows.

Note this is a **different failure mode from Yang et al. (2024)**, who document
cached readings. Both are detectable only because the full sample trace is
retained, which is the second time that design decision has paid for itself.

This also settles that the 2080 Ti's bias is unrelated: it appears at period
ratios of 0.51, 0.70 and 117 alike, so it is not aliasing.

Still unverified on every other GPU model.

### Energy is scoped to the timed region, not the whole run

Implemented 2026-08-18 and **now on `main`** (the branch pointer
`timed-region-energy-window` still exists but is an ancestor of `main`).
**Confirmed on hardware.** Veda's L4 matmul run of 2026-08-19 was pulled off
the results PVC into `data/raw/runs/` on 2026-08-23 and the figures recomputed
from the record, not quoted: `power_window` is `region`, `energy_j` 3200.677 J
against a counter delta of 3194.358 J, agreeing to **0.198%**, and
`power_duration_s` sits 6.1 ms from `runtime_s`. The region integral and the
region counter delta cover the same window, which is what this section asked
for. Note the sign flipped from the whole-run 1080 Ti check: there the counter
read higher, here the integral does. Both are sub-1% and endpoint interpolation
explains either direction.

The same run also shows the trace and the summary correctly disagreeing by
design: 238 samples over 47.56 s in the trace CSV, 223 over 44.57 s in the
summary, because the full window is kept for cached-reading diagnosis while only
the summary is clipped to the region. See `data/raw/runs/README.md`.

The runner previously wrapped `PowerMonitor.start()`/`stop()` around all of
`module.run()`, so `energy_j` integrated the model load (roughly 60 s from
cephfs on gpt2-xl) and the warmup alongside the timed loop, while
`runtime_seconds` covered only the loop. The two windows disagreed and energy
per unit of work was overstated with no visible symptom.

`benchmarks/_context.py` adds `RunContext.timed_region(device)`. Each workload
runs its measured loop inside it; the context syncs at both boundaries and calls
`monitor.mark_region_start()` / `mark_region_end()`. `PowerMonitor.stop()` then
scopes both the trapezoidal integral and the NVML energy counter delta to that
window, clipping the sample trace with interpolated endpoints, and records
`power_window` (`region` when marked, `full_run` when not). The full sample
trace is still written for cached-reading diagnosis; only the summary is scoped.
`runner.py` passes the context in as `run(ctx=..., **kwargs)`, so the benchmark
call contract changed: every `run()` now takes a `ctx` keyword.

Integration stays in the monitor, not the runner, because the monitor owns both
the sample trace and the NVML handle. Scoping the integral while leaving the
counter whole-run would compare two windows and break the 0.59%
counter-versus-integral cross-check.

Verified on CPU with synthetic traces: a constant 200 W trace over a carved
region integrates exactly (7000 J over 35 s), boundary interpolation is exact on
a ramp (9000 J), the counter delta is region-scoped, and an unmarked run still
reports the whole window. **The region path was confirmed on a GPU 2026-08-23**,
on 50 rows across three cards and three workloads: `power_window` is `region` on
every one, and the region counter delta agrees with the region integral to
within the per-card figures in the Measurement contract.

## Library version traps

Established the hard way on 2026-08-11. All of these fail silently, which is
why they are written down.

- transformers 5.x renamed `from_pretrained(torch_dtype=)` to `dtype=`.
  `from_pretrained` takes `**kwargs`, so the old name is not an error. It is
  ignored, the model loads in its checkpoint dtype, and the run records a
  precision it did not use. Select the argument by major version and then
  assert `model.dtype` matches what was requested.
- torch 2.9 deprecates `backends.cuda.matmul.allow_tf32` in favour of
  `fp32_precision`. Set both where both exist, and record the read-back rather
  than the requested value. The image currently has torch 2.5.1+cu121, where
  only `allow_tf32` exists.
- `pynvml` and `nvidia-ml-py` both install a module named `pynvml` and disagree
  on some symbols when both are present. torch itself warns about this on
  import. NVML currently works, but fall back to `nvidia-smi` rather than
  leaving `driver_version` blank, and record which source answered.
- The `cu121` index URL in the Dockerfile is load-bearing, not incidental.
  PyTorch stopped publishing cu121 wheels after 2.5.1 and newer CUDA 12.8 builds
  dropped Pascal, so the GTX 1080 Ti works because of that URL.

  **Corrected 2026-08-18 by measurement.** This entry previously said the cu121
  wheels "still carry `sm_61`". They do not. `torch.cuda.get_arch_list()` on
  torch 2.5.1+cu121 returns `sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90`,
  with no `sm_61`. The GTX 1080 Ti is `sm_61` and runs correctly anyway, because
  CUDA cubins are forward compatible across minor revisions within a major
  generation: the `sm_60` cubin executes on an `sm_61` device, though not the
  reverse. Verified on `k8s-gpu-2.ucsc.edu`, driver 580.159.04, matmul checksum
  finite and 230 W drawn under load. Record in
  `data/raw/preflight/20260818T084227Z-gtx1080ti.json`.

  The practical conclusion is unchanged and the pin still matters. What changes
  is the test: do not check for an exact `sm_XY` match, because that gives a
  false negative on every Pascal consumer card. Check for any `sm_X*` with a
  minor version at or below the device's. `measurement/preflight.py` does this.

  Open question for the paper: the 1080 Ti is running kernels tuned for GP100
  rather than GP102. That does not affect correctness and the measured runtimes
  stand, but "the card is not running its own optimised cubin" is a caveat worth
  a sentence in methods.
- `requirements.txt` pins `transformers==5.15.0`, the version that produced the
  verified `work_hash`. The Dockerfile now installs from it (`COPY
  requirements.txt` then `pip install -r`), so the pin reaches the image; this
  was the gap the 2026-08-18 Dockerfile revert closed. Do not go back to a loose
  `pip install transformers`, which is what let the pin drift before.

## Container builds

kaniko caching is disabled deliberately in `.gitlab-ci.yml`. Do not re-enable
it. With `--cache=true` kaniko pushes the whole image twice, once to the cache
repo and once to the tag. At this image size (7.71 GB, single push measured at
16 minutes) that stalls the shared Nautilus registry: pipeline 86061 applied
every layer, finished its cache push, then went silent for 20+ minutes and had
to be force-cancelled. With caching off the same build completes in 22 minutes.

`:latest` is only tagged on the default branch, so a feature branch build cannot
become the image everyone pulls. Reference images by digest, not tag. Note
`CI_COMMIT_SHORT_SHA` is 8 characters, while `git rev-parse --short` gives 7;
read the tag off the registry rather than deriving it locally.

Any Dockerfile change triggers a 22 minute build, and on `main` it moves
`:latest`. `k8s/interactive.yaml` and `k8s/sample_job.yaml` both pull `:latest`,
so they change underneath anyone using them. Change the Dockerfile
deliberately, not as a side effect of another commit.

## Established results

Phase 3 fixed-work premise, verified 2026-08-11: `work_hash` was bit-identical
across a GTX 1080 Ti (sm_61, driver 580.159.04) and an L4 (sm_89, driver
595.71.05) in fp32, gpt2 at revision 607a30d783df, batch 1, 32 tokens.
Cross-architecture agreement was a hypothesis rather than a guarantee, since
greedy decoding is an argmax and floating point addition is not associative. It
holds at this length. Records in `data/raw/llm_smoke/`.

Not yet established: agreement between two different cards of the same model
(both L4 runs used one physical GPU), and agreement at the full 500-token
length, where a single flipped argmax poisons everything after it.

**Batch-1 decode of a large model is memory-bandwidth bound. Runtime tracks
bandwidth, not architecture generation.** On gpt2-xl the 1080 Ti is within 5% of
an L4 (34.40 s against 32.87 s), while the L4 and L40S share an architecture and
differ by 2.13x (32.87 s against 15.45 s). Bandwidth ordering predicts runtime
ordering; sm version does not. Peak bandwidth figures are verified for L4
(300 GB/s) and L40S (864 GB/s) from NVIDIA product pages; the 1080 Ti's
484 GB/s and the 4090's 1008 GB/s are derived, not published, and need a
citable primary source before use in the paper. At batch 1 gpt2-xl streams all
6.43 GB of weights per token, and the 1080 Ti's 484 GB/s published bandwidth
beats the L4's 300 GB/s. If the replacement case for this workload holds, it
rests on power draw (250 W against 72 W published TDP), not on speed. That is a
hypothesis until Phase 4 measures it.

Corollary: do not scale one GPU's model-to-model cost ratio onto another card.
Doing exactly that predicted the L4 at 14 s against an actual 32.87 s, because
gpt2 and gpt2-xl sit in different regimes (compute bound against bandwidth
bound).

**Cross-model differences are bounded by same-model variance, now measured at
6.43% on one card pair.** This was an open question in this file from
2026-08-11 until 2026-08-23, when two A4000s in the same node happened to differ
by that much while repeating within themselves to 0.30%. Differences comfortably
above it, which includes every matmul and llm ratio the project reports, stand.
Differences near or below it do not. Full figures and the caveats are under
"Same-model variance is real" in this section; the study that would answer it
properly is designed in `docs/tasks/phase6-fleet-selection.md` and was
descoped.

**The fixed-work premise holds across four architectures.** Measured
2026-08-23: matmul's `work_hash` is byte-identical on the GTX 1080 Ti (sm 6.1),
RTX 2080 Ti (sm 7.5), RTX 3090 (sm 8.6) and L4 (sm 8.9); resnet's is identical
across the first three. Both are config-kind hashes, so this proves identical
work was requested, not identical numbers produced. The numbers did differ, as
predicted: two distinct matmul checksums and three distinct resnet final losses.
That closes the cross-card reproducibility question for these two workloads.

**First energy figures, and the premise survives contact with hardware.**
Measured 2026-08-23 on matmul at identical work, above the 30 s floor: the
1080 Ti burns **4.96x** the energy of an L4 while being only 1.30x slower,
because average power differs by 3.82x (274.6 W against 71.8 W). The replacement
case rests on power, not speed. Full tables, including the runs that fell below
the floor and are excluded, live in `paper/methods-notes.md`.

**Peak-FLOPS ratios do not predict runtime ratios.** Published fp32 peaks put
the L4 at 2.7x the 1080 Ti; measured, it is 1.30x, because the 1080 Ti reaches
84% of its peak on this matmul and the L4 reaches 41%. This is the second
instance of the spec-ratio error in this project, after the gpt2-to-gpt2-xl
scaling mistake below. Never size a workload or predict a runtime from a
spec-sheet ratio.

Greedy decoding degenerates with length. `distinct_token_ratio` is 0.938 at 16
tokens, 0.562 at 32, and 0.019 at 960. A run can pass `work_hash`, report
success, and still be measuring a KV-cache loop rather than inference. Validate
workload content separately from workload reproducibility.

The image that produced all of the above was built from the pre-2026-08-13
Dockerfile: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` with the cu121 index URL.
Any image change has to preserve that or the 1080 Ti results are not
reproducible.

### First fleet pass, 2026-08-23 (Arav, `9e6ce60`)

3 cards x 3 workloads x 5 repetitions, image digest
`sha256:9e62c0de`, 50 analysable rows. Read them through
`analysis/fleet_subset.py`, not by hand: it applies the three selection rules
(current sizing by `config_id`, no `exclusion_reason`, above the floor) and
warns per workload and model where a standard deviation comes from one physical
card.

Mean energy per repetition, identical work within each workload:

| Workload | GTX 1080 Ti | RTX 2080 Ti | RTX A4000 |
|---|---|---|---|
| matmul | 63,926 J / 285.2 s | 35,629 J / 205.6 s | 28,648 J / 205.2 s |
| resnet | 40,367 J / 201.1 s | 23,847 J / 139.6 s | 22,289 J / 161.3 s |
| llm | 46,754 J / 329.0 s | 33,337 J / 187.7 s | 28,077 J / 200.9 s |

**The energy ratio exceeds the speed ratio in every card and every workload.**
On matmul the A4000 uses 2.23x less energy than the 1080 Ti while being only
1.39x faster; on resnet, 1.81x less at 1.25x faster. This is the n=5 version of
the n=1 finding above, and it is the project's central result: the replacement
case rests on power draw, not throughput.

**`work_hash` is identical across all three architectures for all three
workloads**, sm_61, sm_75 and sm_86. For the LLM this is at the full 960-token
length and the hash is `output`-kind, so it proves bit-identical generated
tokens across architectures, not merely that the same work was requested. That
closes the 960-token agreement question open since 2026-08-11. matmul and resnet
remain `config`-kind and prove only identical requested work, which is the
distinction `work_hash_kind` exists to carry.

### Same-model variance is real, and it is roughly 21x the run-to-run noise

**Measured incidentally, 2026-08-23, and it bounds every cross-model claim the
project makes.** The A4000's LLM repetitions landed on two different physical
cards in the same node, `ry-gpu-16`, which is the only card-to-card comparison
the project has ever had:

| | mean energy | n | within-card spread |
|---|---|---|---|
| `GPU-5900f3b3` | 29,565.7 J | 1 | not measurable |
| `GPU-d09bcb58` | 27,779.3 J | 5 | 84.3 J, 0.30% |

**Between-card difference 6.43%, within-card spread 0.30%.** Same node, so
co-tenants, cooling and ambient are largely controlled and the gap is the cards
themselves.

The consequence for the paper: **a cross-model difference below about 6% cannot
be distinguished from card-to-card variance.** The matmul and llm ratios (1.69x
to 2.23x) clear that easily and stand. The 7.0% resnet gap between the 2080 Ti
and the A4000 does not, and must not be reported as a real difference.

This is one observation against n=1, not a confidence interval. It establishes
that the effect exists and roughly bounds it. It does not replace the study
designed in `docs/tasks/phase6-fleet-selection.md`, which was descoped by
decision on 2026-08-23; state that as a limitation rather than implying the
variance question is closed.

### The LLM reloads the model on every repetition

Measured 2026-08-23. `runner.py` calls `module.run()` once per repetition and
`llm_inference` loads gpt2-xl inside `run()`, so a 5 repetition pod pays the
load five times. On the 1080 Ti each repetition is roughly 378 s: model load,
one warmup iteration, then a 329 s timed region. The load is outside the timed
region so no energy figure is wrong, but any wall-clock estimate that assumes
one load per pod is low by about 30%. Budget from the per-repetition figure.

## Workload sizing

Settled in `docs/tasks/phase3-workload-sizing.md`, by measurement.

Sizing is set by the **fastest** card in the sweep, not the slowest, because
the work must be identical fleet-wide and the fastest card must still clear the
30 second floor. Longer generations cannot reach that floor: `n_positions` is
1024 for both gpt2 and gpt2-xl, and the L40S runs the 960 token ceiling in
15.45 s, half the floor. **Repetition inside the timed region is the only route
that reaches the floor uniformly.** Degeneracy argues the same way, since short
generations stay non-degenerate while long ones collapse to a KV-cache loop.

That design measures throughput inference, N independent decodes of the same
prompt. It does not measure single-request latency energy and says nothing
about long-context behaviour. The paper has to name which question it answers.

**Sizing constants, measured 2026-08-23.** The fastest card both reachable and
free is the RTX 3090, at 45.54 ms per matmul iteration and 88.65 ms per resnet
batch. Targeting a 90 s region on it gives **matmul `iters=2000`** and **resnet
`NUM_BATCHES` about 1000**, which puts the 1080 Ti at roughly 232 s and 195 s.
The 90 s target rather than 45 s leaves headroom for a card twice the 3090, so a
later 4090 or L40S measurement does not force a resize; resizing changes
`config_id` and `work_hash` and invalidates every row already collected. At
about 1000 batches resnet stays inside CIFAR-10, whose 50,000 rows cap the
design at 1,557 measured batches. **Both are implemented as of 2026-08-23.**
matmul's `DEFAULT_ITERS` is now 2000 rather than 200, so a forgotten `--set` no
longer produces a sub-floor run, and resnet's batch count moved from a module
constant to a `run()` kwarg (`num_batches`, with `warmup_batches` alongside),
which is also what makes the warmup-length axis below possible without another
code change. `_plan_indices` raises before the CIFAR-10 loader is built and the
error names the 1,557 ceiling.

Run duration decisions:

- **No upper bound on run length.** The old "keep runs under 5 minutes" rule is
  withdrawn. It conflicts with the chosen design twice: the oldest cards run
  roughly 300 s of timed region plus equal warmup, and the sweep runs as one
  pod looping internally over all points because the 7.71 GB image pull
  dominates otherwise.
- **Durability replaces the ceiling.** The ceiling was a proxy for not losing
  work to preemption. `runner.py` writes each repetition's row to the PVC as it
  completes, so a preemption costs one repetition.
- **Warmup stays full length until measured.** It is half the compute budget
  and shortening it would save roughly 40% of GPU time, but warmup exists to
  reach thermal and clock steady state, energy is the dependent variable, and
  old and new cards ramp clocks differently. Add a warmup-length axis to the
  sizing sweep rather than cutting it on an assumption.
- **5 repetitions, not 3.** Failed runs are kept with an exclusion reason, so
  effective n is below nominal n; starting at 3 and excluding one leaves a
  standard deviation from n=2. Co-tenant thermal interference on shared nodes
  is an uncontrolled variance source arguing for more samples. The only thing 3
  buys is wall clock, which is the cheap resource here: the sweep is
  courtesy-serialized rather than capacity-limited and blocks nobody.

## Output contract

Settled 2026-08-18. `measurement/runner.py` owns all result writes. Benchmarks
return a `benchmarks._result.WorkloadResult` and touch no files; the runner
flattens it with `.to_row()`.

**Raw JSONL, derived CSV.** `runner.py` appends one JSON line per repetition to
`runs.jsonl`; `analysis/summarize_runs.py` derives
`data/processed/runs_flat.csv` and `data/processed/energy_by_gpu.csv`. This is
the existing convention (`data/raw/` is raw telemetry, `data/processed/` is
committed summary tables) applied to benchmark output. A single CSV cannot hold
three workloads with different natural fields without going sparse or needing a
sidecar that every query then joins back, and JSONL carries lists such as the
per-batch loss sequence inline. The practical payoff: a column nobody thought of
is a re-derive, not a 12-hour re-run of the sweep.

**Grain is one row per repetition.** That is both the unit of exclusion (below
the floor, or crashed) and the unit of independence for a standard deviation.
Anything finer lives inline as a list, or beside the row in the power trace CSV.

**`analysis/fleet_subset.py` is the one definition of the analysable slice.**
Added 2026-08-23. `runs.jsonl` is append-only and now spans four commits and
three sizing generations, so most work wants a subset of it. Selecting by hand
invites two mistakes that look identical in a table: pooling two sizing
generations, and including an excluded run. The module applies three rules
(`config_id` in `KEEP_CONFIGS`, empty `exclusion_reason`, `below_30s_floor` is
`False`), writes `data/processed/fleet_runs.{jsonl,csv}`, and warns per workload
and model wherever the rows come from one physical card. After a sizing change
it selects nothing and exits with an error naming `KEEP_CONFIGS`, rather than
writing an empty table a later step would read as "no runs happened". Add the
new `config_id` there; do not loosen the match to a workload name.

**Two loops, named apart.** `repeat_index` is the runner's `--repeats`, the
outer loop for statistical spread. `inner_iters` is the workload's own loop
inside the timed region, which exists to clear the 30 s floor. Energy per unit
of work is `energy_j / inner_iters`. Conflating them scales every energy figure
by the wrong factor, silently.

**`config_id` states what was asked for; `work_hash` proves it happened.**
Aidan's format, now used by all three workloads:
`gpt2-xl|15ea56dee5df|fp32|b1|n960|p72ef35ff2d6d`. Grouping by workload name
alone would average 32-token and 960-token runs into one meaningless figure.

### How the column set was chosen

Not by listing available fields. By writing the Phase 8 aggregation first and
asking what wrong answer the schema still permits. Each wrong answer implies one
column, and you stop when you cannot invent a new one.

| Wrong answer it would otherwise permit | Column |
|---|---|
| Average fp32 and tf32 runs together | `precision`, `allow_tf32_matmul` |
| Average 32-token and 960-token runs | `config_id` |
| Include a sub-30 s or crashed run | `exclusion_reason`, `below_30s_floor` |
| Average runs that did different work | `work_hash` |
| Report n=5 for a model measured on one card | `gpu_uuid` |
| Compare runs built from different images | `image_ref`, `git_commit` |
| Average region energy against whole-run energy | `power_window` |
| Read the LLM's bit-identical guarantee onto matmul and resnet | `work_hash_kind` |
| Count only working energy for a card that is idle most of the time | `idle_pre_context_avg_w`, `idle_post_context_avg_w` |
| Read a co-tenant's job on a shared node as the card's idle draw | `idle_*_peak_w`, `idle_*_min_w` |
| Trust an idle figure from a window below the floor | `idle_*_duration_s`, `idle_*_n_samples` |

`gpu_uuid` earns its place from evidence: the two L4 runs on 2026-08-11 both
used `GPU-e82f7d3b`, and the 1080 Ti runs used two different physical cards.
Neither is knowable without it, and "agreement between two cards of the same
model" is a named open question in this file.

Derived quantities are not stored. `gflops_per_s` follows from `total_flops` and
`runtime_s`; store inputs and compute outputs in the summary step.

`summarize_runs.py` enforces the validity rules rather than leaving them to
whoever writes the notebook: excluded rows never enter an aggregate, a group
whose rows disagree on `work_hash` is **refused rather than averaged**, and
`n_physical_gpus` is reported next to `n_runs` so a standard deviation from one
card is not read as fleet variation.

**Settled 2026-08-23: llm adopts `inner_iters`, default 8.** It now runs 8
identical `generate()` calls inside the timed region, the same
repetition-inside-the-timed-region design matmul and resnet already use, and
`config_id` carries `i{inner_iters}` so a 1 iteration run can never be pooled
with an 8 iteration one. All three workloads now report a real `inner_iters`.
Rationale and the reason not to retune it are in `docs/tasks/phase3-llm-inference.md`.

### The record schema is convention, not enforcement, and it drifted

Verified 2026-08-18 by comparing the three benchmarks' returned dicts. The
runner-owned spine (identity, power, provenance) is standardized, because
`runner.py` writes it. The benchmark payloads are not, because each `run()`
hand-builds its own dict, so the required set lives only in the table above and
in three separately typed functions. Three fields that mean the same thing are
named or emitted differently:

- `precision` on matmul and resnet, `precision_mode` on llm. The table above
  names `precision` a required column, so llm rows leave it null and the value
  hides under a second name. llm predates `benchmarks/_precision.py` and carries
  its own `_set_precision`.
- `work_hash_covers` on matmul and resnet, nothing on llm. The one field that
  distinguishes the LLM's output-identity guarantee from the other two's
  fixed-work guarantee is absent on the workload that actually has the strong
  one.
- `workload` on matmul and resnet, absent on llm, which reuses the runner's
  `benchmark` key. Two keys for one concept.

None of these crash. JSONL is sparse, so they surface as silent nulls in
analysis, which is exactly the failure class the schema table exists to prevent.
Grouping by `config_id` and the `work_hash` disagreement refusal both still hold,
so nothing measured so far is wrong; the exposure is on the sweep's rows and on
any query that reads a required column by name. The cause is that the contract is
convention, and convention cannot fail loud.

How you know what to record is not by listing available fields. It is the rule
in "How the column set was chosen" above: write the Phase 8 aggregation first,
then for each wrong answer it could still produce, add exactly the one field that
forbids it, and stop when you cannot invent a new wrong answer.

**Decided 2026-08-18, built 2026-08-23** in `benchmarks/_result.py`, adopted by
all three workloads and consumed by `runner.py`. What follows is the design as
decided; the paragraph after it records what building it actually settled.
Replace the convention with an enforced `WorkloadResult`. Required fields become constructor arguments, so a benchmark
physically cannot return a record missing one; the error moves from a runtime
null to an authoring-time construction error. Workload-specific fields go in an
`extra` dict, and the runner reads `.to_row()`. The required set, derived by the
rule above rather than by listing fields:

`workload`, `config_id`, `work_hash`, `work_hash_kind` (`output` or `config`),
`precision` (in `PRECISIONS`), `allow_tf32_matmul`, `allow_tf32_cudnn`,
`inner_iters` (at least 1), `runtime_seconds` (above 0).

This forces one open decision loud rather than silent: llm emits no `inner_iters`
today, since it runs one `generate()` per timed region, so enforcement fails
until it is set. The interim value is 1; the real fix is the
repetition-in-the-timed-region change tracked under Unclaimed side work.

**What building it settled, 2026-08-23.** All three drifts are closed. llm now
emits `precision` rather than `precision_mode`, `workload` rather than reusing
the runner's `benchmark` key, and a `work_hash_kind` of `output` with the prose
`work_hash_covers` it previously lacked; matmul and resnet emit `config`. The
runner treats a benchmark that returns anything other than a `WorkloadResult` as
a recorded failure, so the power trace and a row with an exclusion reason still
reach the PVC before the pod exits non-zero.

Three things the decision did not say, resolved while implementing:

- **`extra` may not shadow a required field**, and every workload has a
  precision or provenance dict that overlaps the required set.
  `benchmarks._result.extra_fields()` filters against `REQUIRED_FIELDS`, so
  promoting a field into the required set later does not have to be undone by
  hand at three call sites.
- **`runtime_seconds` is not written to the row.** The column is `runtime_s`,
  which is what every row already recorded and `summarize_runs.py` use. The
  runner drops the constructor's name rather than carrying both, since two names
  for one number is the drift being removed.
- **`work_hash_kind` and `power_window` were missing from the derived CSV.**
  Both were named in the table above and neither was in
  `summarize_runs.FLAT_FIELDS`, so the distinction existed in the JSONL and
  vanished in the table analysis actually reads. Added, along with `workload`.

`benchmarks/_result.py` deliberately imports no torch, so the contract is
testable in a plain interpreter off the GPU image. `tests/test_workload_result.py`
and `tests/test_idle_power.py` run under `python -m unittest discover -s tests`.

### Idle power is recorded per pod, in two windows

Decided and implemented 2026-08-23, closing Gap 1 of
`docs/tasks/phase8-break-even-inputs.md`, which owns the reasoning at length.
`measure_idle()` in `measurement/runner.py` reuses `PowerMonitor` rather than
sampling separately, and its output is stamped onto every row the pod writes.

Two windows, because idle before and after CUDA context creation are different
quantities and the paper has to name which it reports. `idle_pre_context` is the
card's floor with no context in the process; `idle_post_context` has a live
primary context and allocator but no model and no kernel, which is the NRP case
the project premise is about, a pod holding a GPU it is not using. Neither
includes the benchmark's model load, so neither covers resident weights. That is
a scope boundary to state, not an oversight.

**Once per pod, before the first repetition, cold.** Per repetition would
multiply the cost by `--repeats` for a quantity that does not vary per
repetition. A post-run window is not merely more expensive, it is impossible
here: rows are flushed as each repetition completes, so a figure first known at
the end cannot appear on rows already written. It would also not be comparable
across cards, because the design fixes the work and lets the time vary, so a
slow card idles hotter than a fast one by construction. Cold idle is the same
measurement everywhere. Drift within a pod is therefore not captured; drift
across the fleet is, since the sweep runs many pods per model.

**The window clears the 30 s floor deliberately, at 60 s by default.** An idle
trace is flat and low, which is exactly the regime where a cached reading
(Yang et al., 2024) is indistinguishable from a real one, so this is the case
that most needs the floor rather than the one that least does. `--idle-seconds`
configures it, `--no-idle` skips it, and `--no-power` skips it automatically so a
CPU smoke test still runs. A failure populates `idle_skip_reason` and is never
raised: a missing idle figure is a gap in the carbon model, while a crash there
would cost the whole pod's benchmark time.

`summarize_runs.py` averages idle over **distinct observations, not over rows**,
because one pod's figure is copied onto all of its rows and a mean over rows
would weight a pod by how many repetitions it ran. `n_idle_observations` sits
beside the mean for the same reason `n_physical_gpus` sits beside `n_runs`.

**Never executed against NVML.** The whole path is CPU-tested only.

## Embodied carbon and grid intensity

**All figures in this section are unsourced pending citation.** They came from a
spec draft with no citation attached. Treat them as placeholders and source them
individually before use, in the style `paper/methods-notes.md` used for the
bandwidth figures. Ranges, not point values.

Sources to work from: the ACT model (Gupta et al., 2022) for die size, process
node and fab characteristics; vendor product carbon footprint reports (Dell, HP,
Lenovo), which are whole-system and have to be worked backwards to a GPU
contribution; die sizes from techpowerup and anandtech die shots.

Expected range per GPU: 50 to 400 kg CO2e depending on generation, die size and
memory.

Grid intensity presets, kg CO2 per kWh: CAISO approx 0.200, US national average
approx 0.390, ERCOT approx 0.400, PJM approx 0.550.

## Break-even model

`analysis/summarize_runs.py` prepares the input table.
**`analysis/carbon_model.py` exists as of 2026-08-23**, with
`analysis/grid_intensity.py` beside it and `tests/test_carbon_model.py` covering
both. It reads `data/processed/energy_by_gpu.csv` and re-derives nothing from
`runs.jsonl`, since `analysis/fleet_subset.py` is the one definition of the
analysable slice.

Built before Phase 7 deliberately. Embodied carbon is an argument to the model,
not a prerequisite for writing it, and Phase 9 has to sweep it regardless, so
the model is parameterised over it either way. `EmbodiedEstimate` and
`GridIntensity` carry a `sourced` flag, any False taints
`BreakEven.provisional`, and the CLI refuses to print without
`--allow-unsourced`. Every figure it produces today is arithmetic, not a result.

Three measured findings are enforced in code rather than described in prose:
a per-job energy difference inside the 6.43% same-model variance bound sets
`interpretable=False`; every result carries the note that the integral
understates the saving, so thresholds are pessimistic about replacement; and an
idle differential below 13.57 W is reported as noise, per the scatter finding in
`paper/methods-notes.md`.

**A job is one `inner_iter`**, which is what `energy_j_per_inner_iter` holds.
**Gap 4 is resolved in the signature**: `horizon_years=None` reproduces the
original job-count inequality and drops the idle term, because idle is a rate
and a job count has no time dimension. Including idle needs a horizon. That is
the project premise failing to fit the original equation, not a code
limitation.

Core inequality as originally written:
`embodied_new < (energy_per_job_old - energy_per_job_new) * expected_jobs * grid_intensity`.

Planned entry points in `analysis/carbon_model.py`: `break_even_jobs`,
`break_even_hours_per_year`, `payback_curve`.

**Do not implement this inequality as written without reading
`docs/tasks/phase8-break-even-inputs.md` first.** It walks the equation term by
term and finds six gaps. Two are now closed.

**Gap 1, idle power: closed 2026-08-23.** It is measured per pod in two windows
and recorded on every row. It also reversed the assumption that motivated it:
idle draw is roughly equal across all three cards, so the operational saving
from replacement is in active hours, not idle ones. The 55.03 W "idle" figure
this file quoted from 2026-08-18 was a preflight load-trace minimum, not an idle
measurement. See the Measurement contract and Established results.

**Gap 5, the replacement unit: decided 2026-08-23. The unit is the GPU, not the
node.** Phase 7 sources per-card embodied figures. Grounds: the operational
measurement is board-level, so a whole-system embodied figure would compare two
different scopes across the inequality; the fleet is consumer and workstation
cards that genuinely get swapped inside an existing chassis; and the framing
question is about a card rather than a server. The cost of the choice, which
methods has to state, is that vendor PCF reports are whole-system, so a GPU-only
figure is worked backwards from them and carries that subtraction's error bars.
Prefer the ACT model, which builds up from die size and is naturally GPU-shaped,
and use PCF reports as a cross-check. The decision assumes NRP swaps cards
rather than retiring whole nodes, which is our assumption and not confirmed by
NRP; it reopens if the framing conversation says otherwise.

The other four, summarised: the measurement boundary is the GPU board and
excludes the host CPU feeding it; PUE is absent though we rejected CodeCarbon
for hiding exactly that; the equation is a single-year snapshot while Phase 9
wants declining grid intensity over time; and an NRP-specific recommendation
needs real utilisation data the census does not contain.

Units, written down once, since dropping the conversion is wrong by 3.6 million
while still looking plausible:
`carbon_saved_kg = (delta_energy_j * jobs / 3.6e6) * grid_intensity`.

## How NRP actually acquires hardware, and who decides retirement

Researched 2026-08-23 from NRP's own documentation. This settles the scope
assumption behind the GPU-unit decision and corrects who the Phase 10
deliverable is addressed to.

**The unit of contribution is a whole server, not a GPU.** NRP's contributor
guide: "Contributing a node to the NRP is a trade. You supply the hardware, the
rack space, and the network path." NRP supplies "the operating system, joins the
node to the Nautilus cluster, and runs it from there, patching, monitoring, and
upgrades included", and scopes itself to "security from the OS up". Contributors
keep priority on their own nodes and everyone else gets opportunistic access
when they are idle, which is the mechanism this project has been scheduling
against all along.

**NRP therefore never replaces a GPU or retires a node. The contributing
institution does.** NRP manages software; the hardware stays the owner's, down
to physical work: the guide notes NRP "may ask you to reboot a node or swap a
drive". The owner is the actor who would swap a card or retire a chassis.

**No documented retirement or decommissioning policy exists.** Searched the
admin docs, cluster policies and the 2025 architecture paper
(arXiv:2505.22864). Nodes leaving the cluster are not covered anywhere public.
That absence is itself reportable: it means there is no institutional guidance
for the decision this project is studying.

Three consequences:

- **The GPU unit stands, but as a modelling choice rather than an observed
  practice.** Contribution is node-level, so a whole-node scope would be equally
  defensible on NRP's own terms. What supports the GPU unit for *this* fleet is
  the hardware: the 1080 Ti, 2080 Ti and 3090 are consumer PCIe cards in
  commodity servers, on the FIONA "low-cost, high-performance server-grade"
  reference design, and a PCIe card is physically swappable in a way an
  SXM-socketed A100 is not. State it as a choice, and treat node-level
  replacement as a sensitivity arm rather than pretending the question is
  settled.
- **A caveat the paper has to carry:** a 2017-era node's CPU, RAM, PSU and PCIe
  generation are also aged, so swapping only the GPU may not be realistic on the
  oldest hardware even though it is physically possible. That biases the
  GPU-only scope optimistic, which is the opposite direction from the
  measurement biases.
- **Phase 10 is addressed to the wrong audience as written.** `docs/phases.md`
  promises "a practical recommendation for NRP". NRP does not make this
  decision. The deliverable should target **contributing institutions** deciding
  whether to swap cards or replace a node, with NRP as the context that sets
  utilisation, not as the actor.

Sources: <https://nrp.ai/documentation/admindocs/participating/new-contributor-guide/>,
<https://nrp.ai/documentation/admindocs/links/hardware/>,
<https://nrp.ai/about/>, <https://nrp.ai/documentation/userdocs/start/policies/>,
<https://arxiv.org/abs/2505.22864>.

## Related work

1. Gupta et al. (2022), ACT carbon modeling tool. Embodied carbon methodology.
2. Yang et al. (2024), nvidia-smi power sensor accuracy. Source of the 30 second
   floor and the cached-reading limitation.
3. Uwizeyimana and Jerger (2025), carbon-aware replacement theory. The research
   question.
4. Li et al. (2023), HPC carbon footprint estimation. Broader context.
5. Nguyen et al. (2025), T4 against RTX6000 Ada carbon comparison. Closest prior
   work, modelled rather than measured.
6. Fadel Argerich et al. (2026), Watt Counts energy benchmark across 10 GPUs.
   Similar methodology, no replacement analysis.

## Pitfalls

Sync, warmup, and pre-staged inputs are covered by the correctness requirements
above and are not repeated here.

- pynvml requires paired `nvmlInit()` and `nvmlShutdown()`.
- Old consumer GPUs may quantise power readings in coarse steps, for example
  25 W. Log the observed granularity per GPU model and report it.
  Measured 2026-08-18: this does **not** affect the GTX 1080 Ti, which reports
  at 0.001 W with 130 distinct values across 132 samples. It remains an open
  risk on the older cards (TITAN Xp, Quadro M4000, GTX 1080), so
  `measurement/preflight.py` reports `min_observed_step_w` per model. Run it on
  a card before trusting small energy differences from that card.
- CIFAR-10 and model weights must be pre-staged or cached on the PVC. A download
  inside a timed region invalidates the run.
- Run `measurement/preflight.py` on each GPU model before its first measured
  run. It has already falsified two documented assumptions.

## Implementation order

From the original spec draft, kept as written. It describes intended build
order. The state table below is the authority on what is actually done, and the
two have never matched: steps 1 to 4, 6 and 7 are written, step 5 is the one
still blocking, and the workloads were built in the reverse of this order.

1. `k8s/inventory.sh`, verify you can query the cluster and see GPU models
2. `measurement/power_monitor.py`, test pynvml works in a basic pod
3. `benchmarks/matmul.py`, simplest benchmark, validates the pipeline
4. `measurement/runner.py`, integrate power monitor and benchmark and CSV output
5. `Dockerfile`, `k8s/benchmark-pod.yaml`, `k8s/pvc.yaml`, containerize and run
6. `benchmarks/resnet_train.py`, second benchmark
7. `benchmarks/llm_inference.py`, third benchmark
8. Run full sweep across all GPU models, 5 to 10 reps each
9. `analysis/carbon_model.py`, break-even math
10. `analysis/sensitivity.py` and `analysis/plots.py`, charts and sensitivity

## Current state

Verified against the repo on 2026-08-18, updated 2026-08-22 for Veda's
`2dc4e7e` (Phase 5) and Aidan's `32a1ee5` (census reachable capacity), and
again 2026-08-23 for the first fleet pass (Arav, `9e6ce60`).
Phase numbers follow `docs/phases.md`.

| Phase | Status | Evidence |
|---|---|---|
| 1 Census | Done | `data/processed/census_fleet.csv`, `census_nodes.csv`, `k8s/inventory.sh`, `k8s/summarize_census.py`. Task doc is `docs/tasks/phase0-census.md`; the filename says phase0 but the work is Phase 1. |
| 2 Container | **Done, verified on hardware** | `nvidia/cuda:12.1.0-runtime-ubuntu22.04`, pinned `torch==2.5.1` / `torchvision==0.20.1`, installs from `requirements.txt`, copies both packages. Built and pulled on a 1080 Ti 2026-08-18. Open: which registry is authoritative, see below. |
| 3 Workloads | **Done, all 3 measured through the runner with power** | All three emit `work_hash`, set TF32 explicitly, carry `config_id`, and return an enforced `WorkloadResult`. All three ran 5 repetitions on the 1080 Ti, 2080 Ti and A4000 on 2026-08-23. Sizing constants implemented and verified on hardware. The LLM now has `energy_j`, which was its last gap. |
| 4 Power | **Working end to end on all 3 fleet cards, all 3 workloads** | Region scoping, idle power and the enforced record all verified on hardware 2026-08-23. Counter-against-integral agreement is per-card and per-workload: see the Measurement contract. Preflight done on 1080 Ti, 2080 Ti and A4000; still outstanding on the 3090, which is out of the fleet. |
| 5 Storage | **Done** (Veda, 2026-08-20, `2dc4e7e`) | `k8s/benchmark-pod.yaml` is a real templated pod: both PVCs mounted, `NODE_NAME` / `IMAGE_REF` / `GIT_COMMIT` env, `HF_HOME=/models/hf`, `HF_HUB_OFFLINE=1`, `nodeSelector` on `nvidia.com/gpu.product`, args matching the runner CLI. `k8s/results-pvc.yaml` is the canonical results claim (live name `matmul-results`) and `k8s/STORAGE.md` documents the volume layout. |
| 6 Sweep | **First fleet pass done, 2026-08-23** | 3 cards (1080 Ti, 2080 Ti, A4000) x 3 workloads x 5 repetitions, 50 analysable rows in `data/raw/runs/runs.jsonl`, selected by `analysis/fleet_subset.py` into `data/processed/fleet_runs.{jsonl,csv}`. No 3090, by decision. Every gate passed. Two things it is not: the framing is still unconfirmed with the prof, and same-model variance is measured for exactly one card pair, so cross-model differences below about 6% are not interpretable. |
| 7 Embodied carbon | **Not started, and now the critical path** | Every figure is an unsourced placeholder. Blocks Phase 8. The energy side is done, so this is the only thing between the project and a break-even number. |
| 8 Carbon model | Scoped, not built | `analysis/summarize_runs.py` and `analysis/fleet_subset.py` prepare the input tables. `carbon_model.py` does not exist; its required inputs are reviewed in `docs/tasks/phase8-break-even-inputs.md`. |
| 9 to 10 | Not started | `paper/methods-notes.md` already holds real measured content. |

GPU workloads are no longer restricted. The earlier "read-only census, do not
launch GPU workloads" rule is withdrawn; workloads ran on 2026-08-11.

The root duplicates are resolved: `matmul_benchmark.py` and `power_monitor.py`
moved into `benchmarks/matmul.py` and `measurement/power_monitor.py`, and the
root copies deleted.

### Verified on hardware, 2026-08-18

GTX 1080 Ti on `k8s-gpu-2.ucsc.edu`, image
`ghcr.io/aravadhikari05/gpu-retirement-analysis:sha-d53ab4f7`, via
`k8s/arav-preflight-job.yaml`. Records in `data/raw/preflight/`.

- The image builds and runs on Nautilus. All pins landed: torch 2.5.1+cu121,
  torchvision 0.20.1+cu121, transformers 5.15.0, python 3.10.12.
- The 1080 Ti computes correctly, via `sm_60` forward compatibility rather than
  the `sm_61` this file previously claimed. See the version traps section.
- NVML works through `nvidia-ml-py`, including the energy counter, agreeing with
  the integral to 0.59%.
- Power sampling is sane and finely grained on this card.

**Note on the registry.** This was pulled from GHCR, built by GitHub Actions,
not from `gitlab-registry.nrp-nautilus.io`. The GHCR package is public, so
Nautilus pulls it anonymously. This clone has no `gitlab` remote, so the
Nautilus registry image cannot be built from here at all. Either add that remote
or treat GHCR as the image source and update the note above about GitHub Actions
not building the pod image.

### What is still unverified

Rewritten 2026-08-23 after the first fleet pass. Most of what stood here is now
closed; what remains is listed honestly rather than deleted.

Closed by the fleet pass: preflight on the 2080 Ti and A4000, resnet on a GPU,
timed-region scoping for llm, llm energy, image and commit provenance on every
row, and the sub-floor problem, which the new sizing constants fixed.

Still open:

- **The 3090 has never been preflighted**, and its two rows show the
  cached-reading signature, so its energy figures stay unusable. It is out of
  the fleet by decision, so this only matters if it is added back.
- **Same-model variance is measured for exactly one card pair**, the two A4000s
  above, and only on the llm workload. Every other model in the fleet is one
  physical card, so its standard deviation is run-to-run noise. The designed
  study in `docs/tasks/phase6-fleet-selection.md` was descoped on 2026-08-23 and
  is a stated limitation, not a closed question.
- **The 1080 Ti resnet counter-versus-integral gap is unexplained.** See the
  Measurement contract.
- **The framing has not been confirmed with the prof.** The fleet is a
  consumer and workstation line from 2017 to 2021, not the modern datacenter
  GPUs the project was framed around. This gates what the paper can claim and
  costs no GPU time.
- **L4, L40S, 4090 and every A100 class card remain unreachable.** Quota,
  reservation taints and contention, with different remedies each; see Cluster
  environment.
- **The results PVC is still named `matmul-results`** and now holds three
  workloads. Keep and explain in methods, or migrate. Migrating gets more
  expensive with every run.

### k8s plumbing, landed 2026-08-20

Veda's `2dc4e7e` closed this. `k8s/benchmark-pod.yaml` is a `{{PLACEHOLDER}}`
template, substituted per run with `envsubst` or `sed`: `RUN_NAME`, `IMAGE`,
`GPU_MODEL`, `BENCHMARK`, `REPEATS`, `SET_ARGS`, `GIT_COMMIT`. It mounts
`matmul-results` at `/results` and `aidan-llm-models-pvc` at `/models`, sets
`HF_HOME=/models/hf` to match `aidan-llm-prep-job.yaml`, and relies on the
image's `ENTRYPOINT ["python3", "-m", "measurement.runner"]`, so `args:` are
runner flags.

Two things it leaves open. The results claim is named `matmul-results` for
historical reasons and cannot be renamed without destroying the L4 run it holds,
so either the name stays and methods explains it, or the data is migrated before
the sweep. And `k8s/sample_pvc.yaml` still exists; `k8s/STORAGE.md` says ignore
it, which is documentation rather than deletion.

## To-do list, in pickup order

Written 2026-08-18, revised 2026-08-22 against teammates' commits, revised again
2026-08-23 after the first fleet pass. Items 1 through 7 are now closed; nothing
has been renumbered, so links to "to-do #N" still point where they did. Ordered
by the critical path: each blocked item names what unblocks it. Owners are
suggestions from this file's existing assignments, not fixed. Before starting
any item, check `git log` and live Nautilus pod and PVC state, per the hard rule
about stubs and teammates' in-progress work.

### 1. Phase 5, k8s storage: DONE (Veda, 2026-08-20, `2dc4e7e`)

`k8s/benchmark-pod.yaml`, `k8s/results-pvc.yaml` and `k8s/STORAGE.md` cover every
requirement this item listed: both PVCs mounted, `/results/data/cifar10` writable
on the RWX results claim, `NODE_NAME` from the downward API plus `IMAGE_REF` and
`GIT_COMMIT`, and one canonical results PVC. See "k8s plumbing, landed
2026-08-20" above.

Residual, small, not blocking:

- The canonical results claim is named `matmul-results`. Decide keep against
  migrate before the sweep starts writing into it; afterwards moving it is
  expensive.
- `k8s/sample_pvc.yaml` is documented as ignorable but not deleted.
- The pod pins `{{IMAGE}}` to `:latest`, which moves on every `main` Dockerfile
  build. Container builds says reference by digest. Substitute a digest for sweep
  runs.

### 2. Finish the first-GPU-run validation: DONE (Arav, 2026-08-23, `9e6ce60`)

Veda ran matmul on an L4 through `runner.py` with power on 2026-08-19. The row
and its power trace are now in `data/raw/runs/` (pulled 2026-08-23, md5 matched
against the PVC, PVC copy left in place) and the figures check out when
recomputed: `power_window=region`, integral against counter 0.198%,
`power_duration_s` within 6.1 ms of `runtime_s`. That closes the timed-region
question on hardware.

**All of it is closed as of 2026-08-23** by the first fleet pass, which followed
`docs/tasks/overnight-fleet-run.md` end to end. All three workloads ran through
`runner.py` with power on three cards; llm has `energy_j`; the sizing constants
put every row above the 30 s floor; and `IMAGE_REF` and `GIT_COMMIT` are set on
every one of the 50 rows. The L4 row from 2026-08-19 is still the one record in
the repo with empty provenance, and stays that way as a historical artifact.

`runner.py` still accepts blank `image_ref` and `git_commit` without complaint.
Making it refuse is unclaimed work: the Output contract lists both as the
columns that forbid comparing runs built from different images, and a silent
blank defeats that.

### 3. Idle-power decision: DONE 2026-08-23, and verified on hardware

`measure_idle()` in `measurement/runner.py`, two windows per pod split by CUDA
context creation, 60 s each, reusing `PowerMonitor`. Mechanism and reasoning are
in Output contract, "Idle power is recorded per pod, in two windows", and in
Gap 1 of `docs/tasks/phase8-break-even-inputs.md`.

**Ran against NVML for the first time 2026-08-23, on 14 pod-level observations
across three cards. It works, and it falsified the figure it was going to be
checked against.**

| Card | `idle_pre_context` | `idle_post_context` |
|---|---|---|
| GTX 1080 Ti | 8.75 to 17.39 W | 24.10 to 25.80 W |
| RTX 2080 Ti | 5.97 to 23.00 W | 19.83 to 33.70 W |
| RTX A4000 | 15.52 to 19.73 W | 25.06 to 29.36 W |

`idle_post_context` is above `idle_pre_context` on every one of the 50 rows,
which is the ordering the two windows were built to expose: a live CUDA context
and allocator cost real watts on an otherwise unused card.

**Resolved 2026-08-23: the 55.03 W this section expected was never an idle
figure.** It is `min_power_w` from `measurement/preflight.py`, whose power window
is loaded on purpose: the monitor starts and a sustained matmul runs for the
whole window, under the comment "Light sustained load so the reading is not
idle." That window averaged 229.96 W and peaked at 237.67 W. The second
preflight run on the same node gives 56.34 W the same way, which is why the
number looked reproducible.

A load-trace minimum is an **upper bound** on idle draw, not a measurement of
it, so 55.03 W and the fleet's 25 W never contradicted each other. The same
applies to the 16.52 W once quoted as the A4000's idle floor. Any power figure
attributed to preflight is of this kind: `preflight.py` has no idle field.

**Consequence, and it cuts against the project premise.** With the real figures,
idle draw is essentially flat across the fleet and the newest card is highest.
The 3.3x idle advantage once claimed for the A4000 does not exist, the
difference that remains is inside the 6.43% same-model variance bound, and the
defensible statement is that replacement buys **no measurable idle saving**.
Break-even therefore rests almost entirely on active hours. Corrections were
applied to `paper/methods-notes.md`, `docs/tasks/phase8-break-even-inputs.md`
and `data/raw/runs/README.md`, all of which had inherited the figure.

Still genuinely unmeasured: idle on the specific `k8s-gpu-2` card, and idle on
the L4 and 3090 at all.

One caveat on these windows. `idle_pre_context_peak_w` reaches 64 to 73 W on
every card while the averages sit under 25 W, so co-tenant activity is landing
inside the idle windows on shared nodes. The average is still low enough to be
credible, but the peak is the documented tell that the window was not
exclusively ours, and it is why `idle_*_peak_w` is recorded at all.

### 4. Standardize the record schema (DONE 2026-08-23)

`benchmarks/_result.py` holds the enforced `WorkloadResult`; all three
benchmarks return one and `runner.py` reads `.to_row()`. All three drifts are
closed and llm reports `inner_iters=1`. What building it settled, including the
three points the decision did not cover, is recorded in Output contract under
"The record schema is convention, not enforcement, and it drifted".

Outstanding: **no workload has been constructed on a GPU since the change.** The
contract is unit-tested without torch, but the first GPU run after this must
confirm that all three `run()` functions still return successfully, since a
required field now raises where it previously wrote a null.

### 5. Cross-card reproducibility checks

- **Done 2026-08-23 for matmul (4 cards) and resnet (3 cards).** Both hashes are
  identical across architectures. See Established results.
- **LLM `work_hash` at 960 tokens: closed 2026-08-23.** Identical across three
  architectures, sm_61, sm_75 and sm_86, and it is the `output`-kind hash. See
  Established results.
- **Same-model variance: descoped by decision 2026-08-23, and partially answered
  by accident.** The designed study in `docs/tasks/phase6-fleet-selection.md`
  (matmul, 5 repetitions on each of at least 5 distinct `gpu_uuid` values) was
  **not run**: node pinning was dropped from scope, the project being a
  time-boxed undergraduate one, and the shortfall goes in the presentation as a
  limitation. What exists instead is one accidental card pair, two A4000s in the
  same node on the llm workload, differing by 6.43% against a 0.30% within-card
  spread. That is enough to bound cross-model claims and not enough to close the
  question. The design stays on file for anyone who picks it up.

### 6. Preflight each GPU model before its first measured run (fleet done)

`measurement/preflight.py` is per-card and has already falsified two documented
assumptions.

- 1080 Ti: done 2026-08-18.
- **2080 Ti: done 2026-08-23, and it confirmed the problem rather than clearing
  it.** Over a 60 s window the integral read 17866.658 J against a counter of
  16816.193 J, +6.25%, with 0.001 W granularity and 303 distinct values in 321
  samples. Third independent confirmation of the same per-card bias after matmul
  at +6.94% and resnet at +6.57%, and it rules out both coarse quantisation and
  cached readings as the cause. Which of the two figures to trust on Turing is an
  open question the paper has to address.
- **A4000: done 2026-08-23.** Agrees to -0.894% over 60 s, in the same
  direction as the 1080 Ti, which makes the 2080 Ti an outlier among four cards
  rather than a method problem. It is also the only card measured that stays
  under its power limit. Its 16.52 W was recorded here as an idle floor against
  the 1080 Ti's 55.03 W; both are preflight load-trace minima rather than idle
  measurements, and directly measured idle has the two cards within 2 W of each
  other. Corrected 2026-08-23.
- **3090: still outstanding, and not for want of trying.** Of five GPUs the
  availability feed reported free on 2026-08-23, none could be obtained. Pinning
  to individual nodes returned `Insufficient nvidia.com/gpu` on one and
  `Insufficient memory` on another, and the other two had zero free CPU. Its
  benchmark traces show the cached-reading signature, so its energy numbers stay
  unusable until this runs. A job is queued and will take the first card that
  frees.

`k8s/arav-preflight-job.yaml` now samples for 60 s rather than 20, because
comparing the integral against the counter over a window below the 30 s floor
measures the floor rather than the card.

### 6b. Implement the measured sizing constants (DONE 2026-08-23)

matmul `DEFAULT_ITERS` is 2000, raised from 200 so a forgotten `--set` cannot
produce a sub-floor run. resnet's batch count is now a `run()` kwarg,
`num_batches`, defaulting to 1000, with `warmup_batches` alongside it, and
`_plan_indices` raises before the CIFAR-10 loader is built with an error naming
the 1,557 measured-batch ceiling. See Workload sizing.

Both defaults change `config_id` and `work_hash`, so rows recorded at the old
sizes will not aggregate with new ones. That is the schema working, not a
migration. Neither new default has run on a GPU, so the 90 s target is a
prediction from the 3090 per-iteration figures and not yet an observation.

### 7. Phase 6, the sweep: FIRST PASS DONE (Arav, 2026-08-23, `9e6ce60`)

3 cards x 3 workloads x 5 repetitions, every gate passed, 50 analysable rows.
Results in Established results; read them through `analysis/fleet_subset.py`.
`schema-idle-sizing` is merged.

What the pass deliberately did **not** do, so nobody assumes otherwise:

- **No node pinning**, so repetitions landed on one physical card per model,
  with the single A4000 llm exception. Descoped by decision.
- **No warmup-length axis.** Worth revisiting cheaply: warmup was measured at
  `warmup_iters=10` against 2000 measured iters for matmul and 5 batches against
  1000 for resnet, so it is about 0.5% of the compute budget, not the half this
  file previously assumed. The risk is that 10 iterations is too **short** to
  reach thermal and clock steady state, which is the opposite of the concern
  that was written down. Cheap to test, and energy is the dependent variable.
- **No 3090.**

A re-run is plausible rather than unlikely, since the framing is unconfirmed and
same-model variance is barely sampled. Treat these rows as a real dataset that
may need repeating, not as final.

### The one sizing constant that is an estimate

**LLM `inner_iters`: closed 2026-08-23.** Set to 8, chosen from the measured
L40S runtime plus margin rather than from a 3090 measurement, since no 3090
could be obtained. It remains the one sizing constant that is an estimate: if a
3090 later shows slack, **do not retune it down**, because the headroom is the
insurance and a change to `inner_iters` changes `config_id` and forces a re-run.

Kept here rather than moved to `TODO.md` because it is a standing decision, not
an open task. The 8 iterations were confirmed on hardware across three cards on
2026-08-23.

### 8 onward: Phase 7 and later

**Moved to `TODO.md`.** Everything from Phase 7 (embodied carbon) onward is
open work rather than history, along with the unclaimed side items that used to
sit at the end of this list. `TODO.md` owns them, in dependency order, with what
blocks what.

They were removed from here rather than mirrored because two copies of a task
list drift, and the stale one is always the one somebody reads. What stays above
is the record of how items 1 through 7 were closed and what the closing
measured, which is the part worth keeping in this file.
