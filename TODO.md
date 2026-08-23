# TODO: what to do next

Actionable pickup list. Written 2026-08-23, after the first fleet pass.

This file owns **what to do next**. It does not restate rules, conventions or
measured findings: `CLAUDE.md` owns those and is authoritative wherever the two
appear to disagree. Each item below names the document that owns its detail
rather than copying it.

**Before touching anything, read `CLAUDE.md`, especially Hard rules.** The
cluster is shared with students outside this project and the hygiene rules there
outrank finishing any task in this file.

---

## Where the project stands

The measurement half is done. Three workloads (matmul, resnet, llm) ran 5
repetitions each on three GPUs (GTX 1080 Ti, RTX 2080 Ti, RTX A4000) on
2026-08-23, with power scoped to the timed region and idle power recorded.
Every gate passed. 50 analysable rows.

**Nothing below is blocked on more GPU time.** The critical path is desk work.

Data:

| What | Where |
|---|---|
| Raw records, append-only, all 57 | `data/raw/runs/runs.jsonl` |
| Power traces, one CSV per repetition | `data/raw/runs/{matmul,resnet_train,llm_inference}/` |
| The analysable slice, 50 rows | `data/processed/fleet_runs.{jsonl,csv}` |
| Per-repetition and per-group summaries | `data/processed/runs_flat.csv`, `energy_by_gpu.csv` |

Regenerate both derived sets from the raw file. Neither is edited by hand:

```bash
python3 -m analysis.fleet_subset      # -> data/processed/fleet_runs.{jsonl,csv}
python3 -m analysis.summarize_runs    # -> runs_flat.csv, energy_by_gpu.csv
```

`analysis/fleet_subset.py` is the **one definition of the analysable slice**:
current sizing by `config_id`, no `exclusion_reason`, above the 30 s floor. Do
not hand-filter `runs.jsonl`; pooling two sizing generations and including an
excluded run both look identical in a table.

### Two results that constrain everything downstream

Both are in `CLAUDE.md` under Established results with full figures. Restated
here only because they change what later work is allowed to claim:

1. **Same-model variance is about 6.43%**, measured on two A4000s in one node,
   against a within-card spread of 0.30%. **Any cross-model difference below
   roughly 6% is not interpretable.** The energy ratios (1.69x to 2.23x) clear
   this easily. The 7.0% resnet gap between the 2080 Ti and the A4000 does not.
2. **Idle power is roughly equal across the three cards**, 25.3 W to 27.1 W with
   a live CUDA context, and the newest card is not the lowest. The difference is
   inside the variance bound above, so the honest statement is that replacement
   buys no measurable idle saving, and break-even rests almost entirely on
   active hours. This reverses an earlier claim of a 3.3x idle advantage for the
   newer card, which came from misreading a preflight load-trace minimum as an
   idle figure; see the closed item under Open findings. It matters because the
   project premise is about cards that sit idle.

---

## Critical path

### 1. Replacement unit: DECIDED, the GPU (2026-08-23)

**Gap 5 of `docs/tasks/phase8-break-even-inputs.md`.** Is the thing being
replaced a GPU inside an existing node, or a whole node? Vendor product carbon
footprint reports are whole-system, so a GPU-only figure has to be worked
backwards from them, which is a modelling assumption with its own error bars.

This decision determines **which numbers Phase 7 goes and sources**, so making
it afterwards means sourcing twice.

**Confirmed 2026-08-23: the unit is the GPU, not the node.** Phase 7 sources
per-card embodied figures. Item 2 is unblocked.

Three reasons, all grounded in what the project actually measured:

1. **It matches the operational scope.** `pynvml` reports the GPU board and
   nothing else, which is Gap 2 of the same document. Putting a whole-system
   embodied figure on one side of the inequality and board-only operational
   energy on the other compares two different scopes and quietly favours the
   answer with the larger embodied term.
2. **It matches the hardware.** The fleet is consumer and workstation cards,
   1080 Ti, 2080 Ti and A4000. These are cards that genuinely get swapped inside
   an existing chassis, which is the realistic move for an academic cluster
   running donated hardware.
3. **It matches the question.** "Is this old card worth replacing" rather than
   "is this old server worth replacing."

What accepting it costs, and what therefore has to be stated in methods:

- Vendor PCF reports are whole-system, so a GPU-only figure is worked backwards
  from them. That subtraction is a modelling assumption with its own error bars,
  stacked on the report's. The ACT model builds up from die size instead and is
  naturally GPU-shaped, so prefer it where the die-level inputs exist and use
  PCF reports as a cross-check rather than the primary source.
- Host energy is excluded on the operational side, which understates the slower
  card. Known direction, so state it.

**One thing to carry into the framing conversation below.** This decision
assumes NRP replaces cards inside existing chassis rather than retiring whole
nodes. That assumption is the project's own, not something NRP has confirmed. If
the framing conversation establishes that whole nodes are the real unit, this
reopens and Phase 7 sources whole-system figures instead. Worth one sentence of
confirmation there rather than a separate decision.

Two related boundaries in the same Gap, which need stating rather than solving:
a card retired here may be redeployed elsewhere rather than scrapped, in which
case its remaining operational carbon moves rather than disappears; and
end-of-life and disposal carbon is absent from the model entirely, though it is
usually small next to manufacturing.

### 2. Phase 7: embodied carbon (unblocked, blocks items 3 and 4)

Owner doc: `docs/phases.md` (weeks 5 to 6). Placeholder figures and the sources
to work from are in `CLAUDE.md` under Embodied carbon and grid intensity.

Produce a per-GPU embodied carbon estimate for at least the **GTX 1080 Ti, RTX
2080 Ti and RTX A4000**, from the ACT model (Gupta et al., 2022), vendor PCF
reports, and die sizes.

- **Ranges, not point values.** Honest uncertainty beats false precision.
- **Every current figure in `CLAUDE.md` is an unsourced placeholder.** Treat the
  50 to 400 kg CO2e range as a starting hypothesis, not evidence, and cite each
  number individually in the style `paper/methods-notes.md` used for memory
  bandwidth.
- Never present an estimated number as measured. This is a hard rule.

This is the single thing standing between the project and a break-even number.
It needs no cluster access.

### 3. Phase 8: the carbon model

Owner doc: `docs/tasks/phase8-break-even-inputs.md`. **Read it before writing
any code**; it walks the equation term by term and finds six gaps.

Build `analysis/carbon_model.py` with `break_even_jobs`,
`break_even_hours_per_year`, `payback_curve`.

Use this units line exactly. Dropping the conversion is wrong by 3.6 million
while still looking plausible:

```
carbon_saved_kg = (delta_energy_j * jobs / 3.6e6) * grid_intensity
```

Status of the six gaps as of 2026-08-23:

| Gap | Status |
|---|---|
| 1. Idle power unmeasured | **Closed** by the fleet pass, two windows per pod |
| 2. Boundary is the GPU board only | Open. A statement to make in methods, not work. Excludes host CPU, so it understates the slower card |
| 3. PUE absent | Open. Probably a swept parameter: the fleet spans many institutions |
| 4. Snapshot against integral | Open. Grid intensity declines, so later savings are worth less |
| 5. GPU against whole node | **Closed 2026-08-23**: the GPU. See item 1 |
| 6. Real NRP utilisation | Open. Probably unavailable at user-level access. If so, ship a parameterised curve and say the reader supplies utilisation |

### 4. Phase 9: sensitivity

Owner doc: `docs/phases.md` (weeks 7 to 8). Sweep embodied carbon across its
plausible range and find where the break-even answer flips. Project grid
intensity forward to see whether replacements that fail today pay off by 2030.

Add two axes the original plan did not anticipate, both from this project's own
measurements: the **6.43% same-model variance bound**, and the **1080 Ti resnet
integral against counter disagreement of up to 7.72%**. If the conclusion is
robust to the embodied range but not to those, that is itself the finding.

### 5. Phase 10: writeup and deliverable

Owner doc: `docs/phases.md` (weeks 8 to 10). Practical guidance for NRP:
which donated hardware is worth accepting, which nodes are worth retiring,
parameterised by utilisation and grid location.

`paper/methods-notes.md` already holds real measured content. Put measured facts
there as they are established, not only in task docs.

---

## Decisions that need a human, not a computation

### Confirm the framing with the prof

**This gates what the paper can claim and costs no GPU time.** The project was
framed around replacing an ageing card with a modern datacenter GPU. Those are
unreachable on Nautilus: quota bans the A100 class outright, reservation taints
fence the L4, and contention accounts for the L40S and 4090. What is measurable
is a consumer and workstation line from 2017 to 2021.

That is still a real retirement question, plausibly the one an academic cluster
actually faces, but it is a different one. Reasoning and evidence are in
`docs/tasks/phase6-fleet-selection.md`.

### Decide the results PVC name

The canonical results claim is `matmul-results` and now holds three workloads.
Either keep it and explain the name in methods, or migrate. Migration gets more
expensive with every run that writes into it. See `k8s/STORAGE.md`.

---

## Open findings to resolve

These are unresolved measurements, not bugs. Each needs a decision about what to
report, and two of them affect numbers that reach the paper.

### The 55.03 W idle figure: CLOSED 2026-08-23, it was never an idle figure

Kept as a closed item because the number reached four documents and one of them
was paper-bound, so anyone who read an older copy needs to know it moved.

`min_power_w` from `measurement/preflight.py`, whose window is loaded on
purpose: the monitor starts and a sustained matmul runs throughout, under the
comment "Light sustained load so the reading is not idle." That window averaged
229.96 W. A load-trace minimum is an upper bound on idle draw, not a
measurement, so it never contradicted the fleet's 25 W. The same applies to the
16.52 W once quoted as the A4000's idle floor: `preflight.py` has no idle field,
so every power figure from it is of this kind.

**The correction inverts a claim that was heading for the paper.** The A4000 was
described in `paper/methods-notes.md` as having a 3.3x idle advantage over the
1080 Ti. Measured directly, the two are within 2 W (25.28 W against 27.06 W)
with the newer card slightly higher, and that difference is inside the 6.43%
variance bound. **Replacement buys no measurable idle saving**, so break-even
rests almost entirely on active hours. Corrected in `CLAUDE.md`,
`paper/methods-notes.md`, `docs/tasks/phase8-break-even-inputs.md` and
`data/raw/runs/README.md`.

Genuinely unmeasured, if anyone wants it: idle on the specific `k8s-gpu-2` card,
and idle on the L4 and 3090 at all. Neither blocks anything.

### The 1080 Ti resnet gap: DIAGNOSED 2026-08-23, sampling aliasing

`PowerMonitor` samples every 0.2 s; resnet on the 1080 Ti runs 1000 batches in
about 202 s, or 0.201 s per batch. Sampler and workload share a period, so every
sample lands at nearly the same phase of each batch and the bias never averages
out. Full evidence in `CLAUDE.md` under Measurement contract. It is the only one
of nine workload-and-card pairs with a period ratio near 1.0, and it is the only
one with the anomaly.

**Decided: report `energy_j`, the integral, throughout, and document the
deviations rather than correcting per cell.** Mixing instruments inside one
table is what a reviewer should object to, and switching to counters everywhere
would force a bet on the 2080 Ti, where neither figure is known to be right.

The choice is conservative, which is what settles it. The integral understates
the energy saving in **all six** replacement pairs, by 0.1% (matmul, A4000) to
16.8% (resnet, 2080 Ti), because aliasing makes the old card look cheaper while
the 2080 Ti's bias makes a new card look costlier. Every break-even threshold
from these rows is therefore too pessimistic, never too optimistic.

Phase 8 should read `energy_j`. Keep `energy_j_counter` alongside it: it is the
evidence for the error analysis, not a spare.

**Left deliberately undone:** changing the sampling interval. 0.2 s is a poor
default because it is close to a plausible per-batch time for image training on
mid-range cards. A less round or dithered value would fix it, but it changes
`measurement/power_monitor.py` and therefore the image digest, which separates
any re-run from the existing 50 rows. Worth doing before any future sweep, not
worth invalidating this one over.

**Worth a paragraph in methods.** This is a different instrument failure from
Yang et al. (2024), who document cached readings. Both were detectable only
because the full sample trace is retained.

### The 2080 Ti energy counter bias

Five independent confirmations across three workloads and a 60 s preflight
window, clustered at 5.6% to 6.9%. It is a systematic per-card bias, not a
method problem. **Which of the two figures to trust on Turing is still open**
and the paper has to address it. Figures per card and per workload are in
`CLAUDE.md` under Measurement contract.

---

## Work that is unblocked and unclaimed, no ordering

- **Same-model variance, properly.** Descoped by decision on 2026-08-23 in
  favour of a time-boxed scope, and currently answered by one accidental card
  pair. The design (matmul, 5 repetitions on each of at least 5 distinct
  `gpu_uuid` values) is in `docs/tasks/phase6-fleet-selection.md`. Roughly
  2.2 GPU-hours. Requires pinning `kubernetes.io/hostname` across nodes, since
  without pinning the scheduler puts every repetition on one card. Note you
  cannot request a specific GPU in Kubernetes, only a node; verify with
  `gpu_uuid` afterwards that you actually got distinct cards.
- **Warmup-length axis.** Warmup is `warmup_iters=10` against 2000 measured
  iterations for matmul, and 5 batches against 1000 for resnet, so about 0.5% of
  the compute budget rather than the half `CLAUDE.md` assumed before it was
  measured. The risk is that 10 iterations is too **short** to reach thermal and
  clock steady state, which is the opposite of the concern originally written
  down. Cheap to test, and energy is the dependent variable.
- **The 3090.** Never preflighted, shows the cached-reading signature, energy
  figures unusable. Out of the fleet by decision, so this only matters if it is
  added back.
- **`runner.py` accepts blank `image_ref` and `git_commit`.** Making it refuse
  would enforce what the Output contract already claims those columns do.
- **CI workflow still triggers on `schema-idle-sizing`**, which merged on
  2026-08-23. The workflow comment says to remove it once it lands.
- **`k8s/benchmark-pod.yaml` still pins `{{IMAGE}}` to a tag.** Reference images
  by digest for any measured run.
- **Dockerfile comment.** `Dockerfile` line 4 still says the cu121 wheels "carry
  sm_61". Measurement corrected that; the practical conclusion holds but the
  comment is wrong. See Library version traps in `CLAUDE.md`.
- **Phase numbering.** `docs/phases.md` uses 1 to 10; a separate team plan
  the prof uses has 7 phases, and `matmul.py` cites a Phase 8 that neither
  scheme places cleanly. If the team plan becomes canonical, add a mapping rather
  than renaming task docs, to keep commit and PR links alive.
- **ruff config.** No `pyproject.toml` or `ruff.toml`, so `ruff check` runs on
  defaults and enforces none of the type hint, docstring or naming rules in
  Coding conventions.
- **Registry authority.** GHCR against `gitlab-registry.nrp-nautilus.io`. Decide
  and reconcile the two notes in `CLAUDE.md`.

---

## If you need to run something on the cluster

Read the Hard rules in `CLAUDE.md` first, then
`docs/tasks/overnight-fleet-run.md`, which is a self-contained runbook written
for someone with none of the surrounding conversation. It carries the repair
authority boundary: configuration failures may be fixed in place, a Python
traceback out of `benchmarks/` or `measurement/` stops the run.

The short version, none of which replaces reading those:

- Context `nautilus`, namespace `cmpm118`, requires the UCSC VPN.
- **Prefix every resource with your own name.** The namespace is shared outside
  this project. Never delete anything that is not yours, including pods that
  have sat in `Error` for days.
- **Every GPU job carries `activeDeadlineSeconds` and `ttlSecondsAfterFinished`.**
  The deadline is what releases the card when the client dies or an agent is
  killed. It did exactly that on 2026-08-23 when an LLM job never scheduled.
- **Ask before probing.** `python3 k8s/nrp_availability.py` reports what is free
  at zero cost. Plan against its `gpu_free_open` column, and expect it to be
  wrong sometimes: on 2026-08-23 it reported a free GPU on a node the scheduler
  then refused.
- **Never add a toleration for `nautilus.io/reservation=*`.**
- Before you stop for any reason, run
  `kubectl --context nautilus -n cmpm118 get pods,jobs` and confirm nothing of
  yours remains.

The image that produced the 50 rows, pinned by digest so it stays reproducible:

```
ghcr.io/aravadhikari05/gpu-retirement-analysis@sha256:9e62c0de6a56b995a1de66269cb1d26666f099390fb90c83e7aaca9f360877e4
```

Reference images by digest, never by tag. `:latest` moves on every `main` build
that touches `Dockerfile`, `benchmarks/` or `measurement/`, and two k8s manifests
pull it.
