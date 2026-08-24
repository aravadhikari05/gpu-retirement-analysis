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

Phase 7 landed on 2026-08-23 (Veda, `b98dbe1`) and Phase 8 was built the same day
(Aidan, `aab7021`..`e415ac6`). **The pipeline is end to end**: measured energy,
sourced embodied carbon, and a break-even model with tests.

Grid intensity was sourced on 2026-08-23 from EPA eGRID2023, so **both halves of
the inequality are now cited** and `carbon_model.py` prints `[sourced]`. The
figures are quotable. **The critical path is now Phase 9**, sensitivity.

**Nothing below is blocked on more GPU time.** The critical path is desk work.

Data:

| What | Where |
|---|---|
| Raw records, append-only, all 57 | `data/raw/runs/runs.jsonl` |
| Power traces, one CSV per repetition | `data/raw/runs/{matmul,resnet_train,llm_inference}/` |
| The analysable slice, 50 rows | `data/processed/fleet_runs.{jsonl,csv}` |
| Per-repetition and per-group summaries | `data/processed/runs_flat.csv`, `energy_by_gpu.csv` |
| Embodied carbon per card, low-high | `data/embodied/embodied_carbon_cardlevel.csv` |
| Break-even model and grid presets | `analysis/carbon_model.py`, `analysis/grid_intensity.py` |

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
replaced a GPU inside an existing node, or a whole node?

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

**The embodied number comes from ACT bottom-up, and that is the whole method.**
ACT takes die area and process node and returns kg CO2e, which is `embodied_new`
directly. No whole-system figure is involved at any step.

**The instruction to work backwards from vendor PCF reports is withdrawn.** It
came from the original spec draft and the arithmetic does not survive: a node
total is roughly 1000 kg against 6 to 27 kg for a card, so a 5% error on the
system figure exceeds the answer threefold. `CLAUDE.md` carries the detail under
"Vendor PCF reports are not a source here". Whole-system numbers have one use
left, the Phase 9 node-scope arm, where a total is used as a total.

What accepting the GPU unit costs, and what therefore has to be stated in
methods:

- Host energy is excluded on the operational side, which understates the slower
  card. Known direction, so state it.
- **Published precedent exists.** EcoServe (Li et al., 2025) refreshes GPUs on a
  3-year cycle against 10-year hosts and reports approx 16% cumulative carbon
  saving over a decade. Cite it; the unit is no longer only our assumption.

**Checked against NRP's documentation 2026-08-23, and the decision stands with
its reasoning changed.** Contribution to NRP is **node-level**: contributors
supply a whole server, NRP supplies the OS and runs it. NRP never swaps a GPU or
retires a node; the contributing institution owns the hardware and does the
physical work. There is no published retirement policy at all.

So node-scope would be equally defensible on NRP's own terms, and the GPU unit
is a **modelling choice rather than an observed practice**. What supports it for
this fleet is the hardware: 1080 Ti, 2080 Ti and 3090 are consumer PCIe cards in
commodity servers, physically swappable in a way an SXM-socketed A100 is not.
Treat node-level replacement as a sensitivity arm in Phase 9 rather than a
closed question, and carry the caveat that a 2017-era node's CPU, RAM and PSU
are aged too, which biases the GPU-only scope optimistic. Detail and sources in
`CLAUDE.md` under "How NRP actually acquires hardware".

**Phase 10 needs re-aiming.** `docs/phases.md` promises "a practical
recommendation for NRP", but NRP does not make this decision. Address it to
contributing institutions, with NRP as the context that sets utilisation.

Two related boundaries in the same Gap, which need stating rather than solving:
a card retired here may be redeployed elsewhere rather than scrapped, in which
case its remaining operational carbon moves rather than disappears; and
end-of-life and disposal carbon is absent from the model entirely, though it is
usually small next to manufacturing.

### 2. Phase 7: embodied carbon: FIRST PASS DONE (Veda, 2026-08-23, `b98dbe1`)

`data/embodied/` holds it: `embodied_carbon.py`, `EMBODIED.md` with every input
cited, and card-level and die-only CSVs. Card-level kg CO2e, die plus GDDR:
1080 Ti 6.2 to 17.0, 2080 Ti 9.5 to 26.7, A4000 5.7 to 14.6. Method is the ACT
area model with CPA swept 1.0 to 3.0 kg per cm2, which is where the band comes
from. Figures and the full reasoning are in `CLAUDE.md` under Embodied carbon.

Verified 2026-08-23: running the script regenerates both committed CSVs
byte-identical. **The 50 to 400 kg placeholder is withdrawn**, and the new
figures are an order of magnitude lower by scope rather than by error.

**This is the embodied number. Do not look for a second method.** ACT is
bottom-up, so it yields `embodied_new` directly and no vendor whole-system
figure or subtraction is needed. That instruction is withdrawn everywhere; see
item 1 and `CLAUDE.md`.

**Every constant is sourced, checked 2026-08-23** against the ACT paper and
against ACT's reference implementation at
<https://github.com/facebookresearch/ACT>. Two of them are only in the code:
`DEFAULT_FAB_YIELD = 0.875` and `CARBON_PER_IC_PACKAGE = 150 * g`, both in
`act/core/common.py`. Cite the implementation for those, not the paper's
Table 1, which gives yield only as a 0 to 1 range. Full table in `CLAUDE.md`
under Embodied carbon.

**Two deviations from stock ACT, both conservative, both must reach methods.**
Evaluating ACT's own per-node parameters gives CPA of 0.84 to 1.76 kg/cm2 at
14nm and 0.90 to 2.06 at 8nm, against the 1.0 to 3.0 swept here; and ACT's CPA
already contains `1/Y`, while `embodied_carbon.py` divides by yield again, which
inflates by about 14% **if** Malmodin's figure is yield-inclusive. That last one
is unverified and needs the primary source. Both push embodied carbon up, so
payback is overstated rather than understated, and every conclusion holds under
either. Also worth knowing: ACT tabulates no 16nm or 12nm node, so the 1080 Ti
and 2080 Ti use 14nm as a proxy.

Items 3 and 4 are unblocked.

Three residuals, none blocking, all cheap:

- **Grid intensity: DONE 2026-08-23, from EPA eGRID2023 rev. 2.** Six eGRID
  subregions at their CO2e total output rates, CAMX 0.1950 through RFCM 0.4427,
  US average 0.3497. Three placeholders were high, PJM by 2.02x. The field was
  renamed `kg_co2e_per_kwh` because it was being compared against a CO2e
  embodied figure while named CO2. "PJM" is now refused rather than guessed,
  since it spans three subregions differing by 1.6x. Full reasoning, including
  average-against-marginal and the busbar-against-plug choice, in `CLAUDE.md`
  under Grid intensity.
- **The forward decline rate is the one number still uncited.**
  `EXAMPLE_ANNUAL_DECLINE = 0.03`. It is not used by default, so nothing
  currently printed depends on it, but Phase 9 does. **Use NREL Cambium**
  (<https://www.nrel.gov/analysis/cambium.html>): it publishes projected
  emission factors to 2050 in both average and long-run marginal forms, which
  covers this constant and the marginal sensitivity arm in one download.
- **The memory coefficient is contested, not unsourced.** Ours is 0.065 kg per
  GB (LLMCarbon); EcoServe Table I puts GDDR6 at 0.36 (TechInsights), with its
  DDR4, HBM2 and HBM3e all in the 0.24 to 0.36 band. Both sit inside ACT's
  `E_DRAM` range of 0 to 0.6, so ACT does not settle it. Decision 2026-08-23:
  **keep 0.065** and report the conflict as a sensitivity, because adopting 0.36
  changes no conclusion in any cell. It would stretch payback 35 to 80% and hit
  the A4000 hardest. One sentence in methods covers it.
- **The GPU model strings do not join.** `data/embodied/` uses hyphens
  (`NVIDIA-GeForce-RTX-2080-Ti`), `runs.jsonl` records `gpu_model_observed` with
  spaces. A join without normalisation drops rows silently rather than raising.
  Whoever writes `carbon_model.py` hits this first.
- **`embodied_carbon.py` does not meet Coding conventions.** CSVs go to the
  current working directory rather than a module-level constant, no
  `if __name__ == "__main__":`, not runnable as `python -m`. The numbers are
  sound. Phase 8 reads the committed CSVs rather than importing the module, so
  this is tidiness rather than a blocker.
- **Node-differentiated CPA: partly answered, and it does not do what the ask
  assumed.** ACT ships per-node parameters, giving CPA of 0.84 to 1.76 kg/cm2 at
  14nm and 0.90 to 2.06 at 8nm. But **ACT tabulates no 16nm and no 12nm**, so
  the 1080 Ti and 2080 Ti both fall back to 14nm and cannot be differentiated
  from each other at all. And substituting the per-node values leaves the A4000
  at 4.7 to 9.3 against the 1080 Ti's 4.8 to 9.2, so the replacement is *still*
  at or below the card it replaces. The artifact is not caused by the uniform
  sweep and will not be fixed by node CPA; it has to come from scope or from
  yield falling with die area. Full working in `CLAUDE.md` under Embodied
  carbon. Still worth adopting for accuracy, just not as the deciding input.
- **Yield may be applied twice.** ACT's CPA already contains `1/Y` by
  construction, while `embodied_carbon.py` divides by `YIELD` again. Inflates
  everything by about 14% **if** Malmodin's figure is yield-inclusive.
  Unverified: needs the Malmodin primary source rather than the citation
  through Weppe et al. Conservative direction either way.

One thing to tell Veda: `EMBODIED.md` asks the team to confirm Gap 5, the
GPU-against-node scope. **It was decided on 2026-08-23 and her recommendation
matches it** (card-level as the figure, die-only as the floor, whole-node out of
scope). See item 1 above. Nothing to redo.

Also unpropagated: `docs/tasks/phase8-break-even-inputs.md` line 240 still
carries the 50 to 400 kg placeholder in its inputs table.

### 3. Phase 8: the carbon model: BUILT (Aidan, 2026-08-23, `aab7021`..`e415ac6`)

`analysis/carbon_model.py` and `analysis/grid_intensity.py` exist, with
`tests/test_carbon_model.py` beside them. **Verified 2026-08-23**: 73 tests pass
under `python -m unittest discover -s tests`, and the CLI runs.

```bash
python3 -m analysis.carbon_model --allow-unsourced            # 6-year horizon
python3 -m analysis.carbon_model --allow-unsourced --snapshot # job-count form
```

`--allow-unsourced` is mandatory while grid intensity is a placeholder, and
every line prints `[PROVISIONAL]`. That flag is the gate on quoting any number
from this model, so **do not remove it until grid intensity is sourced**.

**Cross-checked against the independent hand calculation, and they agree
exactly.** For matmul 1080 Ti to A4000 at CAISO with the low embodied bound in
`--snapshot` mode, the module reports 5,816,743 jobs and 2,908 repetitions,
against 2,908 repetitions computed by hand.

**The job unit is now explicit, fixed 2026-08-23.** A job is one inner
iteration, not one repetition, and the two differ by `inner_iters`: 2000 on
matmul, 1000 on resnet, 8 on llm. That was a factor-of-2000 misreading waiting
to happen. `CardEnergy` now carries `inner_iters`, `BreakEven` carries
`jobs_per_repetition` with a `repetitions` property, and every printed line
names the unit and gives both counts. `tests/test_carbon_model.py::JobUnit`
pins the conversion on both the snapshot and horizon paths and through
`payback_curve`.

Two design decisions worth knowing before reading the output:

- **The idle term is suppressed to zero, not annotated.** Against a 100 kg
  placeholder it was a rounding error; against 9.5 kg a 5.60 W differential
  repaid a whole card by itself, so the model said "pays back before you run a
  single job" while the note underneath called that same figure noise. The sign
  flips by workload on the 1080 Ti to A4000 pair, +0.37 W on matmul, +0.36 W on
  resnet, -4.84 W on llm, which is the strongest evidence available that it is
  noise. Each result still reports the value that was removed.
- **The embodied figure is labelled a FLOOR on every line.** die+gddr excludes
  PCB, VRMs, cooler, fan, connectors, assembly and transport.

Remaining gaps, renumbered against `docs/tasks/phase8-break-even-inputs.md`:

| Gap | Status |
|---|---|
| 1. Idle power unmeasured | **Closed** by the fleet pass, then suppressed as noise by the model |
| 2. Boundary is the GPU board only | Open. A statement for methods, not work |
| 3. PUE absent | Open. Probably a swept parameter: the fleet spans many institutions |
| 4. Snapshot against integral | **Closed**: `--horizon-years` and `--annual-decline` implement the integral form, `--snapshot` keeps the original |
| 5. GPU against whole node | **Closed 2026-08-23**: the GPU. See item 1 |
| 6. Real NRP utilisation | Open, and **the dominant unknown**. The model already emits active hours per year, so the parameterised curve is the deliverable |


### 4. Phase 9: sensitivity

Owner doc: `docs/phases.md` (weeks 7 to 8). Sweep embodied carbon across its
plausible range and find where the break-even answer flips. Project grid
intensity forward to see whether replacements that fail today pay off by 2030.

**Phase 7's figures change what this phase is for, and Phase 8 confirmed it.**
The embodied band is small enough that payback lands in days-to-months of
continuous work. Aidan stacked the four named biases at once, node CPA at the
8nm ceiling, a full-card BOM multiplier of 3, ACT yield and a flat grid, and the
matmul 1080 Ti to A4000 break-even moved from 29 to 226 active hours per year,
still only 2.6% of a year. `tests/test_carbon_model.py` pins that, so a change
that flips it fails loudly.

So **replacement pays back under every assumption currently defensible**, and
sweeping embodied carbon will not flip anything. Lead on **utilisation** and
**grid intensity**, the two axes that can, and report the embodied sweep as the
robustness check it turned out to be. Note the low-high spread is a one-parameter
sweep on embodied carbon, not a confidence interval: grid intensity, PUE, the
6.43% variance bound and the integral-versus-counter disagreement are all absent
from it.

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

### Framing: CONFIRMED 2026-08-23

The consumer and workstation fleet is accepted as the paper's subject. It is
what was measurable, the project is a time-boxed test project, and the
limitation gets reported honestly rather than worked around. `CLAUDE.md` under
Cluster environment holds why the modern datacenter cards are unreachable:
namespace quota on the A100 class, reservation taints on the L4, contention on
the L40S and 4090.

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

### The 2080 Ti bias: CHARACTERISED 2026-08-23, still unexplained

**Model-level, not a defective card.** The 16 rows come from two different
physical 2080 Tis on two different nodes: +5.889% and +6.639%. Not a sampling
artifact either, its traces are the healthiest in the fleet. Full detail in
`CLAUDE.md` under Measurement contract.

**Still open:** whether the error is multiplicative (a 6.17% scale error) or
additive (a +10.1 W offset). Both fit the data almost equally well because all
16 rows sit in a narrow 161 to 170 W band.

**Cheap way to settle it, unclaimed.** Record an energy-counter delta across the
idle windows, which sit near 20 W and would separate the two models immediately.
`measure_idle()` already runs `PowerMonitor` for 60 s per window and already
reads the counter, so this is a few lines in `measurement/runner.py`. It changes
the image digest, so bundle it with the sampling-interval change rather than
doing it alone.

Which figure to trust on Turing remains an open question the paper has to
address. It does not block anything: the reported numbers use the integral
throughout and the direction of the error is conservative.

### The distinct-value heuristic is not a proxy for accuracy

**Found 2026-08-23, and it corrects how this project has been reading its own
diagnostics.** The A4000 has a worse distinct-value ratio than the RTX 3090 that
got the 3090's power condemned, 0.28 to 0.35 with runs of 5 identical readings,
and it agrees with its hardware counter to **0.02%**. The 2080 Ti has pristine
traces and is **6%** off. The signature and the accuracy run backwards.

**Use counter against integral as the test**, not distinct-value counting. A
stale reading repeated across a window makes the integral disagree with the
hardware accumulator; agreement to 0.02% means the samples were not stale
however few distinct values they took.

Consequences: the 3090 was set aside on weak evidence and deserves the proper
check when a card can next be obtained, and the methods section should present
counter-against-integral as the diagnostic rather than repeating the
distinct-value heuristic from Yang et al. (2024) as though it were one.

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
