# Task: what the break-even model needs, and what we are not measuring

Status: for team review. Nothing here is decided.
Raised by Arav, 2026-08-18, before the Phase 6 sweep runs.

## Why this exists now rather than at Phase 8

The Phase 6 sweep is roughly 12 to 15 GPU-hours, serialised one card at a time
as a shared-namespace courtesy, realistically spread over several days. If it
finishes and we then find the carbon model needs a quantity the benchmarks never
recorded, the fix is a full re-run.

The same gap found now is a few lines in `measurement/runner.py`.

So the question this doc asks is not "is the model right". It is **"what must
the sweep record so that the model is possible at all"**. Everything else can be
argued about later, once there is data.

## The equation as currently written

From CLAUDE.md, carried over from the original spec draft:

```
replacement_worthwhile =
    embodied_new < (energy_per_job_old - energy_per_job_new)
                   * expected_jobs
                   * grid_intensity
```

What follows walks each term and asks what it assumes and where the number
comes from.

---

## Gap 1: idle power. Not measured anywhere. Blocks the sweep.

**This is the one that matters.** The project premise, from `docs/phases.md`:

> Utilization. A GPU that sits idle most of the time never pays back a
> replacement.

Every benchmark measures energy **inside the timed region**, so every number we
will produce is energy while working. Nothing measures what a card draws while
doing nothing. `PowerResult.min_power_w` is the minimum under load, not idle.

Real annual energy is not one term but two:

```
annual_energy = energy_per_job * jobs_per_year
              + idle_watts * idle_hours_per_year
```

The equation above only contains the first. At the low utilisation that the
whole project says is decisive, the second plausibly dominates.

**Superseded 2026-08-23. The motivating number in this paragraph was wrong, and
measurement reversed its conclusion.**

This originally read: preflight on a GTX 1080 Ti at `k8s-gpu-2.ucsc.edu` caught
it drawing 55.03 W "while effectively idle", so an old card idling at 55 W
against a modern one nearer 15 W leaves a 40 W gap running 8760 hours a year.

That 55.03 W is `min_power_w` from a preflight window that is loaded on purpose
(`measurement/preflight.py` runs a sustained matmul for the whole window), so it
is an upper bound on idle draw rather than a measurement of it. The same applies
to the 16.52 W that was quoted for the A4000.

Idle is now measured directly, per pod, in two 60 s windows split by CUDA
context creation. With a live context the three cards sit at **25.28 W
(1080 Ti), 26.89 W (2080 Ti) and 27.06 W (A4000)**, which is essentially flat,
with the newest card highest. The gap this section was built on does not exist.

**The argument for recording idle still stands and was worth acting on**, but
its conclusion inverts: idle draw does not favour replacement, so break-even
depends almost entirely on active hours. A mostly-idle card is not merely slow
to repay a replacement, it has close to nothing to repay it with.

**Proposal:** add a short idle sampling window to `runner.py`, before warmup and
after the timed region, recorded as `idle_watts_pre` and `idle_watts_post`.
Roughly 30 seconds per run. Cheap now, impossible to add retroactively.

**Implemented 2026-08-23, and the shape changed from that proposal.**
`measurement/runner.py` now has `measure_idle()`, called once per pod before the
first repetition, reusing `measurement/power_monitor.py` rather than sampling
separately. Three departures from the proposal above, each for a reason:

- **Two windows split by CUDA context, not by run position.** Idle before any
  CUDA context exists and idle with a live context but no work are different
  quantities, and the second is the one the project premise is about: an NRP pod
  holding a GPU it is not using. Columns are `idle_pre_context_*` and
  `idle_post_context_*`. Neither includes the benchmark's model load, so neither
  covers the power cost of resident weights; that is a stated scope boundary.
- **Once per pod, not once per run.** Rows are written as each repetition
  completes, so a post-run figure could not appear on rows already flushed. A
  post-run window would also not be comparable across cards: the design fixes
  the work and lets the time vary, so a slow card is hotter for longer before
  its reading than a fast one. Cold idle is the same measurement everywhere.
- **60 s per window, not 30.** The window has to clear the Yang et al. (2024)
  floor with margin rather than sit on it. Idle traces are flat and low, which
  is exactly the regime where a cached reading is indistinguishable from a
  plausible real one.

Each window records `avg_w`, `min_w`, `peak_w`, `duration_s` and `n_samples`.
`peak_w` is there for the open question below: on a shared node a co-tenant's
job inflates `avg_w` with no other symptom, and a peak far above the average is
what exposes it. `analysis/summarize_runs.py` averages idle over distinct
observations rather than over rows, since one pod's figure is copied onto all of
its rows, and reports `n_idle_observations` beside the mean.

Configurable with `--idle-seconds`, skippable with `--no-idle`, and skipped
automatically under `--no-power` so a CPU smoke test still runs.

**Still not measured on a GPU.** The code path has never executed against NVML.

**Open question for the team:** is idle a property of the card, or of the card
plus whatever else the node is doing? Nautilus nodes are shared, so a "idle"
reading may include another tenant's work. That may mean idle has to be measured
per node and per time, not once per card. `peak_w` and `min_w` per window are
recorded so this can be answered from the sweep data rather than assumed.

---

## Gap 2: measurement boundary. Decide before the sweep.

pynvml reports the **GPU board**. It does not include the CPU feeding it, host
RAM, PSU conversion loss, or cooling.

This is not a constant offset across cards. `resnet_train.py` runs a dataloader
with `num_workers=2`, so the CPU is doing real work, and a slower GPU means the
CPU spends longer doing it. Two cards therefore differ in host energy as well as
board energy, and we capture only the latter.

**Options:** accept board-only and state it plainly as a scope boundary; or add
a PUE-style multiplier; or attempt host measurement, which we have no access to
on Nautilus (user-level only, no node access).

Realistically it is option one, but it should be a stated decision in the
methods section rather than an unmentioned omission. It biases in a known
direction: it understates the total cost of the slower card.

---

## Gap 3: PUE. A modeling decision, not a measurement.

We rejected CodeCarbon specifically because it makes its own PUE and grid
assumptions, which are among the things this project exists to examine. That
argument obliges us to state ours explicitly rather than implicitly assume 1.0.

PUE multiplies the operational side and not the embodied side, so it is not
neutral. It shifts where break-even falls.

We do not have a PUE figure for the actual NRP sites, and the fleet spans many
institutions (the census shows 12 distinct sites for the RTX 3090 alone), so one
number will not describe all of them. Most likely this becomes a swept parameter
rather than a constant.

---

## Gap 4: time. The equation is a snapshot; the question is an integral.

`expected_jobs` is a count with no period attached, and `grid_intensity` is a
constant. Both hide a time dimension.

- Over what window do we accumulate savings? The new card's expected life? The
  old card's **remaining** life? Those differ, and comparing a 6 year new card
  against a 2 year old one on equal terms overstates the replacement case.
- Phase 9 explicitly wants grid intensity projected forward as it declines.
  A declining intensity means every future year of savings is worth less carbon
  than the year before, which pushes break-even further out.

The planned signature `break_even_hours_per_year(..., gpu_lifetime_years)`
already implies a per-year framing, so this is mostly a matter of writing the
integral form down rather than a missing input.

---

## Gap 5: what is actually being replaced. Scope boundary.

`embodied_new` assumes a GPU is the unit. But:

- Replacing a card inside an existing node has GPU-only embodied cost.
- Replacing a node has whole-system embodied cost, including CPU, RAM, PSU and
  chassis.
- Vendor product carbon footprint reports are **whole-system** figures.

**Resolved 2026-08-23: the unit is the GPU, and the embodied figure comes from
ACT bottom-up.** `data/embodied/` holds it. ACT takes die area and process node
and returns kg CO2e per card, so no whole-system number enters the calculation
at any point.

**The "work backwards from vendor PCF reports" instruction is withdrawn.** A
node total is roughly 1000 kg and a card is 6 to 27 kg, so the GPU is 1 to 3% of
the total and a 5% error on the system figure is over 3x the answer. EcoServe
(Li et al., 2025) does not subtract either: it uses ACT for dies and takes
per-component coefficients from the Dell R740 LCA for what ACT does not model.
Whole-system figures keep one use, the Phase 9 node-scope arm, where a total is
used directly as a total.

Related: the model assumes a retired card stops existing. In practice Nautilus
runs donated hardware, and a card retired here may be redeployed elsewhere
rather than scrapped, in which case its remaining operational carbon moves
rather than disappears. End-of-life and disposal carbon is also absent, though
it is usually small next to manufacturing.

These are scope boundaries. They do not need solving, but they need stating, and
whichever we pick determines which embodied numbers we go and source in Phase 7.

---

## Gap 6: utilisation data for an NRP-specific recommendation.

Break-even curves are parameterised by utilisation, so as a *figure* this is an
axis rather than an input and nothing is blocked.

But Phase 10 promises a practical recommendation for NRP: which donated hardware
is worth accepting, which nodes are worth retiring. That claim needs real
Nautilus utilisation, and the census records node counts and labels, not usage.

**Open question:** can we get GPU utilisation history from Nautilus at
user-level access? If not, the deliverable is a parameterised curve and the
reader supplies their own utilisation, which is a weaker but honest claim.

---

## Units, so this is written down once

```
energy in joules
1 kWh = 3.6e6 J
grid_intensity in kg CO2 per kWh
embodied in kg CO2e

carbon_saved_kg = (delta_energy_j * jobs / 3.6e6) * grid_intensity
replacement pays back when carbon_saved_kg > embodied_new_kg
```

The `/ 3.6e6` is the step most likely to be silently dropped, and dropping it is
wrong by a factor of 3.6 million while still producing a plausible looking
number. Whatever implements this should have a test that pins one worked example
end to end.

---

## Status of every input number

**Updated 2026-08-23. The embodied side is now sourced; grid intensity is not.**

| Input | Current value | Status |
|---|---|---|
| Embodied per GPU, card level | 5.7 to 26.7 kg CO2e depending on card | **sourced**, ACT bottom-up, `data/embodied/`. Every constant checked against the ACT paper and reference implementation; two deviations recorded in `CLAUDE.md` |
| GDDR embodied intensity | 0.065 kg CO2e per GB | sourced (LLMCarbon), but **contested**: EcoServe Table I says 0.36 for GDDR6. Kept, reported as a sensitivity |
| Grid intensity, 6 eGRID subregions | 0.1950 to 0.4427 kg CO2e/kWh | **sourced** 2026-08-23, EPA eGRID2023 rev. 2 Table 1. Replaces the four placeholders; three were high, PJM by 2.02x |
| Forward decline rate | 0.03 per year | unsourced, unused by default. Phase 9, from NREL Cambium |
| GTX 1080 Ti bandwidth | 484 GB/s | derived, bus width also unverified |
| RTX 4090 bandwidth | 1008 GB/s | derived, NVIDIA publishes none |

Anything the model computes before these are sourced is arithmetic, not a
result. Whatever implements the model should mark placeholders loudly enough
that a number cannot quietly reach the paper.

---

## What we are asking the team to decide

**Before the sweep, because they change what gets recorded:**

1. ~~Add idle power sampling to `runner.py`?~~ Done 2026-08-23, see Gap 1.
   What is left for the team is whether the cold, pre-model-load idle it
   records is the quantity the paper should report.
2. Is board-only measurement an accepted scope boundary, stated in methods?

**Before Phase 7, because they determine which numbers we go and source:**

3. Is the replacement unit a GPU or a whole node?
4. Is PUE a constant, a swept parameter, or explicitly out of scope?

**Can wait:**

5. Snapshot or lifetime-integrated form of the inequality.
6. Whether we can obtain real Nautilus utilisation, and what Phase 10 claims if
   we cannot.

## Deliberately not proposed here

Writing `analysis/carbon_model.py` yet. It cannot produce a meaningful number
until there is measured energy data and sourced carbon figures, and the value of
writing it early was finding missing inputs, which this document does more
cheaply. The one thing code would add that this does not is a units test, which
is worth doing when the model is actually built.
