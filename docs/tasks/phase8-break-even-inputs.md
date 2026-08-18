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

**Observed, not assumed.** Preflight on a GTX 1080 Ti at
`k8s-gpu-2.ucsc.edu` incidentally caught it drawing **55.03 W** while
effectively idle, against a 300 W reported limit
(`data/raw/preflight/20260818T084227Z-gtx1080ti.json`). If an old card idles at
55 W and a modern one idles nearer 15 W, that 40 W gap runs 8760 hours a year no
matter how many jobs are submitted.

**Proposal:** add a short idle sampling window to `runner.py`, before warmup and
after the timed region, recorded as `idle_watts_pre` and `idle_watts_post`.
Roughly 30 seconds per run. Cheap now, impossible to add retroactively.

**Open question for the team:** is idle a property of the card, or of the card
plus whatever else the node is doing? Nautilus nodes are shared, so a "idle"
reading may include another tenant's work. That may mean idle has to be measured
per node and per time, not once per card.

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
- Vendor product carbon footprint reports, which CLAUDE.md names as a source,
  are **whole-system** figures. Working backwards from them to a GPU-only number
  is itself a modelling assumption with its own error bars.

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

**No carbon or grid figure in this project is sourced yet.** From CLAUDE.md, all
currently placeholders inherited from a spec draft with no citation:

| Input | Current value | Status |
|---|---|---|
| Embodied per GPU | 50 to 400 kg CO2e | unsourced |
| CAISO grid intensity | approx 0.200 kg CO2/kWh | unsourced |
| US national average | approx 0.390 | unsourced |
| ERCOT | approx 0.400 | unsourced |
| PJM | approx 0.550 | unsourced |
| GTX 1080 Ti bandwidth | 484 GB/s | derived, bus width also unverified |
| RTX 4090 bandwidth | 1008 GB/s | derived, NVIDIA publishes none |

Anything the model computes before these are sourced is arithmetic, not a
result. Whatever implements the model should mark placeholders loudly enough
that a number cannot quietly reach the paper.

---

## What we are asking the team to decide

**Before the sweep, because they change what gets recorded:**

1. Add idle power sampling to `runner.py`? (Arav proposes yes.)
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
