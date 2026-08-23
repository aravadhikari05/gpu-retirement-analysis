"""Phase 8: carbon break-even between keeping an old GPU and replacing it.

Consumes `data/processed/energy_by_gpu.csv`, which `analysis/summarize_runs.py`
produces and labels as the Phase 8 input. It does not re-derive anything from
`runs.jsonl`: `analysis/fleet_subset.py` is the one definition of the analysable
slice, and duplicating that filter is the specific mistake its docstring warns
about.

## What this module will and will not claim

Every embodied and grid figure in the project is currently an unsourced
placeholder. `EmbodiedEstimate` and `GridIntensity` therefore carry a `sourced`
flag, any False taints `BreakEven.provisional`, and the CLI refuses to print
without `--allow-unsourced`. A number out of here is arithmetic until Phase 7
lands, and the type system says so rather than a comment saying so.

Two measured results from the fleet pass are enforced rather than described:

- **Same-model variance is 6.43%** (two A4000s, 2026-08-23), against 0.30%
  within one card. A replacement pair whose per-job energy difference is inside
  that bound is not interpretable, and `BreakEven.interpretable` says so. This
  is what stops the 7.0% resnet gap between the 2080 Ti and A4000 being read as
  a real saving.
- **The reported energy is the trapezoidal integral throughout**, per the
  decision taken after the 1080 Ti sampling-aliasing diagnosis. The integral
  understates the saving in all six measured replacement pairs, so every
  threshold this module produces is conservative: too pessimistic about
  replacement, never too optimistic. That direction is recorded in the notes on
  every result rather than left in prose.

## Why the snapshot form cannot carry the idle term

The inequality as originally written has no time in it:

    embodied_new < (energy_per_job_old - energy_per_job_new) * jobs * grid

Jobs are a count. Idle draw is a rate, watts against hours, so it has nowhere to
go in that expression. Passing `horizon_years=None` reproduces the original form
exactly and **drops the idle term**, with a note saying so. Including idle needs
a horizon.

That is not a limitation of this code, it is Gap 4 of
`docs/tasks/phase8-break-even-inputs.md` showing up in the type signature: the
project premise is about cards that sit idle, and the snapshot form cannot
express the project premise.

## Fixed work, so the faster card idles longer

Every workload here does a fixed amount of work, which is the whole measurement
design. So for a given number of jobs per year the new card finishes sooner and
spends *more* of the year idle, at roughly the same idle watts as the old card
(25.3 W to 27.1 W across the fleet, newest not lowest).

The idle term therefore works against replacement rather than for it. Writing
the annual difference out and collecting terms in jobs:

    delta_annual_j = a * jobs + b
    a = dE_per_job - (idle_w_old * runtime_old - idle_w_new * runtime_new)
    b = SECONDS_PER_YEAR * (idle_w_old - idle_w_new)

`a` is the effective per-job saving after paying for the extra idle hours the
faster card accrues. `b` is the pure idle differential, which the fleet measured
as approximately zero. Both fall out of the algebra rather than being modelled
separately.

Units, written once, since dropping the conversion is wrong by 3.6 million while
still producing a plausible looking number:

    carbon_saved_kg = (delta_energy_j * jobs / 3.6e6) * grid_intensity
"""

import argparse
import csv
import logging
import os
from dataclasses import dataclass, field

from analysis.grid_intensity import GridIntensity, preset

logger = logging.getLogger(__name__)

# Joules in a kilowatt hour. Named once so no call site writes it out.
J_PER_KWH = 3.6e6

HOURS_PER_YEAR = 8760.0
SECONDS_PER_YEAR = HOURS_PER_YEAR * 3600.0

# Same-model variance measured on two RTX A4000s in one node on 2026-08-23,
# against a within-card spread of 0.30%. A cross-model energy difference smaller
# than this is not distinguishable from two samples of the same model, so the
# model refuses to call it a saving. See CLAUDE.md, Established results.
SAME_MODEL_VARIANCE = 0.0643

# Largest idle spread observed on one physical card across the three workloads
# in the 2026-08-23 fleet pass: the RTX 2080 Ti read 19.83 W under matmul and
# 33.40 W under llm_inference. The between-card spread of per-card means over
# the same table is 1.71 W, so the measurement's own scatter is roughly eight
# times the quantity the model would like to attribute to card identity.
#
# A card's idle draw does not depend on which workload it later ran, so this is
# state that idle_post_context does not hold constant. The likeliest candidate
# is resident VRAM: the LLM pods leave gpt2-xl's 6.43 GB in the allocator, and
# the two highest readings in the table are both llm_inference. That is a
# hypothesis from three cards, not a measured cause.
#
# Until it is explained, an idle differential smaller than this is reported as
# noise rather than as a saving.
IDLE_WITHIN_CARD_SPREAD_W = 13.57

DEFAULT_ENERGY_BY_GPU = "data/processed/energy_by_gpu.csv"

# Recorded on every result. The integral understates the saving in all six
# measured replacement pairs, by 0.1% to 16.8%, because sampling aliasing makes
# the old card look cheaper while the 2080 Ti's bias makes the new card look
# costlier. Both push break-even later, so a threshold from here is an upper
# bound on how long payback takes.
CONSERVATISM_NOTE = (
    "energy_j is the trapezoidal integral, which understates the saving in every "
    "measured pair, so this threshold is pessimistic about replacement"
)


@dataclass(frozen=True)
class EmbodiedEstimate:
    """Embodied carbon for one GPU model, as a range.

    Phase 7 fills these in from the ACT model and vendor PCF reports. Ranges
    rather than point values, per the repo rule on preserving uncertainty.

    Attributes:
      gpu_model: Card this describes, matching gpu_model in the energy table.
      low_kg: Low end of the plausible range, kg CO2e.
      high_kg: High end of the plausible range, kg CO2e.
      sourced: False until a citation is attached. Taints every figure computed
        from it.
      citation: Where the figures came from. Required when sourced is True.
    """

    gpu_model: str
    low_kg: float
    high_kg: float
    sourced: bool
    citation: str = ""

    def __post_init__(self) -> None:
        """Validates at construction, in the style of benchmarks/_result.py.

        Raises:
          ValueError: On an empty model name, a non-positive or inverted range,
            or a sourced estimate with no citation.
        """
        if not self.gpu_model:
            raise ValueError("gpu_model must be a non-empty string")
        for name in ("low_kg", "high_kg"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be above 0, got {value}")
        if self.low_kg > self.high_kg:
            raise ValueError(f"low_kg {self.low_kg} exceeds high_kg {self.high_kg}")
        if self.sourced and not self.citation:
            raise ValueError(
                f"{self.gpu_model} embodied estimate is marked sourced but "
                "carries no citation"
            )


@dataclass(frozen=True)
class CardEnergy:
    """One measured (config_id, gpu_model) group from the energy table.

    Attributes:
      gpu_model: Card as observed at runtime by NVML.
      benchmark: Workload name.
      config_id: Pins the workload sizing. Two cards are comparable only when
        this and work_hash match, which is enforced upstream.
      energy_j_per_job: Energy for one inner_iter, the job unit.
      runtime_s_per_job: Wall clock for one inner_iter.
      idle_w: Idle draw with a live CUDA context, the NRP case of a pod holding
        a GPU it is not using.
      n_runs: Repetitions behind the mean.
      n_physical_gpus: Distinct cards behind the mean. One means the spread is
        run-to-run noise, not fleet variation.
      work_hash: Proof the runs did the same work.
    """

    gpu_model: str
    benchmark: str
    config_id: str
    energy_j_per_job: float
    runtime_s_per_job: float
    idle_w: float
    n_runs: int
    n_physical_gpus: int
    work_hash: str


@dataclass(frozen=True)
class BreakEven:
    """The answer, with everything needed to know whether to believe it.

    Attributes:
      jobs: Total jobs before replacement pays back its embodied carbon, or None
        when it never does.
      years_at_utilisation: Years to reach that job count at the utilisation
        asked for, or None.
      active_hours_per_year: Hours of active work per year this assumes.
      provisional: True when any input was unsourced. A provisional figure is
        arithmetic, not a result.
      interpretable: False when the energy difference is inside the same-model
        variance bound, so the two cards are not distinguishable.
      notes: Why, in words. Always carries the conservatism direction.
    """

    jobs: float | None
    years_at_utilisation: float | None
    active_hours_per_year: float | None
    provisional: bool
    interpretable: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pays_back(self) -> bool:
        """True when replacement pays back at all within the terms given."""
        return self.jobs is not None


def load_card_energy(path: str = DEFAULT_ENERGY_BY_GPU) -> list[CardEnergy]:
    """Reads the Phase 8 input table.

    Rows without a usable per-job energy figure are skipped rather than
    defaulted: `summarize_runs.py` leaves the field blank when a group has mixed
    inner_iters, and inventing a number there would be exactly the fabrication
    the repo rules forbid.

    Args:
      path: Path to energy_by_gpu.csv.

    Returns:
      One CardEnergy per usable row.

    Raises:
      FileNotFoundError: If the table has not been generated. Regenerate with
        `python3 -m analysis.summarize_runs`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Regenerate it with "
            "`python3 -m analysis.summarize_runs`."
        )

    cards = []
    skipped = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            energy = _as_float(row.get("energy_j_per_inner_iter"))
            runtime = _as_float(row.get("runtime_s_mean"))
            inner = _as_float(row.get("inner_iters"))
            if energy is None or runtime is None or not inner:
                skipped += 1
                continue
            cards.append(
                CardEnergy(
                    gpu_model=row.get("gpu_model", ""),
                    benchmark=row.get("benchmark", ""),
                    config_id=row.get("config_id", ""),
                    energy_j_per_job=energy,
                    runtime_s_per_job=runtime / inner,
                    idle_w=_as_float(row.get("idle_post_context_avg_w_mean")) or 0.0,
                    n_runs=int(_as_float(row.get("n_runs")) or 0),
                    n_physical_gpus=int(_as_float(row.get("n_physical_gpus")) or 0),
                    work_hash=row.get("work_hash", ""),
                )
            )

    if skipped:
        logger.info(
            "%d row(s) skipped: no per-job energy, usually mixed inner_iters", skipped
        )
    return cards


def _as_float(value) -> float | None:
    """Parses a CSV cell to float, returning None for blanks and junk."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def annual_energy_j(
    card: CardEnergy, jobs_per_year: float, hours_per_year: float = HOURS_PER_YEAR
) -> float:
    """Energy a card uses in a year, active work plus idle.

    The card is busy for however long its jobs take and idle for the rest of the
    year. Because the work is fixed, a faster card is busy for less of the year
    and idle for more of it, which is why the idle term is not a constant offset
    between two cards.

    Args:
      card: Measured energy and idle for one card and workload.
      jobs_per_year: Jobs completed in the year.
      hours_per_year: Length of the accounting year. Overridable so a card that
        is only powered part of the year can be modelled.

    Returns:
      Total energy in joules.

    Raises:
      ValueError: If jobs_per_year is negative, or if the jobs cannot fit in the
        year. An over-subscribed card is a caller error, not something to clamp
        silently.
    """
    if jobs_per_year < 0:
        raise ValueError(f"jobs_per_year must be non-negative, got {jobs_per_year}")

    busy_s = jobs_per_year * card.runtime_s_per_job
    year_s = hours_per_year * 3600.0
    if busy_s > year_s:
        raise ValueError(
            f"{jobs_per_year:.0f} jobs need {busy_s / 3600:.0f} h on "
            f"{card.gpu_model}, more than the {hours_per_year:.0f} h year"
        )

    idle_s = year_s - busy_s
    return jobs_per_year * card.energy_j_per_job + card.idle_w * idle_s


def cumulative_intensity(grid: GridIntensity, horizon_years: int) -> float:
    """Sums grid intensity over a horizon, one term per year.

    Returns the sum rather than the mean because the caller multiplies it by a
    per-year saving, so the sum is the quantity that has the right units.

    Args:
      grid: Region, carrying its own decline rate.
      horizon_years: Number of whole years, at least 1.

    Returns:
      Sum of kg CO2 per kWh over the horizon.

    Raises:
      ValueError: If horizon_years is below 1.
    """
    if horizon_years < 1:
        raise ValueError(f"horizon_years must be at least 1, got {horizon_years}")
    return sum(grid.at_year(y) for y in range(horizon_years))


def carbon_saved_kg(
    delta_energy_j: float,
    jobs: float,
    grid: GridIntensity,
    horizon_years: int | None = None,
) -> float:
    """Carbon avoided by doing `jobs` jobs on the new card instead of the old.

    This is the units line from CLAUDE.md, and the one place the conversion is
    applied. Dropping the division is wrong by a factor of 3.6 million while
    still producing a plausible looking number, which is why it lives here alone
    and is pinned by a test.

    Args:
      delta_energy_j: Energy saved per job, old minus new.
      jobs: Number of jobs, spread evenly across the horizon.
      grid: Region and decline rate.
      horizon_years: Years to spread the jobs over. None applies the year-zero
        intensity to all of them, which is the snapshot form.

    Returns:
      Carbon avoided, kg CO2e. Negative when the new card uses more.
    """
    if horizon_years is None:
        return (delta_energy_j * jobs / J_PER_KWH) * grid.kg_co2_per_kwh
    # Jobs spread evenly, so each year contributes its own intensity to an equal
    # share of the work. Summing intensity and dividing the jobs by the horizon
    # is the same thing and avoids a loop over jobs.
    total_intensity = cumulative_intensity(grid, horizon_years)
    per_year_jobs = jobs / horizon_years
    return (delta_energy_j * per_year_jobs / J_PER_KWH) * total_intensity


def _linear_terms(old: CardEnergy, new: CardEnergy) -> tuple[float, float]:
    """Collects the annual energy difference into `a * jobs + b`.

    Derivation is in the module docstring. `a` is the per-job saving after
    paying for the extra idle time the faster card accrues; `b` is the pure idle
    differential over a full year.

    Args:
      old: The card being replaced.
      new: The replacement.

    Returns:
      Tuple of (a, b), both in joules.
    """
    a = (
        old.energy_j_per_job
        - new.energy_j_per_job
        - (old.idle_w * old.runtime_s_per_job - new.idle_w * new.runtime_s_per_job)
    )
    b = SECONDS_PER_YEAR * (old.idle_w - new.idle_w)
    return a, b


def _comparable(old: CardEnergy, new: CardEnergy) -> tuple[bool, list[str]]:
    """Checks two cards may be compared at all, and whether the gap is real."""
    notes: list[str] = []
    if old.config_id != new.config_id:
        raise ValueError(
            f"refusing to compare different configurations: {old.config_id!r} "
            f"against {new.config_id!r}. Only runs that did the same work are "
            "comparable."
        )
    if old.work_hash and new.work_hash and old.work_hash != new.work_hash:
        raise ValueError(
            f"refusing to compare {old.gpu_model} and {new.gpu_model}: "
            "work_hash differs, so these runs did not do the same work"
        )

    delta = old.energy_j_per_job - new.energy_j_per_job
    interpretable = True
    if old.energy_j_per_job > 0:
        relative = abs(delta) / old.energy_j_per_job
        if relative < SAME_MODEL_VARIANCE:
            interpretable = False
            notes.append(
                f"energy difference is {relative:.2%}, inside the "
                f"{SAME_MODEL_VARIANCE:.2%} same-model variance bound, so these "
                "two cards are not distinguishable"
            )
    for card in (old, new):
        if card.n_physical_gpus == 1:
            notes.append(
                f"{card.gpu_model} figures come from one physical card, so the "
                "spread is run-to-run noise rather than fleet variation"
            )
    return interpretable, notes


def break_even_jobs(
    embodied: EmbodiedEstimate,
    old: CardEnergy,
    new: CardEnergy,
    grid: GridIntensity,
    horizon_years: int | None = None,
    use_high_estimate: bool = False,
) -> BreakEven:
    """Jobs the new card must run before it repays its embodied carbon.

    With `horizon_years=None` this is the original inequality from CLAUDE.md,
    and the idle term is dropped because idle is a rate with nowhere to go in a
    job count. With a horizon the idle term is included and the faster card is
    charged for the extra hours it spends idle.

    Args:
      embodied: Embodied carbon of the replacement card.
      old: Measured energy for the card being replaced.
      new: Measured energy for the replacement, same config_id.
      grid: Region and decline rate.
      horizon_years: Years to spread the work over, or None for the snapshot.
      use_high_estimate: Take the high end of the embodied range instead of the
        low end. The low end is the default because it is the most favourable
        case for replacement, so a result of "does not pay back" at the low end
        is robust.

    Returns:
      A BreakEven. `jobs` is None when replacement never pays back.

    Raises:
      ValueError: If the two cards did different work, or horizon_years is below 1.
    """
    interpretable, notes = _comparable(old, new)
    notes.append(CONSERVATISM_NOTE)
    provisional = not (embodied.sourced and grid.sourced)
    if provisional:
        unsourced = [
            label
            for label, ok in (
                (f"embodied for {embodied.gpu_model}", embodied.sourced),
                (f"grid intensity for {grid.name}", grid.sourced),
            )
            if not ok
        ]
        notes.append("PROVISIONAL: unsourced " + " and ".join(unsourced))

    embodied_kg = embodied.high_kg if use_high_estimate else embodied.low_kg

    if horizon_years is None:
        notes.append(
            "snapshot form: idle power excluded, because idle is a rate and the "
            "job-count inequality has no time dimension. Pass horizon_years to "
            "include it."
        )
        delta = old.energy_j_per_job - new.energy_j_per_job
        if delta <= 0:
            return _never(provisional, interpretable, notes, old, new, None)
        jobs = embodied_kg * J_PER_KWH / (delta * grid.kg_co2_per_kwh)
        return BreakEven(
            jobs=jobs,
            years_at_utilisation=None,
            active_hours_per_year=None,
            provisional=provisional,
            interpretable=interpretable,
            notes=tuple(notes),
        )

    total_intensity = cumulative_intensity(grid, horizon_years)
    a, b = _linear_terms(old, new)
    if b:
        idle_delta_w = old.idle_w - new.idle_w
        notes.append(
            f"idle term contributes {b * total_intensity / J_PER_KWH:+.2f} kg "
            f"over {horizon_years} y, before any jobs are run"
        )
        if abs(idle_delta_w) < IDLE_WITHIN_CARD_SPREAD_W:
            # The differential is inside the scatter one physical card shows
            # across workloads, so its sign is not established. Say so rather
            # than letting a signed kg figure look like a finding.
            notes.append(
                f"that idle figure is NOISE: the {abs(idle_delta_w):.2f} W "
                f"differential is inside the {IDLE_WITHIN_CARD_SPREAD_W:.2f} W "
                "spread one card shows across workloads, so even its sign is "
                "not established"
            )

    # Solve (a * jobs_per_year + b) * total_intensity / J_PER_KWH >= embodied.
    required_annual_j = embodied_kg * J_PER_KWH / total_intensity
    if a <= 0:
        if b > 0 and b >= required_annual_j:
            # Idle alone repays it, so no jobs are needed at all.
            notes.append("repaid by the idle differential alone, before any jobs")
            return BreakEven(
                jobs=0.0,
                years_at_utilisation=float(horizon_years),
                active_hours_per_year=0.0,
                provisional=provisional,
                interpretable=interpretable,
                notes=tuple(notes),
            )
        notes.append(
            "the replacement saves nothing per job once the extra idle hours "
            "the faster card accrues are charged against it"
        )
        return _never(provisional, interpretable, notes, old, new, horizon_years)

    jobs_per_year = (required_annual_j - b) / a
    if jobs_per_year < 0:
        jobs_per_year = 0.0
    jobs = jobs_per_year * horizon_years
    active_hours = jobs_per_year * new.runtime_s_per_job / 3600.0

    if active_hours > HOURS_PER_YEAR:
        notes.append(
            f"needs {active_hours:.0f} active hours per year, more than the "
            f"{HOURS_PER_YEAR:.0f} h year, so it cannot pay back within "
            f"{horizon_years} y at any utilisation"
        )
        return _never(provisional, interpretable, notes, old, new, horizon_years)

    return BreakEven(
        jobs=jobs,
        years_at_utilisation=float(horizon_years),
        active_hours_per_year=active_hours,
        provisional=provisional,
        interpretable=interpretable,
        notes=tuple(notes),
    )


def _never(
    provisional: bool,
    interpretable: bool,
    notes: list[str],
    old: CardEnergy,
    new: CardEnergy,
    horizon_years: int | None,
) -> BreakEven:
    """Builds the no-payback answer, with the reason attached."""
    delta = old.energy_j_per_job - new.energy_j_per_job
    if delta <= 0:
        notes.append(
            f"{new.gpu_model} uses {-delta:.3g} J more per job than "
            f"{old.gpu_model}, so replacement never pays back"
        )
    return BreakEven(
        jobs=None,
        years_at_utilisation=float(horizon_years) if horizon_years else None,
        active_hours_per_year=None,
        provisional=provisional,
        interpretable=interpretable,
        notes=tuple(notes),
    )


def break_even_hours_per_year(
    embodied: EmbodiedEstimate,
    old: CardEnergy,
    new: CardEnergy,
    grid: GridIntensity,
    gpu_lifetime_years: int,
    use_high_estimate: bool = False,
) -> BreakEven:
    """Active hours per year the new card needs to repay itself in its lifetime.

    The utilisation framing of `break_even_jobs`, and the one the project's own
    premise is about: a card that sits idle most of the time never pays back a
    replacement.

    Args:
      embodied: Embodied carbon of the replacement.
      old: Card being replaced.
      new: Replacement.
      grid: Region and decline rate.
      gpu_lifetime_years: Years over which the replacement must repay itself.
      use_high_estimate: Use the high end of the embodied range.

    Returns:
      A BreakEven whose `active_hours_per_year` is the answer.

    Raises:
      ValueError: If gpu_lifetime_years is below 1, or the cards did different work.
    """
    if gpu_lifetime_years < 1:
        raise ValueError(
            f"gpu_lifetime_years must be at least 1, got {gpu_lifetime_years}"
        )
    return break_even_jobs(
        embodied,
        old,
        new,
        grid,
        horizon_years=gpu_lifetime_years,
        use_high_estimate=use_high_estimate,
    )


def payback_curve(
    embodied: EmbodiedEstimate,
    old: CardEnergy,
    new: CardEnergy,
    grid: GridIntensity,
    utilisations: tuple[float, ...],
    max_years: int = 10,
    use_high_estimate: bool = False,
) -> list[tuple[float, BreakEven]]:
    """Years to payback across a range of utilisations.

    The primitive Phase 9 sweeps and Phase 10 plots. This module does neither:
    it answers one question per utilisation and returns them in order.

    Args:
      embodied: Embodied carbon of the replacement.
      old: Card being replaced.
      new: Replacement.
      grid: Region and decline rate.
      utilisations: Fractions of the year the card is active, each in (0, 1].
      max_years: Longest horizon to search before declaring no payback.
      use_high_estimate: Take the high end of the embodied range. Phase 9 runs
        the curve at both ends, so it has to be reachable here rather than only
        on break_even_jobs.

    Returns:
      List of (utilisation, BreakEven), one per input utilisation. Every entry
      carries the same provenance and interpretability flags break_even_jobs
      would return for the same inputs, whether or not it pays back.

    Raises:
      ValueError: If a utilisation is outside (0, 1].
    """
    for u in utilisations:
        if not 0.0 < u <= 1.0:
            raise ValueError(f"utilisation must be in (0, 1], got {u}")

    curve = []
    for u in utilisations:
        active_hours = u * HOURS_PER_YEAR
        jobs_per_year = active_hours * 3600.0 / new.runtime_s_per_job

        # The no-payback answer is derived from a real evaluation at the longest
        # horizon rather than assembled by hand. An earlier version built it with
        # interpretable=True and an empty note list, so a pair that
        # break_even_jobs correctly refused came back from here claiming to be
        # interpretable, with the PROVISIONAL marker dropped. Every field a
        # caller reads has to come from the same place the answer does.
        longest = break_even_jobs(
            embodied,
            old,
            new,
            grid,
            horizon_years=max_years,
            use_high_estimate=use_high_estimate,
        )
        answer = BreakEven(
            jobs=None,
            years_at_utilisation=None,
            active_hours_per_year=active_hours,
            provisional=longest.provisional,
            interpretable=longest.interpretable,
            notes=longest.notes
            + (f"does not pay back within {max_years} y at {u:.0%} utilisation",),
        )

        # Smallest whole-year horizon at which this utilisation repays. Searched
        # rather than solved because the declining grid makes each extra year
        # worth less than the last, so there is no closed form.
        for years in range(1, max_years + 1):
            candidate = break_even_jobs(
                embodied,
                old,
                new,
                grid,
                horizon_years=years,
                use_high_estimate=use_high_estimate,
            )
            if candidate.jobs is not None and candidate.jobs <= jobs_per_year * years:
                answer = BreakEven(
                    jobs=candidate.jobs,
                    years_at_utilisation=float(years),
                    active_hours_per_year=active_hours,
                    provisional=candidate.provisional,
                    interpretable=candidate.interpretable,
                    notes=candidate.notes,
                )
                break
        curve.append((u, answer))
    return curve


def _pairs(cards: list[CardEnergy]) -> list[tuple[CardEnergy, CardEnergy]]:
    """Every same-config pair of distinct cards, oldest first by energy use."""
    by_config: dict[str, list[CardEnergy]] = {}
    for card in cards:
        by_config.setdefault(card.config_id, []).append(card)

    pairs = []
    for members in by_config.values():
        ordered = sorted(members, key=lambda c: -c.energy_j_per_job)
        for i, old in enumerate(ordered):
            for new in ordered[i + 1 :]:
                pairs.append((old, new))
    return pairs


def main() -> None:
    """CLI: prints break-even for every measured replacement pair."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Carbon break-even between keeping an old GPU and replacing it."
    )
    parser.add_argument("--energy-table", default=DEFAULT_ENERGY_BY_GPU)
    parser.add_argument("--grid", default="CAISO", help="grid region preset")
    parser.add_argument(
        "--embodied-kg",
        type=float,
        default=50.0,
        help="low end of the embodied carbon range for the replacement card. "
        "UNSOURCED placeholder; Phase 7 supplies the real range.",
    )
    parser.add_argument(
        "--embodied-high-kg",
        type=float,
        default=None,
        help="high end of the embodied range. Defaults to the low end, which "
        "makes it a point value. Ranges are the repo convention: the placeholder "
        "50 to 400 kg spans 8x in the answer, so a single number hides most of "
        "the uncertainty.",
    )
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=6,
        help="years over which the replacement must repay itself. Omit the idle "
        "term with --snapshot.",
    )
    parser.add_argument(
        "--annual-decline",
        type=float,
        default=0.0,
        help="fractional grid decarbonisation per year, 0.0 for a constant grid",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="use the original job-count inequality, which excludes idle power",
    )
    parser.add_argument(
        "--allow-unsourced",
        action="store_true",
        help="required while embodied and grid figures are placeholders. Every "
        "figure printed is arithmetic, not a result.",
    )
    args = parser.parse_args()

    grid = preset(args.grid).with_decline(args.annual_decline)
    embodied_sourced = False

    if not args.allow_unsourced and not (embodied_sourced and grid.sourced):
        parser.error(
            "embodied and grid figures are unsourced placeholders, so any number "
            "printed here would be arithmetic rather than a result. Pass "
            "--allow-unsourced to see it anyway, and do not quote it."
        )

    cards = load_card_energy(args.energy_table)
    pairs = _pairs(cards)
    if not pairs:
        logger.warning("no comparable pairs in %s", args.energy_table)
        return

    high_kg = args.embodied_high_kg
    if high_kg is None:
        high_kg = args.embodied_kg
    if high_kg < args.embodied_kg:
        parser.error(
            f"--embodied-high-kg {high_kg} is below --embodied-kg {args.embodied_kg}"
        )
    is_range = high_kg > args.embodied_kg

    horizon = None if args.snapshot else args.horizon_years
    embodied_label = (
        f"{args.embodied_kg} to {high_kg} kg" if is_range else f"{args.embodied_kg} kg"
    )
    print(
        f"grid={grid.name} {grid.kg_co2_per_kwh} kg/kWh "
        f"decline={grid.annual_decline:.1%}/y  "
        f"embodied={embodied_label}  "
        f"horizon={'snapshot' if horizon is None else str(horizon) + ' y'}"
    )
    print()

    for old, new in pairs:
        embodied = EmbodiedEstimate(
            gpu_model=new.gpu_model,
            low_kg=args.embodied_kg,
            high_kg=high_kg,
            sourced=False,
        )
        # Both ends when a range was given. The low end is the most favourable
        # case for replacement, so "does not pay back" at the low end is the
        # robust result, and the spread between the two is what Phase 7's
        # sourcing has to narrow.
        low_answer = break_even_jobs(embodied, old, new, grid, horizon_years=horizon)
        answers = [("low", low_answer)]
        if is_range:
            answers.append(
                (
                    "high",
                    break_even_jobs(
                        embodied,
                        old,
                        new,
                        grid,
                        horizon_years=horizon,
                        use_high_estimate=True,
                    ),
                )
            )

        tag = "PROVISIONAL" if low_answer.provisional else "sourced"
        label = f"{old.benchmark}: {old.gpu_model} -> {new.gpu_model}"
        print(f"[{tag}] {label}")
        for end, answer in answers:
            prefix = f"    {end + ' embodied:':<16}" if is_range else "    "
            if answer.jobs is None:
                print(f"{prefix}never pays back")
            else:
                hours = (
                    f", {answer.active_hours_per_year:,.0f} active h/y"
                    if answer.active_hours_per_year is not None
                    else ""
                )
                print(f"{prefix}{answer.jobs:,.0f} jobs{hours}")
        if not low_answer.interpretable:
            print("    NOT INTERPRETABLE")
        for note in low_answer.notes:
            print(f"    - {note}")
        print()


if __name__ == "__main__":
    main()
