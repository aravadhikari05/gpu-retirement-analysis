"""Unit tests for the carbon break-even model.

Stdlib only, no GPU, no data files. analysis/carbon_model.py imports nothing
heavier than csv and the grid presets, so this runs in a plain interpreter:

  python -m unittest discover -s tests

The energy figures used here are the real ones from the 2026-08-23 fleet pass
(data/processed/energy_by_gpu.csv), so a change to the arithmetic shows up
against numbers the project actually measured rather than against invented ones.
Every embodied and grid figure is a placeholder, which is what the provisional
tests are about.
"""

import unittest

from analysis.carbon_model import (
    HOURS_PER_YEAR,
    J_PER_KWH,
    SAME_MODEL_VARIANCE,
    BreakEven,
    CardEnergy,
    EmbodiedEstimate,
    annual_energy_j,
    break_even_hours_per_year,
    break_even_jobs,
    carbon_saved_kg,
    cumulative_intensity,
    payback_curve,
)
from analysis.grid_intensity import PRESETS, GridIntensity, preset

# Measured, matmul|n8192|fp32|i2000|s20260818, 2026-08-23. energy_j_per_inner_iter
# and idle_post_context_avg_w_mean straight out of energy_by_gpu.csv; the
# per-job runtime is runtime_s_mean / inner_iters.
MATMUL_1080TI = CardEnergy(
    gpu_model="NVIDIA GeForce GTX 1080 Ti",
    benchmark="matmul",
    config_id="matmul|n8192|fp32|i2000|s20260818",
    energy_j_per_job=31.9628456537624,
    runtime_s_per_job=285.20541353583604 / 2000,
    idle_w=25.430789414414402,
    n_runs=7,
    n_physical_gpus=1,
    work_hash="3bbd5bd4",
)
MATMUL_A4000 = CardEnergy(
    gpu_model="NVIDIA RTX A4000",
    benchmark="matmul",
    config_id="matmul|n8192|fp32|i2000|s20260818",
    energy_j_per_job=14.324108542632437,
    runtime_s_per_job=205.2085557779763 / 2000,
    idle_w=25.0573023255814,
    n_runs=5,
    n_physical_gpus=1,
    work_hash="3bbd5bd4",
)

CAISO = preset("CAISO")
PLACEHOLDER = EmbodiedEstimate(
    gpu_model="NVIDIA RTX A4000", low_kg=100.0, high_kg=100.0, sourced=False
)
SOURCED = EmbodiedEstimate(
    gpu_model="NVIDIA RTX A4000",
    low_kg=100.0,
    high_kg=100.0,
    sourced=True,
    citation="placeholder for tests only",
)
SOURCED_GRID = GridIntensity(
    "CAISO", 0.200, sourced=True, citation="placeholder for tests only"
)


class Units(unittest.TestCase):
    """The /3.6e6 conversion, pinned end to end.

    Dropping it is wrong by a factor of 3.6 million while still producing a
    plausible looking number, which is why this is the one thing
    docs/tasks/phase8-break-even-inputs.md says code adds over the document.
    """

    def test_worked_example_by_hand(self):
        # 1 kWh saved per job, 1000 jobs, 0.5 kg/kWh.
        # 3.6e6 J * 1000 / 3.6e6 = 1000 kWh, times 0.5 = 500 kg.
        saved = carbon_saved_kg(
            delta_energy_j=J_PER_KWH,
            jobs=1000,
            grid=GridIntensity("test", 0.5, sourced=True, citation="hand"),
        )
        self.assertAlmostEqual(saved, 500.0, places=9)

    def test_conversion_is_not_dropped(self):
        # Without the division this would be 3.6e6 times larger. Assert the
        # magnitude, so a dropped conversion cannot pass by rounding.
        saved = carbon_saved_kg(1.0, 1.0, PRESETS["CAISO"])
        self.assertLess(saved, 1e-6)
        self.assertGreater(saved, 0.0)

    def test_measured_matmul_pair(self):
        # Real delta from the fleet pass: 31.9628... - 14.3241... J per job.
        delta = MATMUL_1080TI.energy_j_per_job - MATMUL_A4000.energy_j_per_job
        self.assertAlmostEqual(delta, 17.638737111129963, places=9)
        expected_jobs = 100.0 * J_PER_KWH / (delta * 0.200)
        answer = break_even_jobs(PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO)
        self.assertAlmostEqual(answer.jobs, expected_jobs, places=3)


class SnapshotAndIntegral(unittest.TestCase):
    def test_snapshot_excludes_idle_and_says_so(self):
        answer = break_even_jobs(
            PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO, horizon_years=None
        )
        self.assertTrue(any("idle power excluded" in n for n in answer.notes))
        self.assertIsNone(answer.active_hours_per_year)

    def test_constant_grid_makes_cumulative_intensity_linear(self):
        flat = preset("CAISO")
        self.assertAlmostEqual(cumulative_intensity(flat, 5), 5 * 0.200, places=12)

    def test_declining_grid_pushes_break_even_out(self):
        flat = preset("PJM")
        declining = flat.with_decline(0.05)
        near = break_even_jobs(
            PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, flat, horizon_years=6
        )
        far = break_even_jobs(
            PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, declining, horizon_years=6
        )
        self.assertIsNotNone(near.jobs)
        self.assertIsNotNone(far.jobs)
        # A dirtier grid today that cleans up later avoids less carbon overall,
        # so more work is needed to repay the same embodied cost. Never fewer.
        self.assertGreater(far.jobs, near.jobs)

    def test_horizon_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            cumulative_intensity(preset("CAISO"), 0)
        with self.assertRaises(ValueError):
            break_even_hours_per_year(
                PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO, gpu_lifetime_years=0
            )


class IdleTerm(unittest.TestCase):
    def test_annual_energy_includes_idle_for_the_rest_of_the_year(self):
        # One job per year: almost the whole year is idle.
        total = annual_energy_j(MATMUL_1080TI, jobs_per_year=1)
        idle_only = MATMUL_1080TI.idle_w * (
            HOURS_PER_YEAR * 3600.0 - MATMUL_1080TI.runtime_s_per_job
        )
        self.assertAlmostEqual(
            total, MATMUL_1080TI.energy_j_per_job + idle_only, places=6
        )

    def test_faster_card_idles_longer_for_the_same_work(self):
        jobs = 100_000
        busy_old = jobs * MATMUL_1080TI.runtime_s_per_job
        busy_new = jobs * MATMUL_A4000.runtime_s_per_job
        self.assertLess(busy_new, busy_old)

    def test_measured_idle_difference_is_negligible(self):
        # 25.43 W against 25.06 W. The premise was that idle decides the answer;
        # the fleet pass measured that it does not.
        delta_w = MATMUL_1080TI.idle_w - MATMUL_A4000.idle_w
        self.assertLess(abs(delta_w), 1.0)
        relative = abs(delta_w) / MATMUL_1080TI.idle_w
        self.assertLess(relative, SAME_MODEL_VARIANCE)

    def test_idle_differential_inside_measurement_scatter_is_called_noise(self):
        # 25.43 W against 25.06 W, a 0.37 W differential, against a 13.57 W
        # spread one card shows across workloads. The kg figure is reported but
        # must not read as a finding.
        answer = break_even_jobs(
            PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO, horizon_years=6
        )
        self.assertTrue(any("NOISE" in n for n in answer.notes))

    def test_large_idle_differential_is_not_called_noise(self):
        quiet = CardEnergy(
            gpu_model="hypothetical quiet card",
            benchmark="matmul",
            config_id=MATMUL_1080TI.config_id,
            energy_j_per_job=MATMUL_A4000.energy_j_per_job,
            runtime_s_per_job=MATMUL_A4000.runtime_s_per_job,
            idle_w=MATMUL_1080TI.idle_w - 25.0,
            n_runs=5,
            n_physical_gpus=2,
            work_hash=MATMUL_1080TI.work_hash,
        )
        answer = break_even_jobs(
            PLACEHOLDER, MATMUL_1080TI, quiet, CAISO, horizon_years=6
        )
        self.assertFalse(any("NOISE" in n for n in answer.notes))

    def test_over_subscription_is_refused_not_clamped(self):
        with self.assertRaises(ValueError):
            annual_energy_j(MATMUL_1080TI, jobs_per_year=10**9)

    def test_negative_jobs_refused(self):
        with self.assertRaises(ValueError):
            annual_energy_j(MATMUL_1080TI, jobs_per_year=-1)


class VarianceGuard(unittest.TestCase):
    def test_real_pair_is_interpretable(self):
        answer = break_even_jobs(PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO)
        self.assertTrue(answer.interpretable)

    def test_small_difference_is_not_interpretable(self):
        # 3% apart, inside the 6.43% same-model bound.
        near = CardEnergy(
            gpu_model="NVIDIA RTX A4000 (twin)",
            benchmark="matmul",
            config_id=MATMUL_1080TI.config_id,
            energy_j_per_job=MATMUL_1080TI.energy_j_per_job * 0.97,
            runtime_s_per_job=MATMUL_1080TI.runtime_s_per_job,
            idle_w=MATMUL_1080TI.idle_w,
            n_runs=5,
            n_physical_gpus=2,
            work_hash=MATMUL_1080TI.work_hash,
        )
        answer = break_even_jobs(PLACEHOLDER, MATMUL_1080TI, near, CAISO)
        self.assertFalse(answer.interpretable)
        self.assertTrue(any("variance bound" in n for n in answer.notes))

    def test_single_card_spread_is_flagged(self):
        answer = break_even_jobs(PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO)
        self.assertTrue(any("one physical card" in n for n in answer.notes))

    def test_conservatism_direction_always_recorded(self):
        answer = break_even_jobs(PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO)
        self.assertTrue(any("pessimistic" in n for n in answer.notes))


class Provenance(unittest.TestCase):
    def test_unsourced_embodied_taints_result(self):
        answer = break_even_jobs(PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO)
        self.assertTrue(answer.provisional)
        self.assertTrue(any(n.startswith("PROVISIONAL") for n in answer.notes))

    def test_unsourced_grid_taints_result(self):
        answer = break_even_jobs(SOURCED, MATMUL_1080TI, MATMUL_A4000, CAISO)
        self.assertTrue(answer.provisional)

    def test_both_sourced_is_not_provisional(self):
        answer = break_even_jobs(SOURCED, MATMUL_1080TI, MATMUL_A4000, SOURCED_GRID)
        self.assertFalse(answer.provisional)

    def test_sourced_without_citation_is_refused(self):
        with self.assertRaises(ValueError):
            EmbodiedEstimate("card", 1.0, 2.0, sourced=True)
        with self.assertRaises(ValueError):
            GridIntensity("region", 0.2, sourced=True)

    def test_every_shipped_preset_is_unsourced(self):
        # If one of these ever reads True without a citation, the guard is gone.
        for name, grid in PRESETS.items():
            self.assertFalse(grid.sourced, f"{name} claims to be sourced")

    def test_inverted_embodied_range_refused(self):
        with self.assertRaises(ValueError):
            EmbodiedEstimate("card", 400.0, 50.0, sourced=False)


class Degenerate(unittest.TestCase):
    def test_identical_cards_never_pay_back(self):
        answer = break_even_jobs(PLACEHOLDER, MATMUL_1080TI, MATMUL_1080TI, CAISO)
        self.assertIsNone(answer.jobs)
        self.assertFalse(answer.pays_back)

    def test_worse_replacement_never_pays_back(self):
        answer = break_even_jobs(PLACEHOLDER, MATMUL_A4000, MATMUL_1080TI, CAISO)
        self.assertIsNone(answer.jobs)
        self.assertTrue(any("more per job" in n for n in answer.notes))

    def test_different_config_is_refused(self):
        other = CardEnergy(
            gpu_model="NVIDIA L4",
            benchmark="matmul",
            config_id="matmul|n8192|fp32|i500|s20260818",
            energy_j_per_job=44.5,
            runtime_s_per_job=0.1,
            idle_w=20.0,
            n_runs=1,
            n_physical_gpus=1,
            work_hash="afac2e9f",
        )
        with self.assertRaises(ValueError):
            break_even_jobs(PLACEHOLDER, MATMUL_1080TI, other, CAISO)

    def test_differing_work_hash_is_refused(self):
        impostor = CardEnergy(
            gpu_model="NVIDIA RTX A4000",
            benchmark="matmul",
            config_id=MATMUL_1080TI.config_id,
            energy_j_per_job=14.0,
            runtime_s_per_job=0.1,
            idle_w=25.0,
            n_runs=5,
            n_physical_gpus=1,
            work_hash="different",
        )
        with self.assertRaises(ValueError):
            break_even_jobs(PLACEHOLDER, MATMUL_1080TI, impostor, CAISO)


class Curve(unittest.TestCase):
    def test_higher_utilisation_never_takes_longer(self):
        curve = payback_curve(
            PLACEHOLDER,
            MATMUL_1080TI,
            MATMUL_A4000,
            CAISO,
            utilisations=(0.05, 0.25, 0.5, 1.0),
        )
        years = [
            answer.years_at_utilisation
            for _, answer in curve
            if answer.years_at_utilisation is not None
        ]
        self.assertEqual(years, sorted(years, reverse=True))

    def test_utilisation_out_of_range_refused(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                payback_curve(PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO, (bad,))

    def test_curve_returns_one_entry_per_utilisation(self):
        utilisations = (0.1, 0.4, 0.9)
        curve = payback_curve(
            PLACEHOLDER, MATMUL_1080TI, MATMUL_A4000, CAISO, utilisations
        )
        self.assertEqual([u for u, _ in curve], list(utilisations))
        for _, answer in curve:
            self.assertIsInstance(answer, BreakEven)


class GridPresets(unittest.TestCase):
    def test_decline_reduces_intensity_over_time(self):
        grid = preset("PJM").with_decline(0.10)
        self.assertAlmostEqual(grid.at_year(0), 0.550, places=12)
        self.assertAlmostEqual(grid.at_year(1), 0.550 * 0.9, places=12)

    def test_with_decline_preserves_provenance(self):
        grid = SOURCED_GRID.with_decline(0.04)
        self.assertTrue(grid.sourced)
        self.assertEqual(grid.citation, SOURCED_GRID.citation)

    def test_negative_year_refused(self):
        with self.assertRaises(ValueError):
            preset("CAISO").at_year(-1)

    def test_unknown_preset_lists_the_known_ones(self):
        with self.assertRaises(KeyError) as caught:
            preset("NOT_A_REGION")
        self.assertIn("CAISO", str(caught.exception))

    def test_preset_lookup_is_forgiving_about_style(self):
        self.assertIs(preset("us average"), preset("US_AVERAGE"))


if __name__ == "__main__":
    unittest.main()
