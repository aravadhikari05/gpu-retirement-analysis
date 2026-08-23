"""Unit tests for idle power sampling and its aggregation.

Stdlib only, and no GPU. measurement/runner.py imports nothing heavier than the
standard library plus benchmarks/_result.py at module level, and
measurement/power_monitor.py imports pynvml lazily inside start(), so both are
importable in a plain interpreter. The sampler itself is patched out: what is
tested is the column contract and the arithmetic around it, not NVML.

  python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from analysis.summarize_runs import _distinct_numeric, aggregate
from measurement.runner import (
    IDLE_WINDOWS,
    _blank_idle_fields,
    _sample_idle_window,
    measure_idle,
)


class FakePowerResult:
    """The subset of PowerResult that _sample_idle_window reads."""

    def __init__(self):
        self.avg_power_w = 55.03
        self.min_power_w = 54.0
        self.peak_power_w = 56.5
        self.duration_s = 60.0
        self.n_samples = 300


class FakeMonitor:
    def __init__(self, *args, **kwargs):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        return FakePowerResult()


class BlankIdleFields(unittest.TestCase):
    def test_every_window_and_suffix_is_present(self):
        fields = _blank_idle_fields("--no-idle")
        # Emitted rather than omitted: a missing key and an unmeasurable card
        # look identical in sparse JSONL.
        self.assertEqual(len(fields), 2 * 5 + 1)
        for window in IDLE_WINDOWS:
            self.assertEqual(fields[f"{window}_avg_w"], "")
        self.assertEqual(fields["idle_skip_reason"], "--no-idle")

    def test_measured_fields_use_the_same_names_as_blank_ones(self):
        # A drift here would leave half the rows in a sweep with one column set
        # and half with another, which is the failure the schema exists to stop.
        with mock.patch("measurement.power_monitor.PowerMonitor", FakeMonitor):
            with mock.patch("measurement.runner.time.sleep"):
                measured = _sample_idle_window("idle_pre_context", 60.0)
        blank = _blank_idle_fields("")
        self.assertTrue(set(measured).issubset(set(blank)))
        self.assertEqual(measured["idle_pre_context_avg_w"], 55.03)
        self.assertEqual(measured["idle_pre_context_n_samples"], 300)


class MeasureIdle(unittest.TestCase):
    def test_failure_is_recorded_not_raised(self):
        # A missing idle figure is a gap in the carbon model. A crash here would
        # cost the whole pod's benchmark time instead.
        with mock.patch(
            "measurement.power_monitor.PowerMonitor",
            side_effect=RuntimeError("no NVML"),
        ):
            fields = measure_idle(1.0)
        self.assertIn("no NVML", fields["idle_skip_reason"])
        self.assertEqual(fields["idle_pre_context_avg_w"], "")


class IdleAggregation(unittest.TestCase):
    def test_distinct_numeric_ignores_blanks_and_strings(self):
        members = [
            {"idle_post_context_avg_w": 20.0},
            {"idle_post_context_avg_w": 20.0},
            {"idle_post_context_avg_w": ""},
            {},
        ]
        self.assertEqual(_distinct_numeric(members, "idle_post_context_avg_w"), [20.0])

    def test_idle_is_averaged_over_pods_not_over_rows(self):
        # Three repetitions from one pod at 60 W and one from another at 20 W.
        # A mean over rows gives 50 W and is wrong: idle was observed twice.
        runs = []
        for idle, n in ((60.0, 3), (20.0, 1)):
            for i in range(n):
                runs.append(
                    {
                        "config_id": "matmul|n8192|fp32|i2000|s20260818",
                        "gpu_model_observed": "NVIDIA-GeForce-GTX-1080-Ti",
                        "work_hash": "a" * 64,
                        "repeat_index": i,
                        "energy_j": 1000.0,
                        "runtime_s": 91.0,
                        "inner_iters": 2000,
                        "idle_post_context_avg_w": idle,
                        "idle_pre_context_avg_w": idle,
                    }
                )
        rows = aggregate(runs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_runs"], 4)
        self.assertEqual(rows[0]["n_idle_observations"], 2)
        self.assertEqual(rows[0]["idle_post_context_avg_w_mean"], 40.0)


if __name__ == "__main__":
    unittest.main()
