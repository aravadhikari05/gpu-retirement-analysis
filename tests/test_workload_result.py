"""Unit tests for the enforced record contract.

Deliberately stdlib only. benchmarks/_result.py does not import torch precisely
so this can run in a plain interpreter, off the GPU image:

  python -m unittest discover -s tests

Everything here is validation logic. No benchmark, no NVML, no GPU.
"""

import unittest

from benchmarks._result import PRECISION_NAMES, WORK_HASH_KINDS, WorkloadResult


def _valid(**overrides) -> dict:
    """A minimal record that must construct, with named fields overridden."""
    base = {
        "workload": "matmul",
        "config_id": "matmul|n8192|fp32|i2000|s20260818",
        "work_hash": "a" * 64,
        "work_hash_kind": "config",
        "precision": "fp32",
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "inner_iters": 2000,
        "runtime_seconds": 91.2,
    }
    base.update(overrides)
    return base


class WorkloadResultValidation(unittest.TestCase):
    def test_valid_record_constructs(self):
        result = WorkloadResult(**_valid())
        self.assertEqual(result.workload, "matmul")
        self.assertEqual(result.inner_iters, 2000)

    def test_missing_required_field_is_a_construction_error(self):
        # The whole point of the dataclass: a benchmark cannot forget a field.
        args = _valid()
        del args["work_hash_kind"]
        with self.assertRaises(TypeError):
            WorkloadResult(**args)

    def test_empty_strings_rejected(self):
        for name in ("workload", "config_id", "work_hash"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    WorkloadResult(**_valid(**{name: "  "}))

    def test_work_hash_kind_vocabulary(self):
        for kind in WORK_HASH_KINDS:
            WorkloadResult(**_valid(work_hash_kind=kind))
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(work_hash_kind="inputs"))

    def test_precision_vocabulary(self):
        for precision in PRECISION_NAMES:
            WorkloadResult(**_valid(precision=precision))
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(precision="fp8"))

    def test_tf32_flags_must_be_read_back_bools(self):
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(allow_tf32_matmul="False"))
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(allow_tf32_cudnn=None))

    def test_inner_iters_at_least_one(self):
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(inner_iters=0))
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(inner_iters=-3))
        # bool is a subclass of int and must not pass as a count.
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(inner_iters=True))
        # The llm interim value until repetition-in-the-timed-region lands.
        WorkloadResult(**_valid(inner_iters=1))

    def test_runtime_seconds_above_zero(self):
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(runtime_seconds=0.0))
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(runtime_seconds=-1.0))
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(runtime_seconds="91.2"))

    def test_extra_may_not_shadow_a_required_field(self):
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(extra={"precision": "tf32"}))
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(extra={"inner_iters": 5}))

    def test_extra_must_be_a_dict(self):
        with self.assertRaises(ValueError):
            WorkloadResult(**_valid(extra=[("n", 8192)]))


class WorkloadResultRow(unittest.TestCase):
    def test_to_row_merges_extra_under_required(self):
        result = WorkloadResult(
            **_valid(extra={"n": 8192, "total_flops": 2 * 8192**3 * 2000})
        )
        row = result.to_row()
        self.assertEqual(row["n"], 8192)
        self.assertEqual(row["workload"], "matmul")
        self.assertEqual(row["runtime_seconds"], 91.2)

    def test_to_row_carries_every_required_field(self):
        row = WorkloadResult(**_valid()).to_row()
        for name in (
            "workload",
            "config_id",
            "work_hash",
            "work_hash_kind",
            "precision",
            "allow_tf32_matmul",
            "allow_tf32_cudnn",
            "inner_iters",
            "runtime_seconds",
        ):
            self.assertIn(name, row)

    def test_to_row_is_a_copy(self):
        extra = {"n": 8192}
        row = WorkloadResult(**_valid(extra=extra)).to_row()
        row["n"] = 1
        self.assertEqual(extra["n"], 8192)


if __name__ == "__main__":
    unittest.main()
