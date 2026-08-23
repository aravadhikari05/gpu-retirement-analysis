"""The enforced record contract every workload returns.

Decided in CLAUDE.md, "The record schema is convention, not enforcement, and it
drifted". The runner-owned spine (identity, power, provenance) was already
standardized because measurement/runner.py writes it. The benchmark payloads
were not, because each run() hand-built its own dict, so three fields meaning
the same thing were named or emitted differently: `precision` against
`precision_mode`, `work_hash_covers` present on two workloads of three, and
`workload` absent on llm_inference. None of those crash. JSONL is sparse, so
they surface as silent nulls in analysis, which is exactly the failure class the
schema table exists to prevent. Convention cannot fail loud.

WorkloadResult makes the required set constructor arguments, so a benchmark
physically cannot return a record missing one and the error moves from a runtime
null in a summary table to an authoring-time TypeError. Workload-specific fields
go in `extra`, and measurement/runner.py reads `.to_row()`.

The required set is not a list of available fields. It is derived by the rule in
CLAUDE.md, "How the column set was chosen": write the Phase 8 aggregation first,
then for each wrong answer it could still produce, add exactly the one field
that forbids it, and stop when you cannot invent a new wrong answer.

  workload           groups rows without depending on the runner's own key
  config_id          stops 32-token and 960-token runs averaging together
  work_hash          stops runs that did different work averaging together
  work_hash_kind     stops the LLM's output-identity guarantee being read onto
                     matmul and resnet, whose hashes cover inputs and work shape
  precision          stops fp32 and tf32 runs averaging together
  allow_tf32_matmul  read-back, not the request, for the same reason
  allow_tf32_cudnn   likewise
  inner_iters        energy per unit of work is energy_j / inner_iters, and
                     conflating it with the runner's --repeats scales every
                     energy figure by the wrong factor, silently
  runtime_seconds    the measured quantity itself

This module deliberately does not import torch. Validation has to run in a
plain interpreter so the contract is testable without a GPU image, which is why
PRECISION_NAMES lives here and benchmarks/_precision.py builds its dtype table
from it rather than the other way round.
"""

import dataclasses
from typing import Any

# Single source of truth for the precision vocabulary. benchmarks/_precision.py
# maps these names to torch dtypes; adding a name here without a dtype there
# raises at import rather than at record-construction time.
PRECISION_NAMES = ("fp32", "tf32", "fp16", "bf16")

# What a work_hash actually proves, which is not the same across workloads.
#
#   "output" the hash covers the workload's output, so two matching runs were
#            bit-identical. llm_inference hashes generated token IDs, and greedy
#            decoding is an argmax, so the result either matches exactly or
#            diverges visibly.
#   "config" the hash covers the inputs and the shape of the work (seed, dataset
#            indices or input matrices, iteration counts, precision), not the
#            result. Training and repeated matmul are long chains of float
#            reductions and floating point addition is not associative, so
#            results are not bit-identical across architectures and hashing them
#            would fail for reasons unrelated to whether the same work was done.
#
# A reviewer reading one work_hash column would otherwise assume all three carry
# the LLM's guarantee. This field is what forbids that reading.
WORK_HASH_KINDS = ("output", "config")

# Field names the dataclass owns. `extra` may not contain any of them, because a
# shadowed field would make the row disagree with itself. work_hash_covers is
# listed even though it has a default: it is the prose expansion of
# work_hash_kind and belongs to the same concept.
REQUIRED_FIELDS = (
    "workload",
    "config_id",
    "work_hash",
    "work_hash_kind",
    "work_hash_covers",
    "precision",
    "allow_tf32_matmul",
    "allow_tf32_cudnn",
    "inner_iters",
    "runtime_seconds",
)


@dataclasses.dataclass
class WorkloadResult:
    """One repetition's benchmark payload, with the required set enforced.

    Attributes:
      workload: Workload name, for example "matmul". Distinct from the runner's
        own `benchmark` key, which names the module it dispatched to.
      config_id: What was asked for, in the shared pipe-delimited format.
      work_hash: SHA-256 proving what actually happened.
      work_hash_kind: One of WORK_HASH_KINDS, stating what the hash covers.
      work_hash_covers: Human-readable expansion of work_hash_kind.
      precision: One of PRECISION_NAMES.
      allow_tf32_matmul: Read-back of the TF32 matmul flag, never the request.
      allow_tf32_cudnn: Read-back of the TF32 cuDNN flag, never the request.
      inner_iters: The workload's own loop count inside the timed region.
      runtime_seconds: Wall clock of the timed region.
      extra: Workload-specific fields, merged into the row by to_row().
    """

    workload: str
    config_id: str
    work_hash: str
    work_hash_kind: str
    precision: str
    allow_tf32_matmul: bool
    allow_tf32_cudnn: bool
    inner_iters: int
    runtime_seconds: float
    work_hash_covers: str = ""
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Rejects a record that analysis could not safely aggregate.

        Raises:
          ValueError: on any required field that is empty, out of vocabulary or
            out of range.
        """
        for name in ("workload", "config_id", "work_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")

        if self.work_hash_kind not in WORK_HASH_KINDS:
            raise ValueError(
                f"work_hash_kind must be one of {list(WORK_HASH_KINDS)}, got "
                f"{self.work_hash_kind!r}. It states whether the hash covers the "
                "output or only the inputs and work shape."
            )

        if self.precision not in PRECISION_NAMES:
            raise ValueError(
                f"precision must be one of {list(PRECISION_NAMES)}, got "
                f"{self.precision!r}"
            )

        for name in ("allow_tf32_matmul", "allow_tf32_cudnn"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"{name} must be a bool read back from the backend, got "
                    f"{getattr(self, name)!r}"
                )

        # bool is a subclass of int, so it is excluded explicitly rather than
        # sliding through as inner_iters=True.
        if isinstance(self.inner_iters, bool) or not isinstance(self.inner_iters, int):
            raise ValueError(f"inner_iters must be an int, got {self.inner_iters!r}")
        if self.inner_iters < 1:
            raise ValueError(
                f"inner_iters must be at least 1, got {self.inner_iters}. Energy "
                "per unit of work is energy_j / inner_iters, so 0 or a missing "
                "value silently scales every downstream figure."
            )

        if not isinstance(self.runtime_seconds, (int, float)) or isinstance(
            self.runtime_seconds, bool
        ):
            raise ValueError(
                f"runtime_seconds must be a number, got {self.runtime_seconds!r}"
            )
        if self.runtime_seconds <= 0:
            raise ValueError(
                f"runtime_seconds must be above 0, got {self.runtime_seconds}"
            )

        if not isinstance(self.extra, dict):
            raise ValueError(f"extra must be a dict, got {type(self.extra).__name__}")

        collisions = sorted(set(self.extra) & set(self._required_names()))
        if collisions:
            raise ValueError(
                f"extra may not shadow required fields: {collisions}. A shadowed "
                "field would make the row disagree with itself."
            )

    @staticmethod
    def _required_names() -> tuple[str, ...]:
        """Field names owned by the dataclass rather than by `extra`."""
        return REQUIRED_FIELDS

    def to_row(self) -> dict[str, Any]:
        """Flattens to the dict measurement/runner.py writes as one JSONL line.

        Required fields are written last so a stale key left in `extra` cannot
        win, though __post_init__ already refuses that case outright.

        Returns:
          dict of `extra` merged under the required fields.
        """
        row = dict(self.extra)
        row.update({name: getattr(self, name) for name in self._required_names()})
        return row


def extra_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Drops the keys the required set owns, so a metadata dict can be `extra`.

    All three workloads build a precision or provenance dict that overlaps the
    required set, and passing it straight in raises the shadowing error. Filtering
    here rather than in each workload means a field promoted into the required set
    later does not have to be hand-removed from three call sites.

    Args:
      record: Any metadata mapping, for example the read-back dict returned by
        benchmarks._precision.set_precision.

    Returns:
      A new dict with every REQUIRED_FIELDS key removed.
    """
    return {k: v for k, v in record.items() if k not in REQUIRED_FIELDS}
