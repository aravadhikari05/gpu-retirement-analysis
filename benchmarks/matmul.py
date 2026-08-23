"""Benchmark 3: matrix multiplication, pure FLOPS sanity check. Owner: Veda.

Moved here from matmul_benchmark.py at the repo root on 2026-08-18. The root
copy was the real implementation while this path was a one-line stub, so the
Dockerfile's `COPY benchmarks/` shipped the stub.

Three changes were made to the moved code, to satisfy the correctness
requirements in CLAUDE.md that apply to all three workloads:

  1. It no longer writes its own CSV. measurement/runner.py owns all result
     writes; this returns a dict. The CSV header comment claiming a "SHARED
     CONTRACT with Arav/Aidan" described a schema that was never agreed to and
     that could not carry the required runtime provenance fields.
  2. Explicit TF32 control via benchmarks/_precision.py. Previously --dtype
     selected a storage dtype but left the TF32 flags at PyTorch defaults, so
     an fp32 run on an Ampere or Ada card silently used tensor cores while the
     same run on a GTX 1080 Ti did not.
  3. A work_hash, so two runs can be shown to have done the same work.

Design rule: fix the WORK, not the time. N and iters are fixed, so a fast GPU
finishes sooner and a slow one takes longer. That difference is the
measurement. Total ops are known exactly: an N x N matmul is 2 * N^3 FLOPs, so
total_flops = 2 * N^3 * iters, which makes this the clean baseline for
FLOPs-per-joule in Phase 8.

Energy is not recorded here. measurement/runner.py wraps the timed region with
measurement/power_monitor.py.
"""

import argparse
import hashlib
import logging
import platform

import torch

from benchmarks._context import RunContext, sync_device
from benchmarks._precision import set_precision
from benchmarks._result import WorkloadResult, extra_fields

logger = logging.getLogger(__name__)

DEFAULT_N = 8192
# Sized by measurement 2026-08-23, raised from 200. The fastest card both
# reachable and free is the RTX 3090 at 45.54 ms per iteration, so 2000 iters is
# a 90 s region on it and roughly 232 s on a 1080 Ti. At 200 the 3090 finished in
# about 9 s, well under the 30 s floor, so every run at the old default was
# excluded. The target is 90 s rather than the floor itself so a card twice as
# fast as the 3090 still clears it without a resize, and a resize changes
# config_id and work_hash and invalidates every row already collected. See
# Workload sizing in CLAUDE.md.
DEFAULT_ITERS = 2000
DEFAULT_WARMUP = 10

# Fixed seed so the input matrices are identical on every card. The previous
# version used unseeded torch.rand, which does not change the FLOP count but
# does mean no two runs can be proven to have multiplied the same numbers.
SEED = 20260818


def _pick_device() -> str:
    """Returns the best available device. Supports Mac for local smoke tests."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_gpu_model(device: str, override: str | None) -> str:
    """Prefers an explicit override matching the nvidia.com/gpu.product label."""
    if override:
        return override
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0).replace(" ", "-")
    if device.startswith("mps"):
        return f"Apple-{platform.processor() or 'Silicon'}-MPS"
    return f"CPU-{platform.processor() or 'unknown'}"


def run(
    n: int = DEFAULT_N,
    iters: int = DEFAULT_ITERS,
    warmup: int = DEFAULT_WARMUP,
    precision: str = "fp32",
    device: str = "",
    gpu_model: str = "",
    ctx: RunContext | None = None,
) -> WorkloadResult:
    """Runs the fixed-work matrix multiplication benchmark.

    Args:
      n: Matrix dimension. The workload is one n x n by n x n multiply.
      iters: Timed iterations. This plus n is the fixed work.
      warmup: Untimed iterations absorbing CUDA context and clock spin-up.
      precision: One of benchmarks._precision.PRECISIONS. "fp32" disables TF32
        explicitly; "tf32" enables it.
      device: Torch device string. Auto-detected when empty.
      gpu_model: Explicit GPU label. Auto-detected when empty.
      ctx: Timed-region context supplied by measurement/runner.py, which scopes
        power measurement to the timed loop. A monitor-less one is created for a
        standalone CLI run.

    Returns:
      A WorkloadResult. The required fields are enforced by its constructor;
      total_flops and the matmul-specific metadata ride in `extra`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    device = device or _pick_device()
    # A monitor-less context for standalone CLI runs; the runner passes its own.
    ctx = ctx or RunContext()
    precision_record, dtype = set_precision(precision, device)

    # Seeded on the CPU then moved, so the values do not depend on the device
    # RNG implementation. Two different cards then multiply identical inputs.
    generator = torch.Generator().manual_seed(SEED)
    a = torch.rand(n, n, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    b = torch.rand(n, n, generator=generator, dtype=torch.float32).to(
        device=device, dtype=dtype
    )

    for _ in range(warmup):
        c = a @ b
    sync_device(device)

    logger.info(
        "Timed region: %d x %d matmul, %d iters, precision %s", n, n, iters, precision
    )
    # ctx.timed_region syncs at both boundaries and marks them on the power
    # monitor, so energy covers exactly this loop and not the warmup above.
    with ctx.timed_region(device):
        for _ in range(iters):
            c = a @ b
    runtime_seconds = ctx.region_runtime_s

    total_flops = 2 * (n**3) * iters

    # work_hash covers the inputs and the shape of the work, not the product.
    # The product is a float reduction over n terms, so it is not bit-identical
    # across architectures and hashing it would fail for a reason that has
    # nothing to do with whether the same work was done. See resnet_train.py for
    # the same argument stated at length.
    hasher = hashlib.sha256()
    hasher.update(f"matmul|{n}|{iters}|{precision}|{SEED}".encode("utf-8"))
    hasher.update(a.to("cpu", torch.float32).numpy().tobytes())
    hasher.update(b.to("cpu", torch.float32).numpy().tobytes())
    work_hash = hasher.hexdigest()

    # Checksum of the product, recorded but never asserted equal across cards.
    # A large divergence here signals a real numerical problem worth chasing.
    result_checksum = float(c.to("cpu", torch.float64).sum().item())

    del a, b, c
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Everything the required set does not already own. precision and both
    # TF32 read-backs are lifted out of precision_record into the constructor,
    # since extra may not shadow a required field.
    extra = {
        "result_checksum": result_checksum,
        "n": n,
        "iters": iters,
        "warmup_iters": warmup,
        "seed": SEED,
        "total_flops": total_flops,
        "gpu_model_torch": _detect_gpu_model(device, gpu_model or None),
        "dtype": str(dtype),
        **extra_fields(precision_record),
    }

    return WorkloadResult(
        workload="matmul",
        # config_id states what was asked for; work_hash proves it happened.
        # Format follows benchmarks/llm_inference.py so the three workloads are
        # groupable by the same column in analysis.
        config_id=f"matmul|n{n}|{precision}|i{iters}|s{SEED}",
        work_hash=work_hash,
        # Config kind: the hash covers the inputs and the shape of the work, so
        # it proves identical work was requested, not identical numbers produced.
        work_hash_kind="config",
        work_hash_covers="inputs and work shape, not the product",
        precision=precision_record["precision"],
        allow_tf32_matmul=precision_record["allow_tf32_matmul"],
        allow_tf32_cudnn=precision_record["allow_tf32_cudnn"],
        # The loop inside the timed region. Energy per unit of work is
        # energy_j / inner_iters. Distinct from the runner's --repeats, which is
        # the outer loop for statistical spread.
        inner_iters=iters,
        runtime_seconds=runtime_seconds,
        extra=extra,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Matrix multiply benchmark, fixed work"
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--device", default="")
    parser.add_argument("--gpu-model", default="")
    args = parser.parse_args()

    result = run(
        n=args.n,
        iters=args.iters,
        warmup=args.warmup,
        precision=args.precision,
        device=args.device,
        gpu_model=args.gpu_model,
    ).to_row()
    # gflops_per_s is derived, not stored: CLAUDE.md keeps derived quantities out
    # of the record and computes them in the summary step.
    logger.info(
        "runtime=%.4fs %.1f GFLOP/s work_hash=%s",
        result["runtime_seconds"],
        result["total_flops"] / result["runtime_seconds"] / 1e9,
        result["work_hash"][:16],
    )
    if result["runtime_seconds"] < 30:
        logger.warning(
            "Timed region under 30 s. Raise --iters or --n before real runs; "
            "the power sensor may return cached readings below that."
        )
