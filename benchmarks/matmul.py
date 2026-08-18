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

logger = logging.getLogger(__name__)

DEFAULT_N = 8192
DEFAULT_ITERS = 200
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
) -> dict:
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
      dict with runtime_seconds, work_hash, total_flops and the metadata needed
      to prove two runs did the same work.
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

    record = {
        "workload": "matmul",
        # config_id states what was asked for; work_hash proves it happened.
        # Format follows benchmarks/llm_inference.py so the three workloads are
        # groupable by the same column in analysis.
        "config_id": f"matmul|n{n}|{precision}|i{iters}|s{SEED}",
        "runtime_seconds": runtime_seconds,
        "work_hash": work_hash,
        "work_hash_covers": "inputs and work shape, not the product",
        "result_checksum": result_checksum,
        # The loop inside the timed region. Energy per unit of work is
        # energy_j / inner_iters. Distinct from the runner's --repeats, which is
        # the outer loop for statistical spread.
        "inner_iters": iters,
        "n": n,
        "iters": iters,
        "warmup_iters": warmup,
        "seed": SEED,
        "total_flops": total_flops,
        "gflops_per_s": total_flops / runtime_seconds / 1e9 if runtime_seconds else 0.0,
        "gpu_model_torch": _detect_gpu_model(device, gpu_model or None),
        "dtype": str(dtype),
    }
    record.update(precision_record)
    return record


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
    )
    logger.info(
        "runtime=%.4fs %.1f GFLOP/s work_hash=%s",
        result["runtime_seconds"],
        result["gflops_per_s"],
        result["work_hash"][:16],
    )
    if result["runtime_seconds"] < 30:
        logger.warning(
            "Timed region under 30 s. Raise --iters or --n before real runs; "
            "the power sensor may return cached readings below that."
        )
