#!/usr/bin/env python3
"""
Phase 3 - Workload 3: Matrix Multiplication Benchmark (pure-FLOPS sanity check)
GPU Carbon Efficiency project | owner: Veda

WHAT THIS DOES
  Multiplies two N x N matrices a FIXED number of times on the GPU, and records
  how long the identical work took. Every GPU runs the SAME N and the SAME
  iteration count, so runtime + energy differences come purely from the hardware.

DESIGN RULE (the whole point): fix the WORK, not the time.
  N (matrix size) and --iters (loop count) are fixed. A fast GPU finishes sooner,
  a slow GPU takes longer -- that difference IS the measurement.

Total floating-point ops are known exactly: an N x N matmul = 2 * N^3 FLOPs,
so total_flops = 2 * N^3 * iters. That exact count is what makes this the clean
baseline for FLOPs-per-joule (Phase 8).

Energy (joules) is NOT recorded here -- that's Phase 4's power-monitor thread,
which wraps the timed region marked below. This script records the WORK half.
"""

import argparse
import csv
import os
import platform
import time
from datetime import datetime, timezone

import torch


# ----------------------------------------------------------------------------
# Device handling: same script runs on Nautilus (cuda) and a Mac (mps) and CPU.
# ----------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    """Wait for the GPU to ACTUALLY finish. Without this, GPU work is async and
    the timer measures launch time, not compute time -- the #1 benchmarking bug."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    # cpu is synchronous already -- nothing to do


def detect_gpu_model(device, override):
    """Prefer an explicit --gpu-model (so it matches the exact Phase 1
    nvidia.com/gpu.product label on the cluster); else auto-detect."""
    if override:
        return override
    if device == "cuda":
        return torch.cuda.get_device_name(0).replace(" ", "-")
    if device == "mps":
        return f"Apple-{platform.processor() or 'Silicon'}-MPS"
    return f"CPU-{platform.processor() or 'unknown'}"


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# ----------------------------------------------------------------------------
# The benchmark itself.
# ----------------------------------------------------------------------------
def run_once(device, n, iters, warmup, dtype):
    """One measured run: warmup (untimed) -> sync -> timed loop -> sync.
    Returns (runtime_seconds, total_flops)."""
    # Allocate the two input matrices ONCE, on the device. Contents don't matter,
    # only that there are N*N of them. Reusing them keeps the work pure compute
    # (no per-iteration allocation noise).
    a = torch.rand(n, n, device=device, dtype=dtype)
    b = torch.rand(n, n, device=device, dtype=dtype)

    # WARMUP: throw-away iterations absorb one-time costs (CUDA context, memory
    # allocation, clock spin-up) so they don't pollute the measurement.
    for _ in range(warmup):
        c = a @ b
    sync(device)  # make sure warmup is fully done before we start the clock

    # ---- TIMED REGION (Phase 4 wraps its power sampling around exactly this) ----
    start = time.perf_counter()          # perf_counter = high-res monotonic clock
    for _ in range(iters):
        c = a @ b                        # the entire workload, iters times
    sync(device)                         # WAIT for all iters to finish -- critical
    runtime = time.perf_counter() - start
    # ---------------------------------------------------------------------------

    total_flops = 2 * (n ** 3) * iters   # exact FLOP count for the timed region
    del a, b, c
    if device == "cuda":
        torch.cuda.empty_cache()
    return runtime, total_flops


def main():
    p = argparse.ArgumentParser(description="Matrix-multiply GPU benchmark (fixed work).")
    p.add_argument("--n", type=int, default=8192, help="matrix dimension N (NxN). Default 8192.")
    p.add_argument("--iters", type=int, default=200, help="timed iterations (the fixed work). Default 200.")
    p.add_argument("--warmup", type=int, default=10, help="untimed warmup iterations. Default 10.")
    p.add_argument("--dtype", choices=DTYPES.keys(), default="fp32",
                   help="precision. fp32 is the fair cross-generation default; "
                        "lower precisions use tensor cores unevenly across GPU ages.")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat the whole measured run this many times, logging one row each "
                        "(Phase 6 wants 5-10 per GPU for mean/stddev). Default 1.")
    p.add_argument("--gpu-model", default=None,
                   help="exact GPU label (match the Phase 1 nvidia.com/gpu.product value). "
                        "Auto-detected if omitted.")
    p.add_argument("--out", default="results.csv", help="CSV file to append results to.")
    p.add_argument("--workload", default="matmul", help="workload name (shared column). Default matmul.")
    args = p.parse_args()

    device = pick_device()
    dtype = DTYPES[args.dtype]
    gpu_model = detect_gpu_model(device, args.gpu_model)

    print(f"device={device}  gpu_model={gpu_model}  dtype={args.dtype}")
    print(f"work: {args.n}x{args.n} matmul x {args.iters} iters "
          f"(+{args.warmup} warmup) x {args.repeats} repeat(s)")

    # CSV columns. SHARED CONTRACT with Arav/Aidan (must match across all 3 workloads):
    #   run_utc, gpu_model, workload, runtime_s, energy_j   <- energy_j filled in Phase 4
    # Workload-specific extras after that are fine as long as the shared ones line up.
    fields = ["run_utc", "gpu_model", "workload", "runtime_s", "energy_j",
              "device", "dtype", "n", "iters", "total_flops", "gflops_per_s"]
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in range(args.repeats):
            runtime, total_flops = run_once(device, args.n, args.iters, args.warmup, dtype)
            gflops = total_flops / runtime / 1e9
            row = {
                "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "gpu_model": gpu_model,
                "workload": args.workload,
                "runtime_s": round(runtime, 4),
                "energy_j": "",          # populated by Phase 4 power measurement
                "device": device,
                "dtype": args.dtype,
                "n": args.n,
                "iters": args.iters,
                "total_flops": total_flops,
                "gflops_per_s": round(gflops, 1),
            }
            w.writerow(row)
            print(f"  repeat {r+1}/{args.repeats}: {runtime:.3f}s  {gflops:.1f} GFLOP/s")

    print(f"appended {args.repeats} row(s) to {args.out}")
    if device == "cuda" and args.iters and runtime < 30:
        print("WARNING: timed region < 30s -- raise --iters or --n before real runs "
              "(Phase 4 power sensor returns stale readings under ~30s).")


if __name__ == "__main__":
    main()
