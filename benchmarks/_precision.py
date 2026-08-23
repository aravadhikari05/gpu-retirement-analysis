"""Shared precision and TF32 control for the benchmark workloads.

Correctness requirement 3 in CLAUDE.md: Ampere and later can silently route
FP32 matmuls through tensor cores while a GTX 1080 Ti cannot. Left to PyTorch
defaults the two cards compute different arithmetic, which is exactly the
comparison this project rests on, so both flags are set explicitly and the
read-back is recorded rather than the requested value.

benchmarks/llm_inference.py predates this module and carries its own equivalent
`_set_precision`, which is a strict superset: it also pins the torch 2.9
fp32_precision API, disables reduced-precision reductions, and records
sm_capability and tf32_effective. Folding llm into this weaker version would be
a regression, and folding those extras into here changes the arithmetic
recorded by matmul and resnet, so the duplication stays deliberate. What is no
longer duplicated is the vocabulary: both read PRECISION_NAMES from
benchmarks/_result.py, so the two cannot drift on which modes exist.
"""

import logging

import torch

from benchmarks._result import PRECISION_NAMES

logger = logging.getLogger(__name__)

# Keyed off the shared vocabulary rather than restating it. A name added to
# PRECISION_NAMES without a dtype here raises KeyError at import, which is a
# louder failure than a record whose precision the schema silently rejects.
_DTYPE_BY_NAME = {
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

PRECISIONS = {name: _DTYPE_BY_NAME[name] for name in PRECISION_NAMES}


def set_precision(precision: str, device: str) -> tuple[dict, torch.dtype]:
    """Sets both TF32 flags explicitly and returns what the runtime reports.

    Args:
      precision: One of PRECISIONS. "tf32" means fp32 storage with TF32 matmuls
        allowed; "fp32" means TF32 explicitly disabled.
      device: Torch device string, used only to record what was asked for.

    Returns:
      (record, dtype) where record holds the requested precision and the
      read-back state of both TF32 flags.
    """
    if precision not in PRECISIONS:
        raise ValueError(
            f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}"
        )

    dtype = PRECISIONS[precision]
    tf32_requested = precision == "tf32"

    # torch 2.9 deprecates allow_tf32 in favour of fp32_precision. Set whichever
    # exists, and record the read-back, never the request. The image currently
    # has torch 2.5.1+cu121 where only allow_tf32 exists.
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = tf32_requested
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = tf32_requested

    record = {
        "precision": precision,
        "device_requested": device,
        "allow_tf32_matmul": bool(getattr(matmul, "allow_tf32", False)),
        "allow_tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "torch_version": torch.__version__,
    }

    if record["allow_tf32_matmul"] != tf32_requested:
        logger.warning(
            "TF32 matmul flag read back as %s after requesting %s. The run is "
            "not comparable across architectures.",
            record["allow_tf32_matmul"],
            tf32_requested,
        )

    return record, dtype
