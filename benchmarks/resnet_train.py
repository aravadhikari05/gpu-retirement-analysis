# Benchmark 1: ResNet-50 training - Arav
"""ResNet-50 training benchmark: fixed work, not fixed time.

Runs a fixed number of training batches to completion and reports wall-clock
runtime. Energy accounting is done by the caller (measurement/runner.py),
which wraps this function's execution with measurement/power_monitor.py.

Determinism, added 2026-08-18. The first version had no seeding and no TF32
control, which broke three of the correctness requirements in CLAUDE.md:

  - Model weights came from an unseeded random init, and the DataLoader
    shuffled with an unseeded generator, so two runs trained different weights
    on different batches.
  - Both TF32 flags were left at PyTorch defaults. An Ampere or Ada card then
    routes the fp32 convolutions and matmuls through tensor cores while a GTX
    1080 Ti cannot, so the old and new cards were not computing the same
    arithmetic. That is the exact comparison this project rests on, making this
    the most serious of the three.
  - No work_hash, so no run could be shown to match another.

What work_hash covers here, and why it is weaker than the LLM one. Training is
a long chain of float reductions, and floating point addition is not
associative, so final weights are not bit-identical across architectures and
hashing them would fail for reasons unrelated to whether the same work was
done. Instead the hash covers the inputs to the computation: the seed, the
exact dataset indices consumed in order, and the workload shape. Given those,
the FLOP count and the sequence of operations are identical on every card, which
is the fixed-work premise this project needs. The per-batch loss sequence is
recorded alongside for divergence diagnosis but is never asserted equal.
"""

import argparse
import hashlib
import json
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from benchmarks._precision import set_precision

logger = logging.getLogger(__name__)

NUM_WARMUP_BATCHES = 5
NUM_BATCHES = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SEED = 20260818
DEFAULT_DATA_DIR = "/results/data/cifar10"


def _make_loader(data_dir: str, indices: list[int]) -> DataLoader:
    """Builds a loader over an explicit, ordered index list.

    Shuffling is done once up front with a seeded generator and then baked into
    `indices`, rather than left to the DataLoader. That makes the exact batches
    consumed a recorded input to work_hash instead of a side effect of RNG
    state, and removes the need to cycle the iterator.
    """
    # Resize to 224x224: native CIFAR-10 32x32 barely exercises a GPU sized
    # for ResNet-50's expected input, undermining the point of the benchmark.
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    if max(indices) >= len(dataset):
        raise RuntimeError(
            f"index {max(indices)} out of range for CIFAR-10 with {len(dataset)} rows"
        )

    return DataLoader(
        Subset(dataset, indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )


def _plan_indices(dataset_len: int = 50000) -> list[int]:
    """Returns the exact dataset indices this benchmark consumes, in order."""
    needed = (NUM_WARMUP_BATCHES + NUM_BATCHES) * BATCH_SIZE
    if needed > dataset_len:
        raise ValueError(f"need {needed} samples, dataset has {dataset_len}")
    generator = torch.Generator().manual_seed(SEED)
    return torch.randperm(dataset_len, generator=generator)[:needed].tolist()


def run(
    data_dir: str = DEFAULT_DATA_DIR,
    device: str = "cuda:0",
    precision: str = "fp32",
) -> dict:
    """Runs the fixed-work ResNet-50 training benchmark.

    Args:
      data_dir: Directory to download or cache CIFAR-10 into. Must be on the
        PVC; a download inside the timed region would invalidate the run, so it
        happens during setup, before timing starts.
      device: CUDA device string.
      precision: One of benchmarks._precision.PRECISIONS. "fp32" disables TF32
        explicitly, which is required for cross-generation comparability.

    Returns:
      dict with runtime_seconds, work_hash, the loss sequence and the metadata
      needed to prove two runs did the same work.
    """
    torch_device = torch.device(device)
    precision_record, _ = set_precision(precision, device)

    indices = _plan_indices()
    loader = _make_loader(data_dir, indices)

    # Seed immediately before model construction so the initial weights depend
    # only on SEED, not on how much RNG the data pipeline consumed first.
    torch.manual_seed(SEED)
    model = torchvision.models.resnet50(weights=None).to(torch_device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    data_iter = iter(loader)

    def train_step() -> float:
        images, labels = next(data_iter)
        images = images.to(torch_device)
        labels = labels.to(torch_device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    logger.info("Running %d warmup batches (not counted)", NUM_WARMUP_BATCHES)
    for _ in range(NUM_WARMUP_BATCHES):
        train_step()
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch_device)

    logger.info("Running %d measured batches", NUM_BATCHES)
    losses = []
    start = time.perf_counter()
    for _ in range(NUM_BATCHES):
        losses.append(train_step())
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch_device)
    runtime_seconds = time.perf_counter() - start

    # Covers the inputs and the shape of the work. See the module docstring for
    # why the trained weights are deliberately not hashed.
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "workload": "resnet_train",
                "seed": SEED,
                "num_batches": NUM_BATCHES,
                "num_warmup_batches": NUM_WARMUP_BATCHES,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "momentum": MOMENTUM,
                "precision": precision,
                "arch": "resnet50",
                "indices": indices,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    work_hash = hasher.hexdigest()

    return {
        "workload": "resnet_train",
        "runtime_seconds": runtime_seconds,
        "work_hash": work_hash,
        "work_hash_covers": "seed, dataset indices and workload shape, not trained weights",
        "batches_completed": NUM_BATCHES,
        "warmup_batches": NUM_WARMUP_BATCHES,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "final_loss": losses[-1] if losses else 0.0,
        "loss_sequence": losses,
        "arch": "resnet50",
        "gpu_model_torch": (
            torch.cuda.get_device_name(0).replace(" ", "-")
            if device.startswith("cuda") and torch.cuda.is_available()
            else ""
        ),
        **precision_record,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="ResNet-50 training benchmark")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="fp32")
    args = parser.parse_args()

    result = run(data_dir=args.data_dir, device=args.device, precision=args.precision)
    logger.info(
        "runtime=%.4fs final_loss=%.4f work_hash=%s",
        result["runtime_seconds"],
        result["final_loss"],
        result["work_hash"][:16],
    )
