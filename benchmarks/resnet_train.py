# Benchmark 1: ResNet-50 training - Arav
"""ResNet-50 training benchmark: fixed work, not fixed time.

Runs a fixed number of training batches to completion and reports wall-clock
runtime. Energy accounting is done by the caller (measurement/runner.py),
which wraps this function's execution with measurement/power_monitor.py.
"""

import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)

NUM_WARMUP_BATCHES = 5
NUM_BATCHES = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DEFAULT_DATA_DIR = "/results/data/cifar10"


def _make_loader(data_dir: str) -> DataLoader:
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
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


def run(data_dir: str = DEFAULT_DATA_DIR, device: str = "cuda:0") -> dict:
    """Runs the fixed-work ResNet-50 training benchmark.

    Args:
      data_dir: Directory to download/cache CIFAR-10 into.
      device: CUDA device string.

    Returns:
      dict with runtime_seconds, batches_completed, final_loss.
    """
    torch_device = torch.device(device)

    loader = _make_loader(data_dir)
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    model = torchvision.models.resnet50(weights=None).to(torch_device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    def train_step() -> float:
        images, labels = next_batch()
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
    torch.cuda.synchronize(torch_device)

    logger.info("Running %d measured batches", NUM_BATCHES)
    final_loss = 0.0
    start = time.perf_counter()
    for _ in range(NUM_BATCHES):
        final_loss = train_step()
    torch.cuda.synchronize(torch_device)
    runtime_seconds = time.perf_counter() - start

    return {
        "runtime_seconds": runtime_seconds,
        "batches_completed": NUM_BATCHES,
        "final_loss": final_loss,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="ResNet-50 training benchmark")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    result = run(data_dir=args.data_dir, device=args.device)
    logger.info("Result: %s", result)
