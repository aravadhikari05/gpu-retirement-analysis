# k8s Storage & Benchmark Pod (Phase 5) — Owner: Veda, closes CLAUDE.md to-do #1

## Volumes
- matmul-results (Veda) -> /results : canonical results volume, all 3 workloads write here. RWX.
- aidan-llm-models-pvc (Aidan) -> /models : pre-staged HF weights, only llm_inference reads it (HF_HOME=/models/hf).
CIFAR-10 downloads to /results/data/cifar10 during resnet setup (writable, on results PVC).

## PVC yaml consolidation
- results-pvc.yaml : canonical, kept.
- aidan-llm-models-pvc.yaml : kept (Aidan's cache).
- sample_pvc.yaml : unused template, ignore.

Results PVC is named "matmul-results" for historical reasons (holds validated L4 run).
It IS the shared results volume for all workloads. Open question: keep name or migrate to neutral "results-pvc" before sweep?

## Running
benchmark-pod.yaml is a template; substitute {{PLACEHOLDERS}}. Image: ghcr.io/aravadhikari05/gpu-retirement-analysis:latest
Delete every benchmark pod after it completes (holds a GPU). PVCs persist for the project.
Verified 2026-08-19: matmul on L4, energy_j 3200.7 vs counter 3194.4 (0.2%), power_window=region.
