# GPU Retirement Analysis for Nautilus

For each old GPU on Nautilus (NRP), better for climate to keep running it or retire + replace w/ new one? Measure real energy-per-job across GPU generations, calc break-even utilization threshold where replacement wins.

## Project Structure

```
gpu-retirement-analysis/
├── Dockerfile              # Benchmark container image: CUDA base + torch/transformers/pynvml, runs measurement.runner
├── k8s/
│   ├── inventory.sh         # Entry point: snapshots nodes (kubectl), runs summarize_census.py
│   ├── summarize_census.py  # Summarizes a snapshot to data/processed/census_*.csv (Python, not jq)
│   ├── benchmark-pod.yaml   # Pod spec template: nodeSelector picks GPU model, runs container, mounts results PVC
│   └── pvc.yaml             # (planned) PersistentVolumeClaim for /results storage
├── benchmarks/
│   ├── matmul.py            # Benchmark 3: 4096x4096 float32 matmul, 100 iters — pure FLOPS test
│   ├── resnet_train.py      # Benchmark 1: ResNet-50 on CIFAR-10, 100 batches, batch_size=32
│   └── llm_inference.py     # Benchmark 2: GPT-2 (124M), generate 500 tokens from fixed prompt
├── measurement/
│   ├── power_monitor.py     # PowerMonitor class: background thread samples nvmlDeviceGetPowerUsage every 200ms, integrates to joules
│   └── runner.py            # Orchestrator: starts PowerMonitor, runs a benchmark fn, stops monitor, writes CSV to /results
├── analysis/
│   ├── carbon_model.py      # Break-even math: embodied carbon vs saved operational carbon. Provisional until grid intensity is sourced; embodied landed in Phase 7
│   ├── sensitivity.py       # (planned) sweep embodied carbon range, show where break-even flips
│   ├── grid_intensity.py    # Grid carbon intensity presets (CAISO, national avg, ERCOT, PJM), all unsourced placeholders
│   └── plots.py             # (planned) break-even chart generation
└── data/
    ├── raw/                 # (planned) raw CSV output per benchmark run
    ├── embodied/             # (planned) embodied carbon estimates from literature
    └── processed/            # (planned) cleaned/aggregated results
```

## Flow

1. **Inventory the fleet** — run `k8s/inventory.sh` to snapshot nodes and summarize GPU models per-model (node counts, openly schedulable with GPU, allocatable, sites, taints) to `data/processed/census_nodes.csv` and `census_fleet.csv`. Node access is list-only, so this reads `kubectl get nodes` and never describes individual nodes.
2. **Deploy benchmark pods** — for each GPU model, `k8s/benchmark-pod.yaml` templated w/ `{{GPU_MODEL}}`, `{{BENCHMARK}}`, `{{RUN_NUMBER}}`, submitted via nodeSelector to pin to that GPU.
3. **Pod runs container** — built from `Dockerfile`, entrypoint `measurement/runner.py`.
4. **Runner orchestrates single run** — starts `measurement/power_monitor.py` (background pynvml sampling thread), calls one of `benchmarks/{matmul,resnet_train,llm_inference}.py` (fixed work, not fixed time), stops power monitor, writes energy + runtime CSV to PVC-mounted `/results`.
5. **Repeat** 5-10x per GPU model per benchmark for statistical confidence.
6. **Aggregate + analyze** — `analysis/carbon_model.py` combines measured energy-per-job (`data/processed/`) w/ embodied carbon estimates (`data/embodied/`) + grid intensity (`analysis/grid_intensity.py`) to compute break-even.
7. **Sensitivity + plots** — `analysis/sensitivity.py` sweeps embodied carbon assumptions, `analysis/plots.py` renders break-even charts to `results/figures/`.
