# GPU Carbon Break-Even Analysis for Nautilus

## Project Summary

This project answers: **for each old GPU on the Nautilus cluster, would it be better for the climate to keep running it or retire it and replace it with a new one?**

Nautilus (NRP) is a shared Kubernetes GPU cluster where donated hardware from 2018 runs alongside cards from 2025. Replacing an old GPU saves electricity (operational carbon) but costs manufacturing emissions (embodied carbon). We measure real energy-per-job across every GPU generation on the cluster and calculate the break-even utilization threshold where replacement becomes the greener choice.

## Team

Multi-person project — others work on this repo besides you. Missing files/dirs from the structure below may mean teammates have work in progress elsewhere, not that a step is unstarted. Check git log/blame and Nautilus pod/PVC state before assuming something doesn't exist yet.

## Tech Stack

- **Language:** Python 3.10+, Bash
- **ML Framework:** PyTorch (for benchmark workloads only — we are not building models, just running standard ones)
- **Power Measurement:** pynvml (Python bindings to NVIDIA Management Library)
- **Cluster:** Nautilus/Kubernetes (pods, nodeSelector, PersistentVolumeClaims)
- **Containers:** Docker (NVIDIA CUDA base images)
- **Data Analysis:** pandas, matplotlib
- **Style:** [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **Formatter:** ruff format (enforced on all Python files)
- **Shell Style:** [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)

## Repo Structure

```
gpu-carbon-breakeven/
├── CLAUDE.md                  # This file
├── README.md                  # Human-readable project overview
├── pyproject.toml             # Python project config, ruff settings
├── Dockerfile                 # Benchmark container image
├── docs/
│   ├── papers.md              # Related work and APA citations
│   ├── architecture.md        # System design and methodology
│   └── tutorials.md           # Skills table and learning resources
├── k8s/
│   ├── pvc.yaml               # PersistentVolumeClaim for results storage
│   ├── benchmark-pod.yaml     # Template pod spec with nodeSelector
│   └── inventory.sh           # Script to enumerate GPU models on cluster
├── benchmarks/
│   ├── resnet_train.py        # Benchmark 1: ResNet-50 training
│   ├── llm_inference.py       # Benchmark 2: GPT-2 token generation
│   └── matmul.py              # Benchmark 3: Matrix multiplication
├── measurement/
│   ├── power_monitor.py       # pynvml background thread power logger
│   └── runner.py              # Orchestrator: runs benchmark + power monitor, outputs CSV
├── analysis/
│   ├── carbon_model.py        # Break-even calculation and payback math
│   ├── sensitivity.py         # Sensitivity analysis across embodied carbon ranges
│   ├── grid_intensity.py      # Grid carbon intensity data (CAISO, national, coal)
│   └── plots.py               # Break-even chart generation
├── data/
│   ├── raw/                   # Raw CSV outputs from benchmark runs
│   ├── embodied/              # Embodied carbon estimates from literature
│   └── processed/             # Cleaned and aggregated results
└── results/
    └── figures/               # Generated break-even charts and plots
```

## Coding Conventions

- All Python files must pass `ruff check` and `ruff format`
- Type hints on all function signatures
- Docstrings on all public functions (Google-style docstrings)
- No hardcoded paths — use constants at top of file or argparse
- CSV output columns are always lowercase_snake_case
- All scripts must be runnable standalone with `python -m benchmarks.resnet_train` etc.
- Use `if __name__ == "__main__":` in every script
- Logging via Python `logging` module, not print statements (except for CLI output)

## Key Design Decisions

### Benchmarks: Fix the WORK, not the time

Every GPU runs the identical task. We measure total energy to completion, not power over a fixed duration. This gives us joules-per-unit-of-work, which is the number that feeds into the carbon model.

### Three benchmark workloads

1. **resnet_train.py** — ResNet-50 on CIFAR-10, exactly 100 batches, batch size 32. Standard benchmark used in the literature for comparability.
2. **llm_inference.py** — GPT-2 (124M params), generate exactly 500 tokens from a fixed prompt. Represents the fastest-growing workload on Nautilus.
3. **matmul.py** — Multiply two 4096×4096 float32 matrices, 100 iterations. Pure FLOPS test, isolates raw compute from framework overhead.

### Power measurement with pynvml (not CodeCarbon, not nvidia-smi CLI)

We use pynvml directly because:
- CodeCarbon is a black box with its own PUE and grid assumptions — those are what we're analyzing
- nvidia-smi CLI works but pynvml keeps everything in one Python process with synchronized timestamps

Implementation: a background thread calls `nvmlDeviceGetPowerUsage()` every 200ms, writes timestamped readings to a list, integrates power over time to get total joules when the benchmark completes.

**Known limitation:** nvidia-smi/pynvml can return cached readings on some GPU architectures (see Yang et al., 2024). Minimum run duration must be 30 seconds. Report confidence intervals.

### Storage: PersistentVolumeClaim, not local pod filesystem

Nautilus pods are ephemeral and can be preempted. All results must write to a mounted PVC at `/results` inside the pod. Never store data only in the pod's local filesystem.

### Repetitions

Every benchmark runs **5–10 times per GPU model**. Report mean and standard deviation. Log node name and timestamp for every run to correlate anomalies.

## Component Specifications

### `measurement/power_monitor.py`

```
Class: PowerMonitor
  __init__(self, device_index: int = 0, sample_interval_ms: int = 200)
  start(self) -> None          # Starts background sampling thread
  stop(self) -> dict           # Stops sampling, returns summary dict
    Returns:
      total_energy_joules: float
      avg_power_watts: float
      max_power_watts: float
      min_power_watts: float
      duration_seconds: float
      num_samples: int
      readings: list[dict]     # Each: {"timestamp": float, "power_mw": int}

Dependencies: pynvml
Thread: daemon thread, dies when main thread exits
Error handling: if nvmlDeviceGetPowerUsage raises, log warning and skip that sample
```

### `measurement/runner.py`

```
Orchestrates a single benchmark run:
  1. Initialize PowerMonitor
  2. Start power monitoring
  3. Run the benchmark function (passed as callable)
  4. Stop power monitoring
  5. Write results to CSV

Output CSV columns:
  gpu_model, benchmark_name, run_number, runtime_seconds,
  total_energy_joules, avg_power_watts, max_power_watts,
  min_power_watts, num_power_samples, timestamp, node_name

Output path: /results/{gpu_model}/{benchmark_name}/run_{N}.csv
Power trace: /results/{gpu_model}/{benchmark_name}/run_{N}_power.csv
```

### `benchmarks/resnet_train.py`

```
- Model: torchvision.models.resnet50 (pretrained=False)
- Dataset: torchvision.datasets.CIFAR10 (auto-download)
- Fixed work: exactly 100 batches, batch_size=32
- Optimizer: SGD, lr=0.01, momentum=0.9
- Loss: CrossEntropyLoss
- Device: cuda:0
- Returns: dict with runtime_seconds, batches_completed, final_loss
```

### `benchmarks/llm_inference.py`

```
- Model: GPT2LMHeadModel.from_pretrained("gpt2")
- Tokenizer: GPT2Tokenizer.from_pretrained("gpt2")
- Fixed work: generate exactly 500 tokens
- Prompt: "The future of sustainable computing depends on"
- Device: cuda:0
- do_sample=False (greedy decoding for reproducibility)
- Returns: dict with runtime_seconds, tokens_generated, tokens_per_second
```

### `benchmarks/matmul.py`

```
- Matrix size: 4096 x 4096, dtype=torch.float32
- Fixed work: 100 multiplications (torch.mm)
- Warmup: 5 iterations (not counted)
- torch.cuda.synchronize() after each multiply
- Device: cuda:0
- Returns: dict with runtime_seconds, iterations, tflops
```

### `analysis/carbon_model.py`

```
Core equation:
  replacement_worthwhile = embodied_new < (energy_per_job_old - energy_per_job_new) * expected_jobs * grid_intensity

Functions:
  break_even_jobs(embodied_carbon_kg, energy_old_j, energy_new_j, grid_intensity_kgco2_per_kwh) -> float
  break_even_hours_per_year(embodied_carbon_kg, power_old_w, power_new_w, grid_intensity, gpu_lifetime_years) -> float
  payback_curve(gpu_pair, utilization_range, grid_intensity_range) -> DataFrame

Inputs:
  - Measured energy-per-job from benchmarks (data/processed/)
  - Embodied carbon estimates from literature (data/embodied/)
  - Grid intensity values (analysis/grid_intensity.py)

Grid intensity presets (kg CO2/kWh):
  CAISO (California): ~0.200
  US national average: ~0.390
  ERCOT (Texas): ~0.400
  PJM (mid-Atlantic coal): ~0.550
```

### `analysis/sensitivity.py`

```
Sweep embodied carbon across plausible range for each GPU model.
Show where the break-even conclusion flips.
Also project forward with declining grid intensity (CAISO trend ~3-5% annual decline).

Output: multi-panel plots showing break-even threshold vs embodied carbon estimate,
        parameterized by grid intensity and utilization.
```

### `Dockerfile`

```
Base: nvidia/cuda:12.1.0-runtime-ubuntu22.04
Install: python3, pip, pytorch (cu121), torchvision, transformers, pynvml, pandas
Copy: benchmarks/, measurement/
Workdir: /app
Entrypoint: python -m measurement.runner
```

### `k8s/benchmark-pod.yaml`

```
Template fields to substitute per run:
  {{GPU_MODEL}}     — nodeSelector value (e.g., "NVIDIA-A100-SXM4-80GB")
  {{BENCHMARK}}     — which benchmark to run (resnet_train, llm_inference, matmul)
  {{RUN_NUMBER}}    — integer for this repetition

Resource requests:
  nvidia.com/gpu: 1

Volume mount:
  PVC "benchmark-results" mounted at /results

Restart policy: Never (let failures surface, don't retry silently)
```

### `k8s/inventory.sh`

```
Enumerates all GPU models on Nautilus:
  kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .metadata.labels["nvidia.com/gpu.product"]}' | sort | uniq -c

Output: table of GPU models and node counts
```

## Embodied Carbon Sources

These are estimates, not exact numbers. Always present as ranges.

- **ACT model** (Gupta et al., 2022): Framework for estimating based on die size, process node, fab characteristics
- **Vendor product carbon footprint reports**: Dell, HP, Lenovo publish whole-system numbers; work backwards to GPU contribution
- **GPU die sizes**: publicly available from techpowerup, anandtech chip photos

Expected range per GPU: 50–400 kg CO2e depending on generation, die size, memory.

## Related Work (for context, not for code)

1. Gupta et al. (2022) — ACT carbon modeling tool → our embodied carbon methodology
2. Yang et al. (2024) — nvidia-smi power sensor accuracy → our measurement limitations
3. Uwizeyimana & Jerger (2025) — carbon-aware replacement theory → our research question
4. Li et al. (2023) — HPC carbon footprint estimation → broader context
5. Nguyen et al. (2025) — T4 vs RTX6000 Ada carbon comparison → closest prior work, but modeled not measured
6. Fadel Argerich et al. (2026) — Watt Counts energy benchmark across 10 GPUs → similar methodology, no replacement analysis

## Implementation Order

Build and test in this order. Each step should be a working, testable unit before moving to the next.

1. `k8s/inventory.sh` — verify you can query the cluster and see GPU models
2. `measurement/power_monitor.py` — test pynvml works in a basic pod
3. `benchmarks/matmul.py` — simplest benchmark, validates the pipeline
4. `measurement/runner.py` — integrate power monitor + benchmark + CSV output
5. `Dockerfile` + `k8s/benchmark-pod.yaml` + `k8s/pvc.yaml` — containerize and run on cluster
6. `benchmarks/resnet_train.py` — second benchmark
7. `benchmarks/llm_inference.py` — third benchmark
8. Run full sweep across all GPU models (5-10 reps each)
9. `analysis/carbon_model.py` — break-even math
10. `analysis/sensitivity.py` + `analysis/plots.py` — charts and sensitivity analysis

## Common Pitfalls

- **pynvml init:** must call `nvmlInit()` before any queries and `nvmlShutdown()` on exit
- **CUDA sync:** always `torch.cuda.synchronize()` before stopping the timer or power monitor, or you measure submission time not execution time
- **GPU warmup:** first 2-3 iterations on any GPU are always slower (JIT, memory allocation). Either discard them or run warmup iterations before starting measurement
- **Model download:** GPT-2 weights will download on first run. Pre-download in the Dockerfile or handle the network call gracefully
- **CIFAR-10 download:** same — pre-download or cache on the PVC
- **Pod preemption:** keep runs short (< 5 min). The PVC preserves completed results even if the pod dies mid-run
- **Power reading granularity:** older consumer GPUs may report power in coarse steps (e.g., 25W increments). Log this per GPU model and note it in results