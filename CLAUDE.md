# GPU Carbon Payback on NRP Nautilus

Undergraduate research, UC Santa Cruz. Aidan Nguyen, adviser Prof. Jullig.
Collaborators: Arav Adhikari, Veda Satvika.

## Where things are written down

This file holds rules, decisions, and measured findings. It does not duplicate
what another document already owns.

- `README.md` is the authoritative project structure and flow. Do not copy the
  tree into this file or into task docs; one copy, three pointers.
- `docs/phases.md` is the authoritative phase numbering, 1 through 10.
- `docs/tasks/*.md` own per-workload specs.
- `paper/methods-notes.md` owns measured facts destined for the paper. Put them
  there, not only into task docs.
- `docs/claudeback-original.md` is a superseded spec draft, kept for history.
  Its repo tree used the old project name `gpu-carbon-breakeven/` and listed
  files that were never created (`pyproject.toml`, `docs/papers.md`,
  `docs/architecture.md`, `docs/tutorials.md`, `results/figures/`,
  `k8s/pvc.yaml`). Do not reintroduce that tree.

## Hard rules

- Never fabricate numbers. No invented GPU counts, power readings, benchmark
  results, or carbon figures. Label estimated values as estimated.
- Never claim a command succeeded without seeing its actual output.
- No em dashes in any output, code comments included.
- This repo is public. Never commit credentials, kubeconfigs, S3 keys, or tokens.
- Preserve uncertainty. Ranges, not point values, for carbon estimates.
- Shared repo. Read existing files before modifying them. Many files under
  benchmarks/, measurement/, and analysis/ are one-line stubs, not empty.
  A missing file or directory may mean a teammate has work in progress
  elsewhere, not that a step is unstarted. Check `git log`, `git blame`, and
  live Nautilus pod and PVC state before concluding something does not exist.
  On record: Veda's matmul benchmark and power monitor existed as real code at
  the repo root for four days while `benchmarks/matmul.py` and
  `measurement/power_monitor.py` stayed one-line stubs. Reading only the package
  paths would have concluded the work was unstarted.

## Cluster environment

- kubectl context: `nautilus`, namespace: `cmpm118` (shared with other students)
- Requires UCSC VPN
- User-level access only. No admin, no node access, no scheduler changes.
- Prefix every Kubernetes resource name with `aidan`
- Registry: `gitlab-registry.nrp-nautilus.io/aidan/aidan`, built by GitLab CI
  on the `gitlab` remote. GitHub Actions does not build the pod image.
- Fleet reachability matters as much as fleet size. As of the 20260804 census,
  every A100 variant, the H100, H200, RTX A6000, A40 and GH200 report
  `openly_schedulable_with_gpu = 0`. They are in the census and cannot be
  landed on. The fastest card actually reachable is 4090 or L40S class, which
  is what workload sizing is built against.

## Coding conventions

- Python 3.10+ and Bash. PyTorch, pynvml, pandas, matplotlib.
- All Python passes `ruff check` and `ruff format`.
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  and [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html).
- Type hints on all function signatures. Google-style docstrings on public
  functions.
- No hardcoded paths. Module-level constants or argparse.
- CSV columns are lowercase_snake_case.
- Every script runnable as a module: `python -m benchmarks.resnet_train`.
- `if __name__ == "__main__":` in every script.
- `logging`, not print, except for deliberate CLI output.

`benchmarks/resnet_train.py` is the reference example. It also shows the right
shape for a benchmark: it returns a dict and writes no files.

Enforcement is weaker than it looks. Checked 2026-08-18: `ruff check` passes on
`benchmarks/` and `measurement/`, but there is no `pyproject.toml` or
`ruff.toml`, so it runs on ruff defaults (E4, E7, E9, F) and enforces none of
the type hint, docstring or naming rules above. Adding a ruff config that
actually checks them is unclaimed work.

`ruff format --check` is clean across `benchmarks/` and `measurement/` except
`llm_inference.py`, which is left unformatted deliberately: its output is
already committed under `data/raw/llm_smoke/` and reformatting it is a
coordinated change, not a drive-by. `k8s/summarize_census.py` is also
unformatted.

## Data conventions

- `data/raw/` holds raw telemetry and is committed, except node snapshots
  matching `nodes_*.json` which are gitignored for size
- `data/processed/` holds committed summary tables
- UTC timestamps everywhere. Structured output only: JSONL, CSV, or SQLite.
- Record raw telemetry before computing summaries
- Keep failed runs with an explicit exclusion reason rather than deleting them
- The fleet census is a point-in-time snapshot, not a constant. Node labeling
  drifted by one node within an hour between two captures. Phase 3 benchmark
  runs must record the GPU model, node name, and driver version observed at
  runtime from inside the pod, not by joining against a stored census. The
  census decides which models are worth targeting; it is not the source of
  truth for what a given run actually used.

## Working style

Before writing substantial code: restate the task, inspect relevant files,
propose a plan, name the assumptions, wait for approval on architectural
changes. Then implement in small reviewable steps, showing exact commands and
expected output.

## Benchmark correctness requirements

These originate in `docs/tasks/phase3-llm-inference.md` but are not
LLM-specific. Every workload obeys them.

1. Deterministic execution. For decoding that means greedy only,
   `do_sample=False`, `num_beams=1`. Sampling makes different cards produce
   different work.
2. Fixed iteration count, not a fixed stopping condition. For decoding,
   `min_new_tokens == max_new_tokens`, otherwise EOS can stop one card early.
3. Explicit TF32 control. Ampere and later can silently route FP32 matmuls
   through tensor cores; a GTX 1080 Ti cannot. Left to PyTorch defaults, two
   cards compute different arithmetic and the comparison is meaningless. Set
   `torch.backends.cuda.matmul.allow_tf32` and `cudnn.allow_tf32` explicitly.
4. Inputs pre-staged before the timed region, with pinned versions and
   revisions. No network access during measurement.
5. Warmup iterations before timing, excluded from the measurement.
6. `torch.cuda.synchronize()` before starting and before stopping the timer.
7. A `work_hash`: a SHA-256 over the workload output, proving two runs did the
   same work. Compare within a precision mode, not across.

All three workloads emit a `work_hash` as of 2026-08-18, but they do not all
prove the same thing, and the difference matters.

`llm_inference.py` hashes the generated token IDs, so it proves the output was
bit-identical. That works because greedy decoding is an argmax and the
per-token result either matches exactly or diverges visibly.

`resnet_train.py` and `matmul.py` hash the **inputs and the shape of the work**
(seed, dataset indices or input matrices, iteration counts, precision), not the
result. Training and repeated matmul are long chains of float reductions, and
floating point addition is not associative, so results are not bit-identical
across architectures and hashing them would fail for reasons unrelated to
whether the same work was done. Given identical inputs and a fixed iteration
count, the FLOP count and operation sequence are identical on every card, which
is the fixed-work premise the project needs. Both record a result checksum or
loss sequence alongside for divergence diagnosis, never asserted equal.

State that distinction in the methods section. A reviewer will otherwise read
one `work_hash` column and assume all three carry the LLM's guarantee.

The LLM workload's own parameters (model, revision, token counts, prompt) live
in `docs/tasks/phase3-llm-inference.md`, not here. The 500-token gpt2
configuration from the old spec draft is dead: the measured configuration is
gpt2-xl at revision `15ea56dee5df` with 960 new tokens, and the degeneracy
finding below rules out the old one independently.

## Measurement contract

Implemented 2026-08-18 in `measurement/power_monitor.py`, reconciled from the
spec draft and Veda's working implementation. Naming follows the working code;
the spec draft's longer names (`total_energy_joules`, `avg_power_watts`) are not
used.

`PowerMonitor(device_index=0, interval=0.2)`, interval in seconds.
`start()` begins a daemon sampling thread, `stop()` returns a `PowerResult`.

`PowerResult` carries `energy_j`, `avg_power_w`, `peak_power_w`, `min_power_w`,
`n_samples`, `duration_s`, `energy_j_counter`, `readings`, and exposes
`as_dict()` so the CSV writer never reaches into attributes.

- Keep `energy_j_counter`. It reads NVML's hardware energy counter
  (`nvmlDeviceGetTotalEnergyConsumption`, Volta and later) and is an
  independent check on the trapezoidal integral. It is `None` on Pascal, so the
  GTX 1080 Ti has no cross-check and the integral is the only number there.
  Record that asymmetry per run rather than averaging over it.
- Restore `readings`, the timestamped sample list. The working implementation
  discards samples on `stop()`, which makes the per-run power trace CSV
  impossible to write. The trace is also the only way to detect the cached
  reading problem (Yang et al., 2024) after the fact.
- If `nvmlDeviceGetPowerUsage` raises, log a warning and skip that sample. Do
  not let the thread die. A benchmark that loses its power thread mid-run would
  otherwise report success on a truncated trace. The skipped count is recorded
  as `n_failed_power_samples`.

**Unverified.** No measured run has exercised the power monitor yet. The
integral is unit-tested against synthetic samples, but nothing has confirmed on
this cluster that `nvidia-ml-py` resolves
`nvmlDeviceGetTotalEnergyConsumption`, nor that NVML reports sane wattage on
each card. Do that before trusting any `energy_j` figure.

## Library version traps

Established the hard way on 2026-08-11. All of these fail silently, which is
why they are written down.

- transformers 5.x renamed `from_pretrained(torch_dtype=)` to `dtype=`.
  `from_pretrained` takes `**kwargs`, so the old name is not an error. It is
  ignored, the model loads in its checkpoint dtype, and the run records a
  precision it did not use. Select the argument by major version and then
  assert `model.dtype` matches what was requested.
- torch 2.9 deprecates `backends.cuda.matmul.allow_tf32` in favour of
  `fp32_precision`. Set both where both exist, and record the read-back rather
  than the requested value. The image currently has torch 2.5.1+cu121, where
  only `allow_tf32` exists.
- `pynvml` and `nvidia-ml-py` both install a module named `pynvml` and disagree
  on some symbols when both are present. torch itself warns about this on
  import. NVML currently works, but fall back to `nvidia-smi` rather than
  leaving `driver_version` blank, and record which source answered.
- The `cu121` index URL in the Dockerfile is load-bearing, not incidental.
  PyTorch stopped publishing cu121 wheels after 2.5.1, and those wheels still
  carry `sm_61`. Newer CUDA 12.8 builds dropped Pascal. The GTX 1080 Ti works
  because of that URL.
- `requirements.txt` pins `transformers==5.15.0`, the version that produced the
  verified `work_hash`. The Dockerfile does not install from it, so the pin does
  not currently reach the image. Closing that gap is part of the Dockerfile work
  below.

## Container builds

kaniko caching is disabled deliberately in `.gitlab-ci.yml`. Do not re-enable
it. With `--cache=true` kaniko pushes the whole image twice, once to the cache
repo and once to the tag. At this image size (7.71 GB, single push measured at
16 minutes) that stalls the shared Nautilus registry: pipeline 86061 applied
every layer, finished its cache push, then went silent for 20+ minutes and had
to be force-cancelled. With caching off the same build completes in 22 minutes.

`:latest` is only tagged on the default branch, so a feature branch build cannot
become the image everyone pulls. Reference images by digest, not tag. Note
`CI_COMMIT_SHORT_SHA` is 8 characters, while `git rev-parse --short` gives 7;
read the tag off the registry rather than deriving it locally.

Any Dockerfile change triggers a 22 minute build, and on `main` it moves
`:latest`. `k8s/interactive.yaml` and `k8s/sample_job.yaml` both pull `:latest`,
so they change underneath anyone using them. Change the Dockerfile
deliberately, not as a side effect of another commit.

## Established results

Phase 3 fixed-work premise, verified 2026-08-11: `work_hash` was bit-identical
across a GTX 1080 Ti (sm_61, driver 580.159.04) and an L4 (sm_89, driver
595.71.05) in fp32, gpt2 at revision 607a30d783df, batch 1, 32 tokens.
Cross-architecture agreement was a hypothesis rather than a guarantee, since
greedy decoding is an argmax and floating point addition is not associative. It
holds at this length. Records in `data/raw/llm_smoke/`.

Not yet established: agreement between two different cards of the same model
(both L4 runs used one physical GPU), and agreement at the full 500-token
length, where a single flipped argmax poisons everything after it.

**Batch-1 decode of a large model is memory-bandwidth bound. Runtime tracks
bandwidth, not architecture generation.** On gpt2-xl the 1080 Ti is within 5% of
an L4 (34.40 s against 32.87 s), while the L4 and L40S share an architecture and
differ by 2.13x (32.87 s against 15.45 s). Bandwidth ordering predicts runtime
ordering; sm version does not. Peak bandwidth figures are verified for L4
(300 GB/s) and L40S (864 GB/s) from NVIDIA product pages; the 1080 Ti's
484 GB/s and the 4090's 1008 GB/s are derived, not published, and need a
citable primary source before use in the paper. At batch 1 gpt2-xl streams all
6.43 GB of weights per token, and the 1080 Ti's 484 GB/s published bandwidth
beats the L4's 300 GB/s. If the replacement case for this workload holds, it
rests on power draw (250 W against 72 W published TDP), not on speed. That is a
hypothesis until Phase 4 measures it.

Corollary: do not scale one GPU's model-to-model cost ratio onto another card.
Doing exactly that predicted the L4 at 14 s against an actual 32.87 s, because
gpt2 and gpt2-xl sit in different regimes (compute bound against bandwidth
bound).

Greedy decoding degenerates with length. `distinct_token_ratio` is 0.938 at 16
tokens, 0.562 at 32, and 0.019 at 960. A run can pass `work_hash`, report
success, and still be measuring a KV-cache loop rather than inference. Validate
workload content separately from workload reproducibility.

The image that produced all of the above was built from the pre-2026-08-13
Dockerfile: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` with the cu121 index URL.
Any image change has to preserve that or the 1080 Ti results are not
reproducible.

## Workload sizing

Settled in `docs/tasks/phase3-workload-sizing.md`, by measurement.

Sizing is set by the **fastest** card in the sweep, not the slowest, because
the work must be identical fleet-wide and the fastest card must still clear the
30 second floor. Longer generations cannot reach that floor: `n_positions` is
1024 for both gpt2 and gpt2-xl, and the L40S runs the 960 token ceiling in
15.45 s, half the floor. **Repetition inside the timed region is the only route
that reaches the floor uniformly.** Degeneracy argues the same way, since short
generations stay non-degenerate while long ones collapse to a KV-cache loop.

That design measures throughput inference, N independent decodes of the same
prompt. It does not measure single-request latency energy and says nothing
about long-context behaviour. The paper has to name which question it answers.

Run duration decisions:

- **No upper bound on run length.** The old "keep runs under 5 minutes" rule is
  withdrawn. It conflicts with the chosen design twice: the oldest cards run
  roughly 300 s of timed region plus equal warmup, and the sweep runs as one
  pod looping internally over all points because the 7.71 GB image pull
  dominates otherwise.
- **Durability replaces the ceiling.** The ceiling was a proxy for not losing
  work to preemption. `runner.py` writes each repetition's row to the PVC as it
  completes, so a preemption costs one repetition.
- **Warmup stays full length until measured.** It is half the compute budget
  and shortening it would save roughly 40% of GPU time, but warmup exists to
  reach thermal and clock steady state, energy is the dependent variable, and
  old and new cards ramp clocks differently. Add a warmup-length axis to the
  sizing sweep rather than cutting it on an assumption.
- **5 repetitions, not 3.** Failed runs are kept with an exclusion reason, so
  effective n is below nominal n; starting at 3 and excluding one leaves a
  standard deviation from n=2. Co-tenant thermal interference on shared nodes
  is an uncontrolled variance source arguing for more samples. The only thing 3
  buys is wall clock, which is the cheap resource here: the sweep is
  courtesy-serialized rather than capacity-limited and blocks nobody.

## Output contract

`measurement/runner.py` owns all result writes. Benchmarks return a dict and
touch no files. `resnet_train.py` already works this way and is the model to
follow; `matmul_benchmark.py` writes its own CSV and needs changing.

This resolves a three-way conflict. The spec draft had `runner.py` own the file
and write energy at write time; `matmul_benchmark.py` writes its own CSV with
`energy_j` blank for Phase 4 to fill later, under a header comment calling it a
"SHARED CONTRACT with Arav/Aidan" that was never agreed to; `llm_inference.py`
emits JSON instead. Either the benchmark owns the file or the runner does, and
it is the runner.

Still open, and needs Veda and Aidan rather than a unilateral decision: the
exact column set. One wide CSV goes sparse across workloads with different
natural fields (`n`/`iters`/`total_flops` against
`batches`/`batch_size`/`final_loss` against
`work_hash`/`model_revision`/`max_new_tokens`). The likely shape is a shared CSV
carrying run identity, energy, runtime and provenance, plus a per-workload JSON
sidecar, which also preserves the existing `data/raw/llm_smoke/*.json` format
rather than forcing a re-run. Whatever is chosen must satisfy the runtime
provenance rule above: GPU model, node name and driver version observed from
inside the pod. The `matmul_benchmark.py` schema currently cannot.

## Embodied carbon and grid intensity

**All figures in this section are unsourced pending citation.** They came from a
spec draft with no citation attached. Treat them as placeholders and source them
individually before use, in the style `paper/methods-notes.md` used for the
bandwidth figures. Ranges, not point values.

Sources to work from: the ACT model (Gupta et al., 2022) for die size, process
node and fab characteristics; vendor product carbon footprint reports (Dell, HP,
Lenovo), which are whole-system and have to be worked backwards to a GPU
contribution; die sizes from techpowerup and anandtech die shots.

Expected range per GPU: 50 to 400 kg CO2e depending on generation, die size and
memory.

Grid intensity presets, kg CO2 per kWh: CAISO approx 0.200, US national average
approx 0.390, ERCOT approx 0.400, PJM approx 0.550.

## Break-even model

Planned. `analysis/` is empty; these are targets, not code.

Core inequality: replacement is worthwhile when
`embodied_new < (energy_per_job_old - energy_per_job_new) * expected_jobs * grid_intensity`.

Planned entry points in `analysis/carbon_model.py`: `break_even_jobs`,
`break_even_hours_per_year`, `payback_curve`.

## Related work

1. Gupta et al. (2022), ACT carbon modeling tool. Embodied carbon methodology.
2. Yang et al. (2024), nvidia-smi power sensor accuracy. Source of the 30 second
   floor and the cached-reading limitation.
3. Uwizeyimana and Jerger (2025), carbon-aware replacement theory. The research
   question.
4. Li et al. (2023), HPC carbon footprint estimation. Broader context.
5. Nguyen et al. (2025), T4 against RTX6000 Ada carbon comparison. Closest prior
   work, modelled rather than measured.
6. Fadel Argerich et al. (2026), Watt Counts energy benchmark across 10 GPUs.
   Similar methodology, no replacement analysis.

## Pitfalls

Sync, warmup, and pre-staged inputs are covered by the correctness requirements
above and are not repeated here.

- pynvml requires paired `nvmlInit()` and `nvmlShutdown()`.
- Old consumer GPUs may quantise power readings in coarse steps, for example
  25 W. Log the observed granularity per GPU model and report it.
- CIFAR-10 and model weights must be pre-staged or cached on the PVC. A download
  inside a timed region invalidates the run.

## Implementation order

From the original spec draft, kept as written. It describes intended build
order. The state table below is the authority on what is actually done, and the
two disagree: `power_monitor.py` at step 2 and `runner.py` at step 4 are both
still stubs, while `llm_inference.py` at step 7 is complete and measured.

1. `k8s/inventory.sh`, verify you can query the cluster and see GPU models
2. `measurement/power_monitor.py`, test pynvml works in a basic pod
3. `benchmarks/matmul.py`, simplest benchmark, validates the pipeline
4. `measurement/runner.py`, integrate power monitor and benchmark and CSV output
5. `Dockerfile`, `k8s/benchmark-pod.yaml`, `k8s/pvc.yaml`, containerize and run
6. `benchmarks/resnet_train.py`, second benchmark
7. `benchmarks/llm_inference.py`, third benchmark
8. Run full sweep across all GPU models, 5 to 10 reps each
9. `analysis/carbon_model.py`, break-even math
10. `analysis/sensitivity.py` and `analysis/plots.py`, charts and sensitivity

## Current state

Verified against the repo on 2026-08-18. Phase numbers follow `docs/phases.md`.

| Phase | Status | Evidence |
|---|---|---|
| 1 Census | Done | `data/processed/census_fleet.csv`, `census_nodes.csv`, `k8s/inventory.sh`, `k8s/summarize_census.py`. Task doc is `docs/tasks/phase0-census.md`; the filename says phase0 but the work is Phase 1. |
| 2 Container | Reverted to cu121, unbuilt | `Dockerfile` back to `nvidia/cuda:12.1.0-runtime-ubuntu22.04` with pinned `torch==2.5.1` / `torchvision==0.20.1`, installing from `requirements.txt`, copying both packages. **Not yet built or pulled.** |
| 3 Workloads | Written, 1 of 3 measured | All three emit `work_hash` and set TF32 explicitly. LLM measured, `data/raw/llm_smoke/`, 7 runs across 3 GPU models. ResNet and matmul have never run on a GPU. |
| 4 Power | Written, unmeasured | `measurement/power_monitor.py` and `measurement/runner.py` implemented. Integral unit-tested against synthetic samples. **No GPU has been sampled yet.** |
| 5 Storage | Ad hoc | Three PVC yamls: `results-pvc.yaml`, `k8s/sample_pvc.yaml`, `k8s/aidan-llm-models-pvc.yaml`. No canonical one, and `matmul-results` lacks the required `aidan` prefix. |
| 6 to 10 | Not started | `analysis/` is `.gitkeep` only. |

GPU workloads are no longer restricted. The earlier "read-only census, do not
launch GPU workloads" rule is withdrawn; workloads ran on 2026-08-11.

The root duplicates are resolved: `matmul_benchmark.py` and `power_monitor.py`
moved into `benchmarks/matmul.py` and `measurement/power_monitor.py`, and the
root copies deleted.

### What is still unverified

Nothing in Phase 2, 3 or 4 above has run on hardware since the 2026-08-18
changes. Specifically not yet done, and not to be described as working until it
is:

- The image has not been built. The Dockerfile is a revert to a form that built
  before, plus pinned versions and a `requirements.txt` install, so it is
  plausible rather than proven.
- No GPU has been sampled. `energy_j`, `energy_j_counter` and every power column
  are untested against real hardware.
- `nvidia-ml-py` has not been confirmed to resolve
  `nvmlDeviceGetTotalEnergyConsumption` on this cluster.
- `resnet_train.py` and `matmul.py` have never run on a GPU at all. They have
  only been syntax and lint checked.
- The 1080 Ti has not been re-checked against the pinned wheels. `sm_61` support
  is the reason for the cu121 pin and is the thing most worth confirming first.

### Next: k8s plumbing, which is not done

`k8s/benchmark-pod.yaml` is still a one-line stub, so nothing can be scheduled
yet. A working pod spec needs two PVCs mounted, not one: results, plus the
pre-staged model cache for the LLM workload
(`k8s/aidan-llm-models-pvc.yaml`, staged by `aidan-llm-prep-job.yaml`). ResNet
separately needs `/results/data/cifar10` writable, since it downloads CIFAR-10
during setup. It should also set `NODE_NAME` from the downward API, which is
where `runner.py` reads the node name from.
