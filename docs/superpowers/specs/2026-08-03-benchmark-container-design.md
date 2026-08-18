# Phase 2: Benchmark Container — Design

## Goal

Produce a Docker image containing PyTorch, pynvml, and benchmark scripts, buildable and runnable in Nautilus pods. This phase does not implement benchmark or measurement logic — later phases (Implementation Order steps 2–4, 6–7 in CLAUDE.md) fill in real code. This phase only needs the container pipeline to work end-to-end.

## Components

### Package scaffolding

- `benchmarks/__init__.py`, `measurement/__init__.py` — make both importable as packages.
- Empty placeholder files (no logic yet):
  - `benchmarks/resnet_train.py`
  - `benchmarks/llm_inference.py`
  - `benchmarks/matmul.py`
  - `measurement/power_monitor.py`
  - `measurement/runner.py`

These will be implemented in later phases per CLAUDE.md's Component Specifications. Their presence now lets the Dockerfile COPY and `python -m measurement.runner` entrypoint resolve without error (even though runner.py itself does nothing yet).

### Dockerfile

Base image: `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (per CLAUDE.md; confirmed compatible with both older Pascal/Volta-era cards and newer Hopper/Ada cards on Nautilus — CUDA 12.x is the last major line supporting pre-Turing architectures).

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers pynvml pandas

COPY benchmarks/ /app/benchmarks/
COPY measurement/ /app/measurement/

WORKDIR /app

ENTRYPOINT ["python", "-m", "measurement.runner"]
```

### `.dockerignore`

Excludes `.git`, `data/`, `results/`, `docs/` from the build context — keeps the image lean and avoids baking in raw benchmark data or docs.

### Registry & image name

- Registry: GitHub Container Registry (ghcr.io).
- Image: `ghcr.io/<github-username>/gpu-retirement-analysis`, linked to this repo so teammates with repo write access can also push (package inherits repo collaborator permissions).
- Same image is reused for the whole project across all benchmarks/GPU models — k8s pod specs vary only via env vars/args (GPU_MODEL, BENCHMARK, RUN_NUMBER), not via separate images.

### CI: build + push via GitHub Actions

`.github/workflows/docker.yml`:

- Trigger: push to `main` touching `Dockerfile`, `benchmarks/**`, or `measurement/**`.
- Steps: checkout → `docker/login-action` against `ghcr.io` using the built-in `GITHUB_TOKEN` (no PAT needed) → `docker/build-push-action` to build and push.
- Tags: `latest` and `sha-<short-commit-sha>` (traceability — lets a pod spec pin an exact build if needed).

Rationale: multi-person project (per CLAUDE.md Team section) — CI guarantees the pushed image always matches what's in git, rather than depending on whichever teammate last built locally.

## Out of scope for this phase

- Actual benchmark/measurement implementation (later phases).
- `k8s/benchmark-pod.yaml`, `k8s/pvc.yaml` (Implementation Order step 5, after this).
- CIFAR-10/GPT-2 weight pre-downloading/caching strategy (deferred until benchmarks have real logic).

## Testing

- `docker build .` succeeds locally before relying on CI.
- CI workflow run visible in GitHub Actions tab; confirm image appears under the repo's GHCR packages page.
