# Benchmark Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Docker image containing PyTorch, pynvml, and (stubbed) benchmark/measurement scripts, and wire up GitHub Actions to build and push it to GHCR automatically.

**Architecture:** `benchmarks/` and `measurement/` become empty-but-importable Python packages. The Dockerfile installs CUDA 12.1-compatible PyTorch + pynvml + pandas + transformers, copies both packages in, and sets `python -m measurement.runner` as the entrypoint. A GitHub Actions workflow builds and pushes the image to `ghcr.io/aravadhikari05/gpu-retirement-analysis` on every push to `main` that touches the Dockerfile or either package.

**Tech Stack:** Docker, NVIDIA CUDA base image, PyTorch (cu121 wheels), pynvml, GitHub Actions (`docker/login-action`, `docker/build-push-action`).

Reference spec: `docs/superpowers/specs/2026-08-03-benchmark-container-design.md`

---

### Task 1: Package scaffolding (stub files)

**Goal:** Create importable `benchmarks` and `measurement` packages with placeholder files, so the Dockerfile has something real to COPY and the entrypoint module resolves.

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/resnet_train.py`
- Create: `benchmarks/llm_inference.py`
- Create: `benchmarks/matmul.py`
- Create: `measurement/__init__.py`
- Create: `measurement/power_monitor.py`
- Create: `measurement/runner.py`

**Acceptance Criteria:**
- [ ] Both packages exist with `__init__.py`
- [ ] Each placeholder file is a valid (empty-bodied) Python module — no syntax errors
- [ ] `python3 -c "import benchmarks, measurement"` succeeds from repo root

**Verify:** `python3 -m py_compile benchmarks/*.py measurement/*.py && python3 -c "import benchmarks, measurement" && echo OK` → `OK`

**Steps:**

- [ ] **Step 1: Create the packages and placeholder files**

```bash
mkdir -p benchmarks measurement
touch benchmarks/__init__.py measurement/__init__.py
```

`benchmarks/resnet_train.py`:
```python
# Benchmark 1: ResNet-50 training. Implemented in a later phase.
```

`benchmarks/llm_inference.py`:
```python
# Benchmark 2: GPT-2 token generation. Implemented in a later phase.
```

`benchmarks/matmul.py`:
```python
# Benchmark 3: matrix multiplication. Implemented in a later phase.
```

`measurement/power_monitor.py`:
```python
# PowerMonitor: pynvml-based power sampling. Implemented in a later phase.
```

`measurement/runner.py`:
```python
# Benchmark + power monitor orchestrator. Implemented in a later phase.
```

- [ ] **Step 2: Verify imports work**

Run: `python3 -m py_compile benchmarks/*.py measurement/*.py && python3 -c "import benchmarks, measurement" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add benchmarks measurement
git commit -m "Add stub benchmarks/measurement packages for container build"
```

---

### Task 2: Dockerfile + .dockerignore

**Goal:** A Dockerfile that builds successfully from the stub packages, matching the CLAUDE.md container spec.

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Acceptance Criteria:**
- [ ] `docker build` completes without error
- [ ] Image contains `python3`, `torch`, `torchvision`, `transformers`, `pynvml`, `pandas`
- [ ] `.dockerignore` excludes `.git`, `data/`, `results/`, `docs/`

**Verify:** `docker build -t gpu-retirement-analysis:test .` → build succeeds (`Successfully tagged` / final layer `exporting to image` with no errors)

**Steps:**

- [ ] **Step 1: Write `.dockerignore`**

```
.git
data/
results/
docs/
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers pynvml pandas

COPY benchmarks/ /app/benchmarks/
COPY measurement/ /app/measurement/

WORKDIR /app

ENTRYPOINT ["python3", "-m", "measurement.runner"]
```

- [ ] **Step 3: Build locally to verify**

Run: `docker build -t gpu-retirement-analysis:test .`
Expected: build completes, ends with `naming to docker.io/library/gpu-retirement-analysis:test` (or equivalent "Successfully tagged" on older Docker) and no error lines.

If Docker is not installed/running locally, note that and skip to Task 3 — CI will validate the build on push.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "Add benchmark container Dockerfile"
```

---

### Task 3: GitHub Actions build + push to GHCR

**Goal:** Automatically build and push the image to `ghcr.io/aravadhikari05/gpu-retirement-analysis` whenever the Dockerfile or either package changes on `main`.

**Files:**
- Create: `.github/workflows/docker.yml`

**Acceptance Criteria:**
- [ ] Workflow triggers on push to `main` for paths `Dockerfile`, `benchmarks/**`, `measurement/**`
- [ ] Workflow logs into `ghcr.io` using `GITHUB_TOKEN` (no manual PAT/secret needed)
- [ ] Workflow pushes both a `latest` tag and a `sha-<short-sha>` tag
- [ ] Workflow has `packages: write` permission declared

**Verify:** After pushing this file to `main`, the "Actions" tab shows a green run for "Build and Push Container", and `ghcr.io/aravadhikari05/gpu-retirement-analysis:latest` appears under the repo's Packages page.

**Steps:**

- [ ] **Step 1: Write the workflow**

`.github/workflows/docker.yml`:
```yaml
name: Build and Push Container

on:
  push:
    branches: [main]
    paths:
      - "Dockerfile"
      - "benchmarks/**"
      - "measurement/**"

permissions:
  contents: read
  packages: write

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ghcr.io/aravadhikari05/gpu-retirement-analysis:latest
            ghcr.io/aravadhikari05/gpu-retirement-analysis:sha-${{ github.sha }}
```

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/docker.yml
git commit -m "Add CI workflow to build and push container to GHCR"
git push
```

- [ ] **Step 3: Verify the run**

Check the repo's "Actions" tab for a passing "Build and Push Container" run, then check the repo's "Packages" sidebar (or `https://github.com/aravadhikari05?tab=packages`) for `gpu-retirement-analysis:latest`.

---

## Self-Review Notes

- Spec coverage: package scaffolding ✅ (Task 1), Dockerfile/.dockerignore ✅ (Task 2), registry naming ✅ (Task 3 tags), CI build/push ✅ (Task 3). k8s pod/pvc and real benchmark logic explicitly out of scope per spec — not included here.
- No placeholders/TBDs in any step; all file contents are complete.
- Entrypoint uses `python3` (matches what's installed via apt) instead of CLAUDE.md's literal `python` to avoid a missing-binary failure on Ubuntu 22.04, which has no unqualified `python` by default.
