# GPU Carbon Payback on NRP Nautilus

Undergraduate research, UC Santa Cruz. Aidan Nguyen, adviser Prof. Jullig.
Collaborator: Arav Adhikari. See README.md for the full structure and flow.

## Hard rules

- Never fabricate numbers. No invented GPU counts, power readings, benchmark
  results, or carbon figures. Label estimated values as estimated.
- Never claim a command succeeded without seeing its actual output.
- No em dashes in any output, code comments included.
- This repo is public. Never commit credentials, kubeconfigs, S3 keys, or tokens.
- Preserve uncertainty. Ranges, not point values, for carbon estimates.
- Shared repo. Read existing files before modifying them. Many files under
  benchmarks/, measurement/, and analysis/ are one-line stubs, not empty.

## Cluster environment

- kubectl context: `nautilus`, namespace: `cmpm118` (shared with other students)
- Requires UCSC VPN
- User-level access only. No admin, no node access, no scheduler changes.
- Prefix every Kubernetes resource name with `aidan`
- Registry: `gitlab-registry.nrp-nautilus.io/aidan/aidan`, built by GitLab CI
  on the `gitlab` remote. GitHub Actions does not build the pod image.

## Conventions

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

Decode cost at batch 1 scales with **layer count, not parameter count**. gpt2-xl
costs 3.58x gpt2 per token (measured, 1080 Ti), against a 12.6x parameter ratio
and a 4.0x layer ratio. Batch-1 decode is latency bound: layers are serial,
width is absorbed in parallel. Any energy model assuming cost tracks parameter
count overstates large models at batch 1 by roughly 3x.

Greedy decoding degenerates with length. `distinct_token_ratio` is 0.938 at 16
tokens, 0.562 at 32, and 0.019 at 960. A run can pass `work_hash`, report
success, and still be measuring a KV-cache loop rather than inference. Validate
workload content separately from workload reproducibility.

Measured facts destined for the paper go in `paper/methods-notes.md`, not only
into task docs.

## Current phase

Phase 0: read-only GPU fleet census. Do not launch GPU workloads yet.
Active task: docs/tasks/phase0-census.md

Phase 3 in progress for LLM inference: docs/tasks/phase3-llm-inference.md.
The Phase 0 restriction above no longer applies to that task.
