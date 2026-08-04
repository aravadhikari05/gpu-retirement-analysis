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

## Working style

Before writing substantial code: restate the task, inspect relevant files,
propose a plan, name the assumptions, wait for approval on architectural
changes. Then implement in small reviewable steps, showing exact commands and
expected output.

## Current phase

Phase 0: read-only GPU fleet census. Do not launch GPU workloads yet.
Active task: docs/tasks/phase0-census.md
