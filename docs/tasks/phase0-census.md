# Task: Nautilus GPU fleet census

Implements the stub at k8s/inventory.sh. This is step 1 of the README flow.

## Objective

Produce a reproducible read-only census of GPU models on Nautilus, to decide
which GPU generations have enough coverage to benchmark.

## Constraints

- Read-only. No pods, jobs, or PVCs created.
- Large output goes to a file, never the terminal.
- Do not assume the GPU product label key. Discover it from cluster output.
- Snapshots land in data/raw/ and are gitignored. Summaries go to
  data/processed/ and are committed.

## Deliverables

1. `k8s/inventory.sh` takes the snapshot and invokes the summarizer.
2. A Python summarizer that reads a snapshot path, with optional label overrides.
3. Per GPU model: node count, ready nodes, openly schedulable nodes (ready, not
   cordoned, no NoSchedule or NoExecute taint), allocatable GPUs, distinct sites,
   taint variants.
4. Flag nodes with GPU capacity but no product label.
5. Flag models with fewer than three openly schedulable nodes as a coverage risk.
6. Write data/processed/census_nodes.csv and data/processed/census_fleet.csv.

## Definition of done

Runs against a real snapshot, prints a readable table, and label assumptions are
verified against `kubectl describe node` for at least one node.

## Note

README describes this as kubectl + jq. Using Python for the summarization step
because label coverage and taint parsing are unreadable in jq, and CSV output is
needed downstream. The shell script remains the entry point.
