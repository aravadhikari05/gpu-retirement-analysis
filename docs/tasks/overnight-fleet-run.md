# Overnight run plan: branch verification, then first fleet pass

Written 2026-08-23 to be executed unattended overnight. This file is the plan of
record. If it is being read the morning after, jump to "What to check first".

## What this is and is not

It is the **first fleet pass**: 5 repetitions of all three workloads on three GPU
models, with power attached and energy scoped to the timed region.

It is **not** Phase 6 as `docs/tasks/phase6-fleet-selection.md` scopes it. Three
of that file's preconditions are unmet, and every row produced here inherits
that:

1. The fleet framing has not been confirmed with Prof. Jullig.
2. Same-model variance is unmeasured, so no cross-model difference in these rows
   can be interpreted yet.
3. The RTX 3090 is excluded by decision, so the fastest card in the sweep is
   absent and the sizing constants remain justified by a card that is not in it.

Treat the output as a real dataset that may need re-running, not as the result.

## Preconditions, all verified 2026-08-23 unless noted

- Preflight has run on all three target cards: 1080 Ti, 2080 Ti, A4000. This is
  the first time the sweep fleet has been fully preflighted before running.
- CIFAR-10 is staged at `/results/data/cifar10` on the `matmul-results` PVC.
- The Hugging Face cache is staged at `/models/hf` on `aidan-llm-models-pvc`,
  6.5 GB, holding gpt2 and gpt2-xl at their pinned revisions.
- **Not met until Stage 0 completes:** an image containing the branch code.

## Stage 0: wait for the image

The branch head is `3b5198c`, but that commit touches only `k8s/`, which is not
in the workflow's path filter. The image to use is therefore the one built from
**`a7fecce`**, the last commit touching `benchmarks/`. Poll GHCR for
`sha-a7feccefd52a72dfdd8edfa238b3c906b6e8a960` and resolve it to a digest.

Record `GIT_COMMIT=3b5198c...` (what the tree is) and `IMAGE=<a7fecce digest>`
(what actually runs). They differ by a pod spec and no code. State it rather
than pretend they match.

## Stage 1: verification, 1 repetition, GTX 1080 Ti

**Nothing on this branch has ever executed.** The enforced `WorkloadResult`, the
two idle-power windows, the new matmul and resnet sizing defaults and the LLM's
8 generations per timed region are all unverified on hardware. Stage 1 costs
about 15 minutes and Stage 2 costs about 3.3 GPU-hours, so the gate pays for
itself the first time it catches anything.

Two jobs, not one, so an LLM failure cannot take matmul and resnet with it:

- `arav-verify-mm-rn-1080ti`: matmul 1 rep, then resnet 1 rep. Needs 8 Gi.
- `arav-verify-llm-1080ti`: llm 1 rep. Needs 16 Gi, see the memory note below.

### Gate criteria, checked against the written rows and not the exit code

A job can exit zero and still have produced a useless row.

| Check | Required |
|---|---|
| `power_window` | `region` on every row |
| `below_30s_floor` | false on every row |
| idle fields | all 11 populated, `idle_skip_reason` empty |
| `idle_post_context_avg_w` | greater than `idle_pre_context_avg_w` |
| `work_hash_kind` | `config` on matmul and resnet, `output` on llm |
| `iterations_identical` | true on llm |
| `image_ref`, `git_commit` | both non-empty |
| `energy_j` against `energy_j_counter` | recorded per card, not compared to another card's |

Runtimes are informational, not gates: matmul about 232 s, resnet about 195 s,
llm about 275 s. All three are predictions from 3090 rates and none is measured.
A large miss means the sizing needs redoing from 1080 Ti numbers, but a run that
is merely longer than predicted is still valid.

## Stage 2: fleet, 5 repetitions

Only if Stage 1 passes. Per card, two jobs, launched together across cards since
they contend for nothing:

| Job | Contents | Approx duration on the 1080 Ti |
|---|---|---|
| A | matmul 5 reps, then resnet 5 reps | about 40 min |
| B | llm 5 reps | about 26 min |

Cards: **GTX 1080 Ti, RTX 2080 Ti, RTX A4000.** No 3090.

Five repetitions rather than three, per CLAUDE.md: failed runs are kept with an
exclusion reason, so effective n falls below nominal n, and co-tenant thermal
interference on shared nodes is an uncontrolled variance source.

## Resource sizing, decided per card at launch and not copied

`python3 k8s/nrp_availability.py --model <product>` reports free CPU per node.
Free GPUs and free RAM are not correlated on this cluster: a 1080 Ti node was
observed with 39 CPUs free and 5.6 Gi of RAM.

- matmul and resnet: 8 Gi is known sufficient, 6 CPU is comfortable.
- llm: **16 Gi floor.** gpt2-xl is a 6.43 GB fp32 checkpoint staged through host
  RAM before it reaches the device. If no open node of that model has 16 Gi
  free, **do not shrink the request.** Leave the LLM job queued for that card
  and let the other two workloads proceed.

The `/dev/shm` tmpfs counts against the container memory limit, so the 2 Gi
emptyDir is inside the 8 or 16 Gi, not on top of it.

## Decisions taken before execution, 2026-08-23

Four questions were put to Arav before this ran. The answers are binding on
whoever executes it, and the reasoning is recorded so a later reader does not
relitigate them.

- **Stage 1 failure: stop, leave the cluster clean.** No retry, no proceeding.
- **Node pinning on the 1080 Ti: no.** Letting the scheduler place the pods is
  simpler and more likely to place promptly overnight. The consequence is that
  all repetitions may land on one physical card, so this run does **not**
  advance the same-model variance study and may well reproduce the n=1-per-model
  situation that already exists.
- **Results: pull and review, do not commit.** Rows are copied into
  `data/raw/runs/` and reported. Nothing is committed until a human has seen the
  gate results.
- **No other work overnight.** No unreviewed code or documentation changes land
  while nobody is awake. Phase 7 sourcing and the ruff config stay unstarted.

## Failure policy while nobody is watching

- **Stage 1 fails on any gate: stop.** Do not start Stage 2. Leave the cluster
  clean and write up what failed. No retry: a transient failure and a real one
  look the same at 2am, and the cost of guessing wrong is a dataset built on
  broken code.
- **Stage 1 LLM fails but matmul and resnet pass:** start Stage 2 job A on all
  three cards, hold every job B. This is the case the split exists for.
- **A Stage 2 job fails mid-run:** keep going. The runner flushes each
  repetition as it completes, so partial results survive, and a failed
  repetition is written with an `exclusion_reason` rather than lost.
- **A card is unschedulable:** leave the job queued. Pending costs nothing. Do
  not shrink resource requests to force placement.
- **Nothing exceeds its deadline.** Every job carries `activeDeadlineSeconds` so
  no pod holds a GPU past the window, and `ttlSecondsAfterFinished` is long
  enough that logs survive until morning.

## What to check first, the morning after

1. `kubectl -n cmpm118 get jobs` and `get pods`. Anything still Running or
   Pending, and anything that failed.
2. Pull `runs.jsonl` off the PVC into `data/raw/runs/` and diff the row count.
3. Re-run the Stage 1 gate table against every row, not just the Stage 1 ones.
4. `energy_j` against `energy_j_counter` per card. The 2080 Ti is expected to
   disagree by 6 to 7%; that is known and not a new failure.
5. Delete every completed job. TTL should have done it; verify.

## Explicitly out of scope tonight

- The RTX 3090, by decision.
- Same-model variance, which needs 5 distinct `gpu_uuid` values of one model and
  is a separate study. Node pinning that would have made partial progress on it
  was considered and declined in favour of scheduling reliability, so expect
  these rows to be one physical card per model again.
- Merging `schema-idle-sizing`. The branch merges after its output is reviewed,
  not because the pods exited zero.
- Committing results. Rows get pulled and reviewed first.
