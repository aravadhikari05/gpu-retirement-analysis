# Runbook: branch verification, then first fleet pass

Written 2026-08-23 to be executed unattended, possibly by an agent that has
none of the conversation that produced it. It is self-contained on purpose.
Read it start to finish before touching the cluster.

If you are reading this the morning after a run, jump to
"What to check first, the morning after".

---

## 0. Ground rules, which outrank the plan

**Priority order, when anything conflicts: do no damage, then do not waste
shared resources, then complete the run.** Finishing is the least important of
the three. An unfinished run costs a night. A GPU held overnight by a pod
nobody is watching costs another team their work.

1. **Every pod you create, you delete.** Before you stop working for any reason,
   run `kubectl --context nautilus -n cmpm118 get pods,jobs` and confirm nothing
   of yours is left. "Of yours" means anything prefixed `arav-`. Pods named
   `duckdb`, `ollama-jessica` and anything without an owner prefix belong to
   other students in this shared namespace: **never delete them**, even when
   they are in `Error` or have been idle for days.
2. **Every job carries `activeDeadlineSeconds` and `ttlSecondsAfterFinished`.**
   The deadline is what guarantees a GPU is released even if the client dies,
   the VPN drops, or the agent is killed. Never launch a GPU pod without one.
3. **Do not probe the cluster by launching pods when a query will do.**
   `python3 k8s/nrp_availability.py` answers "what is free" from a public NRP
   endpoint at zero cost. Launching one throwaway pod per GPU model to find out
   is a last resort, was done once on 2026-08-23, and should not be repeated
   casually.
4. **Never add a toleration for `nautilus.io/reservation=*`.** Those taints
   fence other institutions' hardware. If a card is only reachable by tolerating
   someone's reservation, it is not reachable.
5. **Never re-run a stage that already passed.** Rows are appended to a shared
   `runs.jsonl` on a PVC. Duplicates are not free: they distort any aggregate
   that does not group correctly, and they cost GPU time.
6. **One card per job, one job per workload group.** Do not request more than
   one GPU. Nothing in this plan needs it.

---

## 1. Context a cold reader needs

### The project

GPU carbon payback on the NRP Nautilus cluster. Undergraduate research, UC Santa
Cruz. The question is whether replacing an old GPU with a newer one pays back
the carbon embodied in manufacturing the new one, given the energy saved per
unit of work. `CLAUDE.md` at the repo root is authoritative for conventions,
decisions and measured findings. Read it before making any judgement call this
file does not cover.

### The cluster

- Context `nautilus`, namespace `cmpm118`, shared with students outside this
  project. Requires the UCSC VPN.
- User-level access only. `kubectl get nodes` (list) works; `kubectl get node
  <name>` (get) is Forbidden, as is listing pods cluster-wide. This is why
  `k8s/nrp_availability.py` exists.
- Every resource name must be prefixed with an owner: `arav-`, `aidan-`,
  `veda-`.
- Namespace quota bans A100, H100, H200 and GH200 outright.

### The two PVCs

| Claim | Mount | Holds |
|---|---|---|
| `matmul-results` | `/results` | `runs.jsonl`, power traces, and staged CIFAR-10 at `/results/data/cifar10` |
| `aidan-llm-models-pvc` | `/models` | 6.5 GB Hugging Face cache, gpt2 and gpt2-xl at pinned revisions |

Both already contain their data. Nothing in this plan stages anything.

### What is being verified and why it is risky

Branch `schema-idle-sizing` (head `3b5198c`, pushed) contains four changes that
**have never executed on any GPU**:

1. **Enforced `WorkloadResult`** (`benchmarks/_result.py`). Required fields are
   constructor arguments. A benchmark that omits one now **raises** where it
   previously wrote a null. This is the change most likely to convert a silent
   gap into a hard pod failure.
2. **Idle power** (`measure_idle()` in `measurement/runner.py`). Two 60 s
   windows before the first repetition, split by CUDA context creation:
   `idle_pre_context` (no context in the process) and `idle_post_context` (a pod
   holding a GPU it is not using). 11 new row fields.
3. **New sizing defaults.** matmul `DEFAULT_ITERS` 200 to 2000; resnet
   `NUM_BATCHES` became a `run()` kwarg defaulting to 1000. Both derived from
   measured RTX 3090 per-iteration rates.
4. **LLM repetition.** `llm_inference` now runs 8 `generate()` calls per timed
   region and defaults to gpt2-xl at 960 tokens, replacing defaults that pointed
   at a superseded configuration.

Local verification only: `ruff check` clean, 21 unit tests pass. torch is not
installed on the workstation, so no benchmark, no runner and no NVML path has
been executed anywhere.

### Why matmul and resnet run separately from the LLM

The LLM is the highest-risk leg: it is the only workload that needs the model
PVC, the only one needing 16 Gi of host RAM, and its 8-iteration loop is new. If
it fails inside a combined job it takes the other two workloads' results with
it. Splitting costs one extra pod per card and nothing else.

---

## 2. What this run is, and is not

It is the **first fleet pass**: 5 repetitions of three workloads on three GPU
models, power attached, energy scoped to the timed region.

It is **not** Phase 6 as `docs/tasks/phase6-fleet-selection.md` scopes it. Three
preconditions are unmet and every row inherits that:

1. The fleet framing has not been confirmed with Prof. Jullig. The project was
   framed around modern datacenter GPUs; those are unreachable, so the fleet is
   a consumer and workstation line.
2. Same-model variance is unmeasured, so **no cross-model difference in these
   rows can be interpreted yet**.
3. The RTX 3090 is excluded by decision, so the card whose measured rates
   justify the sizing constants is absent from the run that uses them.

Treat the output as a real dataset that may need re-running.

---

## 3. Decisions already taken, 2026-08-23

Binding. Recorded so a later reader does not relitigate them.

- **Stage 1 failure: stop.** See section 6 for the one exception added later.
- **No node pinning.** The scheduler places pods. Consequence: repetitions may
  all land on one physical card per model, so this run does **not** advance the
  same-model variance study.
- **Pull results, do not commit.** Copy rows into `data/raw/runs/` and report.
  A human reviews before anything is committed.
- **No other work overnight.** No unreviewed code or documentation changes.

---

## 4. Stage 0: resolve the image

The branch head `3b5198c` touches only `k8s/`, which is not in the CI path
filter (`Dockerfile`, `benchmarks/`, `measurement/`). The image to run is built
from **`a7fecce`**, the last commit touching `benchmarks/`.

```
# poll until present, then resolve to a digest
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:aravadhikari05/gpu-retirement-analysis:pull" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://ghcr.io/v2/aravadhikari05/gpu-retirement-analysis/tags/list"
# expect: sha-a7feccefd52a72dfdd8edfa238b3c906b6e8a960
curl -s -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.v2+json" \
  -D- -o /dev/null \
  "https://ghcr.io/v2/aravadhikari05/gpu-retirement-analysis/manifests/sha-a7fecce..." \
  | awk '/docker-content-digest/{print $2}'
```

A build takes about 22 minutes; kaniko caching is disabled deliberately and must
not be re-enabled. **Pin the digest, never a tag.** Set `GIT_COMMIT=3b5198c...`
(what the tree is) and `IMAGE=<digest>` (what runs). They differ by a pod spec
and no code; state that rather than pretend they match.

If the build failed rather than being slow, stop and report. Do not merge to
`main` to force a build: that moves `:latest`, which `k8s/interactive.yaml` and
`k8s/sample_job.yaml` both pull.

---

## 5. Stage 1: verification, 1 repetition, GTX 1080 Ti

Two jobs from `k8s/arav-verify-job.yaml`, rendered with `envsubst`. That
template runs all three workloads in one pod; **split it** so the LLM is
separate. Read its header comment first: it documents the placeholders and the
memory reasoning.

- `arav-verify-mm-rn-1080ti`: matmul 1 rep, then resnet 1 rep. 8 Gi, 6 CPU.
- `arav-verify-llm-1080ti`: llm 1 rep. 16 Gi.

Roughly 15 minutes total. Stage 2 costs about 3.3 GPU-hours, so this gate pays
for itself the first time it catches anything.

### Sizing the requests

Run `python3 k8s/nrp_availability.py --model NVIDIA-GeForce-GTX-1080-Ti` first.
Free GPUs and free RAM are **not** correlated here: a 1080 Ti node was observed
with 39 CPUs free and 5.6 Gi of RAM. If a pod stays Pending, the scheduler
message is truncated at 1024 characters and hides the real clause; pin
`kubernetes.io/hostname` to one candidate node to get a readable reason.

`/dev/shm` is a 2 Gi tmpfs that counts **against** the container memory limit,
not on top of it. resnet needs it because its DataLoader uses `num_workers=2`
and the k8s default shm of 64 MB is too small for 224x224 batches.

### Gate criteria, checked against the written rows

A job can exit zero and still write a useless row. Check the rows.

| Check | Required |
|---|---|
| `power_window` | `region` on every row |
| `below_30s_floor` | false on every row |
| idle fields | all 11 populated, `idle_skip_reason` empty |
| `idle_post_context_avg_w` | greater than `idle_pre_context_avg_w` |
| `work_hash_kind` | `config` on matmul and resnet, `output` on llm |
| `iterations_identical` | true on llm |
| `image_ref`, `git_commit` | both non-empty |
| `energy_j` vs `energy_j_counter` | recorded per card, never compared across cards |

Runtimes are informational, not gates: matmul about 232 s, resnet about 195 s,
llm about 275 s, all predicted from 3090 rates and none measured. A run longer
than predicted is still valid. A large miss means the sizing should be redone
from 1080 Ti numbers, which is a decision for a human, not a fix to apply.

Known and **not** a failure: the RTX 2080 Ti disagrees between `energy_j` and
`energy_j_counter` by 6 to 7%. Confirmed three times on that card. The 1080 Ti
agrees to about 0.001% and the A4000 to about 0.9%.

---

## 6. Failure handling

### The repair authority, and its boundary

If a stage fails, **diagnose before doing anything**. If the cause is clear and
the fix is configuration, apply it and continue. If the fix is code, stop.

**Fix and continue:**

- Pod Pending because the request exceeds node headroom. Re-check with
  `nrp_availability.py` and lower CPU, or move to another node.
  **Exception: never lower the LLM below 16 Gi.** gpt2-xl is a 6.43 GB fp32
  checkpoint staged through host RAM; shrinking it trades a Pending pod for an
  OOMKill 20 minutes into a run. Leave that job queued instead.
- A rendering mistake: an unsubstituted `{{PLACEHOLDER}}` or `${VAR}`, a wrong
  digest, a missing env var, a missing volume mount.
- A pod evicted or preempted by the cluster. Relaunch once. The runner flushes
  each repetition as it completes, so restart from the next repetition.
- A card becoming unavailable mid-plan. Skip that card, note it, continue with
  the others.

**Stop and report, do not attempt:**

- Any Python traceback from `benchmarks/` or `measurement/`. That is the code
  under test failing, which is the entire point of Stage 1. Editing it to make
  the run proceed destroys the evidence and ships unreviewed code.
- A gate criterion failing on a row that was written successfully. A wrong value
  is a finding, not an error to route around.
- An OOMKill on the LLM leg. Report it with the node's free memory.
- Anything requiring a new image build, a merge to `main`, or a change to a
  benchmark's defaults.

When you stop: delete your jobs, verify the namespace is clean, and write what
failed with the actual output, never a paraphrase.

### Policy by stage

- **Stage 1 fails a gate: stop.** Do not start Stage 2. No retry: at 2am a
  transient failure and a real one look identical, and guessing wrong produces a
  3.3 GPU-hour dataset built on broken code.
- **Stage 1 LLM fails but matmul and resnet pass:** run Stage 2 job A on all
  three cards; hold every job B. This is precisely why the legs are split.
- **A Stage 2 job fails mid-run:** let the others continue. Partial results
  survive because each repetition is flushed as it completes, and a failed
  repetition is written with an `exclusion_reason` rather than lost.
- **A card is unschedulable:** leave the job queued. Pending costs nothing.

---

## 7. Stage 2: fleet, 5 repetitions

Only if Stage 1 passes. Per card, two jobs. Cards contend for nothing, so all
six may be launched together.

| Job | Contents | Approx on 1080 Ti |
|---|---|---|
| A | matmul 5 reps, then resnet 5 reps | about 40 min |
| B | llm 5 reps | about 26 min |

Cards: **GTX 1080 Ti, RTX 2080 Ti, RTX A4000.** No 3090, by decision. All three
have been preflighted, which is the first time the sweep fleet has been fully
preflighted before running.

Five repetitions rather than three, per CLAUDE.md: failed runs are kept with an
exclusion reason so effective n falls below nominal n, and co-tenant thermal
interference on shared nodes is an uncontrolled variance source.

Expect roughly 1.5 hours wall clock and about 3.3 GPU-hours total.

---

## 8. What to check first, the morning after

1. `kubectl --context nautilus -n cmpm118 get pods,jobs`. Anything of ours still
   Running or Pending, and anything that failed. **Delete every finished job of
   ours.**
2. Copy `runs.jsonl` and the power traces off the PVC into `data/raw/runs/`.
   A completed pod cannot be exec'd into, so use `k8s/arav-pvc-shell.yaml`,
   which mounts the results PVC for exactly this. Delete it afterwards.
   Verify with `md5sum` on both sides before trusting the copy.
3. Re-run the section 5 gate table against **every** row, not only Stage 1's.
4. Compare `energy_j` against `energy_j_counter` per card. The 2080 Ti's 6 to 7%
   gap is known.
5. Check `gpu_uuid` per model. If all repetitions share one value, the
   standard deviation is within-card, not fleet variation, and must be reported
   as such. `analysis/summarize_runs.py` reports `n_physical_gpus` for this.
6. Confirm the namespace is clean again.

---

## 9. Out of scope tonight

- The RTX 3090, by decision.
- Same-model variance. Node pinning that would have made partial progress was
  considered and declined in favour of scheduling reliability, so expect one
  physical card per model again.
- Merging `schema-idle-sizing`. It merges after its output is reviewed by a
  human, not because pods exited zero.
- Committing results.
- Any other repo work.
