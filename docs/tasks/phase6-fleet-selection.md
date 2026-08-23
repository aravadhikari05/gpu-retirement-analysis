# Phase 6 scoping: which GPUs the sweep actually runs on

Written 2026-08-23. Owns the decision of which cards the sweep targets and why,
and the same-model variance study that has to happen before any cross-model
difference can be interpreted. `docs/tasks/phase3-workload-sizing.md` owns how
long each run is; this file owns which cards it runs on.

## Decision

**The sweep targets the GTX 1080 Ti, RTX 2080 Ti, RTX 3090 and RTX A4000.** The
L4, L40S and RTX 4090 are dropped from the planned fleet, not because they are
uninteresting but because they cannot be scheduled.

This changes the shape of the paper's claim. The project was framed around
replacing an ageing card with a modern datacenter GPU. What is measurable is a
consumer and workstation line spanning 2017 to 2020 plus a workstation Ampere
part. That is still a real retirement question, and arguably the one an academic
cluster like NRP actually faces, but it is a different question and the paper
has to name it as such rather than quietly substituting one for the other.

**This needs the prof's agreement before the sweep runs.** Collecting 5
repetitions across 3 workloads and 4 cards is a substantial commitment to a
framing, and the framing changed after the cards were checked.

## Evidence for the decision

### Reachability, measured 2026-08-23

`k8s/nrp_availability.py` reads NRP's public `guest.ListNodeInfo` endpoint,
which reports GPUs free per node. Counting only nodes with no blocking taint:

| GPU | Free and reachable | Free overall | Capacity |
|---|---|---|---|
| GTX 1080 Ti | 22 | 43 | 78 |
| RTX A4000 | 20 | 20 | 32 |
| RTX 2080 Ti | 14 | 33 | 93 |
| RTX 3090 | 5 | 126 | 223 |
| NVIDIA L4 | 0 | 49 | 96 |
| NVIDIA L40S | 0 | 4 | 16 |
| RTX 4090 | 0 | 0 | 20 |

Cross-checked the same hour against ten placement probes, one throwaway pod per
GPU model. The feed agreed on eight of ten. The two it missed were the A10,
where it reported two free on untainted nodes while a pod requesting one A10,
one CPU and 1 GiB stayed Pending, and by implication anything else where node
level CPU or memory exhaustion blocks a GPU that is nominally free.

### Three separate mechanisms put a card out of reach

Worth separating, because they have different remedies.

1. **Namespace quota.** `cmpm118` carries hard quotas of
   `requests.nvidia.com/a100: 0/0` and the same for h100, h200 and gh200. These
   are administrative bans. No pod spec gets around them.
2. **Reservation taints.** All 96 L4s sit on six CSU Fullerton nodes tainted
   `nautilus.io/reservation=csuf:NoSchedule`, 49 of them idle at snapshot time.
   **Do not add a toleration for another institution's reservation.** The remedy
   is to ask NRP or CSUF, not to route around it.
3. **Ordinary contention.** The 4090 showed 0 of 20 free. An L40S probe pod sat
   Pending for over 15 minutes even when reduced to 1 GPU, 1 CPU and 1 GiB. A
   3090 preflight pod, at 1 CPU and 4 GiB, was still Pending after 12 minutes.
   Contention is transient, so these cards may become usable opportunistically,
   but a sweep cannot be planned around them.

### The L4 problem specifically

Both L4 measurements this project owns came from
`nautilus-it-gpu03.fullerton.edu`, which now carries the CSUF reservation taint.
The 4.96x energy advantage recorded in `paper/methods-notes.md` therefore rests
on a card the project can no longer schedule against, and cannot currently be
repeated or extended to n>1.

Keep the L4 rows. They are valid measurements and they are the only datacenter
card data the project has. Label them clearly as a single observation from a
pool that later became unreachable.

### Census columns are not enough on their own

`census_fleet.csv` reports `allocatable_gpu_sum_swg`, allocatable GPUs on openly
schedulable nodes, which is the right column for planning against and much
better than raw allocatable. It still cannot see occupancy: it counts what the
namespace could schedule if the cards were idle, not what is idle. The 3090
reports 162 by that column and had 5 actually free. Use the census to decide
which models are worth targeting and `nrp_availability.py` to decide when.

## What the chosen fleet gives us

Measured matmul energy per iteration at identical work, from
`paper/methods-notes.md`:

| GPU | Released | J per iteration | Above the 30 s floor |
|---|---|---|---|
| GTX 1080 Ti | 2017 | 31.78 | yes |
| RTX 2080 Ti | 2018 | 22.69 | yes |
| RTX 3090 | 2020 | 15.67 | no, excluded, must be re-measured |
| RTX A4000 | 2021 | not measured | |

A monotonic generational trend across three consumer generations, with a
workstation Ampere part available as a fourth point. The 1080 Ti to 3090 pair is
the natural replacement case: both consumer, three years apart, and both
reachable at n=5 today.

The A4000 is worth including precisely because it is odd. It is a 140 W
workstation card rather than a 350 W consumer one, so it separates "newer" from
"lower power" in a way the 2080 Ti and 3090 do not.

## Same-model variance, which has to come first

**No cross-model difference can be interpreted until same-model variance is
known.** Every energy figure the project currently holds is n=1 on a single
physical card per model. If two 1080 Tis differ by 15% from each other, then a
1.4x difference between a 1080 Ti and a 2080 Ti means something quite different
than if they differ by 1%.

CLAUDE.md has listed "agreement between two different cards of the same model"
as an open question since 2026-08-11 and no workload has ever tested it.

### Why the 1080 Ti is the card to do it on

It has the most reachable free capacity of any model, 22 GPUs across 6 open
nodes at snapshot time, which is the only way to get a sample of distinct
physical cards without waiting on contention. The project also already has
incidental evidence that its 1080 Ti runs landed on more than one physical card,
which is knowable only because `gpu_uuid` is recorded per row.

### Design

- One workload, matmul, since it is the cheapest and its `work_hash` already
  agrees across four architectures.
- Sized above the 30 s floor per `docs/tasks/phase3-workload-sizing.md`.
- 5 repetitions on each of at least 5 distinct `gpu_uuid` values, spread across
  as many different nodes as possible, since node level thermal and co-tenant
  effects are part of what is being quantified.
- Report within-card standard deviation and between-card spread separately. The
  second is the number that bounds every later claim.

### What it feeds

`analysis/summarize_runs.py` already reports `n_physical_gpus` next to `n_runs`
so that a standard deviation from one card is not read as fleet variation. This
study is what makes that column meaningful rather than merely present.

## Consequences for the to-do list

- Fleet selection above must be confirmed with the prof before Phase 6 runs.
- Same-model variance becomes a prerequisite of Phase 6 interpretation, not a
  nice to have, and it is unblocked today.
- Preflight is still outstanding on the 3090 and A4000. The 2080 Ti is done and
  disagreed with its own energy counter by 6.25%, so this is not a formality.
- If the L4, L40S or 4090 become reachable, take an opportunistic run rather
  than rescheduling the sweep around them. Sizing already leaves headroom for a
  card roughly twice the 3090 so that such a run does not force a resize.
