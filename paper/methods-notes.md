# Methods notes: measured facts for the paper

Running record of things established by measurement that belong in the methods
or results sections. Each entry states what was measured, on what, and what is
inference rather than observation. Task docs describe intent; this file records
what the hardware actually did.

---

## Batch-1 decode of a large model is memory-bandwidth bound, and the 1080 Ti is competitive

**Measured 2026-08-11**, fp32, batch 1, greedy, 60 token prompt, same pinned
revisions on both cards.

| GPU | gpt2, 32 tok | gpt2-xl, 960 tok | gpt2-xl / gpt2 |
|---|---|---|---|
| GTX 1080 Ti (sm 6.1) | 10.00 ms/token | 35.84 ms/token | 3.58x |
| NVIDIA L4 (sm 8.9) | 4.11 ms/token | 34.24 ms/token | 8.33x |
| NVIDIA L40S (sm 8.9) | not measured | 16.09 ms/token | |
| **1080 Ti / L4** | **2.43x slower** | **1.05x slower** | |

**The headline: on gpt2-xl at batch 1 the seven-year-old 1080 Ti is within 5% of
the L4** (34.40 s against 32.87 s for identical work), despite being 2.43x
slower on gpt2.

Runtime tracks memory bandwidth, not architecture generation or age. The L4 and
L40S share an architecture (both sm 8.9, both Ada) and differ by 2.13x in
runtime, while the L4 and the 1080 Ti are three architectures apart and differ
by 1.05x:

| GPU | Peak BW | Source | gpt2-xl 960 tok | Effective BW | Utilisation |
|---|---|---|---|---|---|
| NVIDIA L4 | 300 GB/s | **verified** | 32.87 s | 187.8 GB/s | 63% |
| GTX 1080 Ti | 484 GB/s | **derived** | 34.40 s | 179.4 GB/s | 37% |
| NVIDIA L40S | 864 GB/s | **verified** | 15.45 s | 399.6 GB/s | 46% |

Bandwidth ordering predicts runtime ordering; sm version does not.

### Provenance of the bandwidth figures, checked 2026-08-11

These are the denominators for every utilisation number above, so they are
sourced individually rather than recalled. **Two of four could not be verified
against a live NVIDIA source.**

| GPU | Figure | Status | Source and exact spec label |
|---|---|---|---|
| NVIDIA L4 | 300 GB/s | **verified** | <https://www.nvidia.com/en-us/data-center/l4/>, spec label "GPU memory bandwidth", value "300GB/s" |
| NVIDIA L40S | 864 GB/s | **verified** | <https://www.nvidia.com/en-us/data-center/l40s/>, spec label "Memory Bandwidth", value "864GB/s" |
| GTX 1080 Ti | 484 GB/s | **derived, not verified** | NVIDIA's product and specs pages for this card are retired; `en-us` 404s and `en-gb` and the 10-series specs page now serve current-generation content. Figure is 352-bit bus x 11 Gbps / 8 = 484 GB/s. NVIDIA's launch material confirms "11Gbps" memory speed but states no GB/s figure. **The 352-bit bus width is also unverified.** |
| RTX 4090 | 1008 GB/s | **derived, not verified** | <https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/> publishes "Memory Interface Width: 384-bit" and "Standard Memory Config: 24 GB GDDR6X" but **publishes no memory bandwidth spec at all**. Figure is 384-bit x 21 Gbps / 8 = 1008 GB/s, where the 384-bit width is official and the 21 Gbps data rate is not. |

Both verified figures are **theoretical peak**, which is what NVIDIA labels them
as. No effective-bandwidth figures were used.

The 1080 Ti number is the one carrying the "2017 consumer card beats a 2023
datacenter card on paper bandwidth" claim, and it is the weaker of the two
derivations. **Before this appears in the paper it needs a citable primary
source**, for example the Pascal GP102 architecture whitepaper or an archived
copy of NVIDIA's original specifications page.

What does not depend on these figures: the measured runtimes, and the ordering
result. 1080 Ti and L4 differ by 1.05x while L4 and L40S differ by 2.13x, and
that stands on the runtimes alone. What does depend on them: every utilisation
percentage, and the specific claim that the 1080 Ti has more peak bandwidth than
the L4.

### Caveat from NVIDIA on comparing peak bandwidth across architectures

NVIDIA states that peak bandwidth understates Ada's memory performance, because
its much larger L2 cache "reduced memory bus traffic by just over 50% on
average", letting the GPU use bandwidth "2X more efficiently". Their worked
example: "an Ada GPU with 288 GB/sec of peak memory bandwidth would perform
similarly to an Ampere GPU with 554 GB/sec of peak memory bandwidth"
(<https://www.nvidia.com/en-us/geforce/news/rtx-40-series-vram-video-memory-explained/>).

This cuts against using peak bandwidth as a cross-architecture predictor in
general. It bears less on this workload specifically: gpt2-xl's 6.43 GB of
weights exceeds any of these cards' L2 by two orders of magnitude and is
streamed once per token with no reuse, so cache cannot absorb it. That reasoning
should be stated rather than assumed, and the cleanest way to settle it is to
measure achieved bandwidth directly with a STREAM-style benchmark instead of
inferring it from datasheet peaks.

**Why.** The two models are in different regimes. gpt2 is 0.5 GB and largely
compute and launch-overhead bound, where the L4's newer architecture wins. At
batch 1, gpt2-xl must stream all 6.43 GB of weights from memory for **every
token**, making it memory-bandwidth bound. Published peak bandwidth is **484
GB/s** for the 1080 Ti (GDDR5X, 352-bit) against **300 GB/s** for the L4 (GDDR6,
192-bit). The older card has more bandwidth, which offsets its weaker compute.

Effective weight-streaming bandwidth implied by the measurements:

| GPU | Effective | Published peak | Utilisation |
|---|---|---|---|
| GTX 1080 Ti | 179.4 GB/s | 484 GB/s | 37% |
| NVIDIA L4 | 187.8 GB/s | 300 GB/s | 63% |

Both land near 180 to 190 GB/s. The L4 runs closer to its ceiling; the 1080 Ti
has headroom it cannot exploit, consistent with per-layer launch overhead
limiting an older architecture with 48 serial layers.

**Why it matters for this paper.** This is close to the central question. If a
modern inference card is no faster than a 2017 consumer card on large-model
batch-1 decode, then any replacement case for this workload rests entirely on
**power draw**, not throughput. Published TDP is 250 W for the 1080 Ti against
72 W for the L4, so the energy-per-token gap could be large even though the
time-per-token gap is negligible. **That is a hypothesis until Phase 4 measures
it, not a result.**

**Caveats.** Bandwidth figures are published specifications, not measured. The
per-model ratios come from two points at mismatched generation lengths (32
against 960 tokens), so the gpt2 figures carry proportionally more fixed startup
cost and the true ratios are somewhat compressed. Batch 1 only; at batch 32 the
weight read is amortised across the batch and the regime should shift back
toward compute bound, which would favour the L4 again. Re-measure at matched
token counts and at batch 32 before quoting any of this as headline numbers.

**Superseded.** An earlier version of this note claimed gpt2-xl costs "3.58x
gpt2 per token" as a general fact explained by layer count. That was measured
only on the 1080 Ti; the L4 gives 8.33x. The layer-count story is not wrong so
much as incomplete, and it is not the dominant effect. Scaling one GPU's ratio
to another card, which is how the L4 was predicted at ~14 s against an actual
32.87 s, is invalid across regime boundaries.

---

## Fixed-work premise holds across GPU generations in fp32

**Measured 2026-08-11.** gpt2 at revision `607a30d783df`, fp32, batch 1, 32 new
tokens, TF32 explicitly disabled on both matmul and cudnn.

| GPU | sm | Driver | Runtime | work_hash |
|---|---|---|---|---|
| GTX 1080 Ti | 6.1 | 580.159.04 | 0.3200 s | `65ec51f4…` |
| NVIDIA L4 | 8.9 | 595.71.05 | 0.1317 s | `65ec51f4…` |
| NVIDIA L4 (repeat) | 8.9 | 595.71.05 | 0.1325 s | `65ec51f4…` |

Bit-identical generated token sequences across Pascal and Ada, seven years and
three architectures apart, on different drivers at different sites. The 2.43x
runtime gap is therefore a hardware efficiency difference on demonstrably
identical work, not a difference in what was computed.

This was not assumed. Greedy decoding is an argmax over logits and floating
point addition is not associative, so a near-tied pair of candidate tokens can
flip on one architecture and diverge every token after it. The result is
empirical and length dependent.

### Extended to three cards and three drivers at 960 tokens

**Measured 2026-08-11.** gpt2-xl at revision `15ea56dee5df`, fp32, batch 1, 960
new tokens (1020 of the 1024 context ceiling), TF32 explicitly disabled.

| GPU | sm | Driver | Runtime | work_hash |
|---|---|---|---|---|
| GTX 1080 Ti | 6.1 | 580.159.04 | 34.40 s | `da913d94…` |
| NVIDIA L4 | 8.9 | 595.71.05 | 32.87 s | `da913d94…` |
| NVIDIA L40S | 8.9 | 610.43.02 | 15.45 s | `da913d94…` |

One distinct hash across **three cards, two architectures and three driver
versions**, at 30x the argmax decisions of the 32 token test. Runtimes span
2.23x on bit-identical output.

**Not established.** Agreement between two different cards of the same model:
both L4 runs used the same physical GPU (`gpu_uuid GPU-e82f7d3b`).

---

## Greedy decoding degenerates with length

**Measured 2026-08-11**, same fixed prompt throughout.

| Tokens | Model | distinct_token_ratio |
|---|---|---|
| 16 | gpt2 | 0.938 |
| 32 | gpt2 | 0.562 |
| 960 | gpt2-xl | **0.019** |

At 960 tokens the output is a single sentence repeated to exhaustion ("The
average power consumption of a rack of servers is about 1 watt."), so under 2%
of generated tokens are distinct and the run is overwhelmingly a KV-cache read
loop rather than representative inference.

**Relevance to validity.** The run passed every automated check: `work_hash`
matched, `all_rows_identical` was true, exit code 0. A benchmark can do provably
identical work on every GPU and still be measuring the wrong thing. Workload
content has to be validated separately from workload reproducibility, and the
paper should say how.

---

## First energy measurements: identical work on four GPUs

**Measured 2026-08-23**, all through `measurement/runner.py` with
`measurement/power_monitor.py` attached and energy scoped to the timed region.
These are the first `energy_j` figures for any workload in this project. One
repetition per card, so **n=1 everywhere**: no standard deviations, and every
number below is a single observation, not a mean.

### matmul, n=8192, iters=500, fp32, TF32 explicitly disabled

| GPU | sm | runtime | energy | J per iter | avg power | above 30 s floor |
|---|---|---|---|---|---|---|
| GTX 1080 Ti | 6.1 | 57.89 s | 15,887.6 J | 31.78 | 274.6 W | yes |
| NVIDIA L4 | 8.9 | 44.56 s | 3,200.7 J | 6.40 | 71.8 W | yes |
| RTX 2080 Ti | 7.5 | 40.83 s | 11,344.8 J | 22.69 | 277.8 W | yes |
| RTX 3090 | 8.6 | 22.77 s | 7,834.5 J | 15.67 | 343.6 W | **no, excluded** |

### resnet50 training, CIFAR-10, 100 batches of 32 at 224x224, fp32

| GPU | runtime | energy | J per batch | avg power | above 30 s floor |
|---|---|---|---|---|---|
| GTX 1080 Ti | 19.19 s | 4,747.5 J | 47.48 | 246.8 W | **no, excluded** |
| RTX 2080 Ti | 12.72 s | 3,231.4 J | 32.31 | 254.1 W | **no, excluded** |
| RTX 3090 | 8.87 s | 2,829.7 J | 28.30 | 318.3 W | **no, excluded** |

**Every resnet run and the 3090 matmul run are below the 30 s floor and carry an
`exclusion_reason`.** Their energy figures are recorded but not trustworthy, per
Yang et al. (2024), and must not be quoted as results. Only the first three
matmul rows are above the floor.

### What the trustworthy rows say

Restricting to the three matmul runs above the floor, for identical work:

| Replacement | Energy ratio | Runtime ratio |
|---|---|---|
| 1080 Ti to 2080 Ti | 1.40x less | 1.42x faster |
| 1080 Ti to L4 | **4.96x less** | 1.30x faster |

The 1080 Ti to L4 case is the important one: a 30% speed improvement produces a
**5x energy improvement**, because average power differs by 3.82x (274.6 W
against 71.8 W). This is the first direct measurement supporting the project's
premise that the replacement case rests on power draw rather than speed, which
until now was a hypothesis stated in CLAUDE.md for the LLM workload.

The 3090 suggests the trend continues within the consumer line (15.67 J/iter
against the 1080 Ti's 31.78, a 2.03x improvement) but **that run is excluded and
the figure must be re-measured above the floor before use.**

### Peak-FLOPS ratios do not predict runtime ratios

Published fp32 non-tensor peaks put the L4 at roughly 2.7x the 1080 Ti
(30.3 against 11.3 TFLOPS). The measured ratio on this matmul is **1.30x**. The
1080 Ti achieves 9,497 GFLOP/s, 84% of its peak; the L4 achieves 12,336 GFLOP/s,
41% of its.

This is the second instance of the same error mode in this project. The first
was scaling one GPU's model-to-model cost ratio onto another card, which
predicted the L4 at 14 s against an actual 32.87 s. Both say the same thing:
**do not size a workload, or predict a runtime, from a spec-sheet ratio.**

### Caveats that apply to all of the above

- n=1 per card. No variance estimate.
- One physical GPU per model. Card-to-card variation within a model is still
  unmeasured, and is a named open question.
- The L4 row was produced on 2026-08-19 by a pod that did not set `IMAGE_REF` or
  `GIT_COMMIT`, so it cannot be tied to an image digest. The other three rows
  carry both.
- The L4 row came from `nautilus-it-gpu03.fullerton.edu`, which now sits behind
  `nautilus.io/reservation=csuf:NoSchedule`. That card is not currently
  reachable for a repeat measurement.

---

## The fixed-work premise holds across four architectures

**Measured 2026-08-23.** `work_hash` was byte-identical across every card that
ran each workload:

| Workload | work_hash | Cards |
|---|---|---|
| matmul | `afac2e9f3f01ded4...` | GTX 1080 Ti (sm 6.1), RTX 2080 Ti (sm 7.5), RTX 3090 (sm 8.6), L4 (sm 8.9) |
| resnet50 | `db68a5ff554f811c...` | GTX 1080 Ti, RTX 2080 Ti, RTX 3090 |

Both are config-kind hashes: they cover the seed, the inputs and the shape of
the work, not the numerical result. Agreement therefore proves the cards were
asked to do identical work, which is the fixed-work premise the comparison
rests on. It does not prove they produced identical numbers, and the paper must
say so, because the LLM workload's hash **does** carry the stronger
output-identity guarantee and a reader seeing one column will assume otherwise.

**The numerical results were not identical, as expected.** The matmul
`result_checksum` took two distinct values across the four cards, and the resnet
final loss took three distinct values across three cards. Floating point
addition is not associative and the cards select different kernels, so this is
the predicted behaviour, recorded for divergence diagnosis and never asserted
equal.

---

## The NVML energy counter's accuracy is card-dependent

**Measured 2026-08-23.** Every run above records both a trapezoidal integral of
sampled power (`energy_j`) and NVML's hardware energy counter
(`energy_j_counter`) over the same region. Their disagreement:

| GPU | matmul | resnet | region length |
|---|---|---|---|
| GTX 1080 Ti | **-0.001%** | +1.617% | 57.9 s / 19.2 s |
| NVIDIA L4 | +0.198% | not run | 44.6 s |
| **RTX 2080 Ti** | **+6.940%** | **+6.573%** | 40.8 s / 12.7 s |
| RTX 3090 | -1.388% | -3.939% | 22.8 s / 8.9 s |

Two conclusions, and the first corrects an assumption made earlier the same day.

**It is not purely a duration effect.** The initial reading of the 1080 Ti data
was that agreement degrades with short, spiky regions and is otherwise fine. The
2080 Ti falsifies that: it is ~7% off on both workloads, including a 40.8 s
matmul region that is well above the floor and flat in power. That is a
systematic per-card bias.

**It is not coarse power quantisation either.** The pitfall about old cards
reporting power in 25 W steps does not apply to any card measured here. Observed
minimum step, from the sample traces:

| GPU | Samples | Distinct values | Min step |
|---|---|---|---|
| GTX 1080 Ti | 315 | 298 | 0.004 W |
| RTX 2080 Ti | 224 | 211 | 0.001 W |
| NVIDIA L4 | 238 | 224 | 0.002 W |
| RTX 3090 | 145 | 58 | 0.001 W |

**The 3090 shows the cached-reading signature instead.** Only 58 distinct values
across 145 samples on matmul, and 27 across 79 on resnet, against 211 of 224 on
the 2080 Ti. Repeated identical readings at a 0.2 s sampling interval is exactly
the failure Yang et al. (2024) describe, and exactly what the retained sample
trace exists to detect. The 3090's power numbers should not be trusted until
`measurement/preflight.py` has been run against that model.

Practical consequence: run preflight per GPU model before its first measured
run, and report the counter-against-integral agreement per card in the paper
rather than quoting the 1080 Ti's 0.001% as representative.

---

## Workload sizing constants, set by measurement

**Measured 2026-08-23.** Sizing is set by the fastest card in the sweep, since
the work must be identical fleet-wide and the fastest card must still clear the
30 s floor. The fastest card that is both reachable and actually free is the
RTX 3090:

| Workload | 3090 rate | For a 90 s region on the 3090 | 1080 Ti at that size |
|---|---|---|---|
| matmul | 45.54 ms/iter | `iters=2000` | ~232 s |
| resnet50 | 88.65 ms/batch | ~1000 batches | ~195 s |

The 90 s target rather than a bare 45 s leaves headroom for a card roughly twice
the 3090, so that a future 4090 or L40S measurement does not force a resize.
Resizing after the fact changes `config_id` and `work_hash` and invalidates
every row already collected, so the margin is cheap insurance.

resnet at ~1000 batches stays inside CIFAR-10: `_plan_indices` refuses to exceed
the 50,000 training rows, capping the design at 1,557 measured batches at batch
size 32. That ceiling would have bound if an L40S set the sizing; with the 3090
it does not.

---

## Fleet availability is not fleet capacity, and the census cannot see it

**Measured 2026-08-23.** The kubectl census answers what the fleet contains. It
cannot answer what is free: a `cmpm118` user may not list pods cluster-wide nor
read individual Node objects, and a Node object carries capacity, never
allocation.

NRP publishes the missing half at `guest.ListNodeInfo` on
<https://portal.nrp.ai/rpc>, unauthenticated, which is what nrp.ai/viz/resources
renders. `k8s/nrp_availability.py` reads it. Snapshot taken 2026-08-23, counting
only nodes without a blocking taint:

| GPU | Free on reachable nodes | Free overall | Capacity |
|---|---|---|---|
| GTX 1080 Ti | 22 | 43 | 78 |
| RTX A4000 | 20 | 20 | 32 |
| RTX 2080 Ti | 14 | 33 | 93 |
| RTX 3090 | 5 | 126 | 223 |
| NVIDIA L4 | **0** | 49 | 96 |
| NVIDIA L40S | **0** | 4 | 16 |
| RTX 4090 | **0** | 0 | 20 |

**All 96 L4s sit behind `nautilus.io/reservation=csuf:NoSchedule`**, including
`nautilus-it-gpu03.fullerton.edu`, the node that produced both of this project's
L4 measurements. 49 of them were idle at the time of the snapshot and none were
reachable. The L4 energy figure above therefore comes from a pool this project
can no longer schedule against.

The feed was cross-checked against ten placement probes, one pod per GPU model,
launched the same hour. It agreed on eight of ten. It reported two free A10s on
untainted nodes while a pod requesting one A10, one CPU and 1 GiB stayed
Pending. Treat it as a filter, confirm with a probe.

Consequence for the sweep: the cards that are reachable *and* free are the 1080
Ti, 2080 Ti, 3090 and A4000. A GTX 1080 Ti (2017, Pascal) to RTX 3090 (2020,
Ampere) replacement pair is measurable at n=5 today; anything involving the L4,
L40S or 4090 is not.
