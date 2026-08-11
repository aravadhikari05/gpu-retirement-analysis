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
