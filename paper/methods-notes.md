# Methods notes: measured facts for the paper

Running record of things established by measurement that belong in the methods
or results sections. Each entry states what was measured, on what, and what is
inference rather than observation. Task docs describe intent; this file records
what the hardware actually did.

---

## Decode cost at batch 1 scales with layer count, not parameter count

**Measured 2026-08-11, GTX 1080 Ti (sm 6.1, driver 580.159.04), fp32, batch 1,
greedy, 60 token prompt.**

| Model | Parameters | Layers | Tokens | Runtime | Per token |
|---|---|---|---|---|---|
| gpt2 | 124M | 12 | 32 | 0.3200 s | 10.00 ms |
| gpt2-xl | 1.558B | 48 | 960 | 34.4026 s | 35.84 ms |

**gpt2-xl costs 3.58x gpt2 per token.** The parameter ratio is 12.6x; the layer
ratio is 4.0x. The measured cost tracks the layer ratio.

**Why.** At batch 1 autoregressive decode is latency bound rather than
throughput bound. Each token requires a sequential pass through every layer, and
each layer's cost is dominated by kernel launch overhead and weight streaming
from memory rather than by arithmetic. Widening a layer (768 to 1600 hidden)
adds work that the GPU absorbs largely in parallel; adding layers (12 to 48)
adds serial steps that it cannot.

**Why it matters for this paper.** Any energy-per-token model that assumes cost
scales with parameter count will overstate the cost of large models at batch 1
by roughly 3x. It also means model depth, not size, is the lever that determines
how long a batch-1 benchmark runs, which is what makes reaching a 30 second
measurement floor difficult (see `docs/tasks/phase3-workload-sizing.md`).

**Caveats.** Two points, not a curve, and the two runs differ in generation
length (32 against 960 tokens), so the gpt2 figure carries proportionally more
fixed startup cost and the true ratio is probably somewhat above 3.58x. Both
runs are batch 1; the relationship should not be assumed to hold at batch 32,
where the GPU has parallel work to absorb. This should be re-measured at matched
token counts before being quoted as a headline number.

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

**Not established.** Agreement between two different cards of the same model:
both L4 runs used the same physical GPU (`gpu_uuid GPU-e82f7d3b`). Agreement at
longer generation lengths is being tested separately.

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
