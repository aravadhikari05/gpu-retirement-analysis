# Task: LLM inference benchmark workload

Owner: Aidan. Implements benchmarks/llm_inference.py (currently a stub).
Workload 2 of 3 in Phase 3. Arav owns ResNet-50, Veda owns matmul.

## Objective

Generate a fixed number of tokens from a pretrained model, identically on every
GPU generation, so that measured energy differences reflect hardware efficiency
and nothing else.

## The design rule

Fix the WORK, not the time. Every GPU performs the identical computation and we
record duration and joules. A run that cannot prove it did the same work as
another run is invalid data.

## Correctness requirements

1. Greedy decoding only: do_sample=False, num_beams=1. Sampling would make
   different cards generate different tokens.
2. min_new_tokens == max_new_tokens. Otherwise EOS can stop one card early.
3. Explicit TF32 control. Ampere and later can silently route FP32 matmuls
   through tensor cores; a GTX 1080 Ti cannot. Left to PyTorch defaults, the
   two cards compute different arithmetic and the comparison is meaningless.
   Set torch.backends.cuda.matmul.allow_tf32 and cudnn.allow_tf32 explicitly.
4. Model weights pre-staged before the timed region, with a pinned revision.
   No network access during measurement.
5. Warmup iterations before timing, excluded from the measurement.
6. torch.cuda.synchronize() before starting and before stopping the timer.
7. SHA-256 over the generated token IDs, returned as work_hash. Same precision
   mode should yield the same digest on every GPU. Compare within a precision
   mode, not across.

## Configurations

Measured configuration, and the module defaults as of 2026-08-23:
**gpt2-xl at revision `15ea56dee5df`, batch 1, 960 new tokens, 8 iterations
inside the timed region, fp32.**

The defaults previously read gpt2 at 500 new tokens, which is the superseded
spec draft's configuration. CLAUDE.md had recorded it as dead since 2026-08-18
but the code still carried it, so any run that did not override both silently
measured the abandoned configuration.

### Why 8 iterations

One `generate()` cannot clear the 30 second power sampling floor on the fleet
the sweep targets. Measured: gpt2-xl at 960 tokens takes 34.40 s on a GTX
1080 Ti and 15.45 s on an L40S. The fastest card in the chosen fleet is an RTX
3090, whose memory bandwidth sits in L40S territory, and batch 1 decode is
bandwidth bound, so it lands near 15 s, half the floor.

Neither a longer generation nor a larger model fixes it. `n_positions` is 1024
for both gpt2 and gpt2-xl, so 960 is already near the ceiling, and long greedy
generations degenerate into a KV cache loop (`distinct_token_ratio` 0.019 at
960 tokens). Repetition inside the timed region is the only route, and it is
what matmul and resnet already do.

**8 is an estimate plus margin, not a measurement.** No RTX 3090 could be
obtained on 2026-08-23 despite repeated attempts, so the figure comes from the
measured L40S runtime. At roughly 15 s per generation it puts a 3090 near 120 s;
the 3090 would have to be four times faster than the L40S before the region fell
below 30 s. Overshooting is close to free, since the run length ceiling was
withdrawn and the runner flushes each repetition as it completes, whereas
undershooting costs a re-run. **If a later 3090 run shows slack, do not retune
this down.** `inner_iters` is part of `config_id`, so changing it makes every
earlier row unpoolable.

### What this measures

Throughput inference: 8 independent decodes of one prompt, run back to back. It
is not single request latency, and it says nothing about long context behaviour.
The paper must name which question it answers.

Greedy decoding is an argmax over a fixed prompt, so all 8 iterations produce
identical tokens. `iterations_identical` records that check; it is verified after
the region closes rather than inside it, because comparing per iteration would
force a device sync and a host copy and would time that work as inference.

### Older configurations, kept for context

- gpt2 (124M), batch 1
- gpt2 (124M), batch 32
- gpt2-xl (1.5B), batch 1

Batch 32 exists because GPT-2 124M at batch 1 barely utilizes an L4 or A10. At
batch 1 we would measure idle power on modern cards and near-full load on old
ones, which biases against exactly the hardware the replacement case depends on.

## Metadata returned per run

precision mode, both TF32 flag states, resolved dtype, model id, model revision,
prompt hash, prompt token count, max_new_tokens, inner_iters, batch size,
tokens_generated_total, iterations_identical, work_hash, work_hash_kind, warmup
count, torch version, transformers version, CUDA version, driver version,
GPU model observed at runtime, node name.

## Definition of done

Runs locally on CPU as a smoke test, then runs in a pod on at least two GPU
models, one from the old end and one modern, producing matching work_hash in
fp32. All pod resources deleted afterward.
