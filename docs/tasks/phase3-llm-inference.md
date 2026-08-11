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

- gpt2 (124M), batch 1
- gpt2 (124M), batch 32
- gpt2-xl (1.5B), batch 1

Batch 32 exists because GPT-2 124M at batch 1 barely utilizes an L4 or A10. At
batch 1 we would measure idle power on modern cards and near-full load on old
ones, which biases against exactly the hardware the replacement case depends on.

## Metadata returned per run

precision mode, both TF32 flag states, resolved dtype, model id, model revision,
prompt hash, prompt token count, max_new_tokens, batch size, work_hash, warmup
count, torch version, transformers version, CUDA version, driver version,
GPU model observed at runtime, node name.

## Definition of done

Runs locally on CPU as a smoke test, then runs in a pod on at least two GPU
models, one from the old end and one modern, producing matching work_hash in
fp32. All pod resources deleted afterward.
