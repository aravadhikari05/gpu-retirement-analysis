# Task: workload sizing for the Phase 3 energy sweep

Owner: Aidan. Blocks the full Phase 3 sweep. Depends on Phase 4 (power
measurement), which runs in parallel during weeks 2-3.

## Objective

Decide how much work one benchmark run should do, such that the measured energy
reflects inference rather than startup, and such that the power sensor can be
trusted. Then design a prompt and generation config that fills that duration
without degenerating.

These are one question, not two. Choosing a prompt length for readability and
then discovering the energy integral needs a different length means doing the
prompt work twice.

## Two independent constraints

**The knee.** Joules per token falls steeply with run length and then flattens.
Below the flattening point we are measuring clock ramp, kernel launch and cache
warm-up. Above it we are measuring inference. This is a property of the
hardware and has to be measured.

**The 30 second floor.** From the Phase 4 spec: pynvml can return cached
readings from `nvmlDeviceGetPowerUsage()` on some architectures, so runs must
be at least 30 seconds for the integral to be trustworthy. Cited to Yang et al.
(2024). This is a hard constraint from the measurement side. The
joules-per-token curve does not get a vote on it.

The workload must satisfy both. **The floor is almost certainly binding**, per
the arithmetic below, so the knee experiment is likely to confirm a length we
were already forced into rather than to select one. That is still worth doing:
it tells us whether the chosen length sits comfortably above the knee or barely
clears it, and that difference is the argument the paper needs.

## Arithmetic

Measured this session (timed region only, warmup excluded, gpt2, fp32, batch 1,
32 new tokens, 60 token prompt):

| GPU | Model | Tokens | Runtime | Per token |
|---|---|---|---|---|
| NVIDIA L4 (sm 8.9) | gpt2 | 32 | 0.1317 s | 4.11 ms |
| GTX 1080 Ti (sm 6.1) | gpt2 | 32 | 0.3200 s | 10.00 ms |
| GTX 1080 Ti (sm 6.1) | gpt2-xl | 960 | 34.4026 s | 35.84 ms |

The 1080 Ti runs 2.43x slower than the L4 on identical gpt2 work.

**gpt2-xl costs 3.58x gpt2 per token, measured, not the 8 to 12x first
estimated here.** That estimate assumed cost scales with parameter count
(12.6x). It does not, at batch 1: decode is latency bound and scales closer to
layer count, and gpt2-xl has 48 layers against gpt2's 12, a 4x ratio. 3.58x is
close to that. Correcting this changes the conclusion below.

Linear extrapolation from these, **estimated**:

| Config | L4 | 1080 Ti |
|---|---|---|
| 500 new tokens | ~2.1 s | ~5.0 s |
| 964 new tokens (context ceiling) | ~4.0 s | ~9.6 s |

Two caveats on those estimates. They overstate long-run duration, because at 32
tokens a large share of the time is fixed cost that gets amortised away, so the
true marginal per-token cost is below 4.11 ms. And they understate the
attention term, which grows with context length. The first effect is larger, so
**the real durations will be shorter than shown, making the floor harder to
reach, not easier.**

### The context ceiling forecloses one option

`n_positions` is **1024** for both gpt2 and gpt2-xl, confirmed from their
config.json. With a 60 token prompt, `max_new_tokens` cannot exceed **964**.

At the ceiling an L4 reaches roughly 4.0 s. The floor is 30 s. **Generating more
tokens cannot reach 30 seconds with GPT-2 at batch 1, by a factor of about
7.5.** This is an architectural fact, not a tuning problem. No prompt or
generation config fixes it.

### Measured: gpt2-xl does not rescue it either

gpt2-xl at batch 1 and 960 tokens was measured on a GTX 1080 Ti at **34.4 s**,
which clears the floor **on that card**. That is not sufficient, because sizing
is set by the fastest card in the sweep, not the slowest (see below).

All three cards are now **measured**, at identical work:

| Card | Peak BW | gpt2-xl, 960 tokens | Clears 30 s floor? |
|---|---|---|---|
| GTX 1080 Ti (sm 6.1) | 484 GB/s | **34.40 s** | yes |
| NVIDIA L4 (sm 8.9) | 300 GB/s | **32.87 s** | yes |
| **NVIDIA L40S (sm 8.9)** | **864 GB/s** | **15.45 s** | **no** |

**Settled: Option A is closed.** Sizing is set by the fastest card, and the
L40S comes in at half the floor. The RTX 4090 has more bandwidth still (1008
GB/s published) so it lands lower again, and does not need testing to decide
this. **Repetition inside the timed region is required.**

Note this section previously predicted the L4 at ~14.1 s by scaling the 2.43x
gap measured on gpt2. That was wrong by 2.3x, because gpt2 is compute bound at
0.5 GB while gpt2-xl at batch 1 streams 6.43 GB per token and is bandwidth
bound. The corrected reasoning is in `paper/methods-notes.md`. The L40S estimate
of 12 to 20 s, made from bandwidth rather than by scaling a different model's
ratio, was right.

Degeneracy argues the same way independently: `distinct_token_ratio` is 0.019 at
960 tokens on every card measured, so even where Option A did clear the floor it
was measuring a KV-cache read loop rather than inference.

## Two ways to reach 30 seconds

These answer different research questions. **Not picking one here.**

### Option A: more tokens per generation

Push `max_new_tokens` toward the 964 ceiling.

- Measures **long-context inference**: growing KV cache, attention over an
  increasing sequence, memory pressure rising through the run.
- Closer to summarisation or long-form generation as a real workload.
- **Cannot reach 30 s on modern hardware for any of the three configurations**,
  per the measured table above.
- **Degeneracy is catastrophic at this length, now measured rather than
  predicted.** `distinct_token_ratio` by generation length, gpt2 unless noted:
  0.938 at 16 tokens, 0.562 at 32, and **0.019 at 960** (gpt2-xl). At 960
  tokens the output is one sentence repeated to exhaustion:
  "The average power consumption of a rack of servers is about 1 watt."
  Under 2% of tokens are distinct, so the run is overwhelmingly a KV-cache read
  loop rather than representative inference. `work_hash` matched perfectly and
  the run reported success, which is exactly what makes this failure quiet.
- Energy per token is not constant along the run, since later tokens cost more
  than earlier ones. That complicates the per-token normalisation the whole
  analysis rests on.

### Option B: repeat a fixed generation N times inside one timed region

Keep generation short and clean, loop it N times, time the whole loop.

- Measures **throughput inference**: repeated independent requests, closer to a
  serving workload.
- Reaches any duration, so the floor is satisfiable for every configuration by
  the same mechanism.
- Keeps text short enough to stay non-degenerate, so it addresses the prompt
  problem as a side effect rather than needing a separate fix.
- Energy per token is constant along the run, so normalisation is clean.
- `work_hash` should cover one generation and be asserted identical across all
  N, which is a stronger internal check than we have now.
- Measures nothing about long-context behaviour. If the paper wants to claim
  anything about context length, this design cannot support it.

**The choice is yours.** The factual constraint is only that Option A cannot
reach the floor for either gpt2 configuration; it is not a preference.

## Conclusion of the sizing work

Option A is closed by measurement. No configuration reaches the 30 second floor
by token count on modern hardware, and the 1024 context ceiling means no prompt
or generation config can change that. **Repetition inside the timed region is
the only route that reaches the floor uniformly across the fleet.**

### What that measures, and what it does not

Repeating a fixed generation N times inside one timed region measures
**throughput inference: N independent decodes of the same prompt**. It does not
measure single-request latency energy, and it does not measure long-context
behaviour at all.

Those are different research questions, and **the paper has to name which one
it is answering.** The claim this design supports is of the form "energy per
token for repeated short-form generation at batch 1". It does not support "energy
to serve one long request", nor anything about how energy scales with context
length. Stating that plainly in the methods section is cheaper than having it
raised in review.

The alternative framing, if long-context energy is the question worth asking,
requires either a model with a context window larger than 1024 or accepting
sub-floor runs with a documented measurement caveat. Both are larger changes
than this task, and neither is available within the current model choices.

### The two fixes converge

This also disposes of the degeneracy problem, which is the reason to treat
sizing and prompt design as one question rather than two.

Degeneracy is a function of generation length, now measured:
`distinct_token_ratio` 0.938 at 16 tokens, 0.562 at 32, 0.019 at 960. Short
generations stay clean. Repetition reaches the floor by running many short clean
generations instead of one long degenerate one, so the KV-cache-loop artifact
never appears.

Concretely: at roughly 64 to 128 tokens per generation the text is still mostly
non-repetitive, and N is chosen to reach 30 seconds on the fastest card. The
knee experiment then sets the per-generation length, since below the knee each
individual decode is dominated by fixed costs no matter how many times it is
repeated. Both constraints resolve into one number, which is what the sizing
experiment exists to find.

### N is set by the fastest card, not the slowest

Whichever option is chosen, the work must be identical across GPUs, so N or
`max_new_tokens` is fixed fleet-wide. The **fastest** card must still exceed
30 s, so sizing is driven by the fastest GPU in the sweep and every slower card
then runs proportionally longer.

Worked example, **estimated**, Option B at 500 tokens per generation with the
fastest card assumed ~1.8x an L4 (4090 or L40S class):

| Card | Per generation | N for ~35 s | Actual run length |
|---|---|---|---|
| fastest (est.) | ~1.2 s | 30 | ~35 s |
| L4 (measured basis) | ~2.1 s | 30 | ~63 s |
| GTX 1080 Ti (measured basis) | ~5.0 s | 30 | ~150 s |
| oldest (M4000 class, est.) | ~10 s | 30 | ~300 s |

The oldest hardware, which is the entire point of the study, ends up running
five minutes per benchmark run.

## Revised sweep design

The previous proposal swept `32 … 2048` tokens. That was built on a 2 second
assumption and is now wrong twice over: 2048 exceeds the 1024 context ceiling,
and single-generation runs cannot reach the floor anyway.

**Revised.** One card held fixed, an L4, since it is modern enough that fixed
costs dominate longest; a length that satisfies an L4 comfortably satisfies a
1080 Ti.

- Sweep tokens per generation: **32, 64, 128, 256, 512, 964**, geometric so the
  knee is visible on a log axis, capped at the ceiling.
- At each point, set repetitions so total duration is **at least 35 s**, giving
  margin over the floor.
- **5 repeats per point** for spread, not a single value.
- Record per run: total duration, integrated joules, joules per generated
  token, **power sample count**, and per-generation duration so the knee is
  separable from the total.

Roughly 30 measured runs of 35 s or more. Run them as **one pod looping
internally over all points**, not 30 pods: at this image size pod scheduling
and the 7.71 GB pull dominate everything else.

### Power sample count column

Keep it, reframed. At 200 ms over 35 s that is roughly 175 samples, which is
adequate, so it is no longer the main worry. Its job now is to **flag any run
that fell below the 30 s floor**, whether by misconfiguration or by a card
being faster than assumed. A run under the floor is excluded with a reason
rather than deleted, per the repo convention.

## Cost of the floor across the full sweep

The floor is a per-run cost multiplied by every model, workload and repetition.

Assumptions, all **estimated**: 10 GPU models, 3 workloads, 5 repetitions =
**150 runs**. N sized so the fastest card hits ~35 s. Mean timed region across
the fleet ~100 to 120 s. Warmup is currently one full identical generation, so
it **doubles** compute time. Add model load from cephfs (~10 s gpt2, ~60 s
gpt2-xl) and image pull on cold nodes (~2 to 4 min, amortised).

| | Per run | 150 runs |
|---|---|---|
| Timed region | ~110 s | ~4.6 h |
| Warmup (equal to timed region) | ~110 s | ~4.6 h |
| Load, pull, scheduling | ~60 to 120 s | ~2.5 to 5 h |
| **Total GPU allocation** | **~5 to 6 min** | **~12 to 15 GPU-hours** |

Call it **10 to 20 GPU-hours, estimated**, against roughly **5 GPU-hours** under
the old 2 second assumption, where overhead dominated and almost none of the
time was real compute. So the floor costs roughly **2.5 to 3x** more GPU time,
and shifts most of it from waiting into measuring.

Because we run one GPU at a time as a shared-namespace courtesy, that is **12 to
15 hours of wall clock minimum**, realistically spread over several days.

Two levers if that is too much:

- **Warmup is half the compute budget.** A shorter priming run instead of a full
  identical generation would cut roughly 40% of total GPU time. It trades
  against thermal and clock steady state, which matters because energy is the
  dependent variable. Worth measuring rather than assuming.
- **Repetitions.** 5 gives a usable spread; 3 would cut 40% of the sweep. That
  is a statistics decision, not an engineering one.

## Dependency

This task cannot run until `measurement/power_monitor.py` exists. It is a
one-line stub and Phase 4 ownership is not settled. The 200 ms sampling interval
is itself a variable here: if it proves too coarse, changing it is a change to
that file, not this one.

## Definition of done

A figure plotting joules per token against tokens per generation, with error
bars and sample counts annotated, a marked knee, and the 30 s floor drawn on it.
A stated workload length that clears both, with the margin quantified. That
figure justifies workload sizing in the paper instead of leaving it an arbitrary
choice.
