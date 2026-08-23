# Benchmark 2: LLM token generation - Aidan
"""GPT-2 token generation benchmark: fixed work, not fixed time.

Generates a fixed number of tokens from a pinned pretrained model, identically on
every GPU generation, so that measured energy differences reflect hardware
efficiency and nothing else. Energy accounting is done by the caller
(measurement/runner.py), which wraps this function's execution with
measurement/power_monitor.py.

The proof that two runs did the same work is `work_hash`, a SHA-256 over the
generated token IDs. Runs are comparable only when their `config_id` matches.

Honest caveat on work_hash across architectures:

    Cross-architecture work_hash equality in fp32 is an empirical hypothesis, not
    a guarantee. Greedy decoding is an argmax over logits, floating point addition
    is not associative, and different architectures reduce in different orders. If
    two candidate tokens are nearly tied, the argmax can flip and every token after
    it diverges. This is exactly what the two-GPU pod test is meant to establish.
    When hashes disagree, use the token ID sidecar to find the first divergence
    index rather than discarding the run as merely invalid.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess

# Offline enforcement must be in place before huggingface_hub is imported, so it
# is set here rather than relying only on the pod env. local_files_only=True on
# every from_pretrained call is the authoritative guard; these are belt and braces.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks._context import RunContext
from benchmarks._result import PRECISION_NAMES, WorkloadResult, extra_fields

logger = logging.getLogger(__name__)

# Pinned revisions, resolved from the Hugging Face API on 2026-08-07. Bare "gpt2"
# and "gpt2-xl" redirect to the openai-community org, so the canonical ids are
# used. Both revisions were last modified 2024-02-19 and are stable heads.
MODELS = {
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "gpt2-xl": ("openai-community/gpt2-xl", "15ea56dee5df4983c59b2538573817e1667135e2"),
}

# Fixed prompt. Changing this changes prompt_hash and therefore config_id, which
# invalidates comparison against every previously recorded run.
PROMPT = (
    "The question of when to retire computing hardware is usually framed as a "
    "question of performance, but it is equally a question of carbon. A machine "
    "that is slower per watt may still be the lower emissions choice if the "
    "carbon already embodied in its manufacture has not yet been paid off. "
    "Consider the following analysis:"
)

DEFAULT_CACHE_DIR = "/models/hf"
DEFAULT_MAX_NEW_TOKENS = 500
DEFAULT_WARMUP_ITERS = 1

# The shared vocabulary, not a second copy of it. benchmarks/_result.py owns the
# list because the record contract validates against it, and a mode this module
# accepted but the schema rejected would fail at record construction rather than
# at argument parsing.
PRECISIONS = PRECISION_NAMES

# Each precision fully determines both the dtype and the TF32 flags, so no
# combination is ever left to a library default. tf32 is a separate mode rather
# than a modifier of fp32, so an Ampere-or-later fp32 run can never be silently
# compared against a Pascal fp32 run.
_PRECISION_SPEC = {
    "fp32": (torch.float32, False),
    "tf32": (torch.float32, True),
    "fp16": (torch.float16, False),
    "bf16": (torch.bfloat16, False),
}

if set(_PRECISION_SPEC) != set(PRECISIONS):
    raise RuntimeError(
        f"_PRECISION_SPEC {sorted(_PRECISION_SPEC)} has drifted from the shared "
        f"vocabulary {sorted(PRECISIONS)}"
    )

# Hash encoding version. Bump if the byte encoding below ever changes, because
# doing so silently invalidates comparison against previously recorded hashes.
WORK_HASH_ENCODING = "int64-le-v1"


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _device_index(device: str) -> int:
    parts = str(device).split(":")
    return int(parts[1]) if len(parts) > 1 else 0


def _decode(value) -> str:
    """NVML returns bytes on older bindings and str on newer ones."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _set_precision(precision: str, device: str) -> dict:
    """Applies the precision mode and reads the flags back.

    Returns what the backends actually report after being set, not what we asked
    for. On a GTX 1080 Ti the TF32 flags are a silent no-op, so the read-back plus
    the compute capability is what tells us whether TF32 was really in play.
    """
    if precision not in _PRECISION_SPEC:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

    dtype, tf32_requested = _PRECISION_SPEC[precision]

    matmul = torch.backends.cuda.matmul

    # The legacy API. Present through torch 2.x, deprecated from 2.9 onward.
    if hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = tf32_requested
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = tf32_requested

    # The replacement API. torch 2.9 moved TF32 control to fp32_precision and
    # made allow_tf32 a deprecated shim. Set both where both exist, so the
    # setting is unambiguous no matter which one the installed torch honours.
    # Getting this wrong is not a warning, it silently routes fp32 matmuls
    # through tensor cores on Ampere and later while a 1080 Ti cannot follow.
    fp32_mode = "tf32" if tf32_requested else "ieee"
    if hasattr(matmul, "fp32_precision"):
        matmul.fp32_precision = fp32_mode
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None and hasattr(conv, "fp32_precision"):
        conv.fp32_precision = fp32_mode

    # Reduced precision reduction defaults differ in effect across architectures,
    # so pin them rather than inherit them.
    if hasattr(matmul, "allow_fp16_reduced_precision_reduction"):
        matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(matmul, "allow_bf16_reduced_precision_reduction"):
        matmul.allow_bf16_reduced_precision_reduction = False

    capability = None
    if _is_cuda(device) and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(_device_index(device))
        capability = f"{major}.{minor}"
        tf32_effective = tf32_requested and (major, minor) >= (8, 0)
    else:
        tf32_effective = False

    return {
        # "precision", not "precision_mode". The schema names this column
        # `precision`, and emitting a second name for it left every llm row
        # null in the required column while the value hid elsewhere.
        "precision": precision,
        "resolved_dtype": str(dtype),
        "tf32_requested": tf32_requested,
        # Read back, never echoed from the request. On a 1080 Ti these are a
        # silent no-op, which is why sm_capability is recorded alongside.
        "allow_tf32_matmul": bool(getattr(matmul, "allow_tf32", False)),
        "allow_tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "matmul_fp32_precision": str(getattr(matmul, "fp32_precision", "")),
        "cudnn_conv_fp32_precision": str(getattr(conv, "fp32_precision", "")) if conv else "",
        "tf32_effective": tf32_effective,
        "sm_capability": capability,
        "allow_fp16_reduced_precision_reduction": bool(
            getattr(matmul, "allow_fp16_reduced_precision_reduction", False)
        ),
        "allow_bf16_reduced_precision_reduction": bool(
            getattr(matmul, "allow_bf16_reduced_precision_reduction", False)
        ),
    }, dtype


def _load(model_key: str, dtype: torch.dtype, device: str, cache_dir: str):
    """Loads tokenizer and model at the pinned revision, offline.

    snapshot_download in the prep job stores weights under snapshots/<revision>/,
    so passing revision here resolves to exactly those bytes. The pin enforces
    itself rather than being documentation.
    """
    model_id, revision = MODELS[model_key]

    # transformers 5.x renamed from_pretrained's torch_dtype argument to dtype.
    # from_pretrained takes **kwargs, so passing the wrong name is not an error;
    # it is silently ignored and the model loads in its checkpoint dtype. That
    # would quietly invalidate every fp16 and bf16 comparison, so the choice is
    # made by version and then verified by assertion below.
    major = int(str(transformers.__version__).split(".")[0])
    dtype_kwarg = "dtype" if major >= 5 else "torch_dtype"

    logger.info(
        "Loading %s at revision %s from %s (transformers %s, using %s=)",
        model_id,
        revision,
        cache_dir,
        transformers.__version__,
        dtype_kwarg,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
        **{dtype_kwarg: dtype},
    )

    if model.dtype != dtype:
        raise RuntimeError(
            f"requested dtype {dtype} but model loaded as {model.dtype}. The "
            f"'{dtype_kwarg}' argument was ignored by transformers "
            f"{transformers.__version__}. Refusing to record a run whose "
            "precision is not what was asked for."
        )

    model.to(torch.device(device))
    model.eval()
    return tokenizer, model, model_id, revision


def _build_batch(tokenizer, batch_size: int, device: str):
    """Tiles one tokenized prompt to batch_size, with no padding.

    GPT-2 has no pad token. Rather than paper over that by aliasing eos to pad,
    every row is the same prompt, so the attention mask is all ones and no padding
    exists to get the semantics wrong. Under greedy decoding identical rows must
    generate identically, which doubles as a cheap internal consistency check.
    """
    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].repeat(batch_size, 1).to(torch.device(device))
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _work_hash(new_tokens: torch.Tensor) -> str:
    """SHA-256 over generated token IDs only.

    A pure function of the output, with no configuration mixed in, so that
    comparison is always "same config_id, then same work_hash". The byte encoding
    is fixed explicitly (signed 64 bit, little endian, row major) rather than
    inherited from a numpy default, because the digest has to reproduce across
    machines.
    """
    flat = new_tokens.reshape(-1).tolist()
    digest = hashlib.sha256()
    for token in flat:
        digest.update(int(token).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _longest_repeated_ngram(tokens: list) -> int:
    """Length of the longest token sequence that occurs more than once.

    Replaces an earlier check that counted consecutive identical token ids. That
    version returned 2 on a 960 token gpt2-xl run whose output was one sentence
    repeated to exhaustion, because a looping phrase contains no adjacent
    duplicates. Phrase-level repetition is the failure mode greedy decoding
    actually exhibits, so that is what this measures.

    Binary search over length: "some n-gram of length L repeats" is monotone
    decreasing in L, so the largest such L is findable in O(n log n) hashes.
    """
    n = len(tokens)
    if n < 2:
        return 0

    def has_repeat(length: int) -> bool:
        seen = set()
        for start in range(n - length + 1):
            gram = tuple(tokens[start : start + length])
            if gram in seen:
                return True
            seen.add(gram)
        return False

    low, high = 0, n - 1
    while low < high:
        mid = (low + high + 1) // 2
        if has_repeat(mid):
            low = mid
        else:
            high = mid - 1
    return low


def _observed_hardware(device: str) -> dict:
    """Hardware as observed at runtime from inside the pod.

    CLAUDE.md requires this to be observed at runtime, never joined against the
    stored census, because node labelling drifts. gpu_uuid also catches the case
    where two runs landed on different physical cards on the same node.
    """
    observed = {
        "node_name": os.environ.get("NODE_NAME", ""),
        "pod_name": os.environ.get("POD_NAME", ""),
        "gpu_model_observed": "",
        "gpu_model_torch": "",
        "gpu_uuid": "",
        "driver_version": "",
        "hardware_source": "",
        "nvml_error": "",
    }

    if not (_is_cuda(device) and torch.cuda.is_available()):
        return observed

    index = _device_index(device)
    observed["gpu_model_torch"] = torch.cuda.get_device_name(index)

    # pynvml and nvidia-ml-py both install a module named pynvml, so when both
    # are present in the image whichever landed last wins and the two disagree
    # on some symbols. Driver version is required metadata, so a failure here
    # falls through to nvidia-smi rather than leaving the field blank.
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            observed["gpu_model_observed"] = _decode(pynvml.nvmlDeviceGetName(handle))
            observed["gpu_uuid"] = _decode(pynvml.nvmlDeviceGetUUID(handle))
            observed["driver_version"] = _decode(pynvml.nvmlSystemGetDriverVersion())
            observed["hardware_source"] = "pynvml"
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        observed["nvml_error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("NVML unavailable (%s), falling back to nvidia-smi", exc)
        observed.update(_nvidia_smi_hardware(index))

    return observed


def _nvidia_smi_hardware(index: int) -> dict:
    """Reads GPU name, driver version and UUID from nvidia-smi.

    Fallback for when the pynvml and nvidia-ml-py packages collide. nvidia-smi
    ships in the CUDA runtime image and is present in any GPU pod.
    """
    fields = {}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=name,driver_version,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        parts = [p.strip() for p in completed.stdout.strip().split(",")]
        if len(parts) >= 3:
            fields["gpu_model_observed"] = parts[0]
            fields["driver_version"] = parts[1]
            fields["gpu_uuid"] = parts[2]
            fields["hardware_source"] = "nvidia-smi"
            logger.info("nvidia-smi reports %s driver %s", parts[0], parts[1])
        else:
            fields["nvml_error"] = f"unparsed nvidia-smi output: {completed.stdout!r}"
    except Exception as exc:
        fields["nvml_error"] = f"pynvml and nvidia-smi both failed: {exc}"
        logger.error("nvidia-smi fallback also failed: %s", exc)
    return fields


def _sync(device: str) -> None:
    if _is_cuda(device) and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def run(
    model_key: str = "gpt2",
    batch_size: int = 1,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    precision: str = "fp32",
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    device: str = "cuda:0",
    cache_dir: str = DEFAULT_CACHE_DIR,
    deterministic: bool = False,
    ctx: RunContext | None = None,
) -> WorkloadResult:
    """Runs the fixed-work GPT-2 token generation benchmark.

    Args:
      model_key: One of MODELS ("gpt2" or "gpt2-xl").
      batch_size: Number of identical prompt rows generated in parallel.
      max_new_tokens: Tokens to generate per row. min_new_tokens is pinned to the
        same value so EOS cannot stop one card early.
      precision: One of PRECISIONS. Determines dtype and both TF32 flags.
      warmup_iters: Full identical generate() calls run and discarded before timing.
      device: Torch device string. "cpu" is supported for smoke testing.
      cache_dir: Hugging Face cache holding the pre-staged pinned revisions.
      deterministic: Diagnostic only. Off by default because deterministic
        algorithms change kernel selection and therefore energy, which is the
        quantity the surrounding measurement exists to capture.
      ctx: Timed-region context supplied by measurement/runner.py, which scopes
        power measurement to the timed generate() and excludes the model load
        (roughly 60 s from cephfs on gpt2-xl) and warmup. A monitor-less one is
        created for a standalone CLI run.

    Returns:
      A WorkloadResult. The required fields are enforced by its constructor; the
      full metadata record required by docs/tasks/phase3-llm-inference.md rides
      in `extra`. The generated token IDs are in extra under "token_ids" for
      divergence diagnosis; callers writing tabular output should split that into
      a sidecar rather than a CSV column.
    """
    if model_key not in MODELS:
        raise ValueError(f"model_key must be one of {sorted(MODELS)}, got {model_key!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")

    # A monitor-less context for standalone CLI runs; the runner passes its own.
    ctx = ctx or RunContext()

    if deterministic:
        logger.warning(
            "deterministic=True changes kernel selection and therefore energy. "
            "Diagnostic use only, not for measured runs."
        )
        torch.use_deterministic_algorithms(True)

    precision_record, dtype = _set_precision(precision, device)

    tokenizer, model, model_id, revision = _load(model_key, dtype, device, cache_dir)
    input_ids, attention_mask = _build_batch(tokenizer, batch_size, device)
    prompt_len = int(input_ids.shape[1])

    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        num_beams=1,
        min_new_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Warmup is a full identical generate() call. A shorter one would prime
    # different kernels than the long-sequence decode path and would not reach
    # thermal or clock steady state, which matters because energy is the
    # dependent variable downstream.
    with torch.inference_mode():
        for i in range(warmup_iters):
            logger.info("Warmup iteration %d of %d (not counted)", i + 1, warmup_iters)
            model.generate(**generate_kwargs)
        _sync(device)

        logger.info(
            "Timed region: %s, batch %d, %d new tokens, precision %s",
            model_id,
            batch_size,
            max_new_tokens,
            precision,
        )
        # ctx.timed_region syncs at both boundaries and marks them on the power
        # monitor, so energy covers exactly this generate() and excludes the
        # model load and warmup above.
        with ctx.timed_region(device):
            outputs = model.generate(**generate_kwargs)
        runtime_seconds = ctx.region_runtime_s

    new_tokens = outputs[:, prompt_len:].to("cpu").to(torch.int64).contiguous()
    if int(new_tokens.shape[1]) != max_new_tokens:
        raise RuntimeError(
            f"expected {max_new_tokens} new tokens, got {int(new_tokens.shape[1])}. "
            "The run is not comparable and must not be recorded as valid."
        )

    rows = new_tokens.tolist()
    all_rows_identical = all(row == rows[0] for row in rows)

    # Decoded text is recorded, not just the digest. Greedy decoding from a
    # short prompt degenerates into repetition, and 32 tokens of one phrase on
    # a loop would pass the hash check while being a poor basis for the sweep.
    # distinct_token_ratio is the cheap numeric version of that judgement.
    generated_text = tokenizer.decode(rows[0], skip_special_tokens=True)
    distinct_token_ratio = len(set(rows[0])) / len(rows[0]) if rows[0] else 0.0
    longest_repeat = _longest_repeated_ngram(rows[0])

    prompt_hash = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    config_id = (
        f"{model_key}|{revision[:12]}|{precision}|b{batch_size}"
        f"|n{max_new_tokens}|p{prompt_hash[:12]}"
    )

    # Everything the required set does not already own. precision and both TF32
    # read-backs are lifted out of precision_record into the constructor, since
    # extra may not shadow a required field. The old "benchmark" key is gone:
    # it duplicated the runner's own column for the concept "workload" names.
    extra = {
        "warmup_iters": warmup_iters,
        # Work definition
        "model_key": model_key,
        "model_id": model_id,
        "model_revision": revision,
        "prompt_hash": prompt_hash,
        "prompt_token_count": prompt_len,
        "batch_size": batch_size,
        "min_new_tokens": max_new_tokens,
        "max_new_tokens": max_new_tokens,
        "tokens_generated_total": batch_size * max_new_tokens,
        "use_cache": True,
        "do_sample": False,
        "num_beams": 1,
        "deterministic_algorithms": bool(deterministic),
        "work_hash_encoding": WORK_HASH_ENCODING,
        "all_rows_identical": all_rows_identical,
        # Degeneracy signals. A hash can match perfectly on output that is
        # worthless as a benchmark workload, so these travel with every record.
        "generated_text": generated_text,
        "distinct_token_ratio": distinct_token_ratio,
        "longest_repeated_ngram": longest_repeat,
        # Software versions
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda or "",
        "device_requested": device,
        # Token IDs for divergence diagnosis. Rows are identical under greedy
        # decoding with identical prompts, so store one row unless they are not.
        "token_ids": rows[0] if all_rows_identical else rows,
    }
    extra.update(extra_fields(precision_record))
    extra.update(_observed_hardware(device))

    record = WorkloadResult(
        workload="llm_inference",
        config_id=config_id,
        work_hash=_work_hash(new_tokens),
        # Output kind, and the only one of the three workloads that earns it.
        # The hash is over the generated token IDs, so two matching runs produced
        # bit-identical output. matmul and resnet hash their inputs instead,
        # because a float reduction is not bit-identical across architectures.
        work_hash_kind="output",
        work_hash_covers="generated token ids, so matching runs are bit-identical",
        precision=precision_record["precision"],
        allow_tf32_matmul=precision_record["allow_tf32_matmul"],
        allow_tf32_cudnn=precision_record["allow_tf32_cudnn"],
        # Interim value 1: this workload runs exactly one generate() per timed
        # region, so there is no inner loop yet. Whether it adopts the
        # repetition-inside-the-timed-region design from
        # docs/tasks/phase3-workload-sizing.md is open for Veda and Aidan; see
        # "LLM inner_iters" under Unclaimed side work in CLAUDE.md. Energy per
        # unit of work is energy_j / inner_iters, so 1 is the honest value for a
        # single decode and not a placeholder that scales anything wrongly.
        inner_iters=1,
        runtime_seconds=runtime_seconds,
        extra=extra,
    )

    if not all_rows_identical:
        logger.error(
            "Rows within the batch differ under greedy decoding with identical "
            "prompts. This should not happen. All rows stored for inspection."
        )

    return record


def _write_output(record: dict, out_path: str) -> None:
    """Writes the record, splitting token IDs into a sidecar."""
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    token_ids = record.get("token_ids")
    sidecar_path = f"{os.path.splitext(out_path)[0]}.tokens.json"

    summary = {k: v for k, v in record.items() if k != "token_ids"}
    summary["token_ids_path"] = sidecar_path

    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(sidecar_path, "w") as handle:
        json.dump(
            {
                "config_id": record["config_id"],
                "work_hash": record["work_hash"],
                "work_hash_encoding": record["work_hash_encoding"],
                "all_rows_identical": record["all_rows_identical"],
                "token_ids": token_ids,
            },
            handle,
        )
    logger.info("Wrote %s and %s", out_path, sidecar_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="GPT-2 token generation benchmark")
    parser.add_argument("--model", dest="model_key", default="gpt2", choices=sorted(MODELS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--precision", default="fp32", choices=list(PRECISIONS))
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Diagnostic only. Changes kernel selection and therefore energy.",
    )
    parser.add_argument("--out", default="", help="Path to write the JSON record")
    args = parser.parse_args()

    result = run(
        model_key=args.model_key,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
        warmup_iters=args.warmup_iters,
        device=args.device,
        cache_dir=args.cache_dir,
        deterministic=args.deterministic,
    ).to_row()

    if args.out:
        _write_output(result, args.out)

    # Single-line stdout record so the run script can recover the result from pod
    # logs even if the PVC write fails.
    stdout_record = {k: v for k, v in result.items() if k != "token_ids"}
    print("RESULT_JSON " + json.dumps(stdout_record, sort_keys=True))

    logger.info(
        "work_hash=%s runtime_seconds=%.4f gpu=%s driver=%s (via %s) node=%s",
        result["work_hash"],
        result["runtime_seconds"],
        result["gpu_model_observed"] or result["gpu_model_torch"],
        result["driver_version"],
        result["hardware_source"] or "none",
        result["node_name"],
    )
    logger.info(
        "degeneracy: distinct_token_ratio=%.3f longest_repeated_ngram=%d",
        result["distinct_token_ratio"],
        result["longest_repeated_ngram"],
    )
    logger.info("generated text: %r", result["generated_text"])
