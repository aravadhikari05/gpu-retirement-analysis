#!/usr/bin/env bash
#
# Submits one arm of the llm_inference work_hash validation, streams its logs,
# collects the result record, and cleans up whether the run succeeds, fails, or
# is interrupted.
#
# Usage:
#   k8s/run_llm_smoke.sh --image <ref> --gpu-product NVIDIA-L4
#   k8s/run_llm_smoke.sh --image <ref> --gpu-product NVIDIA-GeForce-GTX-1080-Ti
#   k8s/run_llm_smoke.sh --image <ref> --cpu
#
# --gpu-product is validated against data/processed/census_fleet.csv, so no
# nvidia.com/gpu.product label string is ever typed by hand.
#
# Cleanup layering is documented in k8s/aidan-llm-smoke-job.yaml. The trap here
# is only the fast interactive path; ttlSecondsAfterFinished and
# activeDeadlineSeconds in the manifest are what survive this script being
# SIGKILLed or the machine going away.

set -euo pipefail

NAMESPACE="cmpm118"
NAME_PREFIX="aidan"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CENSUS="${REPO_ROOT}/data/processed/census_fleet.csv"
OUT_DIR="${REPO_ROOT}/data/raw/llm_smoke"

IMAGE=""
GPU_PRODUCT=""
CPU_ONLY="false"
MODEL="gpt2"
BATCH_SIZE="1"
MAX_NEW_TOKENS="32"
PRECISION="fp32"
WARMUP_ITERS="1"
MEMORY="8Gi"
TIMEOUT_SECONDS="900"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-1}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --image)           IMAGE="$2"; shift 2 ;;
    --gpu-product)     GPU_PRODUCT="$2"; shift 2 ;;
    --cpu)             CPU_ONLY="true"; shift ;;
    --model)           MODEL="$2"; shift 2 ;;
    --batch-size)      BATCH_SIZE="$2"; shift 2 ;;
    --max-new-tokens)  MAX_NEW_TOKENS="$2"; shift 2 ;;
    --precision)       PRECISION="$2"; shift 2 ;;
    --warmup-iters)    WARMUP_ITERS="$2"; shift 2 ;;
    --memory)          MEMORY="$2"; shift 2 ;;
    --timeout)         TIMEOUT_SECONDS="$2"; shift 2 ;;
    -h|--help)         usage 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$IMAGE" ] || die "--image is required. Use the per-commit tag or a digest, never :latest."
case "$IMAGE" in
  *:latest) echo "warning: :latest is not reproducible. Prefer a short-sha tag or a digest." >&2 ;;
esac

# ---------------------------------------------------------------------------
# Validate the GPU product label against the census rather than trusting input.
# ---------------------------------------------------------------------------
if [ "$CPU_ONLY" = "false" ]; then
  [ -n "$GPU_PRODUCT" ] || die "--gpu-product is required unless --cpu is given"
  [ -f "$CENSUS" ] || die "census not found at $CENSUS"
  if ! awk -F, -v want="$GPU_PRODUCT" 'NR>1 && $3==want {found=1} END{exit !found}' "$CENSUS"; then
    echo "error: '$GPU_PRODUCT' is not a product value in $CENSUS" >&2
    echo "known values:" >&2
    awk -F, 'NR>1 {print "  " $3}' "$CENSUS" | sort -u >&2
    exit 1
  fi
  # Lowercase, strip non-alphanumerics, for a DNS-1123 safe name fragment.
  GPU_SLUG="$(printf '%s' "$GPU_PRODUCT" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -e 's/^nvidia-//' -e 's/geforce-//' -e 's/[^a-z0-9]//g')"
else
  GPU_SLUG="cpu"
fi

RUN_ID="$(date -u +%Y%m%dt%H%M%Sz)"

if [ "$CPU_ONLY" = "true" ]; then
  JOB="${NAME_PREFIX}-llm-cpusmoke-${RUN_ID}"
  TEMPLATE="${REPO_ROOT}/k8s/aidan-llm-cpu-smoke-job.yaml"
else
  JOB="${NAME_PREFIX}-llm-smoke-${GPU_SLUG}-${RUN_ID}"
  TEMPLATE="${REPO_ROOT}/k8s/aidan-llm-smoke-job.yaml"
fi

# ---------------------------------------------------------------------------
# Deletion guard. Every delete path in this script goes through this function,
# and it refuses any name that is not ours. There is no --all anywhere, and no
# label-only delete that could match another student's work in cmpm118.
# ---------------------------------------------------------------------------
delete_ours() {
  local name="$1"
  case "$name" in
    "${NAME_PREFIX}"-*) ;;
    *) echo "refusing to delete '$name': does not start with ${NAME_PREFIX}-" >&2; return 1 ;;
  esac

  # kubectl rejects a resource name combined with -l, so the ownership label is
  # checked by reading it back and only then deleting by name. Same safety
  # property, valid command.
  local got
  got="$(kubectl get job "$name" -n "$NAMESPACE" \
    -o jsonpath='{.metadata.labels.aidan\.ucsc\.edu/run-id}' 2>/dev/null || true)"
  if [ -z "$got" ]; then
    echo "job '$name' not found or already gone" >&2
    return 0
  fi
  if [ "$got" != "$RUN_ID" ]; then
    echo "refusing to delete '$name': run-id label '$got' is not ours ($RUN_ID)" >&2
    return 1
  fi

  kubectl delete job "$name" -n "$NAMESPACE" \
    --ignore-not-found --wait=false >/dev/null 2>&1 || true
}

RENDERED=""
CLEANED="false"
cleanup() {
  local code=$?
  if [ "$CLEANED" = "false" ]; then
    CLEANED="true"
    [ -n "$RENDERED" ] && rm -f "$RENDERED"
    echo
    echo "cleaning up ${JOB}"
    delete_ours "$JOB" || true
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM HUP

[ "$JOB" != "${JOB#${NAME_PREFIX}-}" ] || die "internal: job name '$JOB' lost its prefix"
[ "${#JOB}" -le 63 ] || die "job name '$JOB' exceeds 63 characters"

# ---------------------------------------------------------------------------
# Render. python3 rather than envsubst, which is not installed by default on
# macOS. string.Template uses ${VAR}, matching the manifests, and raises on any
# placeholder we failed to supply.
# ---------------------------------------------------------------------------
RENDERED="$(mktemp -t aidan-llm-job)"

IMAGE="$IMAGE" RUN_ID="$RUN_ID" GPU_SLUG="$GPU_SLUG" GPU_PRODUCT="$GPU_PRODUCT" \
MODEL="$MODEL" BATCH_SIZE="$BATCH_SIZE" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
PRECISION="$PRECISION" WARMUP_ITERS="$WARMUP_ITERS" MEMORY="$MEMORY" \
python3 - "$TEMPLATE" > "$RENDERED" <<'PY'
import os, string, sys
keys = ("IMAGE", "RUN_ID", "GPU_SLUG", "GPU_PRODUCT", "MODEL", "BATCH_SIZE",
        "MAX_NEW_TOKENS", "PRECISION", "WARMUP_ITERS", "MEMORY")
with open(sys.argv[1]) as fh:
    template = string.Template(fh.read())
sys.stdout.write(template.substitute({k: os.environ[k] for k in keys}))
PY

echo "run id:     ${RUN_ID}"
echo "job:        ${JOB}"
echo "image:      ${IMAGE}"
if [ "$CPU_ONLY" = "false" ]; then
  echo "gpu:        ${GPU_PRODUCT} (nodeSelector nvidia.com/gpu.product)"
else
  echo "gpu:        none, CPU smoke"
fi
echo "config:     model=${MODEL} batch=${BATCH_SIZE} tokens=${MAX_NEW_TOKENS} precision=${PRECISION} warmup=${WARMUP_ITERS}"
echo

kubectl apply -f "$RENDERED" -n "$NAMESPACE"

# ---------------------------------------------------------------------------
# Wait for a pod, then stream. Poll rather than kubectl wait so that Pending,
# Failed, and quota rejections are all reported rather than timing out silently.
# ---------------------------------------------------------------------------
POD=""
deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
while [ -z "$POD" ]; do
  POD="$(kubectl get pods -n "$NAMESPACE" -l "job-name=${JOB}" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [ -n "$POD" ] && break
  [ "$(date +%s)" -lt "$deadline" ] || die "no pod created within ${TIMEOUT_SECONDS}s"
  sleep 3
done
echo "pod:        ${POD}"

phase=""
while :; do
  phase="$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  case "$phase" in
    Running|Succeeded|Failed) break ;;
  esac
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "pod still ${phase:-unknown} after ${TIMEOUT_SECONDS}s. Recent events:" >&2
    kubectl get events -n "$NAMESPACE" \
      --field-selector "involvedObject.name=${POD}" 2>&1 | tail -20 >&2
    die "timed out waiting for pod to start"
  fi
  sleep 5
done

NODE="$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}' 2>/dev/null || true)"
echo "node:       ${NODE}"
echo

mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/${RUN_ID}-${GPU_SLUG}.log"
kubectl logs -f "$POD" -n "$NAMESPACE" 2>&1 | tee "$LOG_FILE" || true

# Image digest, so the run can be pinned reproducibly afterwards. The tag alone
# is not a stable identifier.
IMAGE_ID="$(kubectl get pod "$POD" -n "$NAMESPACE" \
  -o jsonpath='{.status.containerStatuses[0].imageID}' 2>/dev/null || true)"

final_phase="$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)"

RESULT_FILE="${OUT_DIR}/${RUN_ID}-${GPU_SLUG}.json"
if grep -q '^RESULT_JSON ' "$LOG_FILE"; then
  grep '^RESULT_JSON ' "$LOG_FILE" | tail -1 | cut -d' ' -f2- \
    | IMAGE_ID="$IMAGE_ID" NODE="$NODE" POD="$POD" JOB="$JOB" RUN_ID="$RUN_ID" \
      python3 -c '
import json, os, sys
record = json.load(sys.stdin)
record["image_id"] = os.environ.get("IMAGE_ID", "")
record["k8s_node_name"] = os.environ.get("NODE", "")
record["k8s_pod_name"] = os.environ.get("POD", "")
record["k8s_job_name"] = os.environ.get("JOB", "")
record["run_id"] = os.environ.get("RUN_ID", "")
json.dump(record, sys.stdout, indent=2, sort_keys=True)
' > "$RESULT_FILE"
  echo
  echo "result:     ${RESULT_FILE}"
  python3 -c '
import json, sys
r = json.load(open(sys.argv[1]))
print("  work_hash        ", r.get("work_hash"))
print("  config_id        ", r.get("config_id"))
print("  runtime_seconds  ", r.get("runtime_seconds"))
print("  gpu observed     ", r.get("gpu_model_observed") or r.get("gpu_model_torch"))
print("  driver_version   ", r.get("driver_version"))
print("  node_name        ", r.get("node_name"))
print("  resolved_dtype   ", r.get("resolved_dtype"))
print("  tf32 matmul/cudnn", r.get("allow_tf32_matmul"), "/", r.get("allow_tf32_cudnn"))
print("  sm_capability    ", r.get("sm_capability"))
print("  torch/transformers", r.get("torch_version"), "/", r.get("transformers_version"))
' "$RESULT_FILE"
else
  echo "warning: no RESULT_JSON line in pod logs. Full log at ${LOG_FILE}" >&2
fi

echo
echo "pod phase:  ${final_phase}"
[ "$final_phase" = "Succeeded" ] || die "run did not succeed"
