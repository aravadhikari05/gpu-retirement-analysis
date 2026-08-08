#!/usr/bin/env bash
#
# Manual sweep for aidan-prefixed llm benchmark Jobs left behind when every
# automatic layer failed: the TTL controller did not run, the deadline did not
# fire, and run_llm_smoke.sh was SIGKILLed before its trap.
#
# Usage:
#   k8s/reap_aidan.sh              # dry run, lists what would be deleted
#   k8s/reap_aidan.sh --delete     # actually delete
#
# Safety: selects only on app=aidan-llm-smoke or app=aidan-llm-prep, and then
# refuses individually any name not starting with "aidan-". Never uses --all,
# and never deletes the aidan-llm-models-pvc holding the staged weights.
# cmpm118 is shared with other students, so both checks are required, not one.

set -euo pipefail

NAMESPACE="cmpm118"
NAME_PREFIX="aidan"
SELECTORS=("app=aidan-llm-smoke" "app=aidan-llm-prep")
DELETE="false"

case "${1:-}" in
  --delete) DELETE="true" ;;
  "")       ;;
  -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown argument: $1" >&2; exit 1 ;;
esac

found=0
for selector in "${SELECTORS[@]}"; do
  jobs="$(kubectl get jobs -n "$NAMESPACE" -l "$selector" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)"
  [ -n "$jobs" ] || continue

  while IFS= read -r name; do
    [ -n "$name" ] || continue

    case "$name" in
      "${NAME_PREFIX}"-*) ;;
      *)
        echo "SKIP  ${name}  (matched ${selector} but does not start with ${NAME_PREFIX}-, not ours)" >&2
        continue
        ;;
    esac

    found=$((found + 1))
    age="$(kubectl get job "$name" -n "$NAMESPACE" \
      -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || echo unknown)"

    if [ "$DELETE" = "true" ]; then
      echo "DELETE ${name}  (created ${age})"
      # The name came from a query already filtered by "$selector", and the
      # prefix guard above passed. kubectl rejects a name combined with -l, so
      # the delete is by name alone; both checks have already been applied.
      kubectl delete job "$name" -n "$NAMESPACE" --ignore-not-found
    else
      echo "WOULD DELETE ${name}  (created ${age})"
    fi
  done <<< "$jobs"
done

if [ "$found" -eq 0 ]; then
  echo "nothing to reap in ${NAMESPACE}"
elif [ "$DELETE" = "false" ]; then
  echo
  echo "dry run. re-run with --delete to remove the ${found} job(s) above."
fi
