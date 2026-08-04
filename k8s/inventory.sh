#!/usr/bin/env bash
# Entry point for the Phase 0 read-only GPU fleet census.
#
# Usage:
#   k8s/inventory.sh                      take a fresh snapshot, then summarize
#   k8s/inventory.sh path/to/nodes.json   reuse an existing snapshot
#   k8s/inventory.sh [snapshot] --product-key K --site-key S --gpu-resource R
#
# Any --flag args are passed straight through to summarize_census.py. The
# summarization lives in Python, not jq: label coverage and taint parsing are
# unreadable in jq and we need CSV output downstream (see docs/tasks/phase0-census.md).
#
# Access note: the cmpm118 user is list-only on nodes. `kubectl describe node`
# and `kubectl get node <name>` are Forbidden; only `kubectl get nodes` (list)
# works, which is what this script uses.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
PROCESSED_DIR="$REPO_ROOT/data/processed"
SUMMARIZER="$REPO_ROOT/k8s/summarize_census.py"
EXPECTED_CONTEXT="nautilus"

die() { echo "ERROR: $*" >&2; exit 1; }

# Optional first arg is an existing snapshot path. Anything starting with -- is a
# summarizer override, not a snapshot path.
SNAPSHOT=""
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  SNAPSHOT="$1"
  shift
fi

if [[ -n "$SNAPSHOT" ]]; then
  [[ -f "$SNAPSHOT" ]] || die "snapshot not found: $SNAPSHOT"
  echo "Reusing snapshot: $SNAPSHOT" >&2
else
  # Guard: right cluster. Refuse to snapshot anything but nautilus.
  ctx="$(kubectl config current-context 2>/dev/null)" \
    || die "cannot read kubectl context. Is kubectl configured?"
  [[ "$ctx" == "$EXPECTED_CONTEXT" ]] \
    || die "kubectl context is '$ctx', expected '$EXPECTED_CONTEXT'. Refusing to snapshot the wrong cluster."

  # Guard: API reachable. list is the only node verb cmpm118 holds, so it is
  # also the cheapest reachability probe. A failure here usually means VPN down.
  kubectl auth can-i list nodes >/dev/null 2>&1 \
    || die "cannot reach the Kubernetes API or list nodes. Is the UCSC VPN up?"

  mkdir -p "$RAW_DIR"
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  SNAPSHOT="$RAW_DIR/nodes_${ts}.json"
  echo "Taking read-only node snapshot -> $SNAPSHOT (UTC $ts)" >&2
  if ! kubectl get nodes -o json > "$SNAPSHOT"; then
    rm -f "$SNAPSHOT"
    die "kubectl get nodes failed. Is the UCSC VPN up?"
  fi
fi

# Guard: snapshot parses and has at least one node. Never print the snapshot.
node_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("items",[])))' "$SNAPSHOT" 2>/dev/null)" \
  || die "snapshot is not valid JSON: $SNAPSHOT"
[[ "${node_count:-0}" -gt 0 ]] || die "snapshot contains zero nodes: $SNAPSHOT"
echo "Snapshot OK: $node_count nodes" >&2

mkdir -p "$PROCESSED_DIR"
# Default out-dir first so it works from any cwd; a user --out-dir in "$@" wins.
exec python3 "$SUMMARIZER" "$SNAPSHOT" --out-dir "$PROCESSED_DIR" "$@"
