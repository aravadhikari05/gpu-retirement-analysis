"""Reports GPUs that are free on NRP right now, by model.

Why this exists. `k8s/inventory.sh` and `k8s/summarize_census.py` describe what
the fleet *contains*, read through our own namespace's RBAC. They structurally
cannot answer "is a card free at this moment": listing pods cluster-wide and
reading individual node objects are both forbidden to a `cmpm118` user, and a
Node object carries capacity, never allocation. The only way to learn
availability through kubectl was to launch a probe pod per model and see which
ones scheduled.

nrp.ai/viz/resources renders the same numbers in a browser. It is a client-side
page that calls a public JSON-RPC endpoint, `guest.ListNodeInfo` on
https://portal.nrp.ai/rpc, which needs no authentication and returns every node
with GPUCapacity, GPUAvailable, taints, driver and CUDA version. This module
reads that endpoint.

Two cautions, both learned by comparing the feed against real probe pods on
2026-08-23:

  1. It is a snapshot with lag. On that comparison the feed agreed with 8 of 10
     placement probes. It reported 2 free A10s on untainted nodes while an
     actual pod requesting 1 A10, 1 CPU and 1 GiB stayed Pending. Use this to
     filter candidates, then confirm with a probe before planning a run.
  2. Free is not the same as reachable. All 96 L4s sit behind
     `nautilus.io/reservation=csuf:NoSchedule` and showed 49 free while being
     unschedulable for us. The `gpu_free_open` column, not `gpu_free`, is the
     number to plan against. Do not add tolerations for another institution's
     reservation taint to get at them.

A `PreferNoSchedule` taint is soft and is not counted as blocking;
`nvidia.com/gpu=Exists:PreferNoSchedule` is on essentially every GPU node.
"""

import argparse
import collections
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

NRP_RPC_URL = "https://portal.nrp.ai/rpc"
RPC_METHOD = "guest.ListNodeInfo"
DEFAULT_TIMEOUT_S = 60
# Effects that actually stop a pod from landing. PreferNoSchedule is a
# preference, not a barrier, and every GPU node carries one.
BLOCKING_TAINT_EFFECTS = ("NoSchedule", "NoExecute")


def fetch_nodes(
    url: str = NRP_RPC_URL, timeout_s: int = DEFAULT_TIMEOUT_S
) -> list[dict]:
    """Fetches every node record from the NRP portal's JSON-RPC endpoint.

    Args:
      url: JSON-RPC endpoint. Defaults to the public NRP portal.
      timeout_s: Socket timeout in seconds.

    Returns:
      The list of node dicts as returned by the service.

    Raises:
      RuntimeError: if the response carries a JSON-RPC error or no Nodes key.
    """
    payload = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": RPC_METHOD, "params": {}}
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = json.load(response)

    if "error" in body:
        raise RuntimeError(f"{RPC_METHOD} returned an error: {body['error']}")
    nodes = body.get("result", {}).get("Nodes")
    if nodes is None:
        raise RuntimeError(f"{RPC_METHOD} response had no result.Nodes key")
    logger.info("fetched %d nodes from %s", len(nodes), url)
    return nodes


def _as_int(value: object) -> int:
    """Coerces the feed's GPU counts to int. They arrive as strings."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def is_blocked(node: dict) -> bool:
    """True if a plain pod could not land on this node.

    Args:
      node: One node record from the feed.

    Returns:
      True when the node is cordoned or carries a NoSchedule/NoExecute taint.
    """
    if node.get("IsUnschedulable"):
        return True
    return any(
        taint.get("effect") in BLOCKING_TAINT_EFFECTS
        for taint in (node.get("Taints") or [])
    )


def summarize(nodes: list[dict]) -> list[dict]:
    """Aggregates node records into one row per GPU model.

    Args:
      nodes: Node records from fetch_nodes.

    Returns:
      Rows sorted by gpu_free_open descending. Columns are
      lowercase_snake_case per the repo's data conventions.
    """
    by_model: dict[str, dict] = collections.defaultdict(
        lambda: {
            "nodes": 0,
            "gpu_capacity": 0,
            "gpu_free": 0,
            "gpu_free_open": 0,
            "nodes_open_with_free": 0,
            "drivers": set(),
        }
    )
    for node in nodes:
        model = node.get("GPUType") or ""
        if not model:
            continue
        free = _as_int(node.get("GPUAvailable"))
        row = by_model[model]
        row["nodes"] += 1
        row["gpu_capacity"] += _as_int(node.get("GPUCapacity"))
        row["gpu_free"] += free
        if node.get("GPUDriver"):
            row["drivers"].add(node["GPUDriver"])
        if not is_blocked(node):
            row["gpu_free_open"] += free
            if free > 0:
                row["nodes_open_with_free"] += 1

    rows = []
    for model, row in by_model.items():
        rows.append(
            {
                "gpu_model": model,
                "nodes": row["nodes"],
                "gpu_capacity": row["gpu_capacity"],
                "gpu_free": row["gpu_free"],
                "gpu_free_open": row["gpu_free_open"],
                "nodes_open_with_free": row["nodes_open_with_free"],
                "drivers": ",".join(sorted(row["drivers"])),
            }
        )
    return sorted(rows, key=lambda r: (-r["gpu_free_open"], -r["gpu_free"]))


def open_nodes_for(nodes: list[dict], model: str) -> list[dict]:
    """Lists schedulable nodes of one model that have a free GPU.

    Args:
      nodes: Node records from fetch_nodes.
      model: Exact `nvidia.com/gpu.product` value, for example "NVIDIA-L40S".

    Returns:
      Rows with node name, free GPUs, free CPU, driver and CUDA version,
      most free GPUs first.
    """
    out = []
    for node in nodes:
        if node.get("GPUType") != model or is_blocked(node):
            continue
        free = _as_int(node.get("GPUAvailable"))
        if free <= 0:
            continue
        out.append(
            {
                "node_name": node.get("Name", ""),
                "gpu_free": free,
                "cpu_free": node.get("CPUAvailable", 0),
                "gpu_driver": node.get("GPUDriver", ""),
                "cuda": node.get("CUDA", ""),
            }
        )
    return sorted(out, key=lambda r: -r["gpu_free"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Report GPUs free on NRP right now, by model"
    )
    parser.add_argument("--url", default=NRP_RPC_URL)
    parser.add_argument(
        "--model",
        default="",
        help="Exact gpu.product value. Lists that model's open nodes instead "
        "of the fleet summary.",
    )
    parser.add_argument(
        "--min-free",
        type=int,
        default=1,
        help="Hide models with fewer than this many free GPUs on open nodes.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON rather than a table."
    )
    args = parser.parse_args()

    nodes = fetch_nodes(args.url)

    if args.model:
        rows = open_nodes_for(nodes, args.model)
        if args.json:
            print(json.dumps(rows, indent=2))
            return
        if not rows:
            print(f"No schedulable node of {args.model} has a free GPU.")
            return
        print(
            f"{'node_name':40s} {'gpu_free':>8} {'cpu_free':>8} {'driver':>10} {'cuda':>6}"
        )
        for row in rows:
            print(
                f"{row['node_name']:40s} {row['gpu_free']:8d} "
                f"{str(row['cpu_free']):>8} {row['gpu_driver']:>10} {row['cuda']:>6}"
            )
        return

    rows = [r for r in summarize(nodes) if r["gpu_free_open"] >= args.min_free]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(
        f"{'gpu_model':38s} {'free_open':>9} {'free':>6} {'capacity':>8} "
        f"{'nodes':>6} {'open_nodes':>10}"
    )
    for row in rows:
        print(
            f"{row['gpu_model']:38s} {row['gpu_free_open']:9d} {row['gpu_free']:6d} "
            f"{row['gpu_capacity']:8d} {row['nodes']:6d} {row['nodes_open_with_free']:10d}"
        )
    print(
        "\nfree_open is the number to plan against: free counts GPUs behind "
        "reservation taints we cannot use.\nSnapshot with lag. Confirm with a "
        "probe pod before committing a run."
    )


if __name__ == "__main__":
    main()
