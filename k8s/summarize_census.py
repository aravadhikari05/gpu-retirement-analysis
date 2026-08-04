#!/usr/bin/env python3
"""Summarize a read-only Nautilus node snapshot into a GPU fleet census.

Reads a `kubectl get nodes -o json` snapshot and emits two committed tables,
data/processed/census_nodes.csv and data/processed/census_fleet.csv, plus a
readable terminal summary.

Label and resource keys were discovered from a real snapshot, not assumed. They
are overridable so the census survives a cluster relabel. See
docs/tasks/phase0-census.md for the rationale on using Python over jq.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

# Discovered defaults. Confirm against `kubectl describe node` before trusting.
PRODUCT_KEY = "nvidia.com/gpu.product"
SITE_KEY = "netbox.io/site"
REGION_KEY = "topology.kubernetes.io/region"
DRIVER_KEY = "nvidia.com/cuda.driver-version.full"
GPU_RESOURCE = "nvidia.com/gpu"
MIG_RESOURCE_RE = re.compile(r"^nvidia\.com/mig-")

# Taint effects that block an untolerated pod. PreferNoSchedule is soft.
BLOCKING_EFFECTS = {"NoSchedule", "NoExecute"}


def labels(node):
    return node["metadata"].get("labels") or {}


def is_ready(node):
    for c in node["status"].get("conditions") or []:
        if c.get("type") == "Ready":
            return c.get("status") == "True"
    return False


def is_cordoned(node):
    return bool(node["spec"].get("unschedulable"))


def taints(node):
    return node["spec"].get("taints") or []


def has_blocking_taint(node):
    return any(t.get("effect") in BLOCKING_EFFECTS for t in taints(node))


def int_or_zero(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def gpu_adjacent_labels(node):
    """GPU/NVIDIA-related labels a node carries, for hand inspection."""
    pat = re.compile(r"gpu|nvidia", re.I)
    return {k: v for k, v in labels(node).items() if pat.search(k)}


def derive(node, args):
    lab = labels(node)
    cap = node["status"].get("capacity") or {}
    alloc = node["status"].get("allocatable") or {}

    mig_keys = sorted(k for k in cap if MIG_RESOURCE_RE.match(k))
    mig_alloc = sum(int_or_zero(alloc.get(k)) for k in mig_keys)

    product = lab.get(args.product_key)
    is_mig = (product is not None and "MIG" in product.upper()) or mig_alloc > 0

    ready = is_ready(node)
    cordoned = is_cordoned(node)
    blocking = has_blocking_taint(node)

    return {
        "node": node["metadata"]["name"],
        "product": product or "",
        "site": lab.get(args.site_key, ""),
        "region": lab.get(args.region_key, ""),
        "ready": ready,
        "cordoned": cordoned,
        "allocatable_gpu": int_or_zero(alloc.get(args.gpu_resource)),
        "capacity_gpu": int_or_zero(cap.get(args.gpu_resource)),
        "mig_resources": ";".join(mig_keys),
        "mig_allocatable": mig_alloc,
        "measurement_viable": not is_mig,
        # Openly schedulable: the conservative lower bound (node reachability).
        "schedulable": ready and not cordoned and not blocking,
        # Same, but the node must actually advertise an allocatable GPU. This is
        # the number that determines whether we can get repetitions on a model.
        "schedulable_with_gpu": (
            ready and not cordoned and not blocking
            and int_or_zero(alloc.get(args.gpu_resource)) > 0
        ),
        # Ceiling if reserved-pool taints turn out tolerable from cmpm118.
        "tainted_but_ready": ready and not cordoned and blocking,
        "taints": ";".join(
            f"{t.get('key')}:{t.get('effect')}" for t in taints(node)
        ),
        "driver_version": lab.get(args.driver_key, ""),
        "has_product": product is not None,
        "has_gpu_signal": (
            product is not None
            or args.gpu_resource in cap
            or bool(mig_keys)
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="path to nodes_<UTC>.json snapshot")
    ap.add_argument("--product-key", default=PRODUCT_KEY)
    ap.add_argument("--site-key", default=SITE_KEY)
    ap.add_argument("--region-key", default=REGION_KEY)
    ap.add_argument("--driver-key", default=DRIVER_KEY)
    ap.add_argument("--gpu-resource", default=GPU_RESOURCE)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    snapshot_name = os.path.basename(args.snapshot)
    m = re.search(r"nodes_(.+)\.json$", snapshot_name)
    snapshot_ts = m.group(1) if m else ""

    with open(args.snapshot) as f:
        nodes = json.load(f)["items"]

    rows = [derive(n, args) for n in nodes]
    gpu_rows = [r for r in rows if r["has_gpu_signal"]]

    # Nodes with GPU capacity/allocatable but no product label.
    unlabeled = [
        r for r in gpu_rows
        if not r["has_product"]
        and (r["capacity_gpu"] > 0 or r["allocatable_gpu"] > 0 or r["mig_resources"])
    ]
    unlabeled_names = {r["node"] for r in unlabeled}

    write_node_csv(gpu_rows, unlabeled_names, snapshot_name, snapshot_ts, args)
    fleet = build_fleet(gpu_rows, unlabeled_names)
    write_fleet_csv(fleet, snapshot_name, snapshot_ts, args)

    # GPU-adjacent nodes with no product label that also carry no GPU capacity.
    orphans = [
        r for r in gpu_rows
        if not r["has_product"] and r["node"] not in unlabeled_names
    ]

    print_fleet_table(fleet)
    print_gap_analysis(rows, args, len(unlabeled_names))
    print_unlabeled(nodes, unlabeled_names)
    if orphans:
        print(f"\n=== GPU-adjacent but no product label AND no allocatable GPU: "
              f"{len(orphans)} nodes (excluded from fleet) ===")
        for r in sorted(orphans, key=lambda x: x["node"]):
            print(f"  {r['node']}  ready={r['ready']} "
                  f"cap_gpu={r['capacity_gpu']} gpu_resource_key_present")
    print(f"\nwrote {args.out_dir}/census_nodes.csv and census_fleet.csv")
    print(f"provenance: {snapshot_name} (UTC {snapshot_ts})")


def write_node_csv(gpu_rows, unlabeled_names, snap, ts, args):
    path = os.path.join(args.out_dir, "census_nodes.csv")
    cols = [
        "snapshot", "snapshot_ts", "node", "product", "site", "region",
        "ready", "cordoned", "allocatable_gpu", "capacity_gpu",
        "mig_resources", "mig_allocatable", "measurement_viable",
        "schedulable", "schedulable_with_gpu", "tainted_but_ready", "taints",
        "driver_version", "flag",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(gpu_rows, key=lambda x: (x["product"], x["node"])):
            out = dict(r)
            out["snapshot"] = snap
            out["snapshot_ts"] = ts
            out["flag"] = "capacity_no_product" if r["node"] in unlabeled_names else ""
            w.writerow(out)


def build_fleet(gpu_rows, unlabeled_names):
    by_model = defaultdict(list)
    for r in gpu_rows:
        if r["node"] in unlabeled_names:
            continue  # no model to attribute to
        if not r["product"]:
            continue  # GPU-adjacent but unlabeled and no capacity; not a model
        by_model[r["product"]].append(r)

    fleet = []
    for product, members in by_model.items():
        taint_keys = Counter()
        for r in members:
            for t in filter(None, r["taints"].split(";")):
                key = t.split(":", 1)[0]
                taint_keys[key] += 1
        drivers = sorted({r["driver_version"] for r in members if r["driver_version"]})
        openly = sum(1 for r in members if r["schedulable"])
        openly_gpu = sum(1 for r in members if r["schedulable_with_gpu"])
        fleet.append({
            "product": product,
            "node_count": len(members),
            "ready_nodes": sum(1 for r in members if r["ready"]),
            "openly_schedulable": openly,
            "openly_schedulable_with_gpu": openly_gpu,
            "tainted_but_ready": sum(1 for r in members if r["tainted_but_ready"]),
            "allocatable_gpu_sum": sum(r["allocatable_gpu"] for r in members),
            "mig_allocatable_sum": sum(r["mig_allocatable"] for r in members),
            "distinct_sites": len({r["site"] for r in members if r["site"]}),
            "distinct_regions": len({r["region"] for r in members if r["region"]}),
            "taint_keys": ";".join(f"{k}({c})" for k, c in taint_keys.most_common(5)),
            "driver_versions": ";".join(drivers),
            "measurement_viable": all(r["measurement_viable"] for r in members),
            # Risk keys off usable GPUs, not bare node reachability.
            "coverage_risk": openly_gpu < 3,
        })
    return sorted(fleet, key=lambda x: (-x["node_count"], x["product"]))


def write_fleet_csv(fleet, snap, ts, args):
    path = os.path.join(args.out_dir, "census_fleet.csv")
    cols = [
        "snapshot", "snapshot_ts", "product", "node_count", "ready_nodes",
        "openly_schedulable", "openly_schedulable_with_gpu",
        "tainted_but_ready", "allocatable_gpu_sum",
        "mig_allocatable_sum", "distinct_sites", "distinct_regions",
        "taint_keys", "driver_versions", "measurement_viable", "coverage_risk",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in fleet:
            out = dict(r)
            out["snapshot"] = snap
            out["snapshot_ts"] = ts
            w.writerow(out)


def print_fleet_table(fleet):
    print("=== GPU fleet census (per model) ===")
    print("sched = ready+uncordoned+untainted;  swg = sched AND allocatable GPU > 0")
    hdr = f"{'model':<42} {'nodes':>5} {'ready':>5} {'sched':>5} {'swg':>4} {'t+rdy':>5} {'alloc':>5} {'site':>4} {'rgn':>3} {'viable':>6} {'risk':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in fleet:
        print(
            f"{r['product']:<42} {r['node_count']:>5} {r['ready_nodes']:>5} "
            f"{r['openly_schedulable']:>5} {r['openly_schedulable_with_gpu']:>4} "
            f"{r['tainted_but_ready']:>5} "
            f"{r['allocatable_gpu_sum']:>5} {r['distinct_sites']:>4} "
            f"{r['distinct_regions']:>3} {str(r['measurement_viable']):>6} "
            f"{'RISK' if r['coverage_risk'] else '':>4}"
        )
    risk = [r["product"] for r in fleet if r["coverage_risk"]]
    print(f"\ncoverage risk (< 3 openly schedulable WITH GPU): "
          f"{len(risk)} of {len(fleet)} models")


def print_gap_analysis(rows, args, unlabeled_count):
    """Requirement A: product-labeled nodes advertising no allocatable GPU."""
    labeled = [r for r in rows if r["has_product"]]
    zero = [r for r in labeled if r["allocatable_gpu"] == 0]
    labeled_with_gpu = len(labeled) - len(zero)
    print("\n=== 246 vs 185 gap: product-labeled nodes with 0 allocatable "
          f"{args.gpu_resource} ===")
    print(f"labeled nodes: {len(labeled)}; "
          f"with allocatable {args.gpu_resource} > 0: {labeled_with_gpu}; "
          f"with 0: {len(zero)}")
    total_advertising = labeled_with_gpu + unlabeled_count
    print(f"\nreconciliation: {total_advertising} nodes advertise a GPU "
          f"= {labeled_with_gpu} product-labeled with allocatable {args.gpu_resource} "
          f"+ {unlabeled_count} with GPU capacity but no product label. "
          "The two sets are disjoint (product present vs absent), so the "
          "GPU-advertising total does not move; it only splits by label presence.")

    buckets = defaultdict(list)
    for r in zero:
        if r["mig_resources"] or (r["product"] and "MIG" in r["product"].upper()):
            b = "MIG (capacity under nvidia.com/mig-*)"
        elif not r["ready"]:
            b = "NotReady"
        elif r["cordoned"]:
            b = "Cordoned (ready)"
        elif r["capacity_gpu"] > 0:
            # GPU exists on the node but none is allocatable: reserved/withheld.
            b = "Ready, uncordoned, capacity>0 but 0 allocatable (reserved/withheld)"
        else:
            b = "Ready, uncordoned, capacity==0 (device plugin never advertised)"
        buckets[b].append(r)

    for b in [
        "MIG (capacity under nvidia.com/mig-*)",
        "NotReady",
        "Cordoned (ready)",
        "Ready, uncordoned, capacity>0 but 0 allocatable (reserved/withheld)",
        "Ready, uncordoned, capacity==0 (device plugin never advertised)",
    ]:
        members = buckets.get(b, [])
        print(f"\n  [{b}] {len(members)} nodes")
        models = Counter(r["product"] for r in members)
        for model, c in models.most_common():
            print(f"      {c:>3}  {model}")


def print_unlabeled(nodes, unlabeled_names):
    """Requirement 3: name unlabeled-with-capacity nodes and their GPU labels."""
    print(f"\n=== GPU capacity but NO product label: {len(unlabeled_names)} nodes ===")
    by_name = {n["metadata"]["name"]: n for n in nodes}
    for name in sorted(unlabeled_names):
        node = by_name[name]
        adj = gpu_adjacent_labels(node)
        print(f"\n  {name}")
        if adj:
            for k, v in sorted(adj.items()):
                print(f"      {k} = {v}")
        else:
            print("      (no gpu/nvidia labels at all)")


if __name__ == "__main__":
    main()
