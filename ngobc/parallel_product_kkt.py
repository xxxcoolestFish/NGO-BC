"""
psbc/parallel_product_kkt.py — Product KKT for Parallel/混合 DAG topologies.

Strategy: merge parallel sibling groups into "virtual nodes", then apply
standard product KKT on the reduced pipeline.  The virtual node's budget
is split equally among the real siblings.

Math for k homogeneous solvers (σ, κ, τ):
  Q_pool(b) = k·w·σ·(1 - exp(-(κ/k)·(b - k·τ)))
  → σ_pool = k·w·σ,  κ_pool = κ/k,  τ_pool = k·τ

For heterogeneous solvers, use weighted averages.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ngobc.kkt import NodeParams, AllocationResult


# ═══════════════════════════════════════════════════════════════
# Topology analysis: identify parallel groups
# ═══════════════════════════════════════════════════════════════

def _find_parallel_groups(
    adj: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Identify parallel sibling groups in the DAG.

    A parallel group is a set of nodes that share:
      (a) the same single parent, AND
      (b) the same single child.

    Returns {virtual_group_id: [real_node_ids]}.
    """
    # Build reverse adjacency
    rev: Dict[str, List[str]] = defaultdict(list)
    for u, children in adj.items():
        for v in children:
            if u != "node_source" and v in adj:
                rev[v].append(u)

    # Group siblings by (parent_set, child_set)
    groups: Dict[Tuple, List[str]] = defaultdict(list)
    for node in adj:
        if node == "node_source":
            continue
        parents = tuple(sorted(p for p in rev.get(node, []) if p != "node_source"))
        children = tuple(sorted(adj.get(node, [])))
        if parents and children:
            # Nodes with same (parent, child) pattern are parallel siblings
            key = (parents, children)
            groups[key].append(node)

    # Only groups with ≥2 siblings are truly parallel
    result: Dict[str, List[str]] = {}
    for key, nodes in groups.items():
        if len(nodes) >= 2:
            name = "Pool_" + "_".join(n[:4] for n in nodes)
            result[name] = nodes

    return result


@dataclass
class VirtualNode:
    """Parameters of a merged parallel group."""

    group_id: str
    real_nodes: List[str]
    sigma: float
    kappa: float
    tau: float


def _build_virtual_node(
    group_id: str,
    real_nodes: List[str],
    node_params: Dict[str, NodeParams],
) -> VirtualNode:
    """Create a virtual node from a group of parallel siblings.

    Uses the AVERAGE tau (not sum) because each real node's τ constraint
    is enforced at the real-node level, not the virtual level.
    """
    k = len(real_nodes)
    sigma_sum = 0.0
    kappa_harmonic_sum = 0.0
    tau_max = 0.0  # use max τ to ensure minimum per node
    tau_avg = 0.0

    for nid in real_nodes:
        p = node_params.get(nid)
        if p is None:
            continue
        sigma_sum += p.sigma * p.weight
        if p.kappa > 1e-9:
            kappa_harmonic_sum += 1.0 / p.kappa
        tau_max = max(tau_max, p.tau)
        tau_avg += p.tau

    tau_avg /= max(k, 1)

    # Effective κ = harmonic_mean(κ_j) / k  (budget shared, slower saturation)
    if kappa_harmonic_sum > 0:
        kappa_eff = k / kappa_harmonic_sum / k  # harmonic_mean / k
    else:
        kappa_eff = 0.002

    # Use average tau for the virtual node (individual τ enforced at split time)
    tau_eff = tau_max  # use max to ensure pool has enough for the most demanding node

    return VirtualNode(
        group_id=group_id,
        real_nodes=list(real_nodes),
        sigma=sigma_sum,
        kappa=kappa_eff,
        tau=tau_eff,
    )


def _build_reduced_topology(
    adj: Dict[str, List[str]],
    pools: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Replace parallel groups with virtual nodes in adjacency.

    Real nodes in pools are REMOVED; their edges are rerouted through
    the virtual pool node.
    """
    all_pooled: Set[str] = set()
    for nodes in pools.values():
        all_pooled.update(nodes)

    new_adj: Dict[str, List[str]] = {}
    pool_parents: Dict[str, Set[str]] = defaultdict(set)
    pool_children: Dict[str, Set[str]] = defaultdict(set)

    # Collect pool edges from real nodes
    for u, children in adj.items():
        if u == "node_source":
            new_adj["node_source"] = []
            for v in children:
                if v in all_pooled:
                    for pool_id, members in pools.items():
                        if v in members:
                            pool_parents[pool_id].add("node_source")
                else:
                    if "node_source" not in new_adj["node_source"]:
                        new_adj["node_source"].append(v)
            continue

        if u in all_pooled:
            for v in children:
                if v in all_pooled:
                    # pool → pool edge
                    for p1, m1 in pools.items():
                        if u in m1:
                            for p2, m2 in pools.items():
                                if v in m2:
                                    pool_children[p1].add(p2)
                else:
                    for p1, m1 in pools.items():
                        if u in m1:
                            pool_children[p1].add(v)
        else:
            new_adj.setdefault(u, [])
            for v in children:
                if v in all_pooled:
                    for p2, m2 in pools.items():
                        if v in m2:
                            pool_parents[p2].add(u)
                            if p2 not in new_adj[u]:
                                new_adj[u].append(p2)
                else:
                    if v not in new_adj[u]:
                        new_adj[u].append(v)

    # Add pool nodes
    for pool_id, members in pools.items():
        new_adj.setdefault(pool_id, [])
        # Pool children = children of real nodes (already collected)
        for child in pool_children.get(pool_id, set()):
            if child not in new_adj[pool_id]:
                new_adj[pool_id].append(child)

    # Ensure all non-pooled, non-source nodes exist
    for u, children in adj.items():
        if u == "node_source" or u in all_pooled:
            continue
        new_adj.setdefault(u, [])
        for v in children:
            if v not in all_pooled and v not in new_adj.get(u, []):
                new_adj[u].append(v)

    return new_adj


# ═══════════════════════════════════════════════════════════════
# Main solver
# ═══════════════════════════════════════════════════════════════

def parallel_product_kkt_solve(
    adj: Dict[str, List[str]],
    node_params: Dict[str, NodeParams],
    R_free: float,
    p_out: float,
) -> AllocationResult:
    """Solve budget allocation for a DAG with parallel branches.

    1. Identify parallel groups
    2. Merge groups → virtual pipeline
    3. Solve product KKT on the reduced topology
    4. Split virtual-node allocations equally among real siblings
    """
    from ngobc.product_kkt import product_kkt_solve

    pools = _find_parallel_groups(adj)

    if not pools:
        # No parallel groups → fall back to standard product KKT
        return product_kkt_solve(
            list(node_params.values()), R_free, p_out
        )

    # Build virtual nodes
    virtual_nodes: Dict[str, VirtualNode] = {}
    for group_id, real_nodes in pools.items():
        virtual_nodes[group_id] = _build_virtual_node(
            group_id, real_nodes, node_params
        )

    # Build reduced topology and node list
    reduced_adj = _build_reduced_topology(adj, pools)
    all_pooled: Set[str] = set()
    for nodes in pools.values():
        all_pooled.update(nodes)

    # Build NodeParams for reduced topology
    reduced_params: List[NodeParams] = []
    for nid in reduced_adj:
        if nid == "node_source":
            continue
        if nid in virtual_nodes:
            vn = virtual_nodes[nid]
            reduced_params.append(
                NodeParams(
                    node_id=nid,
                    sigma=vn.sigma,
                    kappa=vn.kappa,
                    tau=vn.tau,
                    weight=1.0,
                )
            )
        elif nid in node_params:
            reduced_params.append(node_params[nid])

    # Solve product KKT on the reduced pipeline
    result = product_kkt_solve(reduced_params, R_free, p_out)

    # Distribute virtual-node allocations among real siblings.
    # First ensure each sibling gets its minimum τ, then split the
    # remainder proportionally to 1/κ (slower-saturating nodes need
    # more budget to reach the same quality).
    final_allocs: Dict[str, float] = {}

    for nid in result.allocations:
        if nid in virtual_nodes:
            vn = virtual_nodes[nid]
            pool_budget = result.allocations[nid]
            # Reserve τ for each real node and compute 1/κ-based proportions
            taus = {}
            kappa_inv = {}
            for real_nid in vn.real_nodes:
                p = node_params.get(
                    real_nid, NodeParams(real_nid, 0.9, 0.01, 0.0, 1.0)
                )
                taus[real_nid] = p.tau
                kappa_inv[real_nid] = 1.0 / max(p.kappa, 1e-9)
            tau_reserved = sum(taus.values())
            remaining = max(pool_budget - tau_reserved, 0.0)
            total_w = sum(kappa_inv.values())
            for real_nid in vn.real_nodes:
                share = remaining * kappa_inv[real_nid] / total_w if total_w > 0 else remaining / len(vn.real_nodes)
                final_allocs[real_nid] = taus[real_nid] + share
        else:
            final_allocs[nid] = result.allocations[nid]

    # Budget enforcement: if total exceeds R_free, scale all non-terminal
    # nodes proportionally to the excess (each gets τ, then equal share).
    total_cost = sum(final_allocs.values()) * p_out
    if total_cost > R_free + 1e-9:
        # Scale down proportionally but keep each node ≥ τ
        scale = R_free / total_cost
        for nid in final_allocs:
            tau_j = node_params.get(
                nid, NodeParams(nid, 0.9, 0.01, 0.0, 1.0)
            ).tau
            final_allocs[nid] = max(final_allocs[nid] * scale, tau_j)

    return AllocationResult(
        allocations=final_allocs,
        lambda_star=result.lambda_star,
        normal_mode=result.normal_mode,
    )


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test: Parallel topology D → [S1,S2,S3] → A
    adj = {
        "node_source": ["D"],
        "D": ["S1", "S2", "S3"],
        "S1": ["A"],
        "S2": ["A"],
        "S3": ["A"],
        "A": [],
    }

    import json
    em = json.load(open("psbc_logs/em_params_parallel.json"))
    node_params = {}
    for nid in ["D", "S1", "S2", "S3", "A"]:
        for role, v in em.items():
            if role.lower() in nid.lower():
                sigma, kappa, tau = v["sigma"], v["kappa"], v["tau_quality"]
                break
        else:
            sigma, kappa, tau = 0.9, 0.01, 0.0
        node_params[nid] = NodeParams(nid, sigma, kappa, tau, 1.0)

    p_out = 2.9e-7
    for budget in [0.00042, 0.00092, 0.00133, 0.00308]:
        result = parallel_product_kkt_solve(adj, node_params, budget, p_out)
        total = sum(result.allocations.values()) * p_out
        print(f"Budget {budget:.5f}:")
        for nid in ["D", "S1", "S2", "S3", "A"]:
            print(f"  {nid:4s}: {result.allocations.get(nid, 0):.0f}")
        print(f"  total cost: {total:.5f}")
        print()
