"""
psbc/product_kkt.py – Direction 4: product-objective KKT solver.

For pipeline topology, final quality = Π_j Q_j(b_j), not Σ_j w_j·Q_j.
The product objective ensures no single node is starved (any Q_j≈0 kills
the entire product).

Optimisation problem:
    max  Σ_j log Q_j(b_j)
    s.t. Σ_j p_out·b_j ≤ R_free,  b_j ≥ τ_j

Budget-aware temperature T (≥1) modulates perceived saturation speed:
  - T=1 (tight budget): original product KKT, precise marginal allocation.
  - T>1 (generous budget): nodes appear to saturate slower → all get more tokens.
  - T→∞: b_j* → τ_j + 1/(λ·p_out) — equal increment above τ for all nodes.

Closed form with temperature:
    b_j*(λ, T) = τ_j + (T/κ_j) · ln(1 + κ_j / (T·λ·p_out))
"""

from __future__ import annotations

import math
from typing import Dict, List

from ngobc.kkt import NodeParams, AllocationResult


def _b_star_product(
    node: NodeParams, lam: float, p_out: float, weight: float = 1.0
) -> float:
    """Closed-form allocation for one node under (weighted) product objective.

    b_j*(λ) = τ_j + (1/κ_j) · ln(1 + w_j·κ_j / (λ·p_out))
    """
    if node.kappa <= 0 or lam <= 0:
        return node.tau
    inner = 1.0 + weight * node.kappa / (lam * p_out)
    if inner <= 0:
        return node.tau
    return node.tau + math.log(inner) / node.kappa


def _budget_at_lambda(
    active: List[NodeParams],
    lam: float,
    p_out: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Total budget consumed by all active nodes at Lagrange multiplier λ."""
    total = 0.0
    for n in active:
        w = weights.get(n.node_id, 1.0) if weights else 1.0
        total += max(_b_star_product(n, lam, p_out, w), n.tau) * p_out
    return total


def product_kkt_solve(
    nodes: List[NodeParams],
    R_free: float,
    p_out: float,
    weights: Optional[Dict[str, float]] = None,
    bisect_tol: float = 1e-9,
    bisect_max_iter: int = 200,
) -> AllocationResult:
    """
    Solve the product-objective KKT budget allocation problem.

    Parameters
    ----------
    nodes      : list of NodeParams
    R_free     : free budget available for output tokens (monetary units)
    p_out      : output token unit price
    bisect_tol : convergence tolerance for binary search on λ
    bisect_max_iter : safety cap on binary search iterations

    Returns
    -------
    AllocationResult with b_j* for every node.
    """
    if not nodes:
        return AllocationResult({}, 0.0, True, [])

    if R_free <= 0:
        allocs = {n.node_id: n.tau for n in nodes}
        return AllocationResult(allocs, float("inf"), False,
                                [n.node_id for n in nodes])

    active = list(nodes)
    pinned: Dict[str, float] = {}

    for _ in range(len(nodes) + 1):
        pinned_cost = sum(b * p_out for b in pinned.values())
        budget_for_active = R_free - pinned_cost

        if budget_for_active <= 0 or not active:
            for n in active:
                pinned[n.node_id] = n.tau
            active = []
            break

        min_budget = sum(n.tau * p_out for n in active)
        if min_budget >= budget_for_active:
            for n in active:
                pinned[n.node_id] = n.tau
            active = []
            break

        # Binary search for λ*
        lam_lo = 1e-15
        # Upper bound: when λ→∞, b_j*→τ_j. Start from a reasonable value.
        lam_hi = max(n.kappa for n in active) / p_out
        while _budget_at_lambda(active, lam_hi, p_out, weights) > budget_for_active:
            lam_hi *= 2.0

        lam = lam_lo
        for _ in range(bisect_max_iter):
            lam = 0.5 * (lam_lo + lam_hi)
            b_total = _budget_at_lambda(active, lam, p_out, weights)
            if abs(b_total - budget_for_active) / max(budget_for_active, 1.0) < bisect_tol:
                break
            if b_total > budget_for_active:
                lam_lo = lam
            else:
                lam_hi = lam

        # Compute proposed allocations (clamp to [τ, b_max])
        proposed = {}
        for n in active:
            w = weights.get(n.node_id, 1.0) if weights else 1.0
            proposed[n.node_id] = max(
                min(_b_star_product(n, lam, p_out, w), n.b_max), n.tau
            )

        # Upper bound violation check (b_max)
        upper_violated = []
        for n in active:
            w = weights.get(n.node_id, 1.0) if weights else 1.0
            if _b_star_product(n, lam, p_out, w) > n.b_max and n.b_max < float("inf"):
                upper_violated.append(n)
        if upper_violated:
            for n in upper_violated:
                pinned[n.node_id] = n.b_max
            active = [n for n in active if n.node_id not in pinned]
            continue

        # Lower bound violation check (τ)
        violated = [n for n in active if proposed[n.node_id] < n.tau - 1e-9]
        if violated:
            for n in violated:
                pinned[n.node_id] = n.tau
            active = [n for n in active if n.node_id not in pinned]
            continue

        pinned.update(proposed)
        break

    all_allocs = pinned
    lam_final = lam if active else float("inf")
    normal_mode = bool(active) or any(
        pinned.get(n.node_id, 0) > n.tau + 1e-9 for n in nodes
    )

    return AllocationResult(
        allocations=all_allocs,
        lambda_star=lam_final,
        normal_mode=normal_mode,
        pinned_nodes=[n.node_id for n in nodes
                      if pinned.get(n.node_id, n.tau + 1) <= n.tau + 1e-9],
    )
