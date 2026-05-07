"""
psbc/kkt.py – KKT solver for PSBC budget allocation.

Given a set of nodes with quality-function parameters and a free budget,
solves the weighted KKT optimisation problem:

    max  Σ_j  w_j · σ_j · q̄_pa(j) · (1 - exp(-κ_j (b_j - τ_j)))
    s.t. Σ_j  p_out · b_j  ≤  R_free
         b_j ≥ τ_j   for all j

Closed-form solution per node:
    b_j*(λ) = τ_j + (1/κ_j) · ln( σ̃_j·κ_j / (λ · p_out) )

where  σ̃_j = w_j · σ_j · q̄_pa(j)  (effective weighted quality potential).

λ* is found by binary search on the budget constraint, with a violation-
removal loop to handle nodes that would be assigned below their lower bound τ_j.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeParams:
    """
    Parameters for one node in the KKT problem.

    sigma  : σ_j  – max quality potential ∈ (0, 1]
    kappa  : κ_j  – response rate > 0
    tau    : τ_j  – emergence threshold ≥ 0  (lower bound on b_j)
    weight : w_j  – topological importance weight > 0
    q_pa   : q̄_pa(j) – weighted mean quality of parent nodes (1.0 for source)
    """
    node_id: str
    sigma:   float
    kappa:   float
    tau:     float
    weight:  float
    q_pa:    float = 1.0
    b_max:   float = float("inf")   # optional upper bound on allocation

    def sigma_tilde(self) -> float:
        """σ̃_j = w_j · σ_j · q̄_pa(j)"""
        return self.weight * self.sigma * self.q_pa


@dataclass
class AllocationResult:
    """Output of the KKT solver for one full round."""
    allocations:  Dict[str, float]   # node_id → b_j* (max_tokens)
    lambda_star:  float              # optimal Lagrange multiplier
    normal_mode:  bool               # True = KKT, False = survival
    # Which nodes were pinned at their lower bound τ_j
    pinned_nodes: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Core solver
# ──────────────────────────────────────────────────────────────────────────────

def _b_star(node: NodeParams, lam: float, p_out: float) -> float:
    """Closed-form allocation for one node given λ."""
    st = node.sigma_tilde()
    if st <= 0 or node.kappa <= 0 or lam <= 0:
        return node.tau
    inner = st * node.kappa / (lam * p_out)
    if inner <= 0:
        return node.tau
    return node.tau + math.log(inner) / node.kappa


def _budget_at_lambda(
    active: List[NodeParams],
    lam: float,
    p_out: float,
) -> float:
    """Total budget consumed by all active nodes at Lagrange multiplier λ."""
    return sum(max(_b_star(n, lam, p_out), n.tau) * p_out for n in active)


def kkt_solve(
    nodes: List[NodeParams],
    R_free: float,
    p_out: float,
    bisect_tol: float = 1e-9,
    bisect_max_iter: int = 200,
) -> AllocationResult:
    """
    Solve the KKT budget allocation problem using binary search + violation
    removal.

    Parameters
    ----------
    nodes      : list of NodeParams for all nodes to allocate (current + downstream)
    R_free     : free budget available for output tokens (in monetary units)
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
        # No free budget at all – pin everything at τ
        allocs = {n.node_id: n.tau for n in nodes}
        return AllocationResult(allocs, float("inf"), False, [n.node_id for n in nodes])

    active = list(nodes)
    pinned: Dict[str, float] = {}

    # Violation-removal loop
    for _ in range(len(nodes) + 1):
        # Budget remaining after pinned nodes
        pinned_cost = sum(b * p_out for b in pinned.values())
        budget_for_active = R_free - pinned_cost

        if budget_for_active <= 0 or not active:
            # Pin remaining active nodes at τ as well
            for n in active:
                pinned[n.node_id] = n.tau
            active = []
            break

        # Minimum budget needed if all active nodes are at their lower bound
        min_budget = sum(n.tau * p_out for n in active)
        if min_budget >= budget_for_active:
            # Even τ allocation exceeds budget – pin all at τ proportionally
            # (shouldn't normally happen after reserve logic, but safe fallback)
            for n in active:
                pinned[n.node_id] = n.tau
            active = []
            break

        # Binary search for λ* such that Σ b_j*(λ) · p_out = budget_for_active
        # Upper bound on λ: when λ→∞, b_j*→τ_j  (all pinned at lower bound)
        # Lower bound on λ: when λ→0, b_j*→∞
        # We need a finite upper bound where total budget ≤ budget_for_active
        lam_lo = 1e-15
        lam_hi = max(n.sigma_tilde() * n.kappa for n in active) / p_out

        # Ensure lam_hi gives budget ≤ budget_for_active
        while _budget_at_lambda(active, lam_hi, p_out) > budget_for_active:
            lam_hi *= 2.0

        lam = lam_lo
        for _ in range(bisect_max_iter):
            lam = 0.5 * (lam_lo + lam_hi)
            b_total = _budget_at_lambda(active, lam, p_out)
            if abs(b_total - budget_for_active) / max(budget_for_active, 1.0) < bisect_tol:
                break
            if b_total > budget_for_active:
                lam_lo = lam
            else:
                lam_hi = lam

        # Compute proposed allocations  (clamp to [τ, b_max])
        proposed = {
            n.node_id: max(min(_b_star(n, lam, p_out), n.b_max), n.tau)
            for n in active
        }

        # Violation check: nodes clamped at upper bound (b_max)
        upper_violated = [n for n in active
                          if _b_star(n, lam, p_out) > n.b_max and n.b_max < float("inf")]
        if upper_violated:
            for n in upper_violated:
                pinned[n.node_id] = n.b_max
            active = [n for n in active if n.node_id not in pinned]
            continue

        # Violation check: any node below τ?
        violated = [n for n in active if proposed[n.node_id] < n.tau - 1e-9]
        if violated:
            for n in violated:
                pinned[n.node_id] = n.tau
            active = [n for n in active if n.node_id not in pinned]
            continue

        # No violations – we're done
        pinned.update(proposed)
        break

    all_allocs = pinned
    lam_final = lam if active else float("inf")
    normal = len(pinned) == len(nodes) and all(
        pinned.get(n.node_id, 0) > n.tau + 1e-9 for n in nodes
    )

    return AllocationResult(
        allocations=all_allocs,
        lambda_star=lam_final,
        normal_mode=bool(active) or any(
            pinned.get(n.node_id, 0) > n.tau + 1e-9 for n in nodes
        ),
        pinned_nodes=[n.node_id for n in nodes if pinned.get(n.node_id, n.tau + 1) <= n.tau + 1e-9],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Survival-mode allocator  (Phase A3)
# ──────────────────────────────────────────────────────────────────────────────

def survival_allocate(
    nodes: List[NodeParams],
    R_safe: float,
    p_out: float,
    importance_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Importance-weighted proportional allocation under extreme budget pressure.

    b_i* = (R_safe / p_out) · (imp_i·τ_i / Σ_{k} imp_k·τ_k)

    If importance_weights is None, falls back to node.weight (topo_weights).
    If Σ imp_k·τ_k == 0 (all τ are zero), fall back to equal shares.
    """
    if R_safe <= 0:
        return {n.node_id: 0.0 for n in nodes}

    total_tokens = R_safe / p_out
    if importance_weights is not None:
        denom = sum(importance_weights.get(n.node_id, 0.0) * n.tau for n in nodes)
    else:
        denom = sum(n.weight * n.tau for n in nodes)

    if denom <= 0:
        share = total_tokens / len(nodes) if nodes else 0.0
        return {n.node_id: share for n in nodes}

    if importance_weights is not None:
        return {
            n.node_id: total_tokens * (importance_weights.get(n.node_id, 0.0) * n.tau) / denom
            for n in nodes
        }
    return {n.node_id: total_tokens * (n.weight * n.tau) / denom for n in nodes}
