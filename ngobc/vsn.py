"""
psbc/vsn.py — Virtual Super Node (VSN) 2D optimisation for PSBC-DT.

In PSBC-DT mode, the number of future nodes is unknown.  Instead of
solving an n-dimensional KKT, we reduce to a 2D problem:

    max  w_i · Q_i(b_i) + Q_VSN(B_VSN)
    s.t. p_out · (b_i + B_VSN) ≤ R_free,  b_i ≥ τ_i

where Q_VSN(B) = E_{G ~ P_pred} [ V*(G, B) ] is the expected optimal
quality of allocating B tokens to the unknown future subgraph.

Since Q_VSN(·) is concave (Lemma 7.1), the 2D problem is convex
and solved by 1D search over b_i ∈ [τ_i, R_free/p_out].
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ngobc.kkt import NodeParams, AllocationResult
from ngobc.agent_pool import AgentPool
from ngobc.topology_predictor import TopologyPredictor, TopologySample


def _quality(node: NodeParams, b: float) -> float:
    """Q_j(b) = σ̃_j · (1 − exp(−κ_j (b − τ_j))) for b ≥ τ_j, else 0."""
    if b < node.tau:
        return 0.0
    return node.sigma_tilde() * (1.0 - math.exp(-node.kappa * (b - node.tau)))


def _compute_q_vsn(
    budget_vsn: float,
    samples: List[TopologySample],
    p_out: float,
) -> float:
    """
    Compute Q_VSN(B) = (1/S) Σ_k V*(sample_k, B).

    For each sampled graph, V* is the optimal product-KKT quality.
    """
    if not samples or budget_vsn <= 0:
        return 0.0

    from ngobc.product_kkt import product_kkt_solve

    total_q = 0.0
    for s in samples:
        nodes = list(s.node_params.values())
        if not nodes:
            continue
        result = product_kkt_solve(nodes, budget_vsn, p_out)
        q = 1.0
        for nd in nodes:
            b = result.allocations.get(nd.node_id, nd.tau)
            q *= _quality(nd, b)
        total_q += q

    return total_q / len(samples)


def vsn_solve(
    current_node: NodeParams,
    R_free: float,
    p_out: float,
    predictor: TopologyPredictor,
    observed_node_ids: List[str],
    use_product: bool = True,
    n_grid: int = 50,
) -> Tuple[float, float, float]:
    """
    VSN 2D optimisation: choose (b_i*, B_VSN*) to maximise
    w_i · Q_i(b_i) + Q_VSN(B_VSN).

    Parameters
    ----------
    current_node : NodeParams for the node being decided
    R_free : free output budget (yuan)
    p_out : output token price
    predictor : TopologyPredictor that generates future samples
    observed_node_ids : already-seen nodes
    use_product : if True, use product quality; if False, use sum
    n_grid : number of grid points for 1D search

    Returns
    -------
    (b_i_star, B_VSN_star, max_total_quality)
    Only b_i* is actually used; B_VSN* is passed to next step's R_free seed.
    """
    tau_i = current_node.tau
    max_tokens = max(R_free / p_out, tau_i)

    # Sample future topologies once (shared across all grid points)
    samples = predictor.predict(observed_node_ids)

    best_b, best_bvsn, best_q = tau_i, max_tokens - tau_i, -1.0

    for k in range(n_grid + 1):
        b_i = tau_i + (max_tokens - tau_i) * k / n_grid
        b_vsn_tokens = max_tokens - b_i
        b_vsn_budget = b_vsn_tokens * p_out

        q_i = _quality(current_node, b_i)
        q_vsn = _compute_q_vsn(b_vsn_budget, samples, p_out)
        total = current_node.weight * q_i + q_vsn

        if total > best_q:
            best_q = total
            best_b = b_i
            best_bvsn = b_vsn_tokens

    return best_b, best_bvsn, best_q
