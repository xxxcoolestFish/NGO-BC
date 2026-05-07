"""
psbc/topology_predictor.py — Topology predictor for PSBC-DT.

In PSBC-DT mode, the unobserved future subgraph is a random variable.
The topology predictor generates P_pred(G_unobs | G_obs, AgentPool),
which feeds into the expected-reserve calculation (Step 2) and the
VSN 2D optimisation (Step 3).

Base implementation: uniform sampling from AgentPool (no learning).
Future upgrade: Hybrid DAG-GAT with autoregressive decoder (§5.9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ngobc.agent_pool import AgentPool
from ngobc.kkt import NodeParams


@dataclass
class TopologySample:
    """One sampled future topology trajectory."""
    adj: Dict[str, List[str]]
    node_params: Dict[str, NodeParams]


class TopologyPredictor:
    """
    Generates P_pred(G_unobs) by sampling from the AgentPool.

    Parameters
    ----------
    agent_pool : AgentPool
    n_samples : int (default 10)
        Number of Monte Carlo samples for expected-reserve integration.
    max_future_nodes : int (default 10)
        Maximum number of future nodes per sample.
    """

    def __init__(
        self,
        agent_pool: AgentPool,
        n_samples: int = 10,
        max_future_nodes: int = 10,
    ):
        self._pool = agent_pool
        self.n_samples = n_samples
        self.max_future_nodes = max_future_nodes

    def predict(
        self,
        observed_node_ids: List[str],
        seed: int | None = None,
    ) -> List[TopologySample]:
        """
        Generate S future topology samples given the currently observed nodes.

        Parameters
        ----------
        observed_node_ids : node IDs already seen (used for dedup)
        seed : random seed for reproducibility

        Returns
        -------
        List of S TopologySample objects.
        """
        trajectories = self._pool.sample_trajectories(
            n_samples=self.n_samples,
            max_nodes=self.max_future_nodes,
            seed=seed,
        )
        samples = []
        for adj, params in trajectories:
            # Filter out any nodes already observed (shouldn't happen with
            # "future_X" naming but kept as safety measure)
            adj = {k: v for k, v in adj.items() if k not in observed_node_ids}
            params = {k: v for k, v in params.items() if k not in observed_node_ids}
            if adj:
                samples.append(TopologySample(adj=adj, node_params=params))
        return samples

    def expected_quality(
        self,
        budget: float,
        p_out: float,
        observed_node_ids: List[str],
        seed: int | None = None,
    ) -> float:
        """
        Compute E[V*(G, budget)] over the predicted topology distribution.
        V* = optimal KKT quality for a given graph and budget.

        This is used to construct the VSN quality function Q_VSN(B).
        """
        samples = self.predict(observed_node_ids, seed)
        if not samples:
            return 0.0

        total_q = 0.0
        for s in samples:
            q = self._optimal_quality(s.adj, s.node_params, budget, p_out)
            total_q += q
        return total_q / len(samples)

    @staticmethod
    def _optimal_quality(
        adj: Dict[str, List[str]],
        node_params: Dict[str, NodeParams],
        budget: float,
        p_out: float,
    ) -> float:
        """Compute V*(G, B): optimal KKT quality for deterministic (G, B)."""
        from ngobc.product_kkt import product_kkt_solve as solver
        nodes = list(node_params.values())
        if not nodes:
            return 0.0
        result = solver(nodes, budget, p_out)
        if not result.allocations:
            return 0.0
        # Product quality: Π σ̃_j·(1-exp(-κ_j(b_j-τ_j)))
        q = 1.0
        for nd in nodes:
            b = result.allocations.get(nd.node_id, nd.tau)
            if b > nd.tau:
                q *= nd.sigma * (1.0 - __import__('math').exp(-nd.kappa * (b - nd.tau)))
            else:
                q *= 0.0  # below threshold → zero quality
        return q
