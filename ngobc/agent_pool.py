"""
psbc/agent_pool.py — Agent Pool for PSBC-DT (Dynamic Topology).

In dynamic-topology MAS, the full DAG is unknown at execution time.
The Agent Pool pre-defines the possible agent role classes with their
offline-calibrated physical parameters (σ, κ, τ, C_sys, C_struct).

The pool serves two purposes:
  1. Role registry: lookup (σ, κ, τ) for known agent types.
  2. Topology sampling: generate plausible future subgraphs by drawing
     role instances from the pool (for expected-reserve Monte Carlo).

Assumption (from Part 3 §3.5): edge structure is determined by role types,
so the topology is fully specified by the multiset of role-nodes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ngobc.kkt import NodeParams


@dataclass
class AgentRole:
    """Pre-calibrated parameters for one agent role class."""
    role: str
    sigma: float          # max quality potential ∈ (0, 1]
    kappa: float          # response rate > 0
    tau: float            # emergence threshold ≥ 0
    C_sys: int = 0        # fixed system prompt tokens
    C_struct: int = 0     # framework wrapper tokens

    def to_node_params(self, node_id: str, weight: float = 1.0,
                       q_pa: float = 1.0) -> NodeParams:
        return NodeParams(
            node_id=node_id,
            sigma=self.sigma,
            kappa=self.kappa,
            tau=self.tau,
            weight=weight,
            q_pa=q_pa,
        )


class AgentPool:
    """
    Registry of agent role classes with calibrated parameters.

    Parameters
    ----------
    roles : dict role_name → AgentRole
    topology_rules : optional callable that, given a list of role-names,
                     returns the adjacency edges. If None, defaults to
                     chain topology (each node → next node).
    """

    def __init__(
        self,
        roles: Dict[str, AgentRole],
        topology_rules: Optional[callable] = None,
    ):
        self._roles = roles
        self._topology_rules = topology_rules or self._default_chain_edges

    # -- role registry ---------------------------------------------------------

    def get(self, role_name: str) -> Optional[AgentRole]:
        return self._roles.get(role_name)

    def __contains__(self, role_name: str) -> bool:
        return role_name in self._roles

    def __iter__(self):
        return iter(self._roles.values())

    def __len__(self) -> int:
        return len(self._roles)

    @property
    def role_names(self) -> List[str]:
        return list(self._roles.keys())

    # -- topology sampling (for Monte Carlo expected reserve) ------------------

    def sample_future_subgraph(
        self,
        n_nodes: int,
        seed: int | None = None,
        role_weights: Dict[str, float] | None = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, NodeParams]]:
        """
        Sample a plausible future subgraph with n_nodes role instances.

        Returns (adjacency, node_params) for the sampled subgraph.
        Role instances are drawn with replacement from the pool.
        """
        rng = random.Random(seed)
        role_names = self.role_names
        weights = [role_weights.get(r, 1.0) if role_weights else 1.0
                   for r in role_names]

        # Draw n_nodes roles
        chosen = rng.choices(role_names, weights=weights, k=n_nodes)

        # Generate node IDs
        node_ids = [f"future_{i}_{chosen[i]}" for i in range(n_nodes)]

        # Build adjacency via topology rules
        adj = self._topology_rules(node_ids, chosen)

        # Build NodeParams
        params = {}
        for i, nid in enumerate(node_ids):
            role = self._roles[chosen[i]]
            params[nid] = role.to_node_params(
                nid, weight=1.0 / n_nodes if n_nodes > 0 else 1.0
            )

        return adj, params

    def sample_trajectories(
        self,
        n_samples: int,
        max_nodes: int = 10,
        seed: int | None = None,
        role_weights: Dict[str, float] | None = None,
    ) -> List[Tuple[Dict[str, List[str]], Dict[str, NodeParams]]]:
        """
        Sample S trajectory graphs for Monte Carlo integration.

        Each trajectory samples a random number of nodes (1..max_nodes).
        """
        rng = random.Random(seed)
        trajectories = []
        for s in range(n_samples):
            n_nodes = rng.randint(1, max_nodes)
            adj, params = self.sample_future_subgraph(
                n_nodes=n_nodes,
                seed=None if seed is None else seed + s * 1000,
                role_weights=role_weights,
            )
            trajectories.append((adj, params))
        return trajectories

    # -- internal --------------------------------------------------------------

    @staticmethod
    def _default_chain_edges(
        node_ids: List[str], roles: List[str]
    ) -> Dict[str, List[str]]:
        """Default: chain topology (each node → next node)."""
        adj: Dict[str, List[str]] = {}
        for i, nid in enumerate(node_ids):
            if i + 1 < len(node_ids):
                adj[nid] = [node_ids[i + 1]]
            else:
                adj[nid] = []
        return adj


# ──────────────────────────────────────────────────────────────────────────────
# Factory: ROMA agent pool
# ──────────────────────────────────────────────────────────────────────────────

def roma_agent_pool() -> AgentPool:
    """Create an AgentPool pre-populated with ROMA's 5 agent roles."""
    return AgentPool({
        "atomizer":   AgentRole("atomizer",   sigma=0.85, kappa=0.05, tau=50),
        "planner":    AgentRole("planner",    sigma=0.90, kappa=0.02, tau=100),
        "executor":   AgentRole("executor",   sigma=0.95, kappa=0.01, tau=100),
        "aggregator": AgentRole("aggregator", sigma=0.90, kappa=0.02, tau=100),
        "verifier":   AgentRole("verifier",   sigma=0.85, kappa=0.05, tau=50),
    })
