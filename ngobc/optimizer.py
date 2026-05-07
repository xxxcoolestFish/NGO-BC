"""
psbc/optimizer.py – PSBCOptimizer: the five-step MPC loop.

Step 2 predictor is pluggable:
  - Default (heuristic): EMA-based estimation, no training required
  - Neural (DAG-GAT):    pass oracle=DagGatOracle.load("gat.pt")

Step 3 quality function is pluggable (Direction 1):
  - Default (exponential): Q(b) = σ·(1-exp(-κ(b-τ)))  → b0=None
  - Logarithmic:           Q(b) = σ̃·log(1+(b-τ)/b0)   → b0=200.0 (or any float)

Step 3 objective is pluggable (Direction 3 & 4):
  - Default (sum):    max Σ w_j·Q_j(b_j)  → use_maxmin=False, use_product=False
  - Max-min:          max min_j Q_j(b_j)   → use_maxmin=True
  - Product (Dir 4):  max Π Q_j(b_j)       → use_product=True

Terminal node detection (for product objective / multi-round support):
  - MAS-specified:    terminal_nodes=["Solver", "Verifier"]  → use as given
  - Auto-detected:    terminal_nodes=None (default) → auto-detect sinks from topology

Public API
----------
opt = PSBCOptimizer(
    adj        = topology adjacency list,
    node_params= {node_id: NodeParams},
    budget     = B_t,
    p_in       = ..., p_out = ...,
    oracle     = DagGatOracle.load("psbc_logs/gat.pt"),  # optional
    b0         = 200.0,    # optional; enables log quality function
    use_maxmin = True,     # optional; enables max-min objective
    use_product= True,     # optional; enables product objective (Direction 4)
    terminal_nodes = None, # optional; MAS-specified terminal nodes, None=auto-detect
)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ngobc.gat_oracle import DagGatOracle

from ngobc.dag import (
    topological_sort,
    get_descendants,
    critical_path_nodes,
    topo_weights,
)
from ngobc.kkt import NodeParams, AllocationResult, kkt_solve, survival_allocate


# ──────────────────────────────────────────────────────────────────────────────
# Per-node runtime state  (exported so callers can inject new nodes at runtime)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _NodeState:
    params:            NodeParams
    alpha:             float = 0.7     # utilisation rate EMA (cold start 0.7)
    b_prev:            float = 0.0     # last planned allocation (for prediction)
    exact_in:          int   = 0       # actual input tokens (filled at decide time)
    assigned_max:      int   = 0       # decided max_tokens
    actual_out:        int   = 0       # actual output tokens after LLM call
    status:            str   = "PENDING"  # PENDING / ACTIVE / DONE / SKIPPED


# ──────────────────────────────────────────────────────────────────────────────
# Summary record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RoundSummary:
    total_budget:   float
    total_spent:    float
    budget_ok:      bool           # spent ≤ total_budget
    nodes_executed: int
    nodes_skipped:  int
    nodes_terminated: int
    survival_triggers: int         # how many nodes were in survival mode
    per_node: Dict[str, dict] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class PSBCOptimizer:
    """
    Single-round PSBC with heuristic (EMA) predictor.

    Parameters
    ----------
    adj         : adjacency list  {node_id: [child_id, …]}
    node_params : pre-built NodeParams for every node
    budget      : B_t – total monetary budget (yuan)
    p_in        : input token price (yuan/token)
    p_out       : output token price (yuan/token)
    gamma       : safety margin coefficient for reserves (default 1.0)
    beta        : uncertainty ratio for heuristic predictor (default 0.15)
    alpha_lr    : EMA learning rate for utilisation update (default 0.1)
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        adj:          Dict[str, List[str]],
        node_params:  Dict[str, NodeParams],
        budget:       float,
        p_in:         float,
        p_out:        float,
        gamma:        float = 1.0,
        beta:         float = 0.15,
        alpha_lr:     float = 0.1,
        oracle: "Optional[DagGatOracle]" = None,
        b0:          Optional[float] = None,
        use_maxmin:  bool = False,
        use_product: bool = False,
        one_shot:      bool = False,  # True → solve KKT once, cache, no re-plan
        terminal_nodes: Optional[List[str]] = None,
        topology_mode: str = "STATIC",        # "STATIC" | "DYNAMIC"
        agent_pool:   object = None,           # AgentPool (DYNAMIC only)
        predictor:    object = None,           # TopologyPredictor (DYNAMIC only)
        node_weights: Optional[Dict[str, float]] = None,  # per-node KKT weights
        importance_weights: Optional[Dict[str, float]] = None,  # for survival_allocate
    ) -> None:
        self.adj          = adj
        self.p_in         = p_in
        self.p_out        = p_out
        self.gamma        = gamma
        self.beta         = beta
        self.alpha_lr     = alpha_lr
        self.oracle       = oracle      # None → heuristic; set → DAG-GAT
        self.b0           = b0          # None → exponential Q; float → log Q
        self.use_maxmin   = use_maxmin  # False → sum; True → max-min
        self.use_product  = use_product # False → sum; True → product (Dir 4)
        self.one_shot     = one_shot
        self.topology_mode = topology_mode  # "STATIC" or "DYNAMIC"
        self.agent_pool    = agent_pool
        self.predictor     = predictor
        self.node_weights  = node_weights    # per-node KKT weights (σ or Shapley)
        self.importance_weights = importance_weights  # for survival_allocate

        # Terminal nodes: None=auto-detect, list=MAS-specified
        self._terminal_nodes_input = terminal_nodes
        self._terminal_nodes: Optional[List[str]] = None  # cached

        self.R            = budget          # running remaining budget
        self.B_t          = budget

        self._topo_order  = topological_sort(adj)
        self._crit_nodes  = critical_path_nodes(adj)

        # Per-node state
        self._state: Dict[str, _NodeState] = {}
        for nid, params in node_params.items():
            s = _NodeState(params=params)
            s.b_prev = budget / max(len(node_params), 1) / self.p_out  # tokens
            self._state[nid] = s

        # Execution bookkeeping
        self._executed_order: List[str] = []
        self._terminated     = False
        self._survival_count = 0
        self._cached_allocations: Dict[str, float] = {}  # one-shot mode cache

        if oracle is not None:
            oracle.reset()
        self._skip_count     = 0
        self._last_debug: Dict[str, object] = {}
        self._last_debug_blend = None  # (blend_alpha, b_kkt) stash

    # ── terminal node management ────────────────────────────────────────────

    def set_terminal_nodes(self, nodes: Optional[List[str]]) -> None:
        """Override terminal nodes after construction (e.g. from MAS executor)."""
        self._terminal_nodes_input = nodes
        self._terminal_nodes = None  # invalidate cache

    def get_terminal_nodes(self) -> List[str]:
        """Return terminal nodes: MAS-specified if given, else auto-detect sinks."""
        if self._terminal_nodes is not None:
            return self._terminal_nodes
        if self._terminal_nodes_input is not None and len(self._terminal_nodes_input) > 0:
            self._terminal_nodes = list(self._terminal_nodes_input)
        else:
            self._terminal_nodes = self._detect_terminal_nodes()
        return self._terminal_nodes

    def _detect_terminal_nodes(self) -> List[str]:
        """Auto-detect: nodes with zero out-degree (topological sinks)."""
        all_nodes = set(self.adj.keys())
        children  = set()
        for clist in self.adj.values():
            children.update(clist)
        sinks = all_nodes - children
        return sorted(sinks)

    @property
    def last_debug(self) -> dict:
        """Snapshot of the most recent decide() call (for experiment logging)."""
        return dict(self._last_debug)

    # ── public API ────────────────────────────────────────────────────────────

    def decide(self, agent_id: str, exact_in_tokens: int) -> Tuple[str, int]:
        """
        Steps 1–3: observe → predict → allocate.

        Returns
        -------
        (decision, max_tokens)
        decision ∈ {"EXECUTE", "SKIP", "TERMINATE"}
        max_tokens : int  (only meaningful when decision == "EXECUTE")
        """
        if self._terminated:
            return "TERMINATE", 0

        s = self._state[agent_id]
        s.exact_in = exact_in_tokens
        s.status   = "ACTIVE"

        # ── Step 1: precise observation ───────────────────────────────────────
        R_prime = self.R - self.p_in * exact_in_tokens
        tau_i   = s.params.tau

        # Safety check: can we even pay for τ of this node?
        if R_prime < self.p_out * tau_i:
            _early_dbg = {
                "R_before":    float(self.R + self.p_in * exact_in_tokens),
                "R_prime":     float(R_prime),
                "R_free":      float(R_prime),
                "reserves":    {},
                "normal_mode": False,
            }
            if agent_id in self._crit_nodes:
                # Critical path: allocate what we have, then terminate
                forced = max(int(R_prime / self.p_out), 0)
                s.assigned_max = forced
                s.status = "ACTIVE"
                self._last_debug = {**_early_dbg, "early_terminate": True}
                return "TERMINATE", forced
            else:
                s.status = "SKIPPED"
                self._skip_count += 1
                self._last_debug = {**_early_dbg, "early_skip": True}
                return "SKIP", 0

        # ── One-shot shortcut: use cached allocation if available ──────────
        if self.one_shot and self._cached_allocations:
            cached = self._cached_allocations.get(agent_id, None)
            if cached is not None:
                s.assigned_max = int(max(math.ceil(cached), int(math.ceil(tau_i)), 1))
                s._normal_mode = True
                self._last_debug = {
                    "R_before": float(self.R + self.p_in * exact_in_tokens),
                    "R_prime": float(R_prime),
                    "R_free": float(R_prime),
                    "reserves": {},
                    "normal_mode": True,
                    "one_shot_cached": True,
                }
                return "EXECUTE", s.assigned_max

        # ── Step 2: heuristic forward prediction ──────────────────────────────
        descendants = list(get_descendants(self.adj, agent_id))
        reserves    = self._compute_reserves(agent_id, descendants)

        # ── Step 3: KKT allocation or survival ────────────────────────────────
        R_free = R_prime - sum(reserves.values())

        # Adaptive safety margin: if reserves consume all budget, retry with
        # γ=0 (bare-minimum reserve, no uncertainty padding) to avoid premature
        # TERMINATE when the task naturally needs few tokens.
        if R_free <= 0 and descendants:
            reserves = self._compute_reserves(agent_id, descendants, effective_gamma=0.0)
            R_free = R_prime - sum(reserves.values())

        # Nodes to jointly optimise: current + all descendants
        opt_nodes = [self._state[agent_id].params] + [
            self._state[d].params for d in descendants if d in self._state
        ]

        if R_free > 0:
            # ── Step 3: VSN (DYNAMIC) or n-D KKT (STATIC) ────────────
            if self.topology_mode == "DYNAMIC" and self.predictor is not None:
                # PSBC-DT: 2D VSN optimisation
                from ngobc.vsn import vsn_solve
                b_i_star, b_vsn_star, _ = vsn_solve(
                    current_node=s.params,
                    R_free=R_free,
                    p_out=self.p_out,
                    predictor=self.predictor,
                    observed_node_ids=list(self._state.keys()),
                )
                b_kkt = b_i_star
                self._vsn_budget_seed = b_vsn_star * self.p_out
            else:
                # Static topology: n-D KKT
                if self.use_product:
                    from ngobc.product_kkt import product_kkt_solve
                    result = product_kkt_solve(opt_nodes, R_free, self.p_out,
                                               weights=self.node_weights)
                else:
                    result = kkt_solve(opt_nodes, R_free, self.p_out)
                b_kkt = result.allocations.get(agent_id, tau_i)

                # One-shot: cache all allocations from first solve
                if self.one_shot:
                    self._cached_allocations = dict(result.allocations)

            # Pure KKT allocation (no blending).
            b_assign = max(b_kkt, tau_i)
            normal_mode = True
            self._last_debug_blend = (1.0, float(b_kkt))
        else:
            # Survival mode
            R_safe = R_prime - sum(
                self.p_in * (self._state[d].params.tau)   # C_sys+C_struct ≈ τ
                for d in descendants if d in self._state
            )
            if R_safe <= 0:
                forced = max(int(tau_i), 0)
                s.assigned_max = forced
                self._terminated = True
                return "TERMINATE", forced

            surv_allocs = survival_allocate(opt_nodes, R_safe, self.p_out,
                                             importance_weights=self.importance_weights)
            b_assign   = surv_allocs.get(agent_id, tau_i)
            normal_mode = False
            self._last_debug_blend = None
            self._survival_count += 1

        b_assign = max(int(math.ceil(b_assign)), int(math.ceil(tau_i)), 1)
        s.assigned_max = b_assign
        s._normal_mode = normal_mode   # stash for Step 5

        # Store debug snapshot for experiment logging
        dbg = {
            "R_before":   float(self.R + self.p_in * exact_in_tokens),
            "R_prime":    float(R_prime),
            "R_free":     float(R_free),
            "reserves":   {k: float(v) for k, v in reserves.items()},
            "normal_mode": bool(normal_mode),
        }
        if normal_mode and self._last_debug_blend:
            dbg["blend_alpha"] = float(self._last_debug_blend[0])
            dbg["b_kkt"]       = float(self._last_debug_blend[1])
        self._last_debug = dbg

        return "EXECUTE", b_assign

    def settle(self, agent_id: str, actual_out_tokens: int) -> None:
        """
        Steps 4–5: update budget and utilisation after the LLM call.
        """
        s = self._state[agent_id]
        s.actual_out = actual_out_tokens
        s.status     = "DONE"

        # Step 4: debit actual cost; clamp to 0 if input cost alone exhausted budget
        R_prime = self.R - self.p_in * s.exact_in
        self.R  = max(R_prime - self.p_out * actual_out_tokens, 0.0)

        # Update cache for downstream prediction
        s.b_prev = s.assigned_max

        # Step 5: unbiased α update (only in normal mode)
        if getattr(s, "_normal_mode", True) and s.assigned_max > 0:
            ratio   = actual_out_tokens / s.assigned_max
            s.alpha = (1 - self.alpha_lr) * s.alpha + self.alpha_lr * ratio

        self._executed_order.append(agent_id)

    def settle_skip(self, agent_id: str) -> None:
        """Register a skipped node (no LLM call, no budget deduction)."""
        s = self._state[agent_id]
        s.actual_out  = 0
        s.status      = "SKIPPED"
        s.b_prev      = 0.0

    def settle_terminate(self, agent_id: str, actual_out_tokens: int) -> None:
        """Register the terminal node that exhausted the budget."""
        self.settle(agent_id, actual_out_tokens)
        self._terminated = True

    def summary(self) -> RoundSummary:
        """Return a ledger of the full round."""
        spent = self.B_t - self.R
        per   = {}
        n_exec = n_skip = n_term = 0

        for nid, s in self._state.items():
            per[nid] = {
                "status":       s.status,
                "exact_in":     s.exact_in,
                "assigned_max": s.assigned_max,
                "actual_out":   s.actual_out,
                "alpha":        s.alpha,
            }
            if s.status == "DONE":
                n_exec += 1
            elif s.status == "SKIPPED":
                n_skip += 1

        return RoundSummary(
            total_budget    = self.B_t,
            total_spent     = spent,
            budget_ok       = spent <= self.B_t + 1e-9,
            nodes_executed  = n_exec,
            nodes_skipped   = n_skip,
            nodes_terminated= n_term,
            survival_triggers = self._survival_count,
            per_node        = per,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _compute_reserves(
        self,
        current_id: str,
        descendants: List[str],
        effective_gamma: float | None = None,
    ) -> Dict[str, float]:
        """
        Step-2 reserve computation.  Routes to neural or heuristic predictor.
        effective_gamma overrides self.gamma when set (for adaptive safety margin).
        """
        gamma = self.gamma if effective_gamma is None else effective_gamma
        if self.oracle is not None and descendants:
            return self._neural_reserves(current_id, descendants, gamma)
        return self._heuristic_reserves(current_id, descendants, gamma)

    # ── Neural predictor (DAG-GAT + TTAS) ────────────────────────────────────

    def _neural_reserves(
        self,
        current_id: str,
        descendants: List[str],
        gamma: float = 1.0,
    ) -> Dict[str, float]:
        """Build a GraphData for the current subgraph and call the oracle."""
        from ngobc.dag_gat import GraphData
        from ngobc.gat_trainer import _role_from_id, KNOWN_ROLES
        import torch

        executed_set = set(self._executed_order) | {current_id}
        visible      = [nid for nid in self._topo_order
                        if nid in executed_set or nid in descendants]

        if not visible:
            return {}

        id_to_idx = {nid: i for i, nid in enumerate(visible)}

        # Build parent index map
        parents: Dict[int, List[int]] = {i: [] for i in range(len(visible))}
        for u, children in self.adj.items():
            if u not in id_to_idx:
                continue
            for v in children:
                if v in id_to_idx:
                    parents[id_to_idx[v]].append(id_to_idx[u])

        # Build x_base and role_idxs
        role_map = {r: i for i, r in enumerate(KNOWN_ROLES)}
        rows, role_idxs = [], []
        for nid in visible:
            s    = self._state.get(nid)
            p    = s.params if s else NodeParams(nid, 0.9, 0.01, 0.0, 0.1)
            done = nid in self._executed_order
            rows.append([
                p.sigma, p.kappa, p.tau,
                p.tau, 0.0,                          # C_sys ≈ τ, C_struct ≈ 0
                float(done),
                float(s.actual_out if done and s else 0.0),
            ])
            role_key = _role_from_id(nid)
            role_idxs.append(role_map.get(role_key, role_map["unknown"]))

        x_base    = torch.tensor(rows,      dtype=torch.float32)
        role_tens = torch.tensor(role_idxs, dtype=torch.long)
        g = GraphData(node_ids=visible, x_base=x_base,
                      role_idxs=role_tens, parents=parents)

        e_hat, sigma_hat, _ = self.oracle.predict(g)

        # Update TTAS scale using the current (just-observed) node
        if current_id in id_to_idx:
            s = self._state.get(current_id)
            if s and s.exact_in > 0:
                base_pred = e_hat[id_to_idx[current_id]].item() / max(self.oracle.s_task, 1e-9)
                self.oracle.update_scale(base_pred, float(s.exact_in))

        reserves: Dict[str, float] = {}
        for d in descendants:
            if d not in id_to_idx:
                continue
            sd  = self._state.get(d)
            tau = sd.params.tau if sd else 0.0
            j   = id_to_idx[d]
            reserves[d] = (
                self.p_in * (e_hat[j].item() + gamma * sigma_hat[j].item())
                + self.p_out * tau
            )
        return reserves

    # ── Heuristic predictor (EMA) ─────────────────────────────────────────────

    def _heuristic_reserves(
        self,
        current_id: str,
        descendants: List[str],
        gamma: float = 1.0,
    ) -> Dict[str, float]:
        executed_set = set(self._executed_order) | {current_id}
        reserves: Dict[str, float] = {}

        # Build reverse adjacency (parent lookup)
        parents: Dict[str, List[str]] = {n: [] for n in self.adj}
        for u, children in self.adj.items():
            for v in children:
                parents.setdefault(v, []).append(u)

        for d in descendants:
            if d not in self._state:
                continue
            sd = self._state[d]
            pa_list = parents.get(d, [])

            e_in_hat = sd.params.tau   # C_sys + C_struct ≈ τ (cold-start approx)
            for k in pa_list:
                sk = self._state.get(k)
                if sk is None:
                    continue
                if k in executed_set:
                    e_in_hat += sk.actual_out
                else:
                    e_in_hat += sk.alpha * sk.b_prev

            sigma_j  = self.beta * e_in_hat
            reserve  = (
                self.p_in * (e_in_hat + gamma * sigma_j)
                + self.p_out * sd.params.tau
            )
            reserves[d] = reserve

        return reserves
