"""
psbc/dag_gat.py – Adaptive DAG-GAT Oracle (Phase D2).

Architecture (Part 5 of the method document):
  - Causal directed message passing: each node aggregates only from parents
  - Dual-head output: (Ê_in, σ̂²) per node
  - Loss: Negative log-likelihood (Gaussian)
  - Supports role embeddings for role-aware feature learning

Node features (3 groups):
  x_agent  = [σ, κ, τ]                         static, from EM calibration
  x_task   = [C_sys, C_struct, role_emb_d]      dynamic per task
  x_state  = [I (executed?), L (c_out or b_prev)]  runtime

Input dim = 3 + (2 + role_emb_dim) + 2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════��══════════════════════
# Graph data container
# ═══════════════════════════════════════════════════════════════════════════

class GraphData:
    """
    Holds one DAG's node features and adjacency for a single forward pass.

    node_ids    : list of node IDs in topological order
    x_base      : (N, base_feat_dim) float tensor WITHOUT role embedding
                  (agent[3] + task_scalar[2] + state[2] = 7 dims)
    role_idxs   : (N,) long tensor of role indices (embedded inside DAGGAT.forward)
    parents     : dict {child_idx: [parent_idx, ...]}  (index into node_ids)
    pred_idx    : int, the node index whose E_in is being predicted (set by trainer)
    """
    def __init__(
        self,
        node_ids:  List[str],
        x_base:    torch.Tensor,
        role_idxs: torch.Tensor,
        parents:   Dict[int, List[int]],
    ):
        self.node_ids  = node_ids
        self.x_base    = x_base
        self.role_idxs = role_idxs
        self.parents   = parents
        self.pred_idx  = 0   # set externally

    @property
    def n(self) -> int:
        return len(self.node_ids)


# ════════════════════════════════════════════════════════════════════════��══
# Feature builder
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_ROLES = ["source", "analyst", "synthesiser", "worker",
               "aggregator", "orchestrator", "critic", "unknown"]

class FeatureBuilder:
    """
    Converts per-node metadata dictionaries into a (N, feat_dim) tensor.

    Each node dict should contain:
      sigma, kappa, tau        – quality-function params (float)
      C_sys, C_struct          – fixed token costs (int)
      role                     – role name (str)
      executed                 – bool: has this node run already?
      output_len               – int: c_out if executed, b_prev if not
    """

    def __init__(self, role_emb_dim: int = 8):
        self.role_emb_dim = role_emb_dim
        self._role_map    = {r: i for i, r in enumerate(KNOWN_ROLES)}
        # Learnable role embeddings (initialised externally or via load)
        self.role_emb = nn.Embedding(len(KNOWN_ROLES), role_emb_dim)

    @property
    def feat_dim(self) -> int:
        return 7 + self.role_emb_dim   # base(7) + role_emb

    def _role_idx(self, role: str) -> int:
        key = role.lower().strip()
        return self._role_map.get(key, self._role_map["unknown"])

    def build(self, nodes: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          x_base    : (N, 7)  raw features without role embedding
          role_idxs : (N,)    long tensor of role indices
        """
        rows      = []
        role_idxs = []
        for nd in nodes:
            agent  = [float(nd.get("sigma", 0.9)),
                      float(nd.get("kappa", 0.01)),
                      float(nd.get("tau",   0.0))]
            task_s = [float(nd.get("C_sys", 0)),
                      float(nd.get("C_struct", 0))]
            state  = [float(nd.get("executed", 0)),
                      float(nd.get("output_len", 0))]
            rows.append(agent + task_s + state)
            role_idxs.append(self._role_idx(nd.get("role", "unknown")))

        x_base    = torch.tensor(rows,      dtype=torch.float32)
        role_tens = torch.tensor(role_idxs, dtype=torch.long)
        return x_base, role_tens


# ═══════════════════════════════════════════════════════════════════════════
# DAG-GAT layer
# ═══════════════════════════════════════════════════════════════════════════

class DAGGATLayer(nn.Module):
    """
    One causal graph-attention message-passing layer.

    For each node j, aggregates messages from pa(j) only:
      h_j^{l+1} = MLP_update( h_j^{l} || Σ_{k∈pa(j)} α_{k,j} W h_k^{l} )
    where α_{k,j} is a standard GAT attention weight.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W       = nn.Linear(in_dim, out_dim, bias=False)
        self.a       = nn.Parameter(torch.empty(2 * out_dim))
        self.update  = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_normal_(self.a.unsqueeze(0))

    def forward(
        self,
        h:       torch.Tensor,           # (N, in_dim)
        parents: Dict[int, List[int]],   # {child_idx: [parent_idxs]}
    ) -> torch.Tensor:                   # (N, out_dim)
        N      = h.size(0)
        Wh     = self.W(h)               # (N, out_dim)
        out_dim = Wh.size(1)

        h_new = torch.zeros(N, out_dim, device=h.device)

        for j, pa in parents.items():
            if not pa:
                # Source node: no parents, aggregate = zeros
                agg = torch.zeros(out_dim, device=h.device)
            else:
                # Compute GAT attention scores: e_{k,j} = LeakyReLU(a^T [Wh_j || Wh_k])
                Wh_j = Wh[j].unsqueeze(0).expand(len(pa), -1)  # (|pa|, out_dim)
                Wh_k = Wh[list(pa)]                              # (|pa|, out_dim)
                e    = F.leaky_relu(
                    (torch.cat([Wh_j, Wh_k], dim=1) * self.a).sum(dim=1),
                    negative_slope=0.2,
                )                                                # (|pa|,)
                alpha = F.softmax(e, dim=0)                      # (|pa|,)
                alpha = self.dropout(alpha)
                agg   = (alpha.unsqueeze(1) * Wh_k).sum(dim=0)  # (out_dim,)

            h_new[j] = self.update(torch.cat([h[j], agg], dim=0))

        return h_new


# ═══════════════════════════════════════════════════════════════════════════
# Full DAG-GAT model
# ═══════════════════════════════════════════════════════════════════════════

class DAGGAT(nn.Module):
    """
    Adaptive DAG-GAT Oracle with three heads.

    Input  : GraphData (node features + DAG adjacency)
    Output : per-node (Ê_in, σ̂²) + graph-level P_stop
    """

    def __init__(
        self,
        feat_dim:    int,
        hidden_dim:  int = 64,
        n_layers:    int = 2,
        role_emb_dim: int = 8,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.feat_dim    = feat_dim
        self.role_emb_dim = role_emb_dim
        self.role_emb    = nn.Embedding(len(KNOWN_ROLES), role_emb_dim)

        # Input projection
        self.mlp_in = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # Stacked DAG-GAT layers
        self.gat_layers = nn.ModuleList([
            DAGGATLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        # Three-head readout
        self.head_mean   = nn.Linear(hidden_dim, 1)    # E_in
        self.head_logvar = nn.Linear(hidden_dim, 1)    # log-variance
        self.head_stop   = nn.Linear(hidden_dim, 1)    # P_stop (per-node, pooled to graph)

    def forward(self, g: GraphData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mean   : (N,)  predicted E_in for each node
        stddev : (N,)  predicted std-dev
        p_stop : (N,)  per-node stop scores (mean-pool to graph-level)
        """
        x_role = self.role_emb(g.role_idxs)
        x = torch.cat([g.x_base[:, :5], x_role, g.x_base[:, 5:]], dim=1)

        h = self.mlp_in(x)
        for layer in self.gat_layers:
            h = layer(h, g.parents)

        mean   = self.head_mean(h).squeeze(-1).clamp(min=0.0)
        logvar = self.head_logvar(h).squeeze(-1).clamp(min=-6, max=6)
        stddev = torch.exp(0.5 * logvar).clamp(min=1.0)

        # Per-node stop logits → graph-level P_stop via mean-pool + sigmoid
        stop_logits = self.head_stop(h).squeeze(-1)         # (N,)
        p_stop = torch.sigmoid(stop_logits)                  # (N,) ∈ (0,1)

        return mean, stddev, p_stop

    def forward_stop(self, g: GraphData) -> float:
        """Convenience: return scalar graph-level P_stop."""
        _, _, p_stop = self.forward(g)
        return float(p_stop.mean().item())

    @staticmethod
    def nll_loss(
        mean:    torch.Tensor,   # (N,)
        stddev:  torch.Tensor,   # (N,)
        target:  torch.Tensor,   # (N,) true E_in values
        mask:    Optional[torch.Tensor] = None,  # (N,) bool: True = predict this node
    ) -> torch.Tensor:
        """Gaussian NLL loss for E_in prediction."""
        nll = (target - mean) ** 2 / (2 * stddev ** 2) + torch.log(stddev)
        if mask is not None:
            nll = nll[mask]
        return nll.mean()

    @staticmethod
    def stop_loss(
        p_stop:  torch.Tensor,   # (N,) predicted P_stop per node
        target:  float,          # scalar: 1.0 if final round, 0.0 otherwise
    ) -> torch.Tensor:
        """Binary cross-entropy loss for P_stop prediction (graph-level)."""
        p_stop_graph = p_stop.mean()
        t = torch.tensor(target, device=p_stop.device, dtype=torch.float32)
        return F.binary_cross_entropy(p_stop_graph, t)

    @staticmethod
    def combined_loss(
        mean:    torch.Tensor,
        stddev:  torch.Tensor,
        p_stop:  torch.Tensor,
        e_in_target: torch.Tensor,
        stop_target: float,
        mask:    Optional[torch.Tensor] = None,
        stop_weight: float = 0.1,
    ) -> torch.Tensor:
        """Combined NLL + BCE loss with P_stop weight."""
        l_ein = DAGGAT.nll_loss(mean, stddev, e_in_target, mask)
        l_stop = DAGGAT.stop_loss(p_stop, stop_target)
        return l_ein + stop_weight * l_stop
