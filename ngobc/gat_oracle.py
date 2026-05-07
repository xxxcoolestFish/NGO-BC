"""
psbc/gat_oracle.py – DAG-GAT Oracle: loaded model + TTAS adaptive scaling.

Implements Part 5 §5.3 Level-1 (Test-Time Adaptive Scaling).
Level-2 (residual adapter) is scaffolded but left as a no-op until
enough drift data is collected.

Usage:
    oracle = DagGatOracle.load("psbc_logs/gat.pt")
    oracle.reset()                          # call at start of each new task

    # Inside PSBCOptimizer Step 2:
    e_hat, sigma_hat = oracle.predict(g)    # (N,), (N,)
    oracle.update_scale(node_idx, true_E_in)   # after each node executes
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ngobc.dag_gat import DAGGAT, FeatureBuilder, GraphData


class DagGatOracle:
    """
    Wraps a trained DAGGAT model with Level-1 TTAS adaptive scaling.

    Attributes
    ----------
    s_task   : float  – current task-level scale factor (reset to 1.0 per task)
    lambda_  : float  – TTAS EMA learning rate (default 0.3)
    """

    def __init__(self, model: DAGGAT, feat_dim: int, lambda_: float = 0.3,
                 enable_ttas: bool = True):
        self.model   = model
        self.feat_dim = feat_dim
        self.lambda_ = lambda_
        self.s_task  = 1.0
        self.enable_ttas = enable_ttas
        self.model.eval()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call at the start of each new task/question."""
        self.s_task = 1.0

    # ── Core prediction ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, g: GraphData) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Forward pass + TTAS scaling.

        Returns
        -------
        e_hat    : (N,) scaled predicted E_in
        sigma_hat: (N,) scaled predicted std-dev
        p_stop   : float, graph-level termination probability (§6.2.1)
        """
        mean_base, std_base, p_stop_per_node = self.model(g)
        e_hat    = (self.s_task * mean_base).clamp(min=0.0)
        sigma_hat = (self.s_task * std_base).clamp(min=1.0)
        p_stop = float(p_stop_per_node.mean().item())
        return e_hat, sigma_hat, p_stop

    def update_scale(self, base_pred: float, true_E_in: float) -> None:
        """
        Level-1 TTAS: update s_task after one node executes.

        Parameters
        ----------
        base_pred : Ê_base for the node that just executed (un-scaled)
        true_E_in : actual E_in observed after execution
        """
        if not self.enable_ttas:
            return
        if base_pred > 0:
            ratio    = true_E_in / base_pred
            self.s_task = (1 - self.lambda_) * self.s_task + self.lambda_ * ratio

    # ── Persistence ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path, lambda_: float = 0.3,
             enable_ttas: bool = True) -> "DagGatOracle":
        """Load a saved model checkpoint (backward-compatible with 2-head models)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = DAGGAT(
            feat_dim     = ckpt["feat_dim"],
            hidden_dim   = ckpt["hidden_dim"],
            n_layers     = ckpt["n_layers"],
            role_emb_dim = ckpt.get("role_emb_dim", 8),
        )
        # Handle old 2-head checkpoints (no head_stop)
        state = ckpt["model_state"]
        if "head_stop.weight" not in state:
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state)
        model.eval()
        return cls(model=model, feat_dim=ckpt["feat_dim"], lambda_=lambda_,
                   enable_ttas=enable_ttas)
