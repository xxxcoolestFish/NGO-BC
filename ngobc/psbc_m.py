"""
psbc/psbc_m.py — PSBC-M: Cross-round Macro-level Budget Scheduler.

Implements Part 6 of the PSBC methodology:
  - Projected MWU for unknown-horizon online budget allocation
  - Strategic reserve for inter-round bridging costs
  - Segmented terminal convergence protocol

Usage with ChatDev (single integrator loop)::

    macro = PSBCMacroScheduler(B_total=0.5, K_max=10, p_in=..., p_out=...)

    for k in range(1, K_max + 1):
        C_trans = task_mgr.get_bridging_cost(k)
        B_actual = macro.compute_round_budget(k, C_trans)

        executor = task_mgr.start_round_executor(k)
        # — PSBC single-round loop (PSBCOptimizer) —
        # ... run MPC over executor.get_topological_iterator() ...
        C_true, Q_k = ..., ...

        task_mgr.finalize_round(k)
        macro.update_after_round(C_true, task_is_done)

        if task_is_done:
            break
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Round ledger
# ---------------------------------------------------------------------------

@dataclass
class MacroRoundRecord:
    round_k:          int
    B_rem_before:     float
    B_target:         float
    B_actual:         float
    C_trans:          int               # bridging tokens actually injected
    C_true:           float             # true monetary spend this round
    mu_before:        float
    mu_after:         float
    N_rem:            float
    P_stop:           float
    task_done:        bool
    survival_trigger: bool = False


# ---------------------------------------------------------------------------
# Node bidding (PSBC Part 6 §6.4.1)
# ---------------------------------------------------------------------------

@dataclass
class NodeBid:
    """Per-node optimal output-token bid under macro shadow price μ."""
    node_id:   str
    tau:       float              # minimum output threshold
    b_star:    float              # bid: max(τ, τ + 1/κ · ln(σ̃κ/(μ·p_out)))
    E_in:      float              # estimated input tokens
    cost_in:   float              # p_in * E_in
    cost_out:  float              # p_out * b_star
    total:     float              # cost_in + cost_out


def compute_node_bids(
    nodes:      List[Tuple[str, float, float, float, float]],
    mu:         float,
    p_in:       float,
    p_out:      float,
) -> List[NodeBid]:
    """
    Per-node parallel bidding under macro shadow price μ.

    Each node solves:  max log Q_j(b_j) - μ·p_out·b_j
    which yields the product-KKT consistent closed form:

        b_j*(μ) = τ_j + (1/κ_j)·ln(1 + κ_j/(μ·p_out))

    Note: σ̃_j cancels out of the derivative — this is the key property
    that distinguishes the product-objective bidding from sum-objective.

    Parameters
    ----------
    nodes : list of (node_id, sigma_tilde, kappa, tau, E_in_hat)
    mu : float — global dual variable (macro shadow price)
    p_in, p_out : float — token pricing
    """
    bids = []
    for nid, _sigma, kappa, tau, E_in in nodes:
        if kappa <= 0:
            b_star = tau
        else:
            # b_j*(μ) = τ_j + (1/κ_j)·ln(1 + κ_j/(μ·p_out))
            ratio = kappa / max(mu * p_out, 1e-15)
            # ln(1+ratio) ≈ ln(ratio) for ratio≫1; cap ratio to avoid overflow
            inner = min(1.0 + ratio, 1e15)
            excess = math.log(inner) / kappa
            b_star = max(tau, tau + excess)

        c_in  = p_in * E_in
        c_out = p_out * b_star
        bids.append(NodeBid(
            node_id  = nid,
            tau      = tau,
            b_star   = b_star,
            E_in     = E_in,
            cost_in  = c_in,
            cost_out = c_out,
            total    = c_in + c_out,
        ))
    return bids


# ---------------------------------------------------------------------------
# PSBC-M
# ---------------------------------------------------------------------------

class PSBCMacroScheduler:
    """
    Cross-round budget scheduler with Projected MWU.

    Parameters
    ----------
    B_total : float
        Global budget cap (yuan).
    K_max   : int
        Soft upper-bound on number of rounds (for MWU step-size init).
        When K is completely unknown, use a generous estimate and
        the Doubling Trick is applied internally.
    p_in    : float
        Input token price (yuan / token).
    p_out   : float
        Output token price (yuan / token).
    beta_stop : float
        EMA smoothing coefficient for P_stop (default 0.1).
    beta_trans : float
        EMA smoothing coefficient for C_trans_bar (default 0.3).
    gamma_collapse : float
        Soft-collapse exponent for terminal projection (default 1.5).
    """

    def __init__(
        self,
        B_total:         float,
        K_max:           int,
        p_in:            float,
        p_out:           float,
        *,
        beta_stop:       float = 0.1,
        beta_trans:      float = 0.3,
        gamma_collapse:  float = 1.5,
        oracle           = None,   # DagGatOracle for P_stop prediction (§6.2.1)
    ) -> None:
        if B_total <= 0:
            raise ValueError("B_total must be positive")
        if K_max < 1:
            raise ValueError("K_max must be >= 1")

        self.B_total    = B_total
        self.B_rem      = B_total
        self.K_max      = K_max
        self.p_in       = p_in
        self.p_out      = p_out
        self.oracle     = oracle

        # EMA hyper-parameters
        self.beta_stop       = beta_stop
        self.beta_trans      = beta_trans
        self.gamma_collapse  = gamma_collapse

        # MWU step-size  (Cesa-Bianchi & Lugosi, 2006, Thm 2.2)
        self.eta = math.sqrt(math.log(2.0) / K_max)

        # ── state ──────────────────────────────────────────────────────────
        self.k: int = 0                      # rounds completed so far
        self.P_stop_hat: float = 1.0 / K_max  # EMA-smoothed termination prob
        self.C_trans_bar: float = 0.0         # EMA-smoothed bridging cost (yuan)

        # Global dual variable μ (initialised lazily on first call)
        self.mu: float = 0.0

        # Ledger
        self.history: List[MacroRoundRecord] = []

        # Doubling Trick state
        self._K_current = K_max

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def compute_round_budget(
        self,
        round_k: int,
        C_trans_tokens: int,
        *,
        P_stop_signal: Optional[float] = None,
        available_nodes: Optional[int] = None,
    ) -> float:
        """
        Compute the actual sub-budget for round *round_k*.

        Parameters
        ----------
        round_k : int
            Current round number (1-indexed).
        C_trans_tokens : int
            Bridging tokens injected into the start nodes this round.
            (0 for round 1).
        P_stop_signal : float or None
            If available, the DAG-GAT stopping-head prediction P_stop(𝒢_k).
            When None, the EMA estimate is left unchanged.
        available_nodes : int or None
            Number of agent nodes in this round's topology (for
            minimum-survival check).  When None, a default of 1 is used.

        Returns
        -------
        B_actual : float
            Sub-budget (yuan) that PSBC-P (micro) must respect.
        """
        # ── Doubling Trick ────────────────────────────────────────────────
        if round_k > self._K_current:
            self._K_current *= 2
            self.eta = math.sqrt(math.log(2.0) / self._K_current)

        # ── 1. P_stop via DAG-GAT stopping head (§6.2.1) ────────────────
        if self.oracle is not None and nodes:
            try:
                from ngobc.dag_gat import GraphData, KNOWN_ROLES
                from ngobc.gat_trainer import _role_from_id
                import torch
                visible = [n[0] for n in nodes]
                id_to_idx = {nid: i for i, nid in enumerate(visible)}
                parents = {}
                for i, nid in enumerate(visible):
                    if nid == "Negative":
                        parents[i] = [id_to_idx["Affirmative"]] if "Affirmative" in id_to_idx else []
                    elif nid == "Moderator":
                        pa = []
                        if "Affirmative" in id_to_idx: pa.append(id_to_idx["Affirmative"])
                        if "Negative" in id_to_idx: pa.append(id_to_idx["Negative"])
                        parents[i] = pa
                    else:
                        parents[i] = []
                role_map = {r: i for i, r in enumerate(KNOWN_ROLES)}
                rows, rids = [], []
                for nid in visible:
                    match = [n for n in nodes if n[0] == nid]
                    s, kp, t = match[0][1], match[0][2], match[0][3] if match else (0.9, 0.01, 50)
                    rows.append([s, kp, t, t, 0.0, 0.0, 0.0])
                    rids.append(role_map.get(_role_from_id(nid), role_map["unknown"]))
                g = GraphData(visible, torch.tensor(rows, dtype=torch.float32),
                              torch.tensor(rids, dtype=torch.long), parents)
                p_stop_raw = self.oracle.model.forward_stop(g)
            except Exception:
                p_stop_raw = None
        else:
            p_stop_raw = None

        # Fallback: external signal or raw prediction
        if p_stop_raw is not None:
            self.P_stop_hat = (
                (1.0 - self.beta_stop) * self.P_stop_hat
                + self.beta_stop * p_stop_raw
            )
        elif P_stop_signal is not None:
            self.P_stop_hat = (
                (1.0 - self.beta_stop) * self.P_stop_hat
                + self.beta_stop * max(P_stop_signal, 1e-9)
            )

        N_rem = 1.0 / max(self.P_stop_hat, 1e-9)  # effective remaining horizon

        # ── 2. Update bridging-cost EMA ────────────────────────────────────
        C_trans_yuan = self.p_in * C_trans_tokens
        if round_k > 1 and C_trans_yuan > 0:
            if self.C_trans_bar == 0.0:
                self.C_trans_bar = C_trans_yuan
            else:
                self.C_trans_bar = (
                    (1.0 - self.beta_trans) * self.C_trans_bar
                    + self.beta_trans * C_trans_yuan
                )

        # ── 3. Strategic reserve for future bridging ──────────────────────
        E_future_trans = N_rem * self.C_trans_bar

        # ── 4. Target budget ──────────────────────────────────────────────
        B_target = (self.B_rem - E_future_trans) / N_rem

        # Safety: if target ≤ 0, all future budget is reserved for bridging
        if B_target <= 0 or self.B_rem < C_trans_yuan:
            n_nodes = available_nodes or 1
            # Bare-minimum survival: τ tokens per node output at minimum
            B_min = n_nodes * self.p_out * 50.0  # 50 = default τ per node
            B_actual = min(self.B_rem - C_trans_yuan, B_min)
            if B_actual <= 0:
                B_actual = 0.0
            self._record(round_k, B_target, B_actual, C_trans_tokens,
                         N_rem, bool(P_stop_signal), survival=True)
            return B_actual

        # ── 5. Lazy μ initialisation ──────────────────────────────────────
        if self.mu == 0.0:
            self.mu = self._compute_mu_seed(B_target)

        # ── 6. Physical truncation: cannot exceed available cash ──────────
        B_actual = min(self.B_rem - C_trans_yuan, B_target)

        self._record(round_k, B_target, B_actual, C_trans_tokens,
                     N_rem, bool(P_stop_signal))
        return B_actual

    def compute_bid_round_budget(
        self,
        round_k: int,
        nodes: List[Tuple[str, float, float, float, float]],
        C_trans_tokens: int = 0,
        *,
        P_stop_signal: Optional[float] = None,
    ) -> Tuple[float, List[NodeBid]]:
        """
        Full PSBC-M round budget via per-node bidding (§6.4.1).

        Parameters
        ----------
        round_k : int
        nodes : list of (node_id, sigma, kappa, tau, E_in_hat)
        C_trans_tokens : int
        P_stop_signal : float or None

        Returns
        -------
        B_actual : float     — sub-budget for the round (yuan)
        bids : List[NodeBid] — per-node bid records
        """
        # ── Doubling Trick ──────────────────────────────────────────────────
        if round_k > self._K_current:
            self._K_current *= 2
            self.eta = math.sqrt(math.log(2.0) / self._K_current)

        # ── 1. Horizon estimate ────────────────────────────────────────────
        if P_stop_signal is not None:
            self.P_stop_hat = (
                (1.0 - self.beta_stop) * self.P_stop_hat
                + self.beta_stop * max(P_stop_signal, 1e-9)
            )
        N_rem = 1.0 / max(self.P_stop_hat, 1e-9)

        # ── 2. Bridging-cost EMA ────────────────────────────────────────────
        C_trans_yuan = self.p_in * C_trans_tokens
        if round_k > 1 and C_trans_yuan > 0:
            if self.C_trans_bar == 0.0:
                self.C_trans_bar = C_trans_yuan
            else:
                self.C_trans_bar = (
                    (1.0 - self.beta_trans) * self.C_trans_bar
                    + self.beta_trans * C_trans_yuan
                )

        # ── 3. Strategic reserve ───────────────────────────────────────────
        E_future_trans = N_rem * self.C_trans_bar
        B_target = (self.B_rem - E_future_trans) / N_rem

        # ── 4. Lazy μ initialisation with actual node params ───────────────
        if self.mu == 0.0:
            self.mu = self._compute_mu_seed(B_target, nodes)

        # ── 5. Survival check ──────────────────────────────────────────────
        n_nodes = len(nodes) if nodes else 1
        if B_target <= 0 or self.B_rem < C_trans_yuan:
            # Bare-minimum: τ-only survival
            tau_sum = sum(n[3] for n in nodes) if nodes else 50.0
            B_min = tau_sum * self.p_out
            B_actual = min(max(self.B_rem - C_trans_yuan, 0.0), B_min)
            bids = compute_node_bids(nodes, self.mu, self.p_in, self.p_out)
            self._record(round_k, B_target, B_actual, C_trans_tokens,
                         N_rem, bool(P_stop_signal), survival=True)
            return B_actual, bids

        # ── 6. Per-node parallel bidding (§6.4.1) ──────────────────────────
        bids = compute_node_bids(nodes, self.mu, self.p_in, self.p_out)
        B_request = sum(b.total for b in bids)

        # ── 7. Physical truncation (§6.4.2) ────────────────────────────────
        B_net = max(self.B_rem - C_trans_yuan, 0.0)
        B_actual = min(B_net, B_request)

        self._record(round_k, B_target, B_actual, C_trans_tokens,
                     N_rem, bool(P_stop_signal))
        return B_actual, bids

    def update_after_round(
        self,
        C_true: float,
        task_is_done: bool = False,
    ) -> None:
        """
        MWU dual-variable update after round *k* completes.

        Parameters
        ----------
        C_true : float
            True monetary spend of the round (input + output).
        task_is_done : bool
            Whether the MAS internally considers the task finished.
        """
        if not self.history:
            return

        rec = self.history[-1]
        B_target = rec.B_target

        # ── 1. MWU loss ───────────────────────────────────────────────────
        if B_target > 0:
            loss_ratio = C_true / B_target
            self.mu *= math.exp(self.eta * (loss_ratio - 1.0))

        # ── 2. Segmented projection protocol ──────────────────────────────
        N_rem = 1.0 / max(self.P_stop_hat, 1e-9) if not task_is_done else 1.0
        delta = max(2, self.K_max // 4)

        if N_rem <= 1.0:
            # Hard collapse: final round — release all remaining budget
            self.mu = 1e-9
        elif N_rem <= delta:
            # Soft collapse: projected decay
            decay = (N_rem / delta) ** self.gamma_collapse
            self.mu = min(self.mu, self.mu * decay)

        # ── 3. Deduct spend ───────────────────────────────────────────────
        self.B_rem = max(0.0, self.B_rem - C_true)
        self.k += 1

        # ── 4. Update ledger ───────────────────────────────────────────────
        rec.C_true = C_true
        rec.mu_after = self.mu
        rec.task_done = task_is_done

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_mu_seed(
        self,
        B_target: float,
        nodes: Optional[List[Tuple[str, float, float, float, float]]] = None,
        n_nodes: int = 1,
    ) -> float:
        """
        Hot-start seed for μ₁.

        Derived from the product-KKT consistent bidding formula under
        mean-field node approximation:

            μ₁ = κ̄ / (p_out · (exp(κ̄·excess) − 1))

        where excess = B_out_per_node/p_out − τ̄ is the per-node output
        token budget above the emergence threshold.
        """
        if nodes and len(nodes) > 0:
            kappa_bar  = sum(n[2] for n in nodes) / len(nodes)
            tau_bar    = sum(n[3] for n in nodes) / len(nodes)
            E_in_total = sum(n[4] for n in nodes)
            n          = len(nodes)
        else:
            kappa_bar  = 0.02
            tau_bar    = 50.0
            E_in_total = 400.0
            n          = max(n_nodes, 1)

        # Deduct rigid input costs, compute per-node output budget above τ
        B_out_total = max(B_target - self.p_in * E_in_total, 1e-6)
        B_out_per_node = B_out_total / n
        excess_above_tau = max(B_out_per_node / self.p_out - tau_bar, 1.0)

        inner = min(kappa_bar * excess_above_tau, 30.0)
        mu_seed = kappa_bar / (self.p_out * (math.exp(inner) - 1.0))

        # Heuristic floor for generous budgets (μ → 0⁺)
        if mu_seed < 1e-9:
            mu_seed = 1e-9

        return mu_seed

    def _record(
        self,
        round_k: int,
        B_target: float,
        B_actual: float,
        C_trans_tokens: int,
        N_rem: float,
        P_stop_available: bool,
        survival: bool = False,
    ) -> None:
        self.history.append(MacroRoundRecord(
            round_k       = round_k,
            B_rem_before  = self.B_rem,
            B_target      = B_target,
            B_actual      = B_actual,
            C_trans       = C_trans_tokens,
            C_true        = 0.0,       # filled in update_after_round
            mu_before     = self.mu,
            mu_after      = self.mu,    # updated in update_after_round
            N_rem         = N_rem,
            P_stop        = self.P_stop_hat,
            task_done     = False,
            survival_trigger = survival,
        ))

    # ── query helpers ─────────────────────────────────────────────────────

    def get_mu(self) -> float:
        return self.mu

    def get_remaining_budget(self) -> float:
        return self.B_rem

    def ledger(self) -> List[MacroRoundRecord]:
        return list(self.history)
