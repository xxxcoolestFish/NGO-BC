"""
psbc/em_calibration.py – Empirical (σ, κ) calibration from observation logs.

Method:
  For each role, fit the quality-function curve:
    Q_j(b) = σ_j · (1 - exp(-κ_j · (b - τ_j)))

  Calibration principle: at the natural (unconstrained) output length c_out,
  the agent should reach a target quality level q_target (default 0.85).
  This gives a closed-form estimate of κ from the observed c_out distribution.

  σ is estimated from the task-level accuracy at generous budget
  (where nodes are unconstrained) via back-propagation through the pipeline:
    Q_pipeline = Π σ_j  →  σ_j = Q_pipeline^(1/n)
  For heterogeneous roles, we distribute σ proportional to observed utilization.

Usage:
    from ngobc.em_calibration import calibrate_from_logs
    params = calibrate_from_logs("psbc_logs/psbc_observation.jsonl")
    # params = {"Decomposer": (sigma, kappa, tau), ...}
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def calibrate_from_logs(
    log_path: str | Path,
    q_target: float = 0.85,
    generous_acc: float | None = None,
    kappa_default: float = 0.01,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Estimate (σ, κ, τ) per role from JSONL observation logs.

    κ is calibrated only for roles where median c_out > τ (the quality function
    is well-defined). For roles where c_out < τ frequently (selection bias:
    e.g. Verifier is short because Solver usually succeeds), κ defaults to
    kappa_default to avoid under-allocating tokens under budget pressure.

    Parameters
    ----------
    log_path       : path to psbc_observation.jsonl
    q_target       : assumed quality at median unconstrained output (default 0.85)
    generous_acc   : overall task accuracy at generous budget (bounds σ)
    kappa_default  : fallback κ for roles with selection bias (default 0.01)

    Returns
    -------
    dict mapping role_name → (sigma, kappa, tau)
    """
    records = [json.loads(l) for l in open(log_path) if l.strip()]

    c_outs: Dict[str, List[int]] = defaultdict(list)
    taus:   Dict[str, List[int]] = defaultdict(list)

    for r in records:
        for ex in r["executions"]:
            role = ex["role"]
            c_outs[role].append(ex["c_out"])
            taus[role].append(ex["C_sys"] + ex["C_struct"])

    results = {}
    roles   = sorted(c_outs.keys())

    for role in roles:
        co   = sorted(c_outs[role])
        tau  = float(np.mean(taus[role]))

        c_median    = float(np.median(co))
        frac_below  = sum(1 for v in co if v < tau) / len(co)

        if frac_below > 0.3:
            # Selection bias: most natural outputs are below τ.
            # The quality function is ill-defined; use the default κ.
            kappa = kappa_default
        else:
            denom = max(c_median - tau, 10.0)
            kappa = -math.log(1 - q_target) / denom

        if generous_acc is not None:
            n_roles   = len(roles)
            sigma_pool = generous_acc ** (1.0 / n_roles)
            sigma = min(sigma_pool, 0.95)
        else:
            sigma = 0.9

        results[role] = (sigma, kappa, tau)

    return results


def print_calibration(params: Dict[str, Tuple[float, float, float]]) -> None:
    """Pretty-print calibration results."""
    print(f"\n{'Role':15s}  {'σ':>6}  {'κ':>10}  {'τ':>6}  "
          f"{'90%-quality at':>16}  {'note'}")
    print("-" * 70)
    for role, (sigma, kappa, tau) in sorted(params.items()):
        # tokens needed to reach 90% of sigma
        b_90 = tau + (-math.log(0.1) / kappa)
        note = "fast-saturate" if kappa > 0.01 else "slow-saturate"
        print(f"{role:15s}  {sigma:6.3f}  {kappa:10.5f}  {tau:6.0f}  "
              f"{b_90:>16.0f} tokens  {note}")
