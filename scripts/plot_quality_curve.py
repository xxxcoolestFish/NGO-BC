"""
Plot per-role accuracy vs max_tokens with fitted exponential saturation curves.

Usage: python plot_role_quality_curve.py --data results/role_quality.json
"""

import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict
from pathlib import Path


ROLE_COLORS = {"Decomposer": "#2196F3", "Solver": "#FF9800", "Verifier": "#4CAF50"}
ROLE_MARKERS = {"Decomposer": "o", "Solver": "s", "Verifier": "^"}

# Quality function: Q(b) = σ · (1 − exp(−κ · (b − τ))), b ≥ τ
def quality_fn(b, sigma, kappa, tau):
    return sigma * (1.0 - np.exp(-kappa * np.clip(b - tau, 0, None)))


def fit_quality_curve(max_toks, accs):
    """Fit (sigma, kappa, tau) from accuracy data."""
    x = np.array(max_toks, dtype=float)
    y = np.array(accs, dtype=float)

    # Initial guess
    sigma0 = min(max(y), 0.9)
    kappa0 = 0.01
    tau0 = min(x) * 0.5

    try:
        popt, _ = curve_fit(
            quality_fn, x, y,
            p0=[sigma0, kappa0, tau0],
            bounds=([0.1, 1e-6, 0], [1.0, 1.0, max(x)]),
            maxfev=5000,
        )
        return popt
    except Exception:
        return np.array([sigma0, kappa0, tau0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/role_quality.json")
    ap.add_argument("--out", default="results/fig_role_quality.png")
    args = ap.parse_args()

    data = json.load(open(args.data))
    results = data.get("results", [])
    max_tok_levels = data.get("max_tokens_levels", [])
    roles = data.get("roles", ["Decomposer", "Solver", "Verifier"])

    # Aggregate: per role, per max_tokens → accuracy
    ok = [r for r in results if "error" not in r]
    agg = defaultdict(lambda: defaultdict(list))
    for r in ok:
        agg[r["role"]][r["max_tokens"]].append(r["correct"])

    # ═══════ Plot: 2 rows, 1 column ═══════
    plot_roles = ["Solver", "Verifier"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

    labels = ["(A)", "(B)"]
    fit_summary = {}
    for idx, (role, ax) in enumerate(zip(plot_roles, axes)):
        mt_list = sorted(agg[role].keys())
        acc_mean = [np.mean(agg[role][mt]) for mt in mt_list]
        acc_std = [np.std(agg[role][mt]) / max(np.sqrt(len(agg[role][mt])), 1)
                   for mt in mt_list]

        # Data points with error bars
        ax.errorbar(mt_list, acc_mean, yerr=acc_std,
                    fmt=ROLE_MARKERS[role], color=ROLE_COLORS[role],
                    capsize=6, capthick=2.5, lw=0, ms=20,
                    label="Observed accuracy", alpha=0.85, zorder=5)

        # Fit curve
        popt = fit_quality_curve(mt_list, acc_mean)
        sigma, kappa, tau = popt

        b_smooth = np.linspace(0, 2000, 300)
        q_smooth = quality_fn(b_smooth, sigma, kappa, tau)
        ax.plot(b_smooth, q_smooth, '-', color=ROLE_COLORS[role], lw=4, alpha=0.7)

        # Mark τ (emergence threshold)
        ax.axvline(x=tau, color=ROLE_COLORS[role], linestyle=":", lw=2, alpha=0.5)

        ax.set_ylabel(f"Accuracy ({role})", fontsize=38)
        ax.tick_params(labelsize=28)
        ax.set_xlim(0, 2000)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=26, loc="lower right")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.06, 1.04, labels[idx], transform=ax.transAxes, fontsize=40,
                fontweight='bold', va='bottom')

        fit_summary[role] = {"sigma": float(sigma), "kappa": float(kappa), "tau": float(tau)}

    axes[-1].set_xlabel("max_tokens (output token budget)", fontsize=38)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {args.out}")

    # Print fit parameters
    print("\nFitted quality function parameters Q(b) = σ·(1−exp(−κ(b−τ))):")
    for role in plot_roles:
        fs = fit_summary[role]
        print(f"  {role:12s}: σ={fs['sigma']:.3f}, κ={fs['kappa']:.4f}, τ={fs['tau']:.0f}")

    # Also save a version for the combined figure
    return fit_summary


if __name__ == "__main__":
    main()
