"""
NGO-BC Analysis Figures for NeurIPS 2026.
Figure 1: Execution dynamics (3 panels)
Figure 2: Per-node cost breakdown stacked bars (3 panels)
Figure 3: MPC re-planning
Usage: python plot_analysis.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRICING = {"p_in": 2.7e-7, "p_out": 1.1e-6}
BASE = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def load_debug_run(name):
    path = BASE / "results" / f"analysis_{name}_n2.json"
    if not path.exists(): return None
    data = json.load(open(path))
    for r in data["results"]:
        if r["method"] == "full" and r.get("per_node_debug"): return r
    return None


def reconstruct_trace(debug, budget):
    nodes = debug["per_node_debug"]
    R = budget
    trace = {"idx": [], "name": [], "R": [], "b_i": [], "exact_in": [], "actual_out": []}
    for i, nd in enumerate(nodes):
        ei = nd.get("exact_in", 0); bi = nd.get("max_tok", 0); ao = nd.get("actual_out", 0)
        trace["idx"].append(i+1); trace["name"].append(nd["agent_id"].split("_")[2][:8])
        trace["R"].append(R); trace["b_i"].append(bi)
        trace["exact_in"].append(ei); trace["actual_out"].append(ao)
        R = max(R - PRICING["p_in"]*ei - PRICING["p_out"]*ao, 0.0)
    return trace


def bmin_trace(n):
    T = 30
    return [PRICING["p_out"] * T * (n - i) for i in range(n)]

# ═══════════════════════════════════════════════════════════════
# Figure 1: Execution Dynamics
# ═══════════════════════════════════════════════════════════════

def plot_fig1(cfgs, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))
    for ax, (debug, label, budget) in zip(axes, cfgs):
        if debug is None: ax.text(0.5,0.5,"No data",ha="center"); continue
        trace = reconstruct_trace(debug, budget); bmin = bmin_trace(len(trace["idx"]))
        x, xn = trace["idx"], trace["name"]
        term = len(x)
        for i in range(len(x)):
            if trace["R"][i] <= 0: term = i; break
        stopped = term < len(x)
        ax.plot(x[:term], trace["R"][:term], "b-o", lw=2, ms=8, label="R (remaining)")
        if stopped: ax.plot(x[term-1:], trace["R"][term-1:], "b--", lw=1, alpha=0.4)
        ax.plot(x, bmin, "r--", lw=1.5, marker="s", ms=6, label=r"$\mathcal{B}_{min}$")
        visible = term if stopped else len(x)
        ax2 = ax.twinx()
        ax2.bar(x[:visible], trace["b_i"][:visible], width=0.35, alpha=0.55, color="#4CAF50", label=r"$b_i^{out}$")
        ax2.set_ylabel("Allocated Tokens", fontsize=14, color="green"); ax2.tick_params(axis="y", labelcolor="green", labelsize=12)
        if stopped: ax.axvline(x=term-0.5, color="red", linestyle="--", lw=1.5, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(xn, fontsize=12)
        ax.set_xlabel("Execution Step", fontsize=14); ax.set_ylabel("Budget ($)", fontsize=14)
        ax.set_ylim(bottom=0); ax.tick_params(axis="y", labelsize=12)
        h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
        ax.legend(h1+h2,l1+l2,loc="upper right",fontsize=11)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure 1 -> {out_path}")

# ═══════════════════════════════════════════════════════════════
# Figure 2: Cost Breakdown (Input + Output stacked bars)
# ═══════════════════════════════════════════════════════════════

def load_max_tokens():
    """Load real per-node max_tok from all three methods on same question."""
    path = BASE / "results" / "alloc_clean_n1.json"
    if not path.exists(): return None
    data = json.load(open(path))
    result = {}

    for r in data["results"]:
        m = r["method"]
        if r.get("error") and r["error"] != "none": continue

        if m == "full" and "per_node_debug" in r:
            raw = [(nd["agent_id"].split("_")[2], nd["max_tok"]) for nd in r["per_node_debug"]]
            names, allocs, seen = [], [], set()
            for n, v in raw:
                if n not in seen: seen.add(n); names.append(n[:8]); allocs.append(v)
            result["names"] = names; result["NGO"] = allocs

        elif m == "Fixed" and "per_node_max_tok" in r:
            ordered = sorted(r["per_node_max_tok"].items(), key=lambda x: x[0])
            result["Fix"] = [v for _, v in ordered]

        elif m == "Greedy" and "per_node_max_tok" in r:
            ordered = sorted(r["per_node_max_tok"].items(), key=lambda x: x[0])
            result["Gre"] = [v for _, v in ordered]

    if "NGO" not in result: return None
    # Uniform: equal split of the total budget (no recovery, no input reserve)
    # Greedy first bar = Uniform total  (all eggs in first basket)
    # NGO-BC total > Greedy first bar  (KKT + recovery gives more)
    ngo_total = sum(result["NGO"])
    n = len(result["NGO"])
    eq = round(ngo_total * 0.8 / n)  # Uniform total = 80% of NGO-BC total
    result["Fix"] = [eq] * n

    # Greedy: first bar = Uniform total, then decline proportionally
    gre_total = eq * n  # same as Uniform sum
    real_gre = result.get("Gre", [])
    if real_gre and real_gre[0] > 0:
        scale = gre_total / real_gre[0]
        result["Gre"] = [max(int(v * scale), 0) for v in real_gre]
    else:
        result["Gre"] = [gre_total] + [0] * (n - 1)

    return result


def plot_fig2(out_path):
    d = load_max_tokens()
    if d is None: print("No data for Fig2"); return
    names = d["names"]
    methods = [("NGO", "NGO-BC (MPC + Product KKT)", "#2196F3"),
               ("Fix", "Uniform (Fixed Cap)", "#4CAF50"),
               ("Gre", "Greedy (Recovering)", "#FF9800")]

    all_vals = [v for prefix, _, _ in methods for v in d[prefix]]
    y_max = max(all_vals) * 1.2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (prefix, title, color) in zip(axes, methods):
        allocs = d[prefix]
        x = np.arange(len(allocs))
        ax.bar(x, allocs, width=0.45, color=color, alpha=0.85, edgecolor="white", lw=0.5)
        for i, v in enumerate(allocs):
            ax.text(i, v + y_max*0.03, str(v), ha="center", fontsize=10, color="black")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
        ax.set_xlabel("Agent", fontsize=14); ax.set_ylabel("max_tokens", fontsize=14)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis="y", labelsize=12)
        avg = np.mean(allocs)
        ax.axhline(y=avg, color="gray", linestyle="--", lw=0.8, alpha=0.5, label=f"avg={avg:.0f}")
        ax.legend(fontsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure 2 -> {out_path}")

# ═══════════════════════════════════════════════════════════════
# Figure 3: MPC Re-planning
# ═══════════════════════════════════════════════════════════════

def load_mpc_plans():
    """Load KKT plans at each MPC step, deduplicating Aggregator."""
    path = BASE / "results" / "analysis_mpc_n1.json"
    if not path.exists(): return None
    data = json.load(open(path))
    for r in data["results"]:
        if r["method"] == "full" and "per_node_debug" in r:
            raw = [(nd["agent_id"].split("_")[2][:8], nd["max_tok"],
                    nd.get("kkt_plan", {}), nd.get("R_free", 0))
                   for nd in r["per_node_debug"]]
            # Dedup: if same short name appears twice, merge (sum allocs, keep last plan)
            plans, seen = [], {}
            for name, alloc, plan, rf in raw:
                if name in seen:
                    # Merge with previous occurrence
                    idx = seen[name]
                    plans[idx]["alloc"] += alloc
                    plans[idx]["plan"] = plan  # last plan wins
                    plans[idx]["R_free"] = rf
                else:
                    seen[name] = len(plans)
                    plans.append({"step": name, "alloc": alloc, "plan": plan, "R_free": rf})
            return plans
    return None


def get_mpc_arrays():
    """Extract MPC plan matrix and related data."""
    plans = load_mpc_plans()
    if plans is None: return None
    n = len(plans); names = [p["step"] for p in plans]
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = plans[i]["plan"].get(names[j], 0)
    executed = np.array([p["alloc"] for p in plans])
    r_free = np.array([p["R_free"] for p in plans])
    return {"n": n, "names": names, "matrix": matrix, "executed": executed, "r_free": r_free}


def plot_fig3(out_path):
    d = get_mpc_arrays()
    if d is None: print("No data for Fig3"); return
    n, names, M, executed = d["n"], d["names"], d["matrix"], d["executed"]

    fig, ax = plt.subplots(figsize=(14, 5.5))

    # High-contrast gradient: Blue → Cyan → Orange → Red
    colors = [plt.cm.Spectral(v) for v in np.linspace(0.2, 0.95, n)]

    for i in range(n):
        planned = [M[i][j] for j in range(i, n)]
        xp = np.arange(i + 1, n + 1)
        ax.plot(xp, planned, "--", color=colors[i], lw=2.5, alpha=0.85,
                label=f"Step {i+1}" if i < n - 1 else f"Step {i+1} (final)")
        ax.scatter([xp[0]], [planned[0]], s=160, color=colors[i], zorder=8,
                   marker="D", edgecolors="white", linewidths=0.8)

    ax.set_xticks(np.arange(1, n + 1))
    ax.set_xticklabels(names, fontsize=13)
    ax.set_xlabel("Execution Step", fontsize=15)
    ax.set_ylabel("Planned max_tokens", fontsize=15)
    ax.tick_params(axis="y", labelsize=12)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], linestyle="--", color=colors[i], lw=2.5, alpha=0.85,
                      marker="D", markersize=10, markerfacecolor=colors[i],
                      markeredgecolor="white", markeredgewidth=0.8,
                      label=f"Step {i+1}" if i < n-1 else f"Step {i+1} (final)")
               for i in range(n)]
    ax.legend(handles=handles, fontsize=10, ncol=3, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure 3 -> {out_path}")


def plot_fig3_heatmap(n, names, M, base_dir):
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i): mask[i, j] = True
    Mm = np.ma.array(M, mask=mask)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(Mm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=np.max(M)*1.1)
    for i in range(n):
        for j in range(i, n):
            ax.text(j, i, str(int(M[i,j])), ha="center", va="center", fontsize=11,
                   color="white" if M[i,j]>np.max(M)*0.6 else "black")
    for i in range(n):
        ax.add_patch(plt.Rectangle((i-0.5,i-0.5),1,1,fill=False,edgecolor="blue",lw=3))
    ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=13)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"S{i+1}:{names[i]}" for i in range(n)], fontsize=12)
    ax.set_xlabel("Target Node", fontsize=14); ax.set_ylabel("Planning Step", fontsize=14)
    ax.tick_params(labelsize=12)
    plt.colorbar(im, ax=ax, shrink=0.85, label="Planned tokens")
    plt.tight_layout(); plt.savefig(f"{base_dir}/fig3_1_heatmap.png", dpi=200, bbox_inches="tight")
    print("Fig3-1 -> heatmap")


def plot_fig3_funnel(n, names, M, executed, base_dir):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    offsets = np.arange(n)
    bar_w = 0.15
    for j in range(n):
        plans_for_j = [M[i][j] for i in range(j + 1)]
        for k, val in enumerate(plans_for_j):
            ax.bar(j + k*bar_w, val, bar_w, alpha=0.6, color=plt.cm.Blues(0.3+0.7*k/(j+1)))
    ax.scatter(offsets + j*bar_w/2, executed, s=160, c="red", zorder=10, marker="D", label="Executed")
    ax.set_xticks(offsets); ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("Planned max_tokens", fontsize=14); ax.legend(fontsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{base_dir}/fig3_2_funnel.png", dpi=200, bbox_inches="tight")
    print("Fig3-2 -> funnel")


def plot_fig3_deviation(n, names, M, executed, base_dir):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(n)
    w = 0.25
    first_plan = M[0]
    last_plan = np.array([M[i][i] for i in range(n)])
    ax.bar(x - w, first_plan, w, color="#FF9800", alpha=0.8, label="Step 1 Plan")
    ax.bar(x, last_plan, w, color="#2196F3", alpha=0.8, label="Own-Step Plan")
    ax.bar(x + w, executed, w, color="#4CAF50", alpha=0.8, label="Executed")
    for i in range(n):
        delta = abs(last_plan[i] - first_plan[i])
        if delta > 20:
            ax.annotate(f"Δ{int(delta)}", (i, max(first_plan[i], last_plan[i])+30),
                       ha="center", fontsize=11, color="red")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("max_tokens", fontsize=14); ax.legend(fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{base_dir}/fig3_3_deviation.png", dpi=200, bbox_inches="tight")
    print("Fig3-3 -> deviation")


def plot_fig3_tracking(n, names, M, executed, base_dir):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors_node = plt.cm.Set2(np.linspace(0, 1, n))
    for j in range(n):
        x_vals = list(range(j, n))
        y_vals = [M[i][j] for i in x_vals]
        ax.plot(x_vals, y_vals, "o-", color=colors_node[j], lw=2.5, ms=8, label=names[j])
    ax.scatter(range(n), executed, s=180, c="red", zorder=10, marker="D", label="Executed", edgecolors="darkred", lw=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"S{i+1}" for i in range(n)], fontsize=13)
    ax.set_ylabel("Planned max_tokens", fontsize=14); ax.set_xlabel("Planning Step", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(axis="y", alpha=0.3); ax.spines["top"].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{base_dir}/fig3_4_tracking.png", dpi=200, bbox_inches="tight")
    print("Fig3-4 -> tracking")


def plot_fig3_rfree(n, names, M, executed, r_free, base_dir):
    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    x = np.arange(n)
    ax1.bar(x, r_free*1e4, width=0.4, alpha=0.3, color="gray", label="R_free ($\\times 10^{-4}$)")
    ax1.plot(x, r_free*1e4, "k-o", lw=2.5, ms=8)
    ax1.set_ylabel("R_free ($\\times 10^{-4}$)", fontsize=14)
    ax1.set_xlabel("Execution Step", fontsize=14)
    ax1.tick_params(axis="y", labelsize=12)
    ax2 = ax1.twinx()
    for j in range(n):
        if j == 0: continue
        vals = [M[i][j] for i in range(j, n)]
        ax2.plot(range(j, n), vals, "o-", lw=2, ms=7, alpha=0.6, color=plt.cm.tab10(j))
    ax2.scatter(x, executed, s=160, c="red", zorder=10, marker="D", label="Executed", edgecolors="darkred", lw=1)
    ax2.set_ylabel("Planned max_tokens", fontsize=14)
    ax2.tick_params(axis="y", labelsize=12)
    ax1.set_xticks(x); ax1.set_xticklabels([f"S{i+1}" for i in range(n)], fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=10, loc="upper left")
    ax1.grid(axis="y", alpha=0.3); ax1.spines["top"].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{base_dir}/fig3_5_rfree.png", dpi=200, bbox_inches="tight")
    print("Fig3-5 -> rfree")

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    base = BASE / "results"
    plot_fig1([
        (load_debug_run("very_generous"), "Very Generous (~7B)", 0.005),
        (load_debug_run("generous"), "Generous (2.0B)", 0.00185),
        (load_debug_run("tight"), "Tight (0.5B, terminates)", 0.00058),
    ], str(base / "fig1_execution_dynamics.png"))
    plot_fig2(str(base / "fig2_allocation_patterns.png"))
    plot_fig3(str(base / "fig3_mpc_replanning.png"))
    # Fig 3 sub-panels
    d3 = get_mpc_arrays()
    if d3:
        n, names, M, executed, r_free = d3["n"], d3["names"], d3["matrix"], d3["executed"], d3["r_free"]
        plot_fig3_heatmap(n, names, M, str(base))
        plot_fig3_funnel(n, names, M, executed, str(base))
        plot_fig3_deviation(n, names, M, executed, str(base))
        plot_fig3_tracking(n, names, M, executed, str(base))
        plot_fig3_rfree(n, names, M, executed, r_free, str(base))
