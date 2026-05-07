"""
experiments/run_pipeline.py — 3-node serial pipeline on MATH dataset.

Usage:
  LLM_API_KEY=xxx LLM_BASE_URL=xxx LLM_MODEL=xxx \
  python experiments/run_pipeline.py --n 50 --workers 4 --budget medium
"""

import sys, os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ngobc import (
    PSBCOptimizer, NodeParams,
    FixedAllocator, GreedyAllocator,
)
from ngobc.dag import topo_weights, critical_path_nodes
from ngobc.gat_oracle import DagGatOracle
from ngobc.math_dataset import load_math_problems, sample_problems, answers_match
from mas import LLMClient, DAGExecutor, make_pipeline_agents, extract_boxed

# Budget tiers (yuan)
BUDGETS = {
    "ultra_tight": 0.00025, "tight": 0.00035, "low": 0.00055,
    "medium": 0.00080, "high": 0.00130, "generous": 0.00185,
}

# Default calibration (EM-trained; replace with your own from Phase 0)
DEFAULT_CALIB = {
    "Decomposer": {"sigma": 0.898, "kappa": 0.150, "tau": 13},
    "Solver":     {"sigma": 0.953, "kappa": 0.033, "tau": 56},
    "Verifier":   {"sigma": 0.583, "kappa": 0.001, "tau": 20},
}


def run_one(prob, method, budget_label, budget, task_id, calib, oracle):
    p_in = float(os.environ.get("LLM_P_IN", "1.1e-7"))
    p_out = float(os.environ.get("LLM_P_OUT", "2.9e-7"))
    llm = LLMClient(p_in=p_in, p_out=p_out)
    agents = make_pipeline_agents()
    executor = DAGExecutor(agents, llm)

    # Build node_params
    node_params = {}
    for s in agents:
        c = calib.get(s.role, DEFAULT_CALIB.get(s.role, {"sigma": 0.9, "kappa": 0.01, "tau": 0}))
        node_params[s.node_id] = NodeParams(
            s.node_id, c["sigma"], c["kappa"], c["tau"],
            weight=1.0,
        )

    total_spent = 0.0
    answer = ""
    try:
        if method == "PSBC":
            opt = PSBCOptimizer(
                adj=executor.adjacency,
                node_params=node_params,
                budget=budget, p_in=p_in, p_out=p_out,
                oracle=oracle, use_product=True,
            )
            for nid in executor.iter_topological():
                e_in = executor.get_input_tokens(nid, prob.problem)
                decision, cap = opt.decide(nid, e_in)
                if decision == "TERMINATE" or decision == "SKIP":
                    executor.skip(nid)
                    opt.settle_skip(nid) if decision == "SKIP" else opt.settle_terminate(nid, 0)
                    if decision == "TERMINATE":
                        break
                    continue
                c_out = executor.execute(nid, prob.problem, cap)
                opt.settle(nid, c_out)
                total_spent += p_in * e_in + p_out * c_out
        elif method == "Fixed":
            alloc = FixedAllocator(len(agents), budget, p_in, p_out)
            for nid in executor.iter_topological():
                e_in = executor.get_input_tokens(nid, prob.problem)
                decision, cap = alloc.decide(nid, e_in)
                if decision == "SKIP":
                    executor.skip(nid); alloc.settle_skip(nid); continue
                c_out = executor.execute(nid, prob.problem, cap)
                alloc.settle(nid, c_out)
                total_spent += p_in * e_in + p_out * c_out
        else:  # Greedy
            alloc = GreedyAllocator(budget, p_in, p_out)
            for nid in executor.iter_topological():
                e_in = executor.get_input_tokens(nid, prob.problem)
                decision, cap = alloc.decide(nid, e_in)
                if decision == "SKIP":
                    executor.skip(nid); alloc.settle_skip(nid); continue
                c_out = executor.execute(nid, prob.problem, cap)
                alloc.settle(nid, c_out)
                total_spent += p_in * e_in + p_out * c_out

        answer = executor.get_output("Verifier") or ""
    except Exception as e:
        return {"task_id": task_id, "error": str(e), "correct": False}

    pred = extract_boxed(answer) or answer
    correct = answers_match(pred, prob.answer) if pred else False
    return {
        "task_id": task_id, "method": method, "budget_label": budget_label,
        "budget": budget, "problem_id": prob.problem_id,
        "subject": prob.subject, "level": prob.level,
        "correct": correct, "spent": total_spent,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default="")
    ap.add_argument("--budget", default="all",
                    choices=["all"] + list(BUDGETS.keys()))
    ap.add_argument("--out", default="results/pipeline_results.json")
    ap.add_argument("--gat", default="", help="Path to trained GAT model (.pt)")
    args = ap.parse_args()

    # Data
    data_root = args.data or os.environ.get("MATH_DATA", "./data/math")
    levels = {"Level 3", "Level 4", "Level 5"}
    probs = sample_problems(
        load_math_problems(data_root, levels=levels),
        n=args.n, seed=42, stratified=True,
    )
    print(f"Loaded {len(probs)} problems")

    # GAT oracle
    oracle = None
    if args.gat and Path(args.gat).exists():
        oracle = DagGatOracle.load(args.gat)

    budgets = list(BUDGETS.items()) if args.budget == "all" else \
              [(args.budget, BUDGETS[args.budget])]
    methods = ["PSBC", "Fixed", "Greedy"]

    tasks = []
    for bl, bg in budgets:
        for p in probs:
            for m in methods:
                tasks.append({
                    "prob": p, "method": m, "budget_label": bl,
                    "budget": bg, "task_id": f"{m}_{bl}_{p.problem_id}",
                    "calib": DEFAULT_CALIB, "oracle": oracle,
                })

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, **t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            ok = [x for x in results if "error" not in x]
            acc = sum(x["correct"] for x in ok) / max(len(ok), 1) if ok else 0
            print(f"  [{len(results)}/{len(tasks)}] acc={acc:.1%} "
                  f"[{(time.time()-t0)/60:.0f}m]", flush=True)

    # Summary table
    ok = [r for r in results if "error" not in r]
    print(f"\n{'Method':10s}", end="")
    for bl, _ in budgets:
        print(f"{bl:>14s}", end="")
    print(f"  {'Overall':>8s}")
    print("-" * (10 + 16 * len(budgets) + 10))
    for m in methods:
        print(f"{m:10s}", end="")
        for bl, _ in budgets:
            ss = [r for r in ok if r["method"] == m and r["budget_label"] == bl]
            acc = sum(r["correct"] for r in ss) / len(ss) if ss else 0
            print(f"  {acc:5.0%} n={len(ss):1d}", end="")
        ov = [r for r in ok if r["method"] == m]
        print(f"  {sum(r['correct'] for r in ov)/len(ov):6.1%}" if ov else "")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": {"budgets": dict(budgets)}, "results": results},
              open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
