"""
experiments/run_mad.py — Multi-Agent Debate (MAD) with cross-round MWU scheduling.

Usage:
  LLM_API_KEY=xxx LLM_BASE_URL=xxx LLM_MODEL=xxx \
  python experiments/run_mad.py --n 50 --workers 4
"""

import sys, os, json, re, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ngobc import PSBCMacroScheduler
from ngobc.math_dataset import load_math_problems, sample_problems, answers_match
from mas import LLMClient, DAGExecutor, make_debate_agents, extract_boxed

BUDGETS = {
    "ultra_tight": 0.0003, "tight": 0.0005, "low": 0.0008,
    "medium": 0.0012, "high": 0.0020, "generous": 0.0040,
}
MAX_ROUNDS = 3
DEFAULT_CALIB = {
    "Affirmative": {"sigma": 0.90, "kappa": 0.0029, "tau": 13},
    "Negative":    {"sigma": 0.90, "kappa": 0.0027, "tau": 15},
    "Moderator":   {"sigma": 0.90, "kappa": 0.0116, "tau": 36},
}


def parse_debate_answer(mod_output: str) -> str:
    try:
        return json.loads(mod_output).get("debate_answer", "")
    except Exception:
        m = re.search(r'"debate_answer"\s*:\s*"([^"]*)"', mod_output)
        return m.group(1) if m else ""


def run_one(prob, method, budget_label, budget, task_id, calib):
    p_in = float(os.environ.get("LLM_P_IN", "1.1e-7"))
    p_out = float(os.environ.get("LLM_P_OUT", "2.9e-7"))
    llm = LLMClient(p_in=p_in, p_out=p_out)
    agents = make_debate_agents()
    agent_ids = [s.node_id for s in agents]

    # Build node_list for macro bidding
    node_list = []
    for s in agents:
        c = calib.get(s.role, DEFAULT_CALIB.get(s.role))
        node_list.append((s.node_id, c["sigma"], c["kappa"], c["tau"], 400))

    macro = PSBCMacroScheduler(
        B_total=budget, K_max=MAX_ROUNDS,
        p_in=p_in, p_out=p_out, gamma_collapse=1.5,
    )
    macro.P_stop_hat = 0.33  # prior: ~3 rounds

    total_spent = 0.0
    debate_answer = ""
    try:
        for k in range(1, MAX_ROUNDS + 1):
            if macro.B_rem <= 0:
                break

            executor = DAGExecutor(agents, llm)

            # Update E_in estimates
            for nid in agent_ids:
                node_list[agent_ids.index(nid)] = (
                    nid, calib.get(agents[agent_ids.index(nid)].role,
                                   DEFAULT_CALIB.get(agents[agent_ids.index(nid)].role))["sigma"],
                    calib.get(agents[agent_ids.index(nid)].role,
                              DEFAULT_CALIB.get(agents[agent_ids.index(nid)].role))["kappa"],
                    calib.get(agents[agent_ids.index(nid)].role,
                              DEFAULT_CALIB.get(agents[agent_ids.index(nid)].role))["tau"],
                    executor.get_input_tokens(nid, prob.problem),
                )

            B_actual, bids = macro.compute_bid_round_budget(k, node_list, 0)
            if B_actual <= 0:
                break

            bid_map = {b.node_id: b for b in bids}
            round_spent = 0.0

            for nid in executor.iter_topological():
                e_in = executor.get_input_tokens(nid, prob.problem)
                bid = bid_map.get(nid)

                if method == "Fixed":
                    cap = int(max(B_actual / 3 / p_out, 100))
                elif method == "Greedy":
                    cap = int(max((B_actual - round_spent) / p_out, 100))
                else:
                    cap = int(min(bid.b_star, 4096)) if bid else 100

                c_out = executor.execute(nid, prob.problem, cap)
                round_spent += p_in * e_in + p_out * c_out

            total_spent += round_spent
            result_text = executor.get_output("Moderator") or ""
            da = parse_debate_answer(result_text)
            if da:
                debate_answer = da

            task_done = (k >= MAX_ROUNDS) or bool(da)
            macro.update_after_round(round_spent, task_is_done=task_done)
            if task_done:
                break

    except Exception as e:
        return {"task_id": task_id, "error": str(e), "correct": False}

    correct = answers_match(debate_answer, prob.answer) if debate_answer else False
    return {
        "task_id": task_id, "method": method, "budget_label": budget_label,
        "budget": budget, "problem_id": prob.problem_id,
        "subject": prob.subject, "level": prob.level,
        "correct": correct, "spent": total_spent,
        "rounds": k if 'k' in dir() else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default="")
    ap.add_argument("--budget", default="all")
    ap.add_argument("--out", default="results/mad_results.json")
    args = ap.parse_args()

    data_root = args.data or os.environ.get("MATH_DATA", "./data/math")
    probs = sample_problems(
        load_math_problems(data_root, levels={"Level 3", "Level 4", "Level 5"}),
        n=args.n, seed=42, stratified=True,
    )

    budgets = list(BUDGETS.items()) if args.budget == "all" else \
              [(args.budget, BUDGETS[args.budget])]
    methods = ["PSBC", "Fixed", "Greedy"]

    tasks = [{"prob": p, "method": m, "budget_label": bl, "budget": bg,
              "task_id": f"{m}_{bl}_{p.problem_id}", "calib": DEFAULT_CALIB}
             for bl, bg in budgets for p in probs for m in methods]

    results = []; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, **t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result(); results.append(r)
            ok = [x for x in results if "error" not in x]
            acc = sum(x["correct"] for x in ok) / max(len(ok), 1) if ok else 0
            print(f"  [{len(results)}/{len(tasks)}] acc={acc:.1%} "
                  f"[{(time.time()-t0)/60:.0f}m]", flush=True)

    ok = [r for r in results if "error" not in r]
    print(f"\n{'Method':10s}", end="")
    for bl, _ in budgets: print(f"{bl:>14s}", end="")
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
