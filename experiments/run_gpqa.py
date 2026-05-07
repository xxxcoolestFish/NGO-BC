"""
run_gpqa_adaptation.py — TTAS Adaptation Experiment on GPQA Diamond.

Tests how TTAS (Test-Time Adaptive Scaling) dynamically adjusts when the
MAS encounters out-of-distribution questions (GPQA vs MATH training data).

Key tracking:
  - s_task trajectory: how the scaling factor evolves per question
  - Cumulative accuracy over GPQA questions
  - Per-node allocation patterns

Usage:
  # Quick test
  python run_gpqa_adaptation.py --n 2 --workers 1 --llm deepseek --debug

  # Full experiment
  python run_gpqa_adaptation.py --n 50 --out results/gpqa_adaptation_n50.json \
      --workers 8 --llm deepseek

  # Without TTAS (control)
  python run_gpqa_adaptation.py --n 50 --out results/gpqa_no_adapt_n50.json \
      --workers 8 --llm deepseek --mode tta_off
"""

from __future__ import annotations

import sys, json, time, argparse, threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

METAGPT_ROOT = Path(__file__).parent.parent / "MAS_inter"
sys.path.insert(0, str(METAGPT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from metagpt.actions import Action
from metagpt.roles.role import Role
from metagpt.team import Team
from metagpt.context import Context
from metagpt.config2 import config as global_config
from metagpt.ext.psbc import MetaGPTPSBCAdapter, PSBCRuntime
from metagpt.ext.psbc.llm_preset import use_llm, get_pricing

from ngobc.kkt import NodeParams
from ngobc.dag import topo_weights, critical_path_nodes, get_descendants
from ngobc.optimizer import PSBCOptimizer, _NodeState
from ngobc.gat_oracle import DagGatOracle
from ngobc.parallel_product_kkt import parallel_product_kkt_solve
from gpqa_dataset import (
    load_gpqa_questions, sample_questions, interleave_by_domain,
    extract_answer_from_response, check_answer,
)


# ═══════════════════════════════════════════════════════
# 5-node Parallel MAS (same as ablation)
# ═══════════════════════════════════════════════════════

class Decompose(Action):
    name: str = "Decompose"
    async def run(self, history, *args, **kwargs) -> str:
        msgs = history if isinstance(history, list) else []
        problem = msgs[-1].content if msgs else ""
        return await self._aask(
            f"Analyse the following question. Identify key concepts, "
            f"what is being asked, and outline a strategy. "
            f"Do NOT give the final answer yet.\n\n{problem}"
        )

_SOLVER_PROMPTS = [
    "Solve step by step. Put your final answer in \\boxed{{}}.",
    "Solve using a different approach. Put your final answer in \\boxed{{}}.",
    "Solve and verify each step. Put your final answer in \\boxed{{}}.",
]

class Solve1(Action):
    name: str = "Solve1"
    async def run(self, history, *args, **kwargs) -> str:
        msgs = history if isinstance(history, list) else []
        analysis = msgs[-1].content if msgs else ""
        return await self._aask(f"Using this analysis:\n{analysis}\n\n{_SOLVER_PROMPTS[0]}")

class Solve2(Action):
    name: str = "Solve2"
    async def run(self, history, *args, **kwargs) -> str:
        msgs = history if isinstance(history, list) else []
        analysis = msgs[-1].content if msgs else ""
        return await self._aask(f"Using this analysis:\n{analysis}\n\n{_SOLVER_PROMPTS[1]}")

class Solve3(Action):
    name: str = "Solve3"
    async def run(self, history, *args, **kwargs) -> str:
        msgs = history if isinstance(history, list) else []
        analysis = msgs[-1].content if msgs else ""
        return await self._aask(f"Using this analysis:\n{analysis}\n\n{_SOLVER_PROMPTS[2]}")

class Aggregate(Action):
    name: str = "Aggregate"
    async def run(self, history, *args, **kwargs) -> str:
        msgs = history if isinstance(history, list) else []
        solutions = "\n\n---\n\n".join(
            m.content for m in msgs if hasattr(m, 'content') and 'boxed' in m.content.lower()
        )
        if not solutions:
            solutions = "\n\n---\n\n".join(
                m.content for m in msgs[-3:] if hasattr(m, 'content')
            )
        return await self._aask(
            f"Three solvers produced these solutions:\n\n{solutions}\n\n"
            f"Compare them. If they agree, confirm the answer in \\boxed{{}}. "
            f"If they disagree, determine the best answer. "
            f"Put your final answer inside \\boxed{{}}."
        )

class Decomposer(Role):
    name: str = "Decomposer"; profile: str = "Decomposer"; actions: list = [Decompose]
    def __init__(self, **kw):
        super().__init__(**kw)
        from metagpt.actions.add_requirement import UserRequirement
        self._watch([UserRequirement])

class Solver1(Role):
    name: str = "Solver1"; profile: str = "Solver1"; actions: list = [Solve1]
    def __init__(self, **kw): super().__init__(**kw); self._watch([Decompose])

class Solver2(Role):
    name: str = "Solver2"; profile: str = "Solver2"; actions: list = [Solve2]
    def __init__(self, **kw): super().__init__(**kw); self._watch([Decompose])

class Solver3(Role):
    name: str = "Solver3"; profile: str = "Solver3"; actions: list = [Solve3]
    def __init__(self, **kw): super().__init__(**kw); self._watch([Decompose])

class Aggregator(Role):
    name: str = "Aggregator"; profile: str = "Aggregator"; actions: list = [Aggregate]
    def __init__(self, **kw): super().__init__(**kw); self._watch([Solve1, Solve2, Solve3])

def build_team():
    team = Team(context=Context(config=global_config), use_mgx=False)
    team.hire([Decomposer(), Solver1(), Solver2(), Solver3(), Aggregator()])
    return team


# ═══════════════════════════════════════════════════════
# Worker: run one GPQA question with TTAS tracking
# ═══════════════════════════════════════════════════════

def _run_one_gpqa(question, budget, pricing, calib_params, oracle, enable_ttas: bool):
    """Run one GPQA question through the parallel MAS, tracking TTAS state."""
    from metagpt.ext.psbc.llm_preset import use_llm as _set_llm

    team = build_team()
    runtime = PSBCRuntime()
    adapter = MetaGPTPSBCAdapter(context=team.env.context, runtime=runtime)
    adapter.bind_team(team)
    runtime.extras["send_to"] = "Decomposer"
    executor = adapter.task_manager().start_controlled_execution(
        task_id=f"gpqa_{question.question_id}",
        task_prompt=question.prompt)

    topo = executor.get_initial_topology()
    w = topo_weights(topo)
    node_params = {}
    for nid in topo:
        sigma, kappa, tau = 0.9, 0.01, 0.0
        if calib_params:
            for role, (s, k, t) in calib_params.items():
                if role.lower() in nid.lower():
                    sigma, kappa, tau = s, k, t
                    break
        node_params[nid] = NodeParams(nid, sigma, kappa, tau, w.get(nid, 0.1))

    opt = PSBCOptimizer(
        adj=dict(topo), node_params=node_params,
        budget=budget, p_in=pricing["p_in"], p_out=pricing["p_out"],
        oracle=oracle, use_product=False,
    )

    per_node = {}
    s_task_trajectory = []  # (agent_id, s_task_before, s_task_after, exact_in, e_hat)

    for agent_id in executor.get_topological_iterator():
        meta = executor.get_agent_metadata(agent_id)
        exact_in = executor.get_exact_input_tokens(agent_id)

        # Record s_task BEFORE this node
        s_before = oracle.s_task if oracle else 1.0

        # Extend topology
        added = [x for x in executor.get_initial_topology() if x not in node_params]
        if added:
            for x in added:
                np_new = NodeParams(x, 0.9, 0.01, 0.0, w.get(x, 0.1))
                node_params[x] = np_new
                s = _NodeState(params=np_new)
                s.b_prev = opt.B_t / max(len(node_params), 1) / opt.p_out
                opt._state[x] = s
            opt._crit_nodes = critical_path_nodes(opt.adj)
        node_params[agent_id].tau = max(node_params[agent_id].tau, float(meta.tau))

        # Set exact_in on optimizer state (needed by _neural_reserves for TTAS update)
        opt._state[agent_id].exact_in = exact_in

        # Compute reserves & KKT (with TTAS-active GAT prediction)
        descendants = list(get_descendants(opt.adj, agent_id))
        reserves = opt._compute_reserves(agent_id, descendants)
        R_prime = opt.R - pricing["p_in"] * exact_in
        R_free = R_prime - sum(reserves.values())
        if R_free <= 0 and descendants:
            reserves = opt._compute_reserves(agent_id, descendants, effective_gamma=0.0)
            R_free = R_prime - sum(reserves.values())

        result = parallel_product_kkt_solve(opt.adj, node_params, R_free, pricing["p_out"])
        max_tok = int(result.allocations.get(agent_id, node_params[agent_id].tau))
        max_tok = max(max_tok, 1)

        actual = executor.execute_agent(agent_id, max_tok, True)
        opt.settle(agent_id, actual)
        per_node[agent_id] = {"max_tok": max_tok, "exact_in": exact_in, "actual_out": actual}

        # Record s_task AFTER this node (updated by _neural_reserves → update_scale)
        s_after = oracle.s_task if oracle else 1.0
        s_task_trajectory.append({
            "agent_id": agent_id,
            "s_before": s_before,
            "s_after": s_after,
            "exact_in": exact_in,
        })

        executor.get_newly_discovered_topology()

    summ = opt.summary()
    answer = executor.get_final_result()
    predicted = extract_answer_from_response(answer)
    correct = check_answer(predicted, question.answer)

    return {
        "question_id": question.question_id,
        "domain": question.domain,
        "correct": correct,
        "predicted": predicted,
        "ground_truth": question.answer,
        "spent": summ.total_spent,
        "budget_ok": summ.budget_ok,
        "nodes_exec": summ.nodes_executed,
        "nodes_skip": summ.nodes_skipped,
        "s_task_trajectory": s_task_trajectory,
        "per_node": per_node,
        "answer": answer[:500] if answer else "",
    }


# ═══════════════════════════════════════════════════════
# Worker subprocess
# ═══════════════════════════════════════════════════════

def _worker(args_dict: dict) -> dict:
    """Subprocess worker: one GPQA question."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "MAS_inter"))
    _sys.path.insert(0, str(Path(__file__).parent))

    from metagpt.ext.psbc.llm_preset import use_llm as _set_llm
    _set_llm(args_dict["llm_preset"])

    enable_ttas = args_dict["enable_ttas"]
    gat_path = args_dict["gat_path"]
    oracle = None
    if Path(gat_path).exists():
        from ngobc.gat_oracle import DagGatOracle
        oracle = DagGatOracle.load(gat_path, enable_ttas=enable_ttas)
        oracle.reset()

    # Load calibration params
    em_json = Path(__file__).parent / args_dict["em_params_path"]
    calib_params = None
    if em_json.exists():
        raw = json.loads(em_json.read_text())
        calib_params = {}
        for k, v in raw.items():
            if isinstance(v, (list, tuple)):
                calib_params[k] = tuple(v)
            else:
                calib_params[k] = (v["sigma"], v["kappa"], v["tau_quality"])

    from gpqa_dataset import GPQAQuestion
    qd = args_dict["question"]
    question = GPQAQuestion(
        question_id=qd["question_id"], question=qd["question"],
        correct=qd["correct"], options=qd["options"],
        domain=qd["domain"], subdomain=qd.get("subdomain", ""),
    )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = _run_one_gpqa(question, args_dict["budget"], args_dict["pricing"],
                              calib_params, oracle, enable_ttas)
            return r
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                time.sleep(3 * (2 ** attempt))
                continue
            return {
                "question_id": args_dict["question"]["question_id"],
                "domain": args_dict["question"]["domain"],
                "correct": False, "predicted": None,
                "ground_truth": args_dict["question"]["correct"],
                "spent": 0.0, "budget_ok": True,
                "nodes_exec": 0, "nodes_skip": 0,
                "s_task_trajectory": [],
                "per_node": {},
                "answer": "", "error": err[:200],
            }


# ═══════════════════════════════════════════════════════
# Checkpoint & Driver
# ═══════════════════════════════════════════════════════

def _ckpt_path(out: Path) -> Path:
    return out.with_suffix(".ckpt.json")


def _load_ckpt(out: Path):
    ckpt = _ckpt_path(out)
    if ckpt.exists():
        data = json.loads(ckpt.read_text())
        done = {r["question_id"] for r in data["results"]}
        print(f"Resuming: {len(done)} done")
        return done, data["results"]
    return set(), []


def _save_ckpt(out: Path, results: list, cfg: dict):
    ckpt = _ckpt_path(out)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps({"config": cfg, "results": results}, indent=2))


def run_experiment(questions, budget, pricing, out_path, em_path, gat_path,
                   llm_preset, enable_ttas, n_workers=8, delay=0.8):
    done_ids, results = _load_ckpt(out_path)
    ckpt_lock = threading.Lock()

    batches = []
    for q in questions:
        if q.question_id in done_ids:
            continue
        batches.append({
            "llm_preset": llm_preset,
            "enable_ttas": enable_ttas,
            "gat_path": gat_path,
            "em_params_path": em_path,
            "budget": budget,
            "pricing": pricing,
            "question": {
                "question_id": q.question_id,
                "question": q.question,
                "correct": q.correct,
                "options": q.options,
                "domain": q.domain,
                "subdomain": q.subdomain,
            },
        })

    n_total = len(batches)
    est_min = n_total * 60 / max(n_workers, 1) / 60.0
    print(f"GPQA Adaptation: {'TTAS ON' if enable_ttas else 'TTAS OFF'}")
    print(f"Questions: {n_total}  Budget: {budget:.6f}  Workers: {n_workers}  Est: ~{est_min:.0f}m")

    completed = [0]
    t_start = time.time()

    cfg = {"enable_ttas": enable_ttas, "budget": budget, "pricing": pricing,
           "llm_preset": llm_preset, "n_questions": len(questions)}

    def _on_done(future):
        r = future.result()
        with ckpt_lock:
            results.append(r)
            done_ids.add(r["question_id"])
            completed[0] += 1
            mark = "Y" if r.get("correct") else ("E" if r.get("error") else ".")
            s_final = r.get("s_task_trajectory", [])
            s_last = s_final[-1]["s_after"] if s_final else 0
            elapsed = (time.time() - t_start) / 60.0
            print(f"  [{completed[0]:3d}/{n_total}] {mark} "
                  f"{r.get('domain','?'):10s} "
                  f"pred={str(r.get('predicted','?')):4s} "
                  f"gt={r.get('ground_truth','?'):4s} "
                  f"s_task={s_last:.3f}  [{elapsed:.0f}m]")
            _save_ckpt(out_path, results, cfg)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for i, b in enumerate(batches):
            if delay > 0 and i >= n_workers:
                time.sleep(delay)
            f = pool.submit(_worker, b)
            f.add_done_callback(_on_done)
            futures.append(f)
        for f in as_completed(futures):
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)
    ckpt = _ckpt_path(out_path)
    if ckpt.exists():
        ckpt.unlink()
    print(f"\nSaved -> {out_path}")

    # Quick summary
    n = len(results)
    acc = sum(r["correct"] for r in results) / n if n else 0
    avg_spent = sum(r["spent"] for r in results) / n if n else 0
    print(f"Acc: {acc:.1%}  AvgSpent: \${avg_spent:.6f}  n={n}")

    return results


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50,
                        help="Number of GPQA questions to sample")
    parser.add_argument("--out", default="results/gpqa_adaptation_n50.json")
    parser.add_argument("--gpqa", default="")
    parser.add_argument("--mode", default="tta_on", choices=["tta_on", "tta_off"],
                        help="tta_on = TTAS enabled, tta_off = TTAS frozen (control)")
    parser.add_argument("--budget", type=float, default=0.00058,
                        help="Budget in yuan (default: 0.00058 = 0.5B)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--llm", default="deepseek")
    parser.add_argument("--em-params", default="psbc_logs/em_params_parallel.json")
    parser.add_argument("--gat", default="psbc_logs/gat_parallel.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domains", default=None,
                        help="Comma-separated domains filter (e.g. Physics,Chemistry)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    use_llm(args.llm)
    pricing = get_pricing(args.llm)
    enable_ttas = (args.mode == "tta_on")
    print(f"LLM: {args.llm}  Pricing: p_in={pricing['p_in']:.2e}  p_out={pricing['p_out']:.2e}")
    print(f"TTAS: {'ON' if enable_ttas else 'OFF'}  Budget: {args.budget:.6f}")

    domains = args.domains.split(",") if args.domains else None
    gpqa_path = args.gpqa or os.environ.get("GPQA_DATA", "./data/GPQA/gpqa_diamond.csv")
    all_qs = load_gpqa_questions(gpqa_path, domains=domains)
    questions = sample_questions(all_qs, n=args.n, seed=args.seed)
    questions = interleave_by_domain(questions)
    print(f"Loaded {len(all_qs)} GPQA questions -> sampled {len(questions)} (interleaved)")
    # Show domain distribution
    from collections import Counter
    dist = Counter(q.domain for q in questions)
    print(f"Domains: {dict(dist)}")

    gat_path = str(Path(__file__).parent / args.gat)

    run_experiment(
        questions=questions,
        budget=args.budget,
        pricing=pricing,
        out_path=Path(args.out),
        em_path=args.em_params,
        gat_path=gat_path,
        llm_preset=args.llm,
        enable_ttas=enable_ttas,
        n_workers=args.workers,
        delay=args.delay,
    )
