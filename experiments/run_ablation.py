"""
run_ablation.py — NGO-BC Ablation Study for NeurIPS 2026.

5-node Parallel MetaGPT MAS:
  Decomposer → [Solver1, Solver2, Solver3] → Aggregator

Five ablation variants (all at $0.5B budget):
  1. full        — NGO-BC complete: parallel_product_kkt + GAT + TTAS + MPC
  2. no_gat      — w/o GAT-Lookahead: EMA heuristic predictor
  3. no_ttas     — w/o TTAS: freeze s_task=1.0
  4. no_mpc      — w/o MPC rolling: one-shot KKT at first node
  5. no_pooling  — w/o Structural Pooling: plain product_kkt (no pooling)

Baselines: Fixed, Greedy (for reference).

Run:
  python run_ablation.py --variant full --n 50 --out results/ablation_full_n50.json --workers 8
  python run_ablation.py --variant no_gat --n 50 --out results/ablation_no_gat_n50.json --workers 8
  python run_ablation.py --variant no_ttas --n 50 --out results/ablation_no_ttas_n50.json --workers 8
  python run_ablation.py --variant no_mpc --n 50 --out results/ablation_no_mpc_n50.json --workers 8
  python run_ablation.py --variant no_pooling --n 50 --out results/ablation_no_pooling_n50.json --workers 8
"""

from __future__ import annotations

import sys, json, time, argparse, threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
from ngobc.baselines import FixedAllocator, GreedyAllocator
from ngobc.gat_oracle import DagGatOracle
from ngobc.parallel_product_kkt import parallel_product_kkt_solve
from ngobc.product_kkt import product_kkt_solve
from ngobc.math_dataset import (
    load_math_problems, sample_problems,
    extract_answer_from_response, answers_match,
)


# ═══════════════════════════════════════════════════════
# 5-node Parallel MAS for MATH
# ═══════════════════════════════════════════════════════

class Decompose(Action):
    name: str = "Decompose"
    async def run(self, history, *args, **kwargs) -> str:
        msgs = history if isinstance(history, list) else []
        problem = msgs[-1].content if msgs else ""
        return await self._aask(
            f"Analyse the following math problem. Identify key concepts, "
            f"what is being asked, and outline a solution strategy. "
            f"Do NOT compute the final answer yet.\n\n{problem}"
        )

_SOLVER_PROMPTS = [
    "Solve step by step using algebraic methods. Put answer in \\boxed{{}}.",
    "Solve using a different approach (geometric, case analysis, etc). Put answer in \\boxed{{}}.",
    "Solve step by step. Verify each step. Put answer in \\boxed{{}}.",
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
            f"Three solvers produced the following solutions:\n\n{solutions}\n\n"
            f"Compare the solutions. If they agree, confirm the answer in \\boxed{{}}. "
            f"If they disagree, determine the correct answer. "
            f"Put the final answer inside \\boxed{{}}."
        )

class Decomposer(Role):
    name: str = "Decomposer"; profile: str = "Decomposer"
    actions: list = [Decompose]
    def __init__(self, **kw):
        super().__init__(**kw)
        from metagpt.actions.add_requirement import UserRequirement
        self._watch([UserRequirement])

class Solver1(Role):
    name: str = "Solver1"; profile: str = "Solver1"
    actions: list = [Solve1]
    def __init__(self, **kw):
        super().__init__(**kw)
        self._watch([Decompose])

class Solver2(Role):
    name: str = "Solver2"; profile: str = "Solver2"
    actions: list = [Solve2]
    def __init__(self, **kw):
        super().__init__(**kw)
        self._watch([Decompose])

class Solver3(Role):
    name: str = "Solver3"; profile: str = "Solver3"
    actions: list = [Solve3]
    def __init__(self, **kw):
        super().__init__(**kw)
        self._watch([Decompose])

class Aggregator(Role):
    name: str = "Aggregator"; profile: str = "Aggregator"
    actions: list = [Aggregate]
    def __init__(self, **kw):
        super().__init__(**kw)
        self._watch([Solve1, Solve2, Solve3])

def build_team():
    team = Team(context=Context(config=global_config), use_mgx=False)
    team.hire([Decomposer(), Solver1(), Solver2(), Solver3(), Aggregator()])
    return team


# ── Serialized topology (for no_pooling ablation) ──────────────────────────
# Hard-flatten parallel branches: D → S1 → S2 → S3 → A
# Each solver watches the PREVIOUS solver, not Decomposer directly.

class Solver2Serial(Role):
    name: str = "Solver2"; profile: str = "Solver2"
    actions: list = [Solve2]
    def __init__(self, **kw):
        super().__init__(**kw)
        self._watch([Solve1])           # watches S1, NOT Decomposer

class Solver3Serial(Role):
    name: str = "Solver3"; profile: str = "Solver3"
    actions: list = [Solve3]
    def __init__(self, **kw):
        super().__init__(**kw)
        self._watch([Solve2])           # watches S2, NOT Decomposer

class AggregatorSerial(Role):
    name: str = "Aggregator"; profile: str = "Aggregator"
    actions: list = [Aggregate]
    def __init__(self, **kw):
        super().__init__(**kw)
        self._watch([Solve3])           # watches only S3 (last in chain)

# Serial adjacency: SOURCE → D → S1 → S2 → S3 → A
ADJ_SERIAL = {
    "node_source": ["Decomposer"],
    "Decomposer": ["Solver1"],
    "Solver1": ["Solver2"],
    "Solver2": ["Solver3"],
    "Solver3": ["Aggregator"],
    "Aggregator": [],
}

def build_team_serial():
    team = Team(context=Context(config=global_config), use_mgx=False)
    team.hire([Decomposer(), Solver1(), Solver2Serial(), Solver3Serial(), AggregatorSerial()])
    return team


# ═══════════════════════════════════════════════════════
# Executor
# ═══════════════════════════════════════════════════════

def _get_executor(problem_text, task_id):
    team = build_team()
    runtime = PSBCRuntime()
    adapter = MetaGPTPSBCAdapter(context=team.env.context, runtime=runtime)
    adapter.bind_team(team)
    runtime.extras["send_to"] = "Decomposer"
    return adapter.task_manager().start_controlled_execution(
        task_id=task_id, task_prompt=problem_text)


def _get_executor_serial(problem_text, task_id):
    """Serialized topology: D → S1 → S2 → S3 → A (hard-flatten parallel)."""
    team = build_team_serial()
    runtime = PSBCRuntime()
    adapter = MetaGPTPSBCAdapter(context=team.env.context, runtime=runtime)
    adapter.bind_team(team)
    runtime.extras["send_to"] = "Decomposer"
    return adapter.task_manager().start_controlled_execution(
        task_id=task_id, task_prompt=problem_text)


# ═══════════════════════════════════════════════════════
# Variant worker functions
# ═══════════════════════════════════════════════════════

def _load_calib_params(em_params_path: str):
    em_json = Path(__file__).parent / em_params_path
    if not em_json.exists():
        return None
    raw = json.loads(em_json.read_text())
    calib = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)):
            calib[k] = tuple(v)
        else:
            calib[k] = (v["sigma"], v["kappa"], v["tau_quality"])
    return calib


def _build_node_params(topo, calib_params, weights):
    node_params = {}
    for nid in topo:
        sigma, kappa, tau = 0.9, 0.01, 0.0
        if calib_params:
            for role, (s, k, t) in calib_params.items():
                if role.lower() in nid.lower():
                    sigma, kappa, tau = s, k, t
                    break
        node_params[nid] = NodeParams(nid, sigma, kappa, tau, weights.get(nid, 0.1))
    return node_params


def _run_full(problem_text, budget, pricing, task_id, calib_params, oracle):
    """Variant 1: Full NGO-BC (parallel_product_kkt + GAT + TTAS + MPC)."""
    executor = _get_executor(problem_text, task_id)
    topo = executor.get_initial_topology()
    w = topo_weights(topo)
    node_params = _build_node_params(topo, calib_params, w)

    opt = PSBCOptimizer(
        adj=dict(topo), node_params=node_params,
        budget=budget, p_in=pricing["p_in"], p_out=pricing["p_out"],
        oracle=oracle, use_product=False,
    )
    per_node_tokens = {}
    per_node_debug = []

    for agent_id in executor.get_topological_iterator():
        meta = executor.get_agent_metadata(agent_id)
        exact_in = executor.get_exact_input_tokens(agent_id)
        # Extend topology for newly discovered nodes
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

        # Compute reserves and R_free (reuse optimizer's logic)
        descendants = list(get_descendants(opt.adj, agent_id))
        reserves = opt._compute_reserves(agent_id, descendants)
        R_prime = opt.R - pricing["p_in"] * exact_in
        R_free = R_prime - sum(reserves.values())
        if R_free <= 0 and descendants:
            reserves = opt._compute_reserves(agent_id, descendants, effective_gamma=0.0)
            R_free = R_prime - sum(reserves.values())

        # Solve with parallel product KKT
        result = parallel_product_kkt_solve(opt.adj, node_params, R_free, pricing["p_out"])
        max_tok = int(result.allocations.get(agent_id, node_params[agent_id].tau))
        max_tok = max(max_tok, 1)

        actual = executor.execute_agent(agent_id, max_tok, True)
        opt.settle(agent_id, actual)
        per_node_tokens[agent_id] = actual
        # Record full KKT plan: allocations for ALL remaining nodes at this step
        kkt_plan = {nid.split("_")[2][:8]: int(v) for nid, v in result.allocations.items()}
        per_node_debug.append({
            "agent_id": agent_id, "max_tok": max_tok,
            "exact_in": exact_in, "actual_out": actual,
            "kkt_plan": kkt_plan, "R_free": R_free,
        })
        executor.get_newly_discovered_topology()

    summ = opt.summary()
    answer = executor.get_final_result()
    return {
        "answer": answer, "spent": summ.total_spent,
        "budget_ok": summ.budget_ok,
        "nodes_exec": summ.nodes_executed,
        "nodes_skip": summ.nodes_skipped,
        "per_node_tok": per_node_tokens,
        "per_node_debug": per_node_debug,
    }


def _run_no_gat(problem_text, budget, pricing, task_id, calib_params):
    """Variant 2: w/o GAT-Lookahead — EMA heuristic predictor."""
    return _run_full(problem_text, budget, pricing, task_id, calib_params, oracle=None)


def _run_no_ttas(problem_text, budget, pricing, task_id, calib_params, gat_path):
    """Variant 3: w/o TTAS — freeze s_task=1.0."""
    oracle = DagGatOracle.load(gat_path, enable_ttas=False) if Path(gat_path).exists() else None
    return _run_full(problem_text, budget, pricing, task_id, calib_params, oracle)


def _run_no_mpc(problem_text, budget, pricing, task_id, calib_params, gat_path):
    """Variant 4: w/o MPC rolling — one-shot KKT at first node."""
    executor = _get_executor(problem_text, task_id)
    topo = executor.get_initial_topology()
    w = topo_weights(topo)
    node_params = _build_node_params(topo, calib_params, w)

    oracle = DagGatOracle.load(gat_path) if Path(gat_path).exists() else None

    from ngobc.dag import get_descendants

    opt = PSBCOptimizer(
        adj=dict(topo), node_params=node_params,
        budget=budget, p_in=pricing["p_in"], p_out=pricing["p_out"],
        oracle=oracle, use_product=False, one_shot=True,
    )
    per_node_tokens = {}
    per_node_debug = []
    cached = None  # one-shot: store KKT result on first node

    for agent_id in executor.get_topological_iterator():
        meta = executor.get_agent_metadata(agent_id)
        exact_in = executor.get_exact_input_tokens(agent_id)

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

        # One-shot: compute KKT once at first node, reuse cached result
        if cached is None:
            descendants = list(get_descendants(opt.adj, agent_id))
            reserves = opt._compute_reserves(agent_id, descendants)
            R_prime = opt.R - pricing["p_in"] * exact_in
            R_free = R_prime - sum(reserves.values())
            if R_free <= 0 and descendants:
                reserves = opt._compute_reserves(agent_id, descendants, effective_gamma=0.0)
                R_free = R_prime - sum(reserves.values())

            result = parallel_product_kkt_solve(opt.adj, node_params, R_free, pricing["p_out"])
            cached = dict(result.allocations)
            max_tok = int(cached.get(agent_id, node_params[agent_id].tau))
        else:
            # Recompute R (subtract input cost) but use cached allocation
            opt.R = max(opt.R - pricing["p_in"] * exact_in, 0.0)
            opt._state[agent_id].exact_in = exact_in
            max_tok = int(cached.get(agent_id, node_params[agent_id].tau))

        max_tok = max(max_tok, 1)
        # Safety: cap to available budget
        available_out = max(opt.R / pricing["p_out"], 1)
        max_tok = min(max_tok, int(available_out))

        actual = executor.execute_agent(agent_id, max_tok, True)
        opt.settle(agent_id, actual)
        per_node_tokens[agent_id] = actual
        per_node_debug.append({
            "agent_id": agent_id, "max_tok": max_tok,
            "exact_in": exact_in, "actual_out": actual,
            "one_shot": cached is not None,
        })
        executor.get_newly_discovered_topology()

    summ = opt.summary()
    answer = executor.get_final_result()
    return {
        "answer": answer, "spent": summ.total_spent,
        "budget_ok": summ.budget_ok,
        "nodes_exec": summ.nodes_executed,
        "nodes_skip": summ.nodes_skipped,
        "per_node_tok": per_node_tokens,
        "per_node_debug": per_node_debug,
    }


def _run_no_pooling(problem_text, budget, pricing, task_id, calib_params, oracle):
    """Variant 5: w/o Structural Pooling — hard-flatten parallel to serial chain.

    Original parallel:  D → [S1, S2, S3] → A   (3 solvers in parallel)
    Flattened to serial: D → S1 → S2 → S3 → A   (each solver watches the previous)

    This tests what happens when parallel branches are naively serialized
    instead of using structural pooling. The serial chain causes:
      - S2's input = S1's output (filtered through S1, loses D's direct analysis)
      - S3's input = S2's output (doubly filtered)
      - Early solvers consume budget, later ones get starved
    """
    executor = _get_executor_serial(problem_text, task_id)
    # Use the serial adjacency — standard MPC runs on this chain
    topo_serial = dict(ADJ_SERIAL)
    w = topo_weights(topo_serial)
    node_params = _build_node_params(topo_serial, calib_params, w)

    opt = PSBCOptimizer(
        adj=topo_serial, node_params=node_params,
        budget=budget, p_in=pricing["p_in"], p_out=pricing["p_out"],
        oracle=oracle, use_product=False,
    )
    per_node_tokens = {}
    per_node_debug = []

    for agent_id in executor.get_topological_iterator():
        meta = executor.get_agent_metadata(agent_id)
        exact_in = executor.get_exact_input_tokens(agent_id)

        # Extend topology for newly discovered nodes
        current_topo = executor.get_initial_topology()
        added = [x for x in current_topo if x not in node_params]
        if added:
            for x in added:
                np_new = NodeParams(x, 0.9, 0.01, 0.0, w.get(x, 0.1))
                node_params[x] = np_new
                s = _NodeState(params=np_new)
                s.b_prev = opt.B_t / max(len(node_params), 1) / opt.p_out
                opt._state[x] = s
            opt._crit_nodes = critical_path_nodes(opt.adj)
        node_params[agent_id].tau = max(node_params[agent_id].tau, float(meta.tau))

        # Standard MPC: compute reserves, solve KKT, execute
        descendants = list(get_descendants(opt.adj, agent_id))
        reserves = opt._compute_reserves(agent_id, descendants)
        R_prime = opt.R - pricing["p_in"] * exact_in
        R_free = R_prime - sum(reserves.values())
        if R_free <= 0 and descendants:
            reserves = opt._compute_reserves(agent_id, descendants, effective_gamma=0.0)
            R_free = R_prime - sum(reserves.values())

        opt_nodes = [node_params[agent_id]] + [
            node_params[d] for d in descendants if d in node_params
        ]
        result = product_kkt_solve(opt_nodes, R_free, pricing["p_out"])
        max_tok = int(result.allocations.get(agent_id, node_params[agent_id].tau))
        max_tok = max(max_tok, 1)

        actual = executor.execute_agent(agent_id, max_tok, True)
        opt.settle(agent_id, actual)
        per_node_tokens[agent_id] = actual
        per_node_debug.append({
            "agent_id": agent_id, "max_tok": max_tok,
            "exact_in": exact_in, "actual_out": actual,
        })
        executor.get_newly_discovered_topology()

    summ = opt.summary()
    answer = executor.get_final_result()
    return {
        "answer": answer, "spent": summ.total_spent,
        "budget_ok": summ.budget_ok,
        "nodes_exec": summ.nodes_executed,
        "nodes_skip": summ.nodes_skipped,
        "per_node_tok": per_node_tokens,
        "per_node_debug": per_node_debug,
    }


def _run_baseline(problem_text, budget, pricing, task_id, method):
    executor = _get_executor(problem_text, task_id)
    n_nodes = 5
    if method == "fixed":
        alloc = FixedAllocator(n_nodes=n_nodes, budget=budget,
                               p_in=pricing["p_in"], p_out=pricing["p_out"])
    else:
        alloc = GreedyAllocator(budget=budget,
                                p_in=pricing["p_in"], p_out=pricing["p_out"])
    per_node_tokens = {}
    per_node_max = {}  # allocated max_tokens (not just actual_out)
    for agent_id in executor.get_topological_iterator():
        exact_in = executor.get_exact_input_tokens(agent_id)
        decision, max_tok = alloc.decide(agent_id, exact_in)
        if decision == "SKIP":
            executor.skip_agent(agent_id)
            alloc.settle_skip(agent_id)
            per_node_tokens[agent_id] = 0
            per_node_max[agent_id] = 0
        else:
            actual = executor.execute_agent(agent_id, max_tok, False)
            alloc.settle(agent_id, actual)
            per_node_tokens[agent_id] = actual
            per_node_max[agent_id] = max_tok
        executor.get_newly_discovered_topology()
    summ = alloc.summary()
    answer = executor.get_final_result()
    return {
        "answer": answer, "spent": summ.spent,
        "budget_ok": summ.budget_ok,
        "nodes_exec": summ.nodes_executed,
        "nodes_skip": summ.nodes_skipped,
        "per_node_tok": per_node_tokens,
        "per_node_max_tok": per_node_max,
    }


# ═══════════════════════════════════════════════════════
# Worker (subprocess)
# ═══════════════════════════════════════════════════════

def _worker_batched(args: dict) -> list:
    import sys, json, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "MAS_inter"))
    sys.path.insert(0, str(Path(__file__).parent))

    variant     = args["variant"]
    budget      = args["budget"]
    prob_text   = args["prob_text"]
    prob_answer = args["prob_answer"]
    prob_id     = args["prob_id"]
    prob_subject = args["prob_subject"]
    prob_level  = args["prob_level"]
    pricing     = args["pricing"]
    llm_preset  = args["llm_preset"]
    em_params_path = args["em_params_path"]
    gat_path    = args["gat_path"]
    max_retries = 5

    from metagpt.ext.psbc.llm_preset import use_llm
    use_llm(llm_preset)

    from ngobc.gat_oracle import DagGatOracle
    from ngobc.math_dataset import extract_answer_from_response, answers_match

    calib_params = _load_calib_params(em_params_path)

    # Load oracle for variants that need it
    oracle = None
    if variant in ("full", "no_pooling"):
        oracle = DagGatOracle.load(gat_path) if Path(gat_path).exists() else None
    elif variant == "no_ttas":
        oracle = DagGatOracle.load(gat_path, enable_ttas=False) if Path(gat_path).exists() else None
    elif variant == "no_mpc":
        oracle = DagGatOracle.load(gat_path) if Path(gat_path).exists() else None

    method_names = [variant, "Fixed", "Greedy"]

    def _run_one(method: str) -> dict:
        tid = f"{method}_{prob_id}"
        for attempt in range(max_retries):
            try:
                if method == "full":
                    r = _run_full(prob_text, budget, pricing, tid, calib_params, oracle)
                elif method == "no_gat":
                    r = _run_no_gat(prob_text, budget, pricing, tid, calib_params)
                elif method == "no_ttas":
                    r = _run_no_ttas(prob_text, budget, pricing, tid, calib_params, gat_path)
                elif method == "no_mpc":
                    r = _run_no_mpc(prob_text, budget, pricing, tid, calib_params, gat_path)
                elif method == "no_pooling":
                    r = _run_no_pooling(prob_text, budget, pricing, tid, calib_params, oracle)
                elif method == "Fixed":
                    r = _run_baseline(prob_text, budget, pricing, tid, "fixed")
                else:  # Greedy
                    r = _run_baseline(prob_text, budget, pricing, tid, "greedy")

                predicted = extract_answer_from_response(r["answer"])
                correct = answers_match(predicted or "", prob_answer)
                r.update({
                    "task_id": tid, "method": method,
                    "variant": variant, "budget": budget,
                    "problem_id": prob_id, "subject": prob_subject,
                    "level": prob_level, "ground_truth": prob_answer,
                    "predicted": predicted, "correct": correct,
                })
                return r
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    time.sleep(3 * (2 ** attempt))
                    continue
                return {
                    "task_id": tid, "method": method,
                    "variant": variant, "budget": budget,
                    "problem_id": prob_id, "subject": prob_subject,
                    "level": prob_level, "ground_truth": prob_answer,
                    "predicted": None, "correct": False,
                    "spent": 0.0, "budget_ok": True,
                    "nodes_exec": 0, "nodes_skip": 0,
                    "per_node_tok": {}, "answer": "", "error": err[:200],
                }

    from concurrent.futures import ThreadPoolExecutor, as_completed as thread_as_completed
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_run_one, m): m for m in method_names}
        results = [f.result() for f in thread_as_completed(futures)]
    return results


# ═══════════════════════════════════════════════════════
# Checkpoint & driver
# ═══════════════════════════════════════════════════════

def _ckpt_path(out_path: Path) -> Path:
    return out_path.with_suffix(".ckpt.json")


def _load_checkpoint(out_path: Path):
    ckpt = _ckpt_path(out_path)
    if ckpt.exists():
        data = json.loads(ckpt.read_text())
        done = {r["task_id"] for r in data["results"]}
        print(f"Resuming: {len(done)} done")
        return done, data["results"]
    return set(), []


def _save_checkpoint(out_path, results, pricing, budget, variant):
    ckpt = _ckpt_path(out_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps({
        "variant": variant, "pricing": pricing, "budget": budget,
        "results": results,
    }, indent=2))


def run_experiment(problems, budget, pricing, variant, out_path,
                   em_path, gat_path, llm_preset,
                   n_workers=8, delay=0.8, debug=False):
    method_names = [variant, "Fixed", "Greedy"]
    done_ids, results = _load_checkpoint(out_path)
    ckpt_lock = threading.Lock()

    batches, n_total = [], 0
    for prob in problems:
        remaining = [m for m in method_names
                     if f"{m}_{prob.problem_id}" not in done_ids]
        if not remaining:
            continue
        batches.append({
            "variant": variant,
            "budget": budget,
            "prob_text": prob.problem,
            "prob_answer": prob.answer,
            "prob_id": prob.problem_id,
            "prob_subject": prob.subject,
            "prob_level": prob.level,
            "pricing": pricing,
            "llm_preset": llm_preset,
            "em_params_path": em_path,
            "gat_path": gat_path,
        })
        n_total += len(remaining)

    n_batches = len(batches)
    est_min = n_batches * 45 / max(n_workers, 1) / 60.0
    print(f"Variant: {variant}  Budget: {budget:.6f}")
    print(f"Batches: {n_batches}  runs: {n_total}  workers: {n_workers}  est: ~{est_min:.0f}m")

    completed = [0]
    t_start = time.time()

    def _on_done(future, _):
        batch_results = future.result()
        with ckpt_lock:
            for r in batch_results:
                results.append(r)
                done_ids.add(r["task_id"])
                completed[0] += 1
                mark = "Y" if r.get("correct") else ("E" if r.get("error") else ".")
                elapsed = (time.time() - t_start) / 60.0
                print(f"  [{completed[0]:3d}/{n_total}] {mark} "
                      f"{r['method']:12s} | "
                      f"{r['subject'][:8]:8s} {r['level']} | "
                      f"pred={str(r.get('predicted','?'))[:6]:6s} "
                      f"gt={r['ground_truth'][:6]:6s}  [{elapsed:.0f}m]")
            _save_checkpoint(out_path, results, pricing, budget, variant)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for b in batches:
            f = pool.submit(_worker_batched, b)
            f.add_done_callback(lambda fut, ba=b: _on_done(fut, ba))
            futures.append(f)
            if delay > 0:
                time.sleep(delay)
        for f in as_completed(futures):
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip_keys = [] if debug else ["per_node_tok", "per_node_debug"]
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant, "pricing": pricing, "budget": budget,
            "results": [{k: v for k, v in r.items() if k not in strip_keys}
                        for r in results],
        }, f, indent=2)
    ckpt = _ckpt_path(out_path)
    if ckpt.exists():
        ckpt.unlink()
    print(f"\nSaved -> {out_path}")
    return results


def print_summary(results, budget, variant):
    methods = [variant, "Fixed", "Greedy"]
    print(f"\n{'='*70}")
    print(f"Ablation: {variant}  |  Budget: {budget:.6f}")
    print(f"{'='*70}")
    print(f"{'Method':15s}  {'Acc':>6s}  {'AvgSpent':>10s}  {'BudgetOK':>8s}")
    print("-" * 50)
    for m in methods:
        ss = [r for r in results if r["method"] == m]
        if not ss:
            continue
        acc = sum(r["correct"] for r in ss) / len(ss)
        avg_spent = sum(r["spent"] for r in ss) / len(ss)
        budget_ok_rate = sum(r["budget_ok"] for r in ss) / len(ss)
        print(f"{m:15s}  {acc:6.1%}  {avg_spent:10.6f}  {budget_ok_rate:8.1%}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["full", "no_gat", "no_ttas", "no_mpc", "no_pooling"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", default="results/ablation_full_n50.json")
    parser.add_argument("--data", default="")
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--levels", default="3,4,5")
    parser.add_argument("--em-params", default="psbc_logs/em_params_parallel.json")
    parser.add_argument("--gat", default="psbc_logs/gat_parallel.pt")
    parser.add_argument("--llm", default="deepseek",
                        choices=["deepseek", "doubao", "kimi"])
    parser.add_argument("--budget", type=float, default=None,
                        help="Override budget (yuan). Default: 0.00058 (0.5B for parallel)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    use_llm(args.llm)
    pricing = get_pricing(args.llm)
    print(f"LLM: {args.llm}  Pricing: p_in={pricing['p_in']:.2e}  p_out={pricing['p_out']:.2e}")

    # Default 0.5B budget for parallel topology (calibrated for DeepSeek pricing)
    if args.budget is not None:
        BUDGET = args.budget
    else:
        BUDGET = 0.00058  # ~0.5B for parallel MetaGPT
    print(f"Budget: {BUDGET:.6f}")

    data_root = args.data or os.environ.get("MATH_DATA", "./data/math")
    levels = {f"Level {l}" for l in args.levels.split(",")}
    all_problems = load_math_problems(data_root, levels=levels)
    problems = sample_problems(all_problems, n=args.n, seed=42, stratified=True)
    print(f"Loaded {len(all_problems)} -> sampled {len(problems)}")

    gat_path = str(Path(__file__).parent / args.gat)

    results = run_experiment(
        problems, BUDGET, pricing, args.variant,
        out_path=Path(args.out),
        em_path=args.em_params,
        gat_path=gat_path,
        llm_preset=args.llm,
        n_workers=args.workers,
        delay=args.delay,
        debug=args.debug,
    )
    print_summary(results, BUDGET, args.variant)
