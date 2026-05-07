"""
Run per-role quality curve experiment: test each agent role independently
on MATH at varying max_tokens to measure accuracy vs token budget curve.

Roles: Decomposer, Solver, Verifier (from MetaGPT Pipeline)
Dataset: MATH Level 3-5
max_tokens levels: 30, 50, 80, 120, 200, 350, 600, 1000, 1600, 2500, 4000

Usage on server:
  python run_role_quality_curve.py --n 50 --workers 8 --out results/role_quality.json
"""

import sys, os, json, time, argparse, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ngobc.math_dataset import load_math_problems, sample_problems
from mas.llm import LLMClient

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Role prompts (each role solves independently) ───────────────────

ROLE_PROMPTS = {
    "Decomposer": (
        "You are a math problem analyst. Your task is to analyze the problem, "
        "identify key concepts, devise a solution strategy, and then solve the problem. "
        "Put your final answer inside \\boxed{{}}."
    ),
    "Solver": (
        "You are a math problem solver. Solve the following problem step by step. "
        "Show your reasoning clearly. "
        "Put your final answer inside \\boxed{{}}."
    ),
    "Verifier": (
        "You are a careful math verifier. Read the problem, solve it carefully, "
        "check each step for errors, and confirm the correct answer. "
        "Put your final answer inside \\boxed{{}}."
    ),
}

ROLES = ["Decomposer", "Solver", "Verifier"]

# max_tokens levels (widely spaced to capture curve shape)
MAX_TOKENS_LEVELS = [30, 50, 80, 120, 200, 350, 600, 1000, 1600, 2500, 4000]

def run_one(prob, role, max_tok, task_id):
    """Single evaluation: one role, one problem, one max_tokens level."""
    try:
        p_in = float(os.environ.get("LLM_P_IN", "1.1e-7"))
        p_out = float(os.environ.get("LLM_P_OUT", "2.9e-7"))
        llm = LLMClient(p_in=p_in, p_out=p_out)
        system = ROLE_PROMPTS[role]
        prompt = prob.problem.replace("{", "{{").replace("}", "}}")
        answer, c_out = llm.chat(system, prompt, max_tok)
        e_in = llm.count_messages(system, prompt)

        # Extract answer
        pred = None
        matches = re.findall(r"\\boxed\{([^}]+)\}", answer)
        if matches:
            pred = matches[-1].strip()

        from ngobc.math_dataset import answers_match
        correct = answers_match(pred or "", prob.answer)

        return {
            "task_id": task_id, "role": role, "max_tokens": max_tok,
            "problem_id": prob.problem_id, "subject": prob.subject,
            "level": prob.level, "correct": correct,
            "e_in": e_in, "c_out": c_out,
            "pred": pred or "", "gt": prob.answer,
        }
    except Exception as e:
        return {
            "task_id": task_id, "role": role, "max_tokens": max_tok,
            "problem_id": prob.problem_id, "error": str(e),
            "correct": False,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--math-data", default="")
    ap.add_argument("--out", default="results/role_quality.json")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    # Load problems
    data_root = args.math_data or os.environ.get("MATH_DATA", "./data/math")
    levels = {"Level 3", "Level 4", "Level 5"}
    probs = sample_problems(
        load_math_problems(data_root, levels=levels),
        n=args.n, seed=42, stratified=True,
    )
    print(f"Loaded {len(probs)} problems")

    # Build task list
    tasks = []
    for p in probs:
        for role in ROLES:
            for mt in MAX_TOKENS_LEVELS:
                tasks.append({
                    "prob": p, "role": role, "max_tok": mt,
                    "task_id": f"{role}_{p.problem_id}_{mt}",
                })

    out_path = Path(args.out)
    completed_ids = set()
    prior_results = []
    if args.resume and out_path.exists():
        try:
            prior = json.load(open(out_path))
            prior_results = prior.get("results", [])
            completed_ids = set(r["task_id"] for r in prior_results)
            print(f"Resume: {len(completed_ids)} already done")
        except Exception:
            pass

    pending = [t for t in tasks if t["task_id"] not in completed_ids]
    results = list(prior_results)
    print(f"Total: {len(tasks)} (n={args.n} × 3 roles × {len(MAX_TOKENS_LEVELS)} levels)")
    print(f"Pending: {len(pending)}, workers={args.workers}")

    if not pending:
        print("All done!")
        return

    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_every = max(len(pending) // 50, 1)
    last_saved = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, **t): t for t in pending}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            n_done = len([x for x in results
                         if x["task_id"] not in completed_ids])
            if n_done - last_saved >= save_every or n_done in (1, len(pending)):
                json.dump({"results": results,
                          "max_tokens_levels": MAX_TOKENS_LEVELS,
                          "roles": ROLES},
                         open(out_path, "w"), indent=2)
                last_saved = n_done
            if n_done % 20 == 0:
                ok = [x for x in results if "error" not in x]
                acc = sum(x["correct"] for x in ok) / max(len(ok), 1) if ok else 0
                elapsed = (time.time() - t0) / 60
                print(f"  [{n_done}/{len(pending)}] acc={acc:.1%} [{elapsed:.0f}m]",
                      flush=True)

    # Final save
    json.dump({"results": results,
              "max_tokens_levels": MAX_TOKENS_LEVELS,
              "roles": ROLES},
             open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")
    print(f"Total time: {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
