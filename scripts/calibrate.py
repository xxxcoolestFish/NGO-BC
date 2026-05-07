"""
scripts/calibrate.py — Phase 0: observation + EM calibration + GAT training.

Usage:
  # Step 1: Collect observation logs (no budget control)
  LLM_API_KEY=xxx LLM_BASE_URL=xxx LLM_MODEL=xxx \
  python scripts/calibrate.py --phase obs --n 30 --topology pipeline

  # Step 2: EM calibration from logs
  python scripts/calibrate.py --phase em --obs-dir logs/obs

  # Step 3: Train GAT-Lookahead
  python scripts/calibrate.py --phase train --obs-dir logs/obs
"""

import sys, os, json, time, argparse, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ngobc.em_calibration import calibrate_from_logs
from ngobc.dag_gat import DAGGAT, FeatureBuilder, GraphData, KNOWN_ROLES
from ngobc.math_dataset import load_math_problems, sample_problems
from mas import LLMClient, DAGExecutor
from mas.executor import make_pipeline_agents, make_parallel_agents


ROLE_MAP = {"Decomposer": "analyst", "Solver": "worker",
            "Verifier": "synthesiser", "Aggregator": "aggregator",
            "Affirmative": "worker", "Negative": "critic", "Moderator": "aggregator"}


def run_obs_pipeline(prob, task_id, llm):
    """Single-observation for pipeline topology."""
    agents = make_pipeline_agents()
    executor = DAGExecutor(agents, llm)
    entries = []

    for nid in executor.iter_topological():
        e_in = executor.get_input_tokens(nid, prob.problem)
        c_out = executor.execute(nid, prob.problem, 4096)
        entries.append({
            "node_id": nid, "role": agents[[s.node_id for s in agents].index(nid)].role,
            "C_sys": llm.count_tokens(agents[[s.node_id for s in agents].index(nid)].system_prompt),
            "C_struct": 0, "E_in": e_in, "c_out": c_out, "assigned_max": 4096,
        })

    return {"task_id": task_id, "problem_id": prob.problem_id,
            "round_id": 1, "is_final_round": True,
            "executions": entries}


def run_obs_parallel(prob, task_id, llm):
    """Single-observation for parallel topology."""
    agents = make_parallel_agents()
    executor = DAGExecutor(agents, llm)
    entries = []

    for nid in executor.iter_topological():
        e_in = executor.get_input_tokens(nid, prob.problem)
        c_out = executor.execute(nid, prob.problem, 4096)
        entries.append({
            "node_id": nid, "role": agents[[s.node_id for s in agents].index(nid)].role,
            "C_sys": llm.count_tokens(agents[[s.node_id for s in agents].index(nid)].system_prompt),
            "C_struct": 0, "E_in": e_in, "c_out": c_out, "assigned_max": 4096,
        })

    return {"task_id": task_id, "problem_id": prob.problem_id,
            "round_id": 1, "is_final_round": True,
            "executions": entries}


def build_gat_samples(entries, calib):
    """Build training samples for DAG-GAT."""
    samples = []
    for entry in entries:
        executions = entry.get("executions", [])
        if not executions:
            continue
        node_ids = [e["node_id"] for e in executions]

        nodes = []
        for e in executions:
            role = e.get("role", "unknown")
            cp = calib.get(role, {"sigma": 0.9, "kappa": 0.01, "tau": 50})
            nodes.append({
                "sigma": cp["sigma"], "kappa": cp["kappa"], "tau": cp["tau"],
                "C_sys": e.get("C_sys", 0), "C_struct": e.get("C_struct", 0),
                "role": ROLE_MAP.get(role, "unknown"),
                "executed": False, "output_len": e.get("c_out", 0),
                "E_in": e.get("E_in", 0),
            })

        parents = {}
        for i, nid in enumerate(node_ids):
            parents[i] = list(range(i)) if i > 0 else []

        for j in range(len(nodes) - 1):
            node_states = [dict(n) for n in nodes]
            for i in range(j + 1):
                node_states[i]["executed"] = True
            samples.append({
                "node_ids": list(node_ids), "nodes": node_states,
                "parents": dict(parents), "pred_idx": j + 1,
                "is_final": 1.0,
            })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="obs", choices=["obs", "em", "train"])
    ap.add_argument("--topology", default="pipeline",
                    choices=["pipeline", "parallel"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default="")
    ap.add_argument("--obs-dir", default="logs/obs")
    ap.add_argument("--out-em", default="logs/em_params.json")
    ap.add_argument("--out-gat", default="logs/gat.pt")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    p_in = float(os.environ.get("LLM_P_IN", "1.1e-7"))
    p_out = float(os.environ.get("LLM_P_OUT", "2.9e-7"))

    if args.phase == "obs":
        data_root = args.data or os.environ.get("MATH_DATA", "./data/math")
        probs = sample_problems(
            load_math_problems(data_root, levels={"Level 3", "Level 4", "Level 5"}),
            n=args.n, seed=42, stratified=True,
        )
        obs_fn = run_obs_pipeline if args.topology == "pipeline" else run_obs_parallel
        obs_dir = Path(args.obs_dir)
        obs_dir.mkdir(parents=True, exist_ok=True)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        llm = LLMClient(p_in=p_in, p_out=p_out)
        results = []; t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(obs_fn, p, f"obs_{i}", llm): i
                       for i, p in enumerate(probs)}
            for f in as_completed(futures):
                r = f.result(); results.append(r)
                with open(obs_dir / f"{r['task_id']}.jsonl", "w") as fout:
                    fout.write(json.dumps(r) + "\n")
                print(f"  [{len(results)}/{len(probs)}] [{(time.time()-t0)/60:.0f}m]",
                      flush=True)

        print(f"Saved {len(results)} obs files -> {obs_dir}")

    elif args.phase == "em":
        obs_dir = Path(args.obs_dir)
        all_entries = []
        for f in obs_dir.glob("*.jsonl"):
            for line in open(f):
                if line.strip():
                    all_entries.append(json.loads(line))

        # Write concatenated JSONL for calibrate_from_logs
        tmp = Path("/tmp/obs_all.jsonl")
        with open(tmp, "w") as f:
            for e in all_entries:
                f.write(json.dumps(e) + "\n")

        params = calibrate_from_logs(str(tmp))
        out = {role: {"sigma": s, "kappa": k, "tau": t}
               for role, (s, k, t) in params.items()}
        Path(args.out_em).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out_em, "w"), indent=2)
        print(f"EM params: {out}")
        print(f"Saved -> {args.out_em}")

    elif args.phase == "train":
        obs_dir = Path(args.obs_dir)
        em_path = Path(args.out_em)
        calib = json.load(open(em_path)) if em_path.exists() else {}

        all_entries = []
        for f in obs_dir.glob("*.jsonl"):
            for line in open(f):
                if line.strip():
                    all_entries.append(json.loads(line))

        samples = build_gat_samples(all_entries, calib)
        random.shuffle(samples)
        split = int(0.8 * len(samples))
        train_s, val_s = samples[:split], samples[split:]
        print(f"GAT samples: {len(samples)} (train={len(train_s)}, val={len(val_s)})")

        fb = FeatureBuilder(role_emb_dim=8)
        model = DAGGAT(feat_dim=fb.feat_dim, hidden_dim=64, n_layers=2, role_emb_dim=8)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        best_loss = float("inf")

        for epoch in range(args.epochs):
            model.train(); total_loss = 0.0; n_batches = 0
            for s in train_s:
                x_base, role_idxs = fb.build(s["nodes"])
                g = GraphData(s["node_ids"], x_base, role_idxs, s["parents"])
                mean, stddev, p_stop = model(g)
                ti = s["pred_idx"]
                target = torch.tensor([s["nodes"][ti]["E_in"]], dtype=torch.float32)
                mask = torch.zeros(len(s["nodes"]), dtype=torch.bool); mask[ti] = True
                loss = DAGGAT.combined_loss(mean, stddev, p_stop, target,
                                           s["is_final"], mask, stop_weight=0.1)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); n_batches += 1

            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for s in val_s:
                    x_base, role_idxs = fb.build(s["nodes"])
                    g = GraphData(s["node_ids"], x_base, role_idxs, s["parents"])
                    mean, stddev, p_stop = model(g)
                    ti = s["pred_idx"]
                    target = torch.tensor([s["nodes"][ti]["E_in"]], dtype=torch.float32)
                    mask = torch.zeros(len(s["nodes"]), dtype=torch.bool); mask[ti] = True
                    val_loss += DAGGAT.combined_loss(
                        mean, stddev, p_stop, target, s["is_final"], mask).item()

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"feat_dim": fb.feat_dim, "hidden_dim": 64, "n_layers": 2,
                            "role_emb_dim": 8, "model_state": model.state_dict()},
                           args.out_gat)
            if epoch % 50 == 0:
                print(f"  Epoch {epoch}: train={total_loss/max(n_batches,1):.4f}  "
                      f"val={val_loss/max(len(val_s),1):.4f}")

        print(f"Best val_loss={best_loss:.4f} -> {args.out_gat}")


if __name__ == "__main__":
    main()
