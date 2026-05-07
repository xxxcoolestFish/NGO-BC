"""
psbc/gat_trainer.py – Dataset loader and training loop for DAG-GAT.

Loads JSONL observation logs produced by collect_logs.py and trains
the DAG-GAT model to predict E_in for each node.

Key design choices:
  - Leave-one-out style: for each node j in a record, mask out j and
    all its descendants to simulate the "not-yet-executed" prediction scenario.
  - The SOURCE node (E_in=0) is always excluded from the prediction target.
  - Train/val split: 80/20 on records (not nodes).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ngobc.dag_gat import DAGGAT, FeatureBuilder, GraphData, KNOWN_ROLES


# ═══════════════════════════════════════════════════════════════════════════
# JSONL → GraphData conversion
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_ID = "node_source"

def _role_from_id(node_id: str) -> str:
    """Infer role from node_id string (e.g. node_1_Analyst_Analyse → analyst)."""
    parts = node_id.split("_")
    if len(parts) >= 3:
        return parts[2].lower()
    return "unknown"


def record_to_graphs(
    record: dict,
    fb:     FeatureBuilder,
    # default quality-function params (will be replaced by EM in Phase D3)
    default_sigma: float = 0.9,
    default_kappa: float = 0.01,
) -> List[Tuple[GraphData, torch.Tensor]]:
    """
    Convert one JSONL record into a list of (GraphData, target_E_in) pairs.

    For a record with n non-source nodes, we generate n training samples:
    sample k = predict node k's E_in given nodes 0..k-1 already executed.

    Returns list of (graph, target) where target is the E_in of the node
    being predicted (scalar tensor).
    """
    topo   = record["topology"]
    execs  = {ex["node_id"]: ex for ex in record["executions"]}

    # Build adjacency: parent → children
    all_nodes: List[str] = [n for n in topo["nodes"] if n != SOURCE_ID]
    edges: Dict[str, List[str]] = {n: [] for n in topo["nodes"]}
    for u, v in topo["edges"]:
        edges.setdefault(u, []).append(v)

    # Topological order (exclude SOURCE)
    exec_order = [ex["node_id"] for ex in record["executions"]]

    samples = []
    for pred_idx, pred_node in enumerate(exec_order):
        if pred_node not in execs:
            continue
        target_E_in = float(execs[pred_node]["E_in"])

        # Build node list: source + all nodes up to and including pred_node
        visible = [SOURCE_ID] + exec_order[: pred_idx + 1]
        node_ids = visible

        # Build node feature dicts
        node_dicts = []
        for nid in node_ids:
            if nid == SOURCE_ID:
                nd = {
                    "sigma": default_sigma, "kappa": default_kappa, "tau": 0.0,
                    "C_sys": 0, "C_struct": 0, "role": "source",
                    "executed": 1, "output_len": 0.0,
                }
            else:
                ex = execs.get(nid, {})
                is_executed = (nid != pred_node)
                nd = {
                    "sigma":      default_sigma,
                    "kappa":      default_kappa,
                    "tau":        float(ex.get("C_sys", 0) + ex.get("C_struct", 0)),
                    "C_sys":      float(ex.get("C_sys", 0)),
                    "C_struct":   float(ex.get("C_struct", 0)),
                    "role":       ex.get("role", _role_from_id(nid)),
                    "executed":   float(is_executed),
                    "output_len": float(ex.get("c_out", 0)) if is_executed else 0.0,
                }
            node_dicts.append(nd)

        # Build feature tensor (base only, role embedding applied in forward)
        x_base, role_idxs = fb.build(node_dicts)

        # Build parent index map
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        parents: Dict[int, List[int]] = {i: [] for i in range(len(node_ids))}
        for u, vs in edges.items():
            if u not in id_to_idx:
                continue
            for v in vs:
                if v in id_to_idx:
                    parents[id_to_idx[v]].append(id_to_idx[u])

        g           = GraphData(node_ids=node_ids, x_base=x_base,
                                role_idxs=role_idxs, parents=parents)
        g.pred_idx  = len(node_ids) - 1
        target      = torch.tensor(target_E_in, dtype=torch.float32)

        samples.append((g, target))

    return samples


def load_dataset(
    jsonl_path: Path,
    val_ratio:  float = 0.2,
    seed:       int   = 42,
) -> Tuple[List, List, FeatureBuilder]:
    """
    Load all records, convert to (GraphData, target) samples, split train/val.

    Returns (train_samples, val_samples, feature_builder).
    """
    records = [json.loads(l) for l in open(jsonl_path) if l.strip()]

    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_ratio))
    val_records   = records[:n_val]
    train_records = records[n_val:]

    fb = FeatureBuilder(role_emb_dim=8)
    # Remove the embedding from FeatureBuilder – it lives in DAGGAT now
    del fb.role_emb

    train_samples = []
    for r in train_records:
        train_samples.extend(record_to_graphs(r, fb))

    val_samples = []
    for r in val_records:
        val_samples.extend(record_to_graphs(r, fb))

    return train_samples, val_samples, fb


# ══════════════════════════════��════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(
    model:        DAGGAT,
    fb:           FeatureBuilder,
    train_data:   List,
    val_data:     List,
    n_epochs:     int   = 100,
    lr:           float = 1e-3,
    batch_size:   int   = 16,
    patience:     int   = 20,
    seed:         int   = 42,
    verbose:      bool  = True,
) -> Dict[str, List[float]]:
    """
    Train DAG-GAT with MSE loss on the mean head.
    Early stopping tracks val_mae (not val_nll) for stability on small data.
    """
    torch.manual_seed(seed)
    params = list(model.parameters())
    opt    = Adam(params, lr=lr, weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    history    = {"train_mse": [], "val_mae": [], "val_nll": []}
    best_mae   = float("inf")
    wait       = 0
    best_state = None
    rng        = random.Random(seed)

    for epoch in range(1, n_epochs + 1):
        # ── Train (MSE on mean prediction) ───────────────────────────────
        model.train()
        rng.shuffle(train_data)
        train_loss = 0.0
        n_batches  = 0

        for start in range(0, len(train_data), batch_size):
            batch = train_data[start: start + batch_size]
            opt.zero_grad()
            losses = []

            for g, target in batch:
                mean, _, _ = model(g)
                pred_mean = mean[g.pred_idx]
                mse       = (target - pred_mean) ** 2
                losses.append(mse)

            loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            train_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg_train = train_loss / max(n_batches, 1)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_mae = 0.0
        val_nll = 0.0
        with torch.no_grad():
            for g, target in val_data:
                mean, std, _ = model(g)
                pm  = mean[g.pred_idx]
                ps  = std[g.pred_idx]
                val_mae += abs((pm - target).item())
                val_nll += ((target - pm) ** 2 / (2 * ps ** 2) + torch.log(ps)).item()

        avg_mae = val_mae / max(len(val_data), 1)
        avg_nll = val_nll / max(len(val_data), 1)

        history["train_mse"].append(avg_train)
        history["val_mae"].append(avg_mae)
        history["val_nll"].append(avg_nll)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:4d}  "
                  f"train_mse={avg_train:.1f}  "
                  f"val_mae={avg_mae:.1f} tokens  "
                  f"val_nll={avg_nll:.3f}")

        # Early stopping on val_mae
        if avg_mae < best_mae - 0.5:
            best_mae   = avg_mae
            wait       = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}  "
                          f"(best val_mae={best_mae:.1f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

