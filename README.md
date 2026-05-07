# NGO-BC: Neural-Guided Online Budget Control

**Neural-Guided Online Budget Control for Multi-Agent LLM Systems**

NGO-BC is a plug-in budget control layer for multi-agent LLM systems. It
sets `max_tokens` for each agent before LLM invocation **without modifying
agent internals, prompts, or interaction topology**.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Across-Round Layer (MWU Scheduling)        │
│  μ_k → B_actual^(k) → C_true^(k) → μ_{k+1} │
├─────────────────────────────────────────────┤
│  Within-Round Layer (MPC + Product KKT)     │
│  Step 1: Observe → 2: Predict → 3: KKT →   │
│  Step 4: Execute → 5: Update                │
├─────────────────────────────────────────────┤
│  Perception Layer (GAT-Lookahead)           │
│  Head 1/2: E_in prediction + uncertainty    │
│  Head 3: Topology prediction (dynamic)      │
│  Head 4: P_stop (multi-round horizon)       │
└─────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/xxx/NGO-BC.git
cd NGO-BC
pip install -r requirements.txt
```

### Environment Variables

Set your LLM API credentials:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"   # or any OpenAI-compatible endpoint
export LLM_MODEL="gpt-4"
export LLM_P_IN="1.0e-6"    # input token price (yuan or USD)
export LLM_P_OUT="2.0e-6"   # output token price
```

### Data Preparation

Download the MATH dataset (Hendrycks et al., 2021) and place under `data/math/`:

```bash
# The directory should contain HuggingFace datasets subdirectories:
#   data/math/algebra/
#   data/math/geometry/
#   ...
export MATH_DATA="./data/math"
```

### Run an Experiment

**Single-round Pipeline** (3-node serial: Decomposer → Solver → Verifier):
```bash
python experiments/run_pipeline.py --n 50 --workers 4 --budget all
```

**Single-round Parallel** (5-node parallel: Decomposer → [S1, S2, S3] → Aggregator):
```bash
python experiments/run_parallel.py --n 50 --workers 4
```

**Multi-round Debate** (3-node debate × 3 rounds, with cross-round MWU):
```bash
python experiments/run_mad.py --n 50 --workers 4
```

### Full Calibration Pipeline (Phase 0 → EM → GAT)

```bash
# Step 1: Collect observation logs (no budget control, ~30 problems)
python scripts/calibrate.py --phase obs --n 30 --topology pipeline

# Step 2: EM calibration of quality function parameters (σ, κ, τ)
python scripts/calibrate.py --phase em --obs-dir logs/obs

# Step 3: Train GAT-Lookahead for input token prediction
python scripts/calibrate.py --phase train --obs-dir logs/obs

# Step 4: Run budget-controlled experiment with trained GAT
python experiments/run_pipeline.py --n 50 --gat logs/gat.pt
```

## Code Structure

```
NGO-BC/
├── README.md
├── requirements.txt
├── ngobc/                        # Core algorithm
│   ├── __init__.py
│   ├── optimizer.py              # 5-step MPC main loop
│   ├── product_kkt.py            # Product KKT closed-form solver
│   ├── parallel_product_kkt.py   # Structural pooling for parallel DAGs
│   ├── psbc_m.py                 # Multi-round MWU macro scheduler
│   ├── kkt.py                    # NodeParams, AllocationResult, sum-KKT
│   ├── dag.py                    # DAG utilities (topo sort, weights, etc.)
│   ├── dag_gat.py                # GAT-Lookahead network (3-head)
│   ├── gat_oracle.py             # TTAS online adaptation wrapper
│   ├── gat_trainer.py            # GAT training utilities
│   ├── em_calibration.py         # Variational EM parameter estimation
│   ├── baselines.py              # Fixed / Greedy baseline allocators
│   ├── agent_pool.py             # Agent pool (for dynamic topology)
│   ├── topology_predictor.py     # Monte-Carlo topology sampling
│   ├── vsn.py                    # Virtual Super-Node 2D solver
│   └── math_dataset.py           # MATH dataset loader
├── mas/                          # Minimal multi-agent system
│   ├── __init__.py
│   ├── llm.py                    # LLM client (OpenAI SDK compatible)
│   └── executor.py               # DAG-based agent executor
├── experiments/                  # Experiment scripts
│   ├── run_pipeline.py           # Single-round serial pipeline
│   ├── run_parallel.py           # Single-round parallel aggregation
│   └── run_mad.py                # Multi-round debate (MAD)
├── scripts/
│   └── calibrate.py              # Phase 0 → EM → GAT calibration pipeline
├── data/                         # Dataset directory (placeholder)
└── results/                      # Experiment output directory
```

## Key Method

### Within-Round: Product KKT + Structural Pooling

For a DAG topology, each agent $v_j$ has a quality function:

$$Q_j(b_j) = \tilde{\sigma}_j \cdot (1 - e^{-\kappa_j(b_j - \tau_j)}), \quad b_j \ge \tau_j$$

The optimizer maximizes $\prod_j Q_j(b_j)$ under a budget constraint.
The closed-form allocation is:

$$b_j^*(\lambda) = \tau_j + \frac{1}{\kappa_j} \ln\left(1 + \frac{\kappa_j}{\lambda \cdot p_{out}}\right)$$

Parallel sibling groups are pooled into virtual nodes before KKT solving,
then budgets are split proportionally to $1/\kappa_j$.

### Across-Round: MWU with Terminal Projection

A global dual variable $\mu_k$ (shadow price) is maintained across rounds:

$$\mu_{k+1} = \mu_k \cdot \exp(\eta \cdot (\ell_k - 1)), \quad \ell_k = C_{true}^{(k)} / B_{target}^{(k)}$$

Per-node bidding under $\mu_k$: $b_j^*(\mu_k) = \tau_j + \frac{1}{\kappa_j}\ln(1 + \frac{\kappa_j}{\mu_k \cdot p_{out}})$

Terminal projection prevents budget hoarding in late rounds.

### GAT-Lookahead

A causal graph-attention network predicts:
- Downstream input token costs (Head 1 + Head 2 for uncertainty)
- Future topology (Head 3, for dynamic MAS)
- Task termination probability (Head 4, for horizon estimation)

Online adaptation via Test-Time Adaptive Scaling (TTAS) with EMA correction.

## Datasets

We use the [MATH](https://github.com/hendrycks/math) dataset (Hendrycks et al.,
NeurIPS 2021), MMLU-Pro and GPQA.
## Baselines

- **Fixed (Uniform)**: Equal `max_tokens` for all agents
- **Greedy**: Each agent receives all remaining budget (no downstream reserve)

## Citation

```bibtex
@inproceedings{ngobc2026,
  title={NGO-BC: Neural-Guided Online Budget Control for Multi-Agent LLM Systems},
  author={...},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT
