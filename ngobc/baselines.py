"""
psbc/baselines.py – Budget allocation baselines for comparison with PSBC.

FixedAllocator  : divide total budget equally among all nodes (static pre-split).
GreedyAllocator : give current node as many tokens as remaining budget allows,
                  no reserve for downstream nodes (demonstrates budget breakdown risk).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BaselineSummary:
    method:          str
    budget:          float
    spent:           float
    budget_ok:       bool
    nodes_executed:  int
    nodes_skipped:   int
    per_node:        Dict[str, dict] = field(default_factory=dict)


class FixedAllocator:
    """
    Static equal-share allocator.

    Each node receives:  max_tokens = floor((B_t - input_reserve) / n / p_out)

    where input_reserve = n * avg_E_in * p_in  (estimated from a cold-start
    heuristic of avg_E_in tokens per node).

    This is the simplest possible baseline: no topology awareness, no
    adaptation, no reserve.
    """

    def __init__(
        self,
        n_nodes:        int,
        budget:         float,
        p_in:           float,
        p_out:          float,
        avg_input_est:  int = 300,   # cold-start estimate for input tokens per node
    ) -> None:
        self.n_nodes  = max(n_nodes, 1)
        self.budget   = budget
        self.p_in     = p_in
        self.p_out    = p_out
        self.B_t      = budget
        self.R        = budget

        # Pre-compute per-node cap
        input_reserve = self.n_nodes * avg_input_est * p_in
        output_budget = max(budget - input_reserve, 0.0)
        self._cap     = max(int(output_budget / self.n_nodes / p_out), 1)

        self._nodes_executed = 0
        self._nodes_skipped  = 0
        self._per_node: Dict[str, dict] = {}

    def decide(self, agent_id: str, exact_in_tokens: int) -> Tuple[str, int]:
        cost_in = self.p_in * exact_in_tokens
        if self.R - cost_in < self.p_out * self._cap:
            # Remaining budget can't cover the fixed cap; give what's left
            remaining_out = max(self.R - cost_in, 0.0)
            cap = max(int(remaining_out / self.p_out), 0)
            if cap == 0:
                self._nodes_skipped += 1
                self._per_node[agent_id] = {
                    "status": "SKIPPED", "exact_in": exact_in_tokens,
                    "assigned_max": 0, "actual_out": 0,
                }
                return "SKIP", 0
        else:
            cap = self._cap

        self._per_node[agent_id] = {
            "status": "ACTIVE", "exact_in": exact_in_tokens,
            "assigned_max": cap, "actual_out": 0,
        }
        return "EXECUTE", cap

    def settle(self, agent_id: str, actual_out: int) -> None:
        info = self._per_node[agent_id]
        info["actual_out"] = actual_out
        info["status"]     = "DONE"
        self.R -= self.p_in * info["exact_in"] + self.p_out * actual_out
        self._nodes_executed += 1

    def settle_skip(self, agent_id: str) -> None:
        self._per_node.setdefault(agent_id, {})["status"] = "SKIPPED"

    def summary(self) -> BaselineSummary:
        spent = self.B_t - self.R
        return BaselineSummary(
            method="Fixed",
            budget=self.B_t,
            spent=spent,
            budget_ok=spent <= self.B_t + 1e-9,
            nodes_executed=self._nodes_executed,
            nodes_skipped=self._nodes_skipped,
            per_node=self._per_node,
        )


class GreedyAllocator:
    """
    Greedy allocator: give each node as many tokens as the remaining budget
    allows, with NO reserve for downstream nodes.

    This demonstrates the "budget breakdown" failure mode: early nodes may
    consume all budget, leaving later nodes with nothing.
    """

    def __init__(
        self,
        budget:       float,
        p_in:         float,
        p_out:        float,
        max_cap:      int = 4096,   # hard ceiling per node
    ) -> None:
        self.budget   = budget
        self.B_t      = budget
        self.p_in     = p_in
        self.p_out    = p_out
        self.max_cap  = max_cap
        self.R        = budget

        self._nodes_executed = 0
        self._nodes_skipped  = 0
        self._per_node: Dict[str, dict] = {}

    def decide(self, agent_id: str, exact_in_tokens: int) -> Tuple[str, int]:
        cost_in       = self.p_in * exact_in_tokens
        remaining_out = self.R - cost_in
        if remaining_out <= 0:
            self._nodes_skipped += 1
            self._per_node[agent_id] = {
                "status": "SKIPPED", "exact_in": exact_in_tokens,
                "assigned_max": 0, "actual_out": 0,
            }
            return "SKIP", 0

        cap = min(int(remaining_out / self.p_out), self.max_cap)
        cap = max(cap, 1)
        self._per_node[agent_id] = {
            "status": "ACTIVE", "exact_in": exact_in_tokens,
            "assigned_max": cap, "actual_out": 0,
        }
        return "EXECUTE", cap

    def settle(self, agent_id: str, actual_out: int) -> None:
        info = self._per_node[agent_id]
        info["actual_out"] = actual_out
        info["status"]     = "DONE"
        self.R -= self.p_in * info["exact_in"] + self.p_out * actual_out
        self._nodes_executed += 1

    def settle_skip(self, agent_id: str) -> None:
        self._per_node.setdefault(agent_id, {})["status"] = "SKIPPED"

    def summary(self) -> BaselineSummary:
        spent = self.B_t - self.R
        return BaselineSummary(
            method="Greedy",
            budget=self.B_t,
            spent=spent,
            budget_ok=spent <= self.B_t + 1e-9,
            nodes_executed=self._nodes_executed,
            nodes_skipped=self._nodes_skipped,
            per_node=self._per_node,
        )
