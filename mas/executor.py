"""
mas/executor.py — DAG-based agent executor for budget-controlled MAS.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

from .llm import LLMClient


@dataclass
class AgentSpec:
    """Specification of one agent node in the DAG."""
    node_id: str
    role: str
    system_prompt: str
    depends_on: List[str] = None   # parent node IDs in the DAG

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


class DAGExecutor:
    """
    Execute agents in topological order through a DAG.

    Each agent receives outputs from its parent nodes.  The executor
    tracks per-node input/output token counts and exposes the API
    that NGO-BC's PSBCOptimizer expects.
    """

    def __init__(
        self,
        agent_specs: List[AgentSpec],
        llm: LLMClient,
        *,
        terminal_nodes: Optional[List[str]] = None,
    ):
        self._specs = {s.node_id: s for s in agent_specs}

        # Build adjacency
        self._adj: Dict[str, List[str]] = {}
        self._rev: Dict[str, List[str]] = {}
        for s in agent_specs:
            self._adj.setdefault(s.node_id, [])
            for p in s.depends_on:
                self._rev.setdefault(s.node_id, []).append(p)
                self._adj.setdefault(p, []).append(s.node_id)

        self._llm = llm

        # Terminal nodes
        if terminal_nodes:
            self._terminal = terminal_nodes
        else:
            children = set()
            for clist in self._adj.values():
                children.update(clist)
            self._terminal = sorted(set(self._adj) - children)

        # Execution state
        self._outputs: Dict[str, str] = {}
        self._executed: Set[str] = set()
        self._skipped: Set[str] = set()

    @property
    def adjacency(self) -> Dict[str, List[str]]:
        return dict(self._adj)

    @property
    def terminal_nodes(self) -> List[str]:
        return list(self._terminal)

    def get_input_tokens(self, node_id: str, task_prompt: str) -> int:
        """Compute exact input tokens for a node before execution."""
        spec = self._specs[node_id]
        parent_outputs = ""
        for p in self._rev.get(node_id, []):
            parent_outputs += self._outputs.get(p, "")
        prompt = task_prompt + parent_outputs
        return self._llm.count_messages(spec.system_prompt, prompt)

    def execute(self, node_id: str, task_prompt: str, max_tokens: int) -> int:
        """Execute one agent. Returns actual output token count."""
        spec = self._specs[node_id]
        parent_outputs = ""
        for p in self._rev.get(node_id, []):
            parent_outputs += self._outputs.get(p, "")
        prompt = task_prompt + parent_outputs

        answer, c_out = self._llm.chat(spec.system_prompt, prompt, max_tokens)
        self._outputs[node_id] = answer
        self._executed.add(node_id)
        return c_out

    def skip(self, node_id: str) -> None:
        self._skipped.add(node_id)

    def get_output(self, node_id: str) -> str:
        return self._outputs.get(node_id, "")

    def iter_topological(self) -> Iterator[str]:
        """Yield node IDs in topological order, as they become ready."""
        ready = deque()
        yielded: Set[str] = set()

        def _enqueue():
            done = self._executed | self._skipped
            for nid in self._specs:
                if nid in yielded:
                    continue
                parents = self._rev.get(nid, [])
                if all(p in done for p in parents):
                    ready.append(nid)
                    yielded.add(nid)

        _enqueue()
        while ready:
            yield ready.popleft()
            _enqueue()


# ── Pre-built topologies ──────────────────────────────────────────

def make_pipeline_agents() -> List[AgentSpec]:
    """3-node serial pipeline for math problem solving."""
    return [
        AgentSpec("Decomposer", "Decomposer",
                  "Analyse the math problem. Identify key concepts and outline a strategy. "
                  "Do NOT compute the final answer yet.",
                  depends_on=[]),
        AgentSpec("Solver", "Solver",
                  "Solve step by step. Put the final answer inside \\boxed{{}}.",
                  depends_on=["Decomposer"]),
        AgentSpec("Verifier", "Verifier",
                  "Review the solution. If correct, confirm in \\boxed{{}}. "
                  "If wrong, provide corrected answer in \\boxed{{}}.",
                  depends_on=["Solver"]),
    ]


def make_parallel_agents() -> List[AgentSpec]:
    """5-node parallel aggregation for math problem solving."""
    return [
        AgentSpec("Decomposer", "Decomposer",
                  "Analyse the math problem. Identify key concepts and outline a strategy. "
                  "Do NOT compute the final answer yet.",
                  depends_on=[]),
        AgentSpec("Solver1", "Solver",
                  "Solve step by step using algebraic methods. Put answer in \\boxed{{}}.",
                  depends_on=["Decomposer"]),
        AgentSpec("Solver2", "Solver",
                  "Solve using a different approach (geometric, case analysis, etc). "
                  "Put answer in \\boxed{{}}.",
                  depends_on=["Decomposer"]),
        AgentSpec("Solver3", "Solver",
                  "Solve step by step. Verify each step. Put answer in \\boxed{{}}.",
                  depends_on=["Decomposer"]),
        AgentSpec("Aggregator", "Aggregator",
                  "Compare the three solutions. Confirm consensus or determine the correct answer. "
                  "Put final answer in \\boxed{{}}.",
                  depends_on=["Solver1", "Solver2", "Solver3"]),
    ]


def make_debate_agents() -> List[AgentSpec]:
    """3-node debate pipeline (MAD-style)."""
    return [
        AgentSpec("Affirmative", "Affirmative",
                  "You are a debater. Provide your reasoning and answer to the debate topic.",
                  depends_on=[]),
        AgentSpec("Negative", "Negative",
                  "You disagree with the affirmative side. Provide your counter-argument and answer.",
                  depends_on=["Affirmative"]),
        AgentSpec("Moderator", "Moderator",
                  "Evaluate both debaters' arguments. "
                  'Output JSON: {"debate_answer": "<final answer>", "Reason": "..."}',
                  depends_on=["Affirmative", "Negative"]),
    ]


# ── Answer extraction ─────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    """Extract \\boxed{} answer from model output."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def answers_match(pred: str, gt: str) -> bool:
    """Check if predicted answer matches ground truth."""
    p = pred.strip()
    g = gt.strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    return False
