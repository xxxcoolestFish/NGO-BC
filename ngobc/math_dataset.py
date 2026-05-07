"""
psbc/math_dataset.py – MATH dataset loader for PSBC experiments.

Loads Level 3-5 problems from the local MATH dataset, extracts ground-truth
answers via \boxed{}, and provides a deterministic sampler for reproducibility.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

TARGET_LEVELS = {"Level 3", "Level 4", "Level 5"}


@dataclass
class MathProblem:
    problem_id:  str
    subject:     str
    level:       str
    problem:     str
    solution:    str          # full reference solution
    answer:      str          # extracted \boxed{} content


def _extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} in the solution."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def _normalize_answer(ans: str) -> str:
    """Light normalisation for answer comparison."""
    ans = ans.strip()
    # Remove surrounding $...$ or \(...\)
    ans = re.sub(r"^\\\(|\\\)$", "", ans).strip()
    ans = re.sub(r"^\$|\$$", "", ans).strip()
    # Collapse whitespace
    ans = re.sub(r"\s+", " ", ans)
    return ans


def _latex_to_sympy_str(s: str) -> str:
    """Convert common LaTeX math to sympy-parseable string."""
    s = s.strip()
    # \frac{a}{b} → (a)/(b)
    while True:
        m = re.search(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
        if not m:
            break
        s = s[:m.start()] + f"({m.group(1)})/({m.group(2)})" + s[m.end():]
    # \sqrt{a} → sqrt(a)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    # Remove remaining backslashes and LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "(").replace("}", ")")
    return s


def answers_match(predicted: str, ground_truth: str) -> bool:
    """
    Compare two math answers after normalisation.
    Tries string equality, numeric, then sympy symbolic evaluation.
    """
    p = _normalize_answer(predicted)
    g = _normalize_answer(ground_truth)
    if p == g:
        return True

    # Try direct numeric comparison
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass

    # Try sympy symbolic evaluation with LaTeX conversion
    try:
        import sympy
        ps = _latex_to_sympy_str(p)
        gs = _latex_to_sympy_str(g)
        pv = sympy.sympify(ps)
        gv = sympy.sympify(gs)
        diff = sympy.simplify(pv - gv)
        return diff == 0
    except Exception:
        pass

    return False


def load_math_problems(
    data_root: str | Path,
    subjects:  Optional[List[str]] = None,
    levels:    Optional[set] = None,
    split:     str = "test",
) -> List[MathProblem]:
    """Load all MATH problems matching the given subjects and levels."""
    from datasets import load_from_disk

    data_root = Path(data_root)
    subjects  = subjects or MATH_SUBJECTS
    levels    = levels   or TARGET_LEVELS
    problems: List[MathProblem] = []

    for subj in subjects:
        path = data_root / subj
        if not path.exists():
            continue
        ds = load_from_disk(str(path))
        if split not in ds:
            continue
        for row in ds[split]:
            if row["level"] not in levels:
                continue
            answer = _extract_boxed(row["solution"])
            if answer is None:
                continue
            problems.append(MathProblem(
                problem_id = f"{subj}_{row.get('type', 'unknown')[:20]}_{len(problems)}",
                subject    = subj,
                level      = row["level"],
                problem    = row["problem"],
                solution   = row["solution"],
                answer     = answer,
            ))

    return problems


def sample_problems(
    problems:   List[MathProblem],
    n:          int,
    seed:       int = 42,
    stratified: bool = True,
) -> List[MathProblem]:
    """
    Sample n problems, optionally stratified by subject × level.
    Uses a fixed seed for reproducibility.
    """
    rng = random.Random(seed)

    if not stratified or n >= len(problems):
        shuffled = list(problems)
        rng.shuffle(shuffled)
        return shuffled[:n]

    # Stratified: equal quota per (subject, level) bucket
    buckets: dict = {}
    for p in problems:
        key = (p.subject, p.level)
        buckets.setdefault(key, []).append(p)

    per_bucket = max(1, n // len(buckets))
    sampled = []
    for key, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_bucket])

    # Top-up to exactly n
    sampled_ids = {id(p) for p in sampled}
    remaining   = [p for p in problems if id(p) not in sampled_ids]
    rng.shuffle(remaining)
    sampled.extend(remaining[: max(0, n - len(sampled))])
    return sampled[:n]


def extract_answer_from_response(response: str) -> Optional[str]:
    """
    Try to extract a \\boxed{} answer from a model response.
    Falls back to the last number/fraction found if no boxed answer.
    """
    boxed = _extract_boxed(response)
    if boxed:
        return _normalize_answer(boxed)

    # Fallback: last number or simple fraction
    numbers = re.findall(r"-?\d+(?:[./]\d+)?", response)
    return numbers[-1] if numbers else None
