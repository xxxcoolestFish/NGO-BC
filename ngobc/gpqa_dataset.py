"""
GPQA Diamond dataset loader for PSBC experiments.

GPQA (Google-Proof Q&A) is a multiple-choice QA dataset of graduate-level
questions in Physics, Chemistry, and Biology. The "Diamond" subset contains
the hardest questions where non-experts rarely answer correctly.

Usage:
    from gpqa_dataset import load_gpqa_questions, sample_questions, check_answer

    questions = load_gpqa_questions("data/GPQA/gpqa_diamond.csv")
    sampled   = sample_questions(questions, n=50, seed=42)
    for q in sampled:
        print(q.prompt)  # formatted MC question
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GPQAQuestion:
    question_id: str
    question:    str          # cleaned question text
    correct:     str          # e.g. "A"
    options:     List[str]    # [option_A_text, option_B_text, option_C_text, option_D_text]
    domain:      str          # Physics / Chemistry / Biology
    subdomain:   str

    @property
    def prompt(self) -> str:
        """Build a formatted MC prompt for the MAS."""
        lines = [self.question, "", "Options:"]
        letters = ["A", "B", "C", "D"]
        for letter, opt in zip(letters, self.options):
            lines.append(f"  {letter}. {opt}")
        lines.append("")
        lines.append("Please reason step by step and put your final answer in \\boxed{{}} (e.g., \\boxed{{B}}).")
        return "\n".join(lines)

    @property
    def answer(self) -> str:
        """The correct letter, e.g. 'A'."""
        return self.correct


def load_gpqa_questions(
    csv_path: str | Path,
    domains: Optional[List[str]] = None,
) -> List[GPQAQuestion]:
    """
    Load GPQA Diamond questions from CSV.

    Parameters
    ----------
    csv_path : path to gpqa_diamond.csv
    domains  : filter by domain, e.g. ["Physics", "Chemistry"]. None = all.

    Returns
    -------
    List of GPQAQuestion objects.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"GPQA dataset not found: {path}")

    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row.get("High-level domain", "").strip()
            if domains and domain not in domains:
                continue

            question_text = row.get("Question", "").strip()
            if not question_text:
                continue

            # Collect all 4 options
            correct = row.get("Correct Answer", "").strip()
            wrong1  = row.get("Incorrect Answer 1", "").strip()
            wrong2  = row.get("Incorrect Answer 2", "").strip()
            wrong3  = row.get("Incorrect Answer 3", "").strip()

            all_options = [correct, wrong1, wrong2, wrong3]
            # Shuffle deterministically by question_id to mix correct position
            qid = row.get("Record ID", "")
            rng = random.Random(hash(qid) % (2**31))
            shuffled = list(enumerate(all_options))
            rng.shuffle(shuffled)

            correct_letter = None
            options = [""] * 4
            for new_idx, (orig_idx, text) in enumerate(shuffled):
                letter = chr(ord("A") + new_idx)
                options[new_idx] = text
                if orig_idx == 0:  # this was the correct answer
                    correct_letter = letter

            questions.append(GPQAQuestion(
                question_id=qid,
                question=question_text,
                correct=correct_letter or "A",
                options=options,
                domain=domain,
                subdomain=row.get("Subdomain", "").strip(),
            ))

    return questions


def sample_questions(
    questions: List[GPQAQuestion],
    n: int = 50,
    seed: int = 42,
    stratified: bool = True,
) -> List[GPQAQuestion]:
    """
    Sample n questions (stratified by domain if possible).
    """
    if n >= len(questions):
        return list(questions)

    if not stratified:
        rng = random.Random(seed)
        return rng.sample(questions, n)

    # Stratify by domain
    from collections import defaultdict
    by_domain = defaultdict(list)
    for q in questions:
        by_domain[q.domain].append(q)

    domains = sorted(by_domain.keys())
    per_domain = max(1, n // len(domains))

    rng = random.Random(seed)
    sampled = []
    for domain in domains:
        pool = by_domain[domain]
        k = min(per_domain, len(pool))
        sampled.extend(rng.sample(pool, k))

    # Fill remaining from random
    if len(sampled) < n:
        remaining = [q for q in questions if q not in sampled]
        sampled.extend(rng.sample(remaining, n - len(sampled)))

    rng.shuffle(sampled)
    return sampled[:n]


def interleave_by_domain(questions: List[GPQAQuestion]) -> List[GPQAQuestion]:
    """
    Arrange questions in round-robin order by domain (B, C, P, B, C, P, ...)
    so that any contiguous segment has balanced domain difficulty.
    """
    from collections import defaultdict
    by_domain = defaultdict(list)
    for q in questions:
        by_domain[q.domain].append(q)

    domains = sorted(by_domain.keys())
    result = []
    while True:
        added = False
        for d in domains:
            if by_domain[d]:
                result.append(by_domain[d].pop(0))
                added = True
        if not added:
            break
    return result


def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract the answer letter from an LLM response."""
    if not response:
        return None

    # Try \boxed{X} format first
    m = re.search(r"\\boxed\{([A-Da-d])\}", response)
    if m:
        return m.group(1).upper()

    # Try "Answer: X" or "The answer is X"
    m = re.search(r"(?:answer\s*(?:is|:)?\s*)([A-Da-d])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Try bare letter near the end
    m = re.search(r"\b([A-D])\b\s*$", response.strip())
    if m:
        return m.group(1).upper()

    return None


def check_answer(predicted: Optional[str], ground_truth: str) -> bool:
    """Compare predicted letter with ground truth."""
    if not predicted:
        return False
    return predicted.upper() == ground_truth.upper()
