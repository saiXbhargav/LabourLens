"""Evaluation dataset: loading and validation shared by drafting, review and the eval harness.

Files (evals/):
    handwritten.jsonl   calculation and out-of-scope questions written by hand
    drafts.jsonl        LLM-drafted questions (python -m evals.draft_questions)
    reviews.jsonl       review decisions, one per question id (python -m evals.review)
    golden_set.jsonl    the frozen test set: accepted questions with edits applied
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import PROCESSED

EVALS = Path(__file__).resolve().parent
HANDWRITTEN = EVALS / "handwritten.jsonl"
DRAFTS = EVALS / "drafts.jsonl"
REVIEWS = EVALS / "reviews.jsonl"
GOLDEN = EVALS / "golden_set.jsonl"

TYPES = {"definition", "rule", "everyday", "calculation", "multi_section", "what_changed", "out_of_scope"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sections() -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(PROCESSED / "sections.jsonl")}


def normalize(text: str) -> str:
    """Compare quotes loosely: case, whitespace, quote styles and dashes may differ."""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[–—-]+", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def validate(item: dict, sections: dict[str, dict]) -> list[str]:
    """Problems with a question; an empty list means it passes the automatic checks."""
    problems = []
    for key in ("id", "type", "question", "answer", "gold_sections", "answerable"):
        if key not in item:
            problems.append(f"missing field {key}")
    if problems:
        return problems
    if item["type"] not in TYPES:
        problems.append(f"unknown type {item['type']}")
    missing = [s for s in item["gold_sections"] if s not in sections]
    if missing:
        problems.append(f"unknown gold sections {missing}")
    if item["answerable"] and not item["gold_sections"]:
        problems.append("answerable question without gold sections")
    if quote := item.get("quote"):
        gold_text = " ".join(normalize(sections[s]["text"]) for s in item["gold_sections"] if s in sections)
        if normalize(quote) not in gold_text:
            problems.append("quote not found verbatim in the gold sections")
    elif item["answerable"]:
        problems.append("answerable question without a supporting quote")
    if item["answerable"] and re.search(r"\b(section|sub-section|rule|clause)\s+\d", item["question"], re.I):
        problems.append("question names a section/rule number (gives the answer away)")
    return problems
