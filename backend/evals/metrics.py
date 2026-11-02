"""Pure metric functions (no I/O), so they are easy to test."""

from __future__ import annotations

import re
import statistics


# ---------------------------------------------------------------- retrieval

def first_hit_rank(retrieved: list[list[str]], gold: set[str]) -> int | None:
    """1-based rank of the first retrieved chunk that overlaps a gold section."""
    for rank, sections in enumerate(retrieved, 1):
        if gold & set(sections):
            return rank
    return None


def hit_at_k(retrieved: list[list[str]], gold: set[str], k: int) -> bool:
    """Any gold section appears in the top k chunks."""
    rank = first_hit_rank(retrieved, gold)
    return rank is not None and rank <= k


def recall_at_k(retrieved: list[list[str]], gold: set[str], k: int) -> float:
    """Fraction of gold sections covered by the top k chunks (matters for multi-section questions)."""
    covered = set().union(*map(set, retrieved[:k])) if retrieved else set()
    return len(gold & covered) / len(gold) if gold else 0.0


def reciprocal_rank(retrieved: list[list[str]], gold: set[str]) -> float:
    rank = first_hit_rank(retrieved, gold)
    return 1.0 / rank if rank else 0.0


# ---------------------------------------------------------------- answers

_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?", re.I)


def numbers_in(text: str) -> list[float]:
    """Numbers in an answer, understanding Indian formats: 1,50,000 / 1.5 lakh / Rs. 2 crore."""
    out = []
    for num, unit in _NUMBER.findall(text):
        try:
            value = float(num.replace(",", ""))
        except ValueError:
            continue
        if unit:
            value *= 100_000 if unit.lower().startswith("lakh") else 10_000_000
        out.append(value)
    return out


def calc_match(answer: str, expected: float, tolerance: float = 1.0) -> bool:
    """The expected value appears in the answer (within ±tolerance, e.g. ₹1)."""
    return any(abs(v - expected) <= tolerance for v in numbers_in(answer))


_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")


def citations(answer: str) -> list[int]:
    """Passage numbers cited as [1], [2][3], [1, 2] or [1-3]."""
    cited = []
    for group in _CITATION.findall(answer):
        for part in re.split(r"\s*,\s*", group):
            if m := re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part):
                cited += range(int(m.group(1)), int(m.group(2)) + 1)
            else:
                cited.append(int(part))
    return sorted(set(cited))


_REFUSAL = re.compile(
    r"could not (find|locate|be found)|cannot (be )?(determine|calculat|answer|find)|not (specified|mentioned|"
    r"provided|available|stated|found) in the (provided )?(texts|passages)|do(es)? not (contain|specify|provide|"
    r"mention|state)", re.I)


def is_refusal(answer: str) -> bool:
    """The answer says the information isn't available (used to measure over-refusal)."""
    return bool(_REFUSAL.search(answer[:400]))


# ---------------------------------------------------------------- summaries

def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else None


def percentile(values, p: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    idx = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[idx]
