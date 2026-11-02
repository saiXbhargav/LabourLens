"""Blind spot-check of the LLM judge against your own grading.

Shows ~30 answers from a run (a mix of judge scores, plus out-of-scope questions). You
grade each one BEFORE seeing the judge's verdict; then the agreement is reported. Report
this agreement next to the judge-based numbers, so readers know how far to trust them.

Usage (from backend/):
    python -m evals.spotcheck baseline           # grade (resumable; x to stop)
    python -m evals.spotcheck baseline --report  # agreement so far
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from collections import Counter
from datetime import datetime

from evals.dataset import GOLDEN, read_jsonl, write_jsonl
from evals.run import RESULTS

WIDTH = 100


def wrap(text: str) -> str:
    return "\n".join(textwrap.fill(p, WIDTH, initial_indent="    ", subsequent_indent="    ")
                     for p in str(text).split("\n"))


def sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Stratified: roughly equal numbers of judge scores 0/1/2, plus a few out-of-scope items."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = f"score{r['correctness']}" if r["answerable"] else "out_of_scope"
        if r.get("correctness") is not None or not r["answerable"]:
            groups.setdefault(key, []).append(r)
    per_group = max(1, n // max(1, len(groups)))
    picked = []
    for key in sorted(groups):
        picked += rng.sample(groups[key], min(per_group, len(groups[key])))
    rest = [r for g in groups.values() for r in g if r not in picked]
    picked += rng.sample(rest, min(n - len(picked), len(rest)))
    rng.shuffle(picked)
    return picked


def kappa(pairs: list[tuple]) -> float | None:
    """Cohen's kappa between your grades and the judge's."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    you, judge = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(you[c] * judge[c] for c in set(you) | set(judge)) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def report(grades: list[dict]) -> None:
    ans = [(g["you"], g["judge"]) for g in grades if g["answerable"]]
    oos = [(g["you"], g["judge"]) for g in grades if not g["answerable"]]
    if ans:
        exact = sum(a == b for a, b in ans) / len(ans)
        binary = sum((a == 2) == (b == 2) for a, b in ans) / len(ans)  # agree on "fully correct or not"
        k = kappa(ans)
        print(f"Answerable ({len(ans)}): exact agreement {exact:.0%}, agree on fully-correct {binary:.0%}, "
              f"Cohen's kappa {k:.2f}" if k is not None else "")
        disagreements = Counter(f"you {a} / judge {b}" for a, b in ans if a != b)
        if disagreements:
            print("  disagreements:", dict(disagreements))
    if oos:
        print(f"Out-of-scope ({len(oos)}): agreement {sum(a == b for a, b in oos) / len(oos):.0%}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="experiment name under evals/results/")
    parser.add_argument("-n", type=int, default=30)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    run_dir = RESULTS / args.run
    path = run_dir / "spotcheck.jsonl"
    grades = read_jsonl(path)
    if args.report:
        report(grades)
        return

    rows = [r for r in read_jsonl(run_dir / "per_question.jsonl") if "error" not in r]
    golden = {g["id"]: g for g in read_jsonl(GOLDEN)}
    done = {g["id"] for g in grades}
    todo = [r for r in sample(rows, args.n) if r["id"] not in done]
    for i, r in enumerate(todo, 1):
        item = golden[r["id"]]
        print("\n" + "=" * WIDTH + f"\n[{len(done) + i}/{args.n}] {r['id']}  type={r['type']}")
        print("QUESTION:\n" + wrap(item["question"]))
        print("REFERENCE ANSWER:\n" + wrap(item["answer"]))
        print("SYSTEM ANSWER:\n" + wrap(r["answer"]))
        if item["answerable"]:
            prompt = "\nYour grade: 2 = fully correct, 1 = partly, 0 = wrong/no answer  (x = stop) > "
            valid = {"0": 0, "1": 1, "2": 2}
        else:
            prompt = "\nDid it correctly decline / not invent an answer? y/n  (x = stop) > "
            valid = {"y": True, "n": False}
        while (key := input(prompt).strip().lstrip("﻿").lower()) not in valid and key != "x":
            print("Please answer", "/".join(valid))
        if key == "x":
            break
        judge = r["correctness"] if item["answerable"] else r["correct_behaviour"]
        print(f"Judge said: {judge}  ({r.get('judge_reason', '')})")
        grades.append({"id": r["id"], "answerable": item["answerable"], "you": valid[key], "judge": judge,
                       "at": datetime.now().isoformat(timespec="seconds")})
        write_jsonl(path, grades)
    report(grades)


if __name__ == "__main__":
    main()
