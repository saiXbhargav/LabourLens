"""Review candidate questions one by one and build the golden set.

Candidates = evals/handwritten.jsonl + evals/drafts.jsonl. Every decision is saved
immediately to evals/reviews.jsonl, so you can stop (x) and continue later. A decision
records the version of the question it was made on (item_hash); if the question is
changed afterwards, the decision lapses and the question is pending again.

Usage (from backend/):
    python -m evals.review              # review everything not yet decided
    python -m evals.review --redo ID    # review one question again
    python -m evals.review --status     # counts by type and decision
    python -m evals.review --export     # write evals/golden_set.jsonl from accepted questions

For each question, check: is the answer legally correct? Do the quotes support EVERY claim
in it? Is the question unambiguous and not a duplicate? Are conditions/exceptions included?

Keys while reviewing:
    a  accept            r  reject (asks for a reason)     s  skip for now
    q  edit question     w  edit answer                    t  show full gold section text
    x  exit
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter
from datetime import datetime

from evals.dataset import (
    DRAFTS, GOLDEN, HANDWRITTEN, REVIEWS, item_hash, item_quotes, load_sections, read_jsonl, validate, write_jsonl,
)

WIDTH = 100


def candidates() -> list[dict]:
    return read_jsonl(HANDWRITTEN) + read_jsonl(DRAFTS)


def valid_decisions(items: list[dict]) -> dict[str, dict]:
    """Decisions that still apply: made on the current version of each question."""
    by_id = {r["id"]: r for r in read_jsonl(REVIEWS)}  # last decision per id wins
    return {i["id"]: by_id[i["id"]] for i in items
            if i["id"] in by_id and by_id[i["id"]].get("base_hash") == item_hash(i)}


def save_decision(decision: dict) -> None:
    rows = [r for r in read_jsonl(REVIEWS) if r["id"] != decision["id"]] + [decision]
    write_jsonl(REVIEWS, rows)


def apply_edits(item: dict, decision: dict | None) -> dict:
    item = dict(item)
    if decision:
        item.update(decision.get("edits", {}))
    return item


def wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(textwrap.fill(p, WIDTH, initial_indent=indent, subsequent_indent=indent)
                     for p in str(text).split("\n"))


def show(item: dict, sections: dict, n: int, total: int) -> None:
    print("\n" + "=" * WIDTH)
    print(f"[{n}/{total}] {item['id']}  type={item['type']}  answerable={item['answerable']}")
    print("QUESTION:\n" + wrap(item["question"]))
    print("ANSWER:\n" + wrap(item["answer"]))
    if item.get("expected_value") is not None:
        print(f"EXPECTED VALUE: {item['expected_value']}")
    quotes = item_quotes(item)
    for i, quote in enumerate(quotes, 1):
        print((f"QUOTE {i}/{len(quotes)}:" if len(quotes) > 1 else "QUOTE (supporting text):") + "\n" + wrap(quote))
    for sid in item["gold_sections"]:
        s = sections.get(sid)
        print(f"GOLD: {s['label']} - {s['title']}" if s else f"GOLD: {sid} (unknown!)")
    if item.get("ai_recommendation"):
        print(f"AI PRE-CHECK: recommends {item['ai_recommendation'].upper()}"
              + (f"\n{wrap(item['ai_note'])}" if item.get("ai_note") else ""))
    if problems := validate(item, sections):
        print("AUTOMATIC CHECK PROBLEMS:\n" + wrap("; ".join(problems)))


def review(items: list[dict], sections: dict) -> None:
    prior = {r["id"]: r for r in read_jsonl(REVIEWS)}
    for n, original in enumerate(items, 1):
        item = apply_edits(original, prior.get(original["id"]))
        edits: dict = dict(prior.get(original["id"], {}).get("edits", {}))
        while True:
            show(item, sections, n, len(items))
            prompt = "\n[a]ccept [r]eject [s]kip  edit [q]uestion / ans[w]er  [t]ext  e[x]it > "
            key = input(prompt).strip().lstrip("﻿").lower()  # piped input on Windows may start with a BOM
            if key == "t":
                for sid in item["gold_sections"]:
                    if sid in sections:
                        print(f"\n--- {sections[sid]['label']} ---\n" + wrap(sections[sid]["text"], ""))
            elif key in ("q", "w"):
                field = "question" if key == "q" else "answer"
                new = input(f"New {field} (empty = keep): ").strip()
                if new:
                    item[field] = edits[field] = new
            elif key in ("a", "r"):
                if key == "a" and (problems := validate(item, sections)):
                    print(f"Can't accept yet: {problems}. Edit it or reject it.")
                    continue
                reason = input("Reason for rejecting: ").strip() if key == "r" else ""
                save_decision({"id": original["id"], "decision": "accept" if key == "a" else "reject",
                               "reason": reason, "edits": edits, "base_hash": item_hash(original),
                               "at": datetime.now().isoformat(timespec="seconds")})
                break
            elif key == "s":
                break
            elif key == "x":
                return
            else:
                print(f"Unknown key {key!r}. Use a, r, s, q, w, t or x.")


def status() -> None:
    items = candidates()
    done = valid_decisions(items)
    rows = Counter((i["type"], done.get(i["id"], {}).get("decision", "pending")) for i in items)
    print(f"{'type':15} {'accept':>7} {'reject':>7} {'pending':>8}")
    for t in sorted({t for t, _ in rows}):
        print(f"{t:15} {rows[(t, 'accept')]:>7} {rows[(t, 'reject')]:>7} {rows[(t, 'pending')]:>8}")
    totals = Counter(d for (_, d), v in rows.items() for _ in range(v))
    print(f"{'TOTAL':15} {totals['accept']:>7} {totals['reject']:>7} {totals['pending']:>8}")


def export(sections: dict) -> None:
    items = candidates()
    done = valid_decisions(items)
    pending = [i["id"] for i in items if i["id"] not in done]
    if pending:
        print(f"Note: {len(pending)} questions are still pending and are left out: {pending[:10]}...")
    golden, errors = [], []
    for item in items:
        d = done.get(item["id"])
        if not d or d["decision"] != "accept":
            continue
        item = apply_edits(item, d)
        if problems := validate(item, sections):
            errors.append((item["id"], problems))
            continue
        keep = ("id", "type", "question", "answer", "expected_value", "expected_behavior", "gold_sections",
                "gold_definition", "answerable")
        row = {k: item[k] for k in keep if k in item}
        row["quotes"] = item_quotes(item)
        golden.append(row)
    if errors:
        raise SystemExit(f"Not exported; fix these first: {errors}")
    write_jsonl(GOLDEN, golden)
    print(f"Wrote {len(golden)} questions to {GOLDEN.name}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--redo", help="review this question id again")
    args = parser.parse_args()
    sections = load_sections()

    if args.status:
        status()
    elif args.export:
        export(sections)
    else:
        items = candidates()
        if args.redo:
            items = [i for i in items if i["id"] == args.redo]
        else:
            done = valid_decisions(items)
            items = [i for i in items if i["id"] not in done]
        if not items:
            print("Nothing left to review. Run with --status or --export.")
            return
        review(items, sections)
        status()


if __name__ == "__main__":
    main()
