"""Run one experiment over the golden set and record the results.

Usage (from backend/):
    python -m evals.run --config evals/configs/baseline.yaml
    python -m evals.run --config evals/configs/baseline.yaml --limit 10      # quick smoke test
    python -m evals.run --config evals/configs/baseline.yaml --retrieval-only

Writes evals/results/<name>/per_question.jsonl and summary.json, and regenerates
evals/results/RESULTS.md (one row per experiment).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the embedding model is cached by indexing; skip Hub checks

import yaml  # noqa: E402

from evals.cache import CachedLLM  # noqa: E402
from evals.dataset import EVALS, GOLDEN, read_jsonl, write_jsonl  # noqa: E402
from evals.judge import JUDGE_SYSTEM, judge_prompt, parse_verdict  # noqa: E402
from evals.metrics import (  # noqa: E402
    calc_match, citations, hit_at_k, is_refusal, mean, percentile, recall_at_k, reciprocal_rank,
)
from src.config import PROCESSED  # noqa: E402

RESULTS = EVALS / "results"


def evaluate_item(item: dict, cfg: dict, embedder, store, rag, judge) -> dict:
    k, depth = cfg["top_k"], cfg["retrieval_depth"]
    row = {"id": item["id"], "type": item["type"], "answerable": item["answerable"], "question": item["question"]}

    t0 = time.perf_counter()
    hits = store.search(embedder.embed_query(item["question"]), depth)
    row["retrieve_ms"] = (time.perf_counter() - t0) * 1000
    retrieved = [h.payload["section_ids"] for h in hits]
    row["retrieved_chunks"] = [h.id for h in hits]
    gold = set(item["gold_sections"])

    if item["answerable"] and gold:
        row.update({f"hit@{n}": hit_at_k(retrieved, gold, n) for n in (1, 3, 5, 10) if n <= depth})
        row["recall@5"] = recall_at_k(retrieved, gold, 5)
        row["rr"] = reciprocal_rank(retrieved, gold)

    if rag is None:
        return row

    answer = rag.generate(item["question"], hits[:k])
    meta = rag.llm.last_meta
    row.update(answer=answer.text, generator_model=answer.model, generate_ms=meta.get("latency_ms"),
               prompt_tokens=meta.get("prompt_tokens"), output_tokens=meta.get("output_tokens"),
               generation_cached=meta.get("cached"))

    cited = citations(answer.text)
    row["cited"] = cited
    row["invalid_citation"] = any(c < 1 or c > len(hits[:k]) for c in cited)
    if gold and item["answerable"]:
        row["cites_gold"] = any(gold & set(hits[c - 1].payload["section_ids"]) for c in cited if 1 <= c <= k)
    if item.get("expected_value"):  # 0 ("not entitled") is left to the judge
        row["calc_match"] = calc_match(answer.text, item["expected_value"])

    if judge is not None:
        from src.generation.prompt import format_context

        prompt = judge_prompt(item, answer.text, format_context(hits[:k]))
        raw, judge_model = judge.generate(prompt, system=JUDGE_SYSTEM, json_mode=True)
        verdict = parse_verdict(raw, item["answerable"])
        if "judge_error" in verdict:  # malformed/truncated JSON: ask once more, replacing the cached reply
            raw, judge_model = judge.generate(prompt, system=JUDGE_SYSTEM, json_mode=True, refresh=True)
            verdict = parse_verdict(raw, item["answerable"])
        row.update(verdict, judge_model=judge_model)
    return row


def summarize(rows: list[dict], cfg: dict, dataset_id: str, args) -> dict:
    ans = [r for r in rows if r["answerable"] and "rr" in r]
    unans = [r for r in rows if not r["answerable"]]
    judged = [r for r in rows if r["answerable"] and r.get("correctness") is not None]
    s = {
        "name": cfg["name"], "description": cfg.get("description", ""), "run_at": datetime.now().isoformat("T", "seconds"),
        "dataset": dataset_id, "n_questions": len(rows), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "config": cfg, "partial": bool(args.limit or args.retrieval_only or args.no_judge),
        "retrieval": {
            **{f"hit@{n}": mean(r.get(f"hit@{n}") for r in ans) for n in (1, 3, 5, 10)},
            "recall@5": mean(r["recall@5"] for r in ans),
            "mrr@10": mean(r["rr"] for r in ans),
        },
    }
    if not args.retrieval_only:
        s["answers"] = {
            "accuracy": mean(r["correctness"] == 2 for r in judged),
            "correctness": mean(r["correctness"] / 2 for r in judged),
            "faithful": mean(r.get("faithful") for r in judged),
            "calc_match": mean(r["calc_match"] for r in rows if "calc_match" in r),
            "refusal_accuracy": mean(r.get("correct_behaviour") for r in unans),
            "cites_gold": mean(r["cites_gold"] for r in rows if "cites_gold" in r),
            "invalid_citation": mean(r["invalid_citation"] for r in rows if "invalid_citation" in r),
            "no_citation": mean(not r["cited"] for r in rows if r["answerable"] and "cited" in r),
            # answerable questions where the system said it couldn't find the answer
            "over_refusal": mean(is_refusal(r["answer"]) for r in rows if r["answerable"] and "answer" in r),
            "judge_errors": sum("judge_error" in r for r in rows),
        }
        s["latency_ms"] = {
            "retrieve_p50": percentile([r["retrieve_ms"] for r in rows], 50),
            "generate_p50": percentile([r.get("generate_ms") for r in rows], 50),
            "generate_p95": percentile([r.get("generate_ms") for r in rows], 95),
        }
        s["tokens"] = {"prompt": sum(r.get("prompt_tokens") or 0 for r in rows),
                       "output": sum(r.get("output_tokens") or 0 for r in rows)}
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    s["by_type"] = {
        t: {"n": len(rs), "hit@5": mean(r.get("hit@5") for r in rs if "hit@5" in r),
            "accuracy": mean(r["correctness"] == 2 for r in rs if r.get("correctness") is not None),
            "refusal_accuracy": mean(r.get("correct_behaviour") for r in rs if not r["answerable"])}
        for t, rs in sorted(by_type.items())
    }
    return s


def pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def write_results_table() -> None:
    """RESULTS.md: one row per experiment, rebuilt from every summary.json."""
    summaries = sorted((json.loads(p.read_text(encoding="utf-8")) for p in RESULTS.glob("*/summary.json")),
                       key=lambda s: s["run_at"])
    lines = [
        "# Experiment results",
        "",
        "Generated by `python -m evals.run`. Retrieval metrics are over answerable questions at section level;",
        "Accuracy = judge score 2/2; Refusal = correct behaviour on out-of-scope questions;",
        "Over-refusal = answerable questions where the system said it couldn't find the answer (lower is better).",
        "",
        "| Experiment | Dataset (n) | Hit@1 | Hit@5 | Recall@5 | MRR@10 | Accuracy | Faithful | Calc | Refusal | "
        "Over-refusal | Cites gold | Gen p50 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        r, a, lat = s["retrieval"], s.get("answers", {}), s.get("latency_ms", {})
        mrr = "–" if r["mrr@10"] is None else f"{r['mrr@10']:.2f}"
        gen = "–" if lat.get("generate_p50") is None else f"{lat['generate_p50'] / 1000:.1f}s"
        name = s["name"] + (" *(partial)*" if s.get("partial") else "")
        lines.append(
            f"| {name} | {s['dataset']} ({s['n_questions']}) | {pct(r['hit@1'])} | {pct(r['hit@5'])} | "
            f"{pct(r['recall@5'])} | {mrr} | {pct(a.get('accuracy'))} | {pct(a.get('faithful'))} | "
            f"{pct(a.get('calc_match'))} | {pct(a.get('refusal_accuracy'))} | {pct(a.get('over_refusal'))} | "
            f"{pct(a.get('cites_gold'))} | {gen} |"
        )
    lines += ["", "Descriptions:", ""] + [f"- **{s['name']}**: {s['description']}" for s in summaries]
    lines += judge_check_lines(summaries)
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def judge_check_lines(summaries: list[dict]) -> list[str]:
    """Human-vs-judge agreement for runs that have a blind spot-check (evals.spotcheck)."""
    from evals.spotcheck import kappa  # local import: spotcheck imports this module

    out = []
    for s in summaries:
        grades = read_jsonl(RESULTS / s["name"] / "spotcheck.jsonl")
        ans = [(g["you"], g["judge"]) for g in grades if g["answerable"]]
        oos = [(g["you"], g["judge"]) for g in grades if not g["answerable"]]
        if not ans:
            continue
        k = kappa(ans)
        exact = sum(a == b for a, b in ans) / len(ans)
        line = (f"- **{s['name']}**: {len(grades)} answers graded blind by a human. Answerable: exact agreement "
                f"{exact:.0%}" + (f", Cohen's κ {k:.2f}" if k is not None else ""))
        if oos:
            line += f". Out-of-scope: agreement {sum(a == b for a, b in oos) / len(oos):.0%}"
        out.append(line + ".")
    return ["", "Judge check (how far to trust the Accuracy column):", ""] + out if out else []


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--retrieval-only", action="store_true", help="no LLM calls")
    parser.add_argument("--no-judge", action="store_true", help="generate answers but skip the judge")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    golden = read_jsonl(GOLDEN)
    if not golden:
        raise SystemExit("No golden set: run python -m evals.review --export first.")
    dataset_id = hashlib.sha1(GOLDEN.read_bytes()).hexdigest()[:8]
    if args.limit:
        golden = golden[: args.limit]

    from src.rag import RAGPipeline
    from src.retrieval.embed import Embedder
    from src.retrieval.store import QdrantStore

    meta = json.loads((PROCESSED / f"index_{cfg['collection']}.json").read_text(encoding="utf-8"))
    embedder = Embedder(meta["model"])
    store = QdrantStore(cfg["collection"])
    rag = judge = None
    if not args.retrieval_only:
        from src.generation.llm import GeminiLLM

        interval = cfg.get("min_seconds_between_calls", 0)
        rag = RAGPipeline(embedder, store, CachedLLM(GeminiLLM(models=cfg["generator_models"]), interval),
                          k=cfg["top_k"])
        if not args.no_judge:
            judge = CachedLLM(GeminiLLM(models=cfg["judge_models"]), interval)

    out_dir = RESULTS / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Running '{cfg['name']}' on {len(golden)} questions (dataset {dataset_id})")
    for i, item in enumerate(golden, 1):
        try:
            row = evaluate_item(item, cfg, embedder, store, rag, judge)
        except Exception as e:  # keep going; the failure is recorded in the results
            row = {"id": item["id"], "type": item["type"], "answerable": item["answerable"],
                   "question": item["question"], "error": f"{type(e).__name__}: {str(e)[:300]}"}
        rows.append(row)
        write_jsonl(out_dir / "per_question.jsonl", rows)
        verdict = (f"score={row.get('correctness')}" if item["answerable"]
                   else f"behaviour={'ok' if row.get('correct_behaviour') else row.get('correct_behaviour')}")
        print(f"[{i:3}/{len(golden)}] {item['id']:10} hit@5={row.get('hit@5', '-')!s:5} {verdict}"
              + (f"  ERROR {row['error']}" if "error" in row else ""))

    ok_rows = [r for r in rows if "error" not in r]
    summary = summarize(ok_rows, cfg, dataset_id, args)
    summary["errors"] = len(rows) - len(ok_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_results_table()
    store.close()

    r, a = summary["retrieval"], summary.get("answers", {})
    print(f"\nRetrieval: Hit@1 {pct(r['hit@1'])}  Hit@5 {pct(r['hit@5'])}  Recall@5 {pct(r['recall@5'])}  "
          f"MRR@10 {r['mrr@10']:.2f}")
    if a:
        print(f"Answers:   accuracy {pct(a['accuracy'])}  faithful {pct(a['faithful'])}  calc {pct(a['calc_match'])}  "
              f"refusal {pct(a['refusal_accuracy'])}  cites gold {pct(a['cites_gold'])}  judge errors {a['judge_errors']}")
    print(f"Errors: {summary['errors']}. Results in {out_dir.relative_to(EVALS.parent)}; table in evals/results/RESULTS.md")


if __name__ == "__main__":
    main()
