"""Ask LabourLens a question from the command line.

Usage (from backend/):
    python -m src.ask "Am I eligible for gratuity on a fixed-term contract?"
    python -m src.ask "..." --retrieve-only        # show retrieved passages, no LLM call
    python -m src.ask "..." --source social_security_code_2020
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import DEFAULT_COLLECTION, DEFAULT_TOP_K, PROCESSED
from src.rag import RAGPipeline
from src.retrieval.embed import Embedder
from src.retrieval.store import QdrantStore


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--source", action="append", help="restrict to a source_id (repeatable)")
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    meta = json.loads((PROCESSED / f"index_{args.collection}.json").read_text(encoding="utf-8"))
    llm = None
    if not args.retrieve_only:
        from src.generation.llm import GeminiLLM
        llm = GeminiLLM()
    store = QdrantStore(args.collection)
    rag = RAGPipeline(Embedder(meta["model"]), store, llm, k=args.k)
    filters = {"source_id": args.source} if args.source else None

    if args.retrieve_only:
        hits, answer = rag.retrieve(args.question, filters), None
    else:
        answer = rag.answer(args.question, filters)
        hits = answer.hits
        print(f"\n{answer.text}\n")
        print(f"[{answer.model}; retrieve {answer.timings_ms['retrieve']:.0f} ms, "
              f"generate {answer.timings_ms['generate']:.0f} ms]\n")

    print("Sources:")
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"  [{i}] {h.score:.3f}  {p['source_title']}: {', '.join(p['labels'][:5])}  (pp. {p['pages']})")
        if args.retrieve_only:
            print("      " + p["text"][:300].replace("\n", " ") + " ...")
    store.close()


if __name__ == "__main__":
    main()
