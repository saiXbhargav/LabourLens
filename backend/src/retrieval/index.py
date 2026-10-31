"""Build a vector index from data/processed/sections.jsonl.

Usage (from backend/):
    python -m src.retrieval.index                         # baseline: 500-token chunks, 50 overlap
    python -m src.retrieval.index --size 300 --overlap 30 --collection fixed300
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from src.config import DEFAULT_COLLECTION, EMBED_MODEL, PROCESSED
from src.retrieval.chunking import fixed_size_chunks
from src.retrieval.embed import Embedder
from src.retrieval.store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--size", type=int, default=500, help="chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=50, help="overlap in tokens")
    parser.add_argument("--model", default=EMBED_MODEL)
    args = parser.parse_args()

    sections = [json.loads(l) for l in (PROCESSED / "sections.jsonl").open(encoding="utf-8")]
    t0 = time.perf_counter()
    embedder = Embedder(args.model)
    if args.size + 2 > embedder.max_tokens:  # +2 for the [CLS]/[SEP] tokens
        raise SystemExit(f"--size {args.size} exceeds {args.model}'s limit of {embedder.max_tokens} tokens")

    store = QdrantStore(args.collection)
    store.recreate(embedder.dim)  # connects now, so a bad store config fails before the slow embedding

    chunks = fixed_size_chunks(sections, embedder.token_offsets, args.size, args.overlap)
    print(f"{len(sections)} sections -> {len(chunks)} chunks "
          f"(median {statistics.median(c.n_tokens for c in chunks):.0f} tokens)")

    out = PROCESSED / f"chunks_{args.collection}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    # Embeddings are the slow part (minutes on CPU), so save them before touching the store.
    vectors = embedder.embed_documents([c.text for c in chunks])
    np.save(PROCESSED / f"vectors_{args.collection}.npy", vectors)

    store.add([c.id for c in chunks], vectors, [c.to_dict() for c in chunks])

    meta = {"collection": args.collection, "model": args.model, "size": args.size,
            "overlap": args.overlap, "chunks": len(chunks), "dim": embedder.dim}
    (PROCESSED / f"index_{args.collection}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Indexed {store.count()} chunks into '{args.collection}' in {time.perf_counter() - t0:.0f}s; "
          f"chunks saved to {out.name}")
    store.close()


if __name__ == "__main__":
    main()
