"""Baseline prompt: answer only from the retrieved passages and cite them by number."""

from __future__ import annotations

from src.retrieval.store import SearchHit

SYSTEM = """You answer questions about India's labour codes and the central rules made under them.
Use ONLY the numbered passages provided. Cite the passages you rely on like [1] or [2][3].
If the passages do not contain the answer, say you could not find it in the provided texts.
Be concise and use plain language. This is general information, not legal advice."""


def format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        p = hit.payload
        where = ", ".join(p.get("labels", [])[:6]) or p.get("source_title", "")
        blocks.append(f"[{i}] {p.get('source_title', '')} ({where})\n{p['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    return f"Passages:\n\n{format_context(hits)}\n\nQuestion: {question}\n\nAnswer:"
