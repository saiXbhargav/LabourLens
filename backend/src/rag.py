"""The question-answering pipeline: retrieve passages, then ask the LLM to answer from them."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config import DEFAULT_TOP_K
from src.generation.llm import LLM
from src.generation.prompt import SYSTEM, build_prompt
from src.retrieval.store import SearchHit, VectorStore


class QueryEmbedder:  # the part of Embedder the pipeline needs
    def embed_query(self, query: str): ...


@dataclass
class Answer:
    question: str
    text: str
    hits: list[SearchHit]
    model: str | None
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, embedder: QueryEmbedder, store: VectorStore, llm: LLM | None, k: int = DEFAULT_TOP_K):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.k = k

    def retrieve(self, question: str, filters: dict | None = None) -> list[SearchHit]:
        return self.store.search(self.embedder.embed_query(question), self.k, filters)

    def answer(self, question: str, filters: dict | None = None) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question, filters)
        answer = self.generate(question, hits)
        answer.timings_ms["retrieve"] = (time.perf_counter() - t0) * 1000 - answer.timings_ms["generate"]
        return answer

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        """Answer from already-retrieved passages (lets the eval reuse one deeper retrieval)."""
        if self.llm is None:
            raise RuntimeError("no LLM configured; use retrieve() for retrieval only")
        t0 = time.perf_counter()
        text, model = self.llm.generate(build_prompt(question, hits), system=SYSTEM)
        return Answer(question, text, hits, model, {"generate": (time.perf_counter() - t0) * 1000})
