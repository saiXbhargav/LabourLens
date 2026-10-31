import re

import numpy as np
import pytest

from src.generation.prompt import build_prompt
from src.rag import RAGPipeline
from src.retrieval.chunking import fixed_size_chunks
from src.retrieval.store import QdrantStore, SearchHit


def word_offsets(text):
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def section(sid, number, text, source="doc", pages=(1, 1)):
    return {"id": f"{source}:s{number}", "source_id": source, "source_title": "Doc", "label": f"Doc §{number}",
            "text": text, "pages": list(pages), "url": "http://x"}


# ---------------------------------------------------------------- chunking


def test_chunks_respect_size_and_overlap():
    words = [f"w{i}" for i in range(25)]
    chunks = fixed_size_chunks([section("a", 1, " ".join(words))], word_offsets, size=10, overlap=3)
    toks = [c.text.split() for c in chunks]
    assert all(len(t) <= 10 for t in toks)
    assert toks[0][-3:] == toks[1][:3]  # 3-token overlap
    assert toks[-1][-1] == "w24"  # nothing lost at the end
    # windows start at 0, 7, 14, 21; the third ends at w23, so a fourth is needed for w24
    assert [c.id for c in chunks] == ["doc:c0", "doc:c1", "doc:c2", "doc:c3"]
    assert toks[-1] == ["w21", "w22", "w23", "w24"]


def test_no_redundant_last_chunk():
    chunks = fixed_size_chunks([section("a", 1, " ".join(f"w{i}" for i in range(24)))], word_offsets, 10, 3)
    assert len(chunks) == 3  # 0-9, 7-16, 14-23 already cover everything


def test_chunks_record_the_sections_they_span():
    secs = [section("a", 1, "one two three four", pages=(1, 1)),
            section("b", 2, "five six seven eight", pages=(2, 3))]
    chunks = fixed_size_chunks(secs, word_offsets, size=6, overlap=0)
    assert chunks[0].section_ids == ["doc:s1", "doc:s2"]
    assert chunks[0].labels == ["Doc §1", "Doc §2"]
    assert chunks[0].pages == [1, 3]
    assert chunks[1].section_ids == ["doc:s2"]


def test_chunks_never_cross_documents():
    secs = [section("a", 1, "alpha beta", source="d1"), section("b", 1, "gamma delta", source="d2")]
    chunks = fixed_size_chunks(secs, word_offsets, size=10, overlap=2)
    assert [(c.source_id, c.text) for c in chunks] == [("d1", "alpha beta"), ("d2", "gamma delta")]


def test_short_document_gives_one_chunk():
    chunks = fixed_size_chunks([section("a", 1, "just three words")], word_offsets, size=500, overlap=50)
    assert len(chunks) == 1 and chunks[0].n_tokens == 3


def test_invalid_overlap():
    with pytest.raises(ValueError):
        fixed_size_chunks([], word_offsets, size=10, overlap=10)


# ---------------------------------------------------------------- store


@pytest.fixture
def store():
    s = QdrantStore("test", location=":memory:")
    s.recreate(dim=3)
    s.add(
        ["a:c0", "b:c0", "b:c1"],
        np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=float),
        [{"source_id": "a", "text": "A"}, {"source_id": "b", "text": "B"}, {"source_id": "b", "text": "B2"}],
    )
    return s


def test_store_returns_nearest_first(store):
    hits = store.search(np.array([1.0, 0, 0]), k=2)
    assert [h.id for h in hits] == ["a:c0", "b:c1"]
    assert hits[0].score > hits[1].score
    assert store.count() == 3


def test_store_filters(store):
    assert [h.id for h in store.search(np.array([1.0, 0, 0]), k=3, filters={"source_id": "b"})] == ["b:c1", "b:c0"]
    assert len(store.search(np.array([1.0, 0, 0]), k=3, filters={"source_id": ["a", "b"]})) == 3


def test_recreate_empties_collection(store):
    store.recreate(dim=3)
    assert store.count() == 0


# ---------------------------------------------------------------- pipeline


class FakeEmbedder:
    def embed_query(self, q):
        return np.array([1.0, 0, 0])


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system=None):
        self.prompts.append((prompt, system))
        return "Answer [1]", "fake-model"


def test_pipeline_answers_from_retrieved_passages(store):
    llm = FakeLLM()
    answer = RAGPipeline(FakeEmbedder(), store, llm, k=2).answer("What is A?")
    assert answer.text == "Answer [1]" and answer.model == "fake-model"
    assert [h.id for h in answer.hits] == ["a:c0", "b:c1"]
    prompt, system = llm.prompts[0]
    assert "[1]" in prompt and "What is A?" in prompt and "ONLY" in system


def test_prompt_numbers_passages_with_their_labels():
    hits = [SearchHit("x", 0.9, {"source_title": "SS Code", "labels": ["SS Code §53"], "text": "Gratuity..."})]
    assert build_prompt("q", hits).startswith("Passages:\n\n[1] SS Code (SS Code §53)\nGratuity...")
