"""Sentence-transformers embedding wrapper (CPU)."""

from __future__ import annotations

import numpy as np

from src.config import EMBED_MODEL

# BGE v1.5 models expect this instruction on queries (not on passages).
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # slow import; keep it lazy

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = QUERY_PREFIXES.get(model_name, "")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.max_tokens = self.model.max_seq_length

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token, using the model's own tokenizer."""
        tok = self.model.tokenizer
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        return [tuple(o) for o in enc["offset_mapping"]]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 100
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + query, normalize_embeddings=True)
