"""Vector store interface and its Qdrant implementation.

The rest of the code depends only on `VectorStore`, so another backend (pgvector,
Elasticsearch, ...) can be added as one new class without touching the pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import QDRANT_API_KEY, QDRANT_PATH, QDRANT_URL


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    def recreate(self, dim: int) -> None: ...
    def add(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...
    def search(self, vector: np.ndarray, k: int, filters: dict | None = None) -> list[SearchHit]: ...
    def count(self) -> int: ...


class QdrantStore:
    """Qdrant collection. Defaults to local mode (data/qdrant); set QDRANT_URL for a server."""

    def __init__(self, collection: str, location: str | None = None):
        self.collection = collection
        if location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif location or QDRANT_URL:
            self.client = QdrantClient(url=location or QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(QDRANT_PATH))

    def recreate(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def add(self, ids: list[str], vectors: np.ndarray, payloads: list[dict], batch: int = 256) -> None:
        for i in range(0, len(ids), batch):
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, cid)),  # Qdrant ids must be int/UUID
                        vector=vec.tolist(),
                        payload={**payload, "chunk_id": cid},
                    )
                    for cid, vec, payload in zip(ids[i : i + batch], vectors[i : i + batch], payloads[i : i + batch])
                ],
            )

    def search(self, vector: np.ndarray, k: int, filters: dict | None = None) -> list[SearchHit]:
        result = self.client.query_points(
            self.collection, query=vector.tolist(), limit=k, query_filter=_to_filter(filters), with_payload=True
        )
        return [SearchHit(p.payload["chunk_id"], p.score, p.payload) for p in result.points]

    def count(self) -> int:
        return self.client.count(self.collection).count

    def close(self) -> None:
        self.client.close()


def _to_filter(filters: dict | None) -> models.Filter | None:
    """{"source_id": "x"} -> exact match; {"source_id": ["x", "y"]} -> any of."""
    if not filters:
        return None
    must = []
    for key, value in filters.items():
        match = models.MatchAny(any=list(value)) if isinstance(value, (list, tuple, set)) else models.MatchValue(value=value)
        must.append(models.FieldCondition(key=key, match=match))
    return models.Filter(must=must)
