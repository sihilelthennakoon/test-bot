from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Sequence

import numpy as np

from ragbot.embeddings.providers import EmbeddingProvider
from ragbot.schemas import DocumentChunk, RetrievalHit

try:
    import faiss
except Exception:  # pragma: no cover - optional native dependency
    faiss = None


class _NumpyIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch.")
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int):
        similarities = query_vector @ self._vectors.T
        if similarities.size == 0:
            scores = np.empty((1, 0), dtype=np.float32)
            indices = np.empty((1, 0), dtype=np.int64)
            return scores, indices
        indices = np.argsort(-similarities, axis=1)[:, :top_k]
        scores = np.take_along_axis(similarities, indices, axis=1)
        return scores, indices

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors


def _create_index(dimension: int):
    if faiss is not None:
        return faiss.IndexFlatIP(dimension)
    return _NumpyIndex(dimension)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


@dataclass(slots=True)
class FaissVectorStore:
    embedding_provider: EmbeddingProvider
    index_dir: Path
    _index: faiss.Index | None = None
    _records: list[DocumentChunk] = field(default_factory=list)

    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "records.json"

    @classmethod
    def load_or_create(cls, embedding_provider: EmbeddingProvider, index_dir: Path) -> "FaissVectorStore":
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "records.json"
        if index_path.exists() and metadata_path.exists():
            if faiss is not None:
                index = faiss.read_index(str(index_path))
            else:
                payload = json.loads(index_path.read_text())
                index = _NumpyIndex(payload["dimension"])
                if payload["vectors"]:
                    index.add(np.array(payload["vectors"], dtype=np.float32))
            records = [DocumentChunk.model_validate(item) for item in json.loads(metadata_path.read_text())]
            return cls(embedding_provider=embedding_provider, index_dir=index_dir, _index=index, _records=records)
        return cls(embedding_provider=embedding_provider, index_dir=index_dir)

    def clear(self) -> None:
        self._index = None
        self._records = []

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
        matrix = np.array(embeddings, dtype=np.float32)
        matrix = _normalize_rows(matrix)
        if self._index is None:
            self._index = _create_index(matrix.shape[1])
        self._index.add(matrix)
        self._records.extend(chunks)

    def search(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
        if self._index is None or not self._records:
            return []
        query_vector = np.array([self.embedding_provider.embed_query(query)], dtype=np.float32)
        query_vector = _normalize_rows(query_vector)
        scores, indices = self._index.search(query_vector, top_k)
        hits: list[RetrievalHit] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0 or index >= len(self._records):
                continue
            record = self._records[index]
            hits.append(
                RetrievalHit(
                    chunk_id=record.chunk_id,
                    source_path=record.source_path,
                    text=record.text,
                    score=float(score),
                    chunk_index=record.chunk_index,
                    metadata=record.metadata,
                )
            )
        return hits

    def persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            if faiss is not None:
                faiss.write_index(self._index, str(self.index_path))
            else:
                self.index_path.write_text(
                    json.dumps(
                        {
                            "dimension": self._index.dimension,
                            "vectors": self._index.vectors.tolist(),
                        },
                        indent=2,
                    )
                )
        self.metadata_path.write_text(json.dumps([record.model_dump() for record in self._records], indent=2))
