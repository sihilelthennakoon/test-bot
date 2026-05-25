from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
import math
from typing import Sequence

import numpy as np


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        values = []
        while len(values) < self._dimension:
            for byte_value in digest:
                centered = (byte_value / 255.0) * 2.0 - 1.0
                values.append(centered)
                if len(values) >= self._dimension:
                    break
            digest = sha256(digest).digest()
        vector = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if not math.isfinite(float(norm)) or norm == 0.0:
            return [0.0] * self._dimension
        return (vector / norm).tolist()


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    @property
    def dimension(self) -> int:
        return 768

    def _client_instance(self):
        if self._client is not None:
            return self._client
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Gemini embeddings are unavailable. Install langchain-google-genai and set GOOGLE_API_KEY."
            ) from exc

        self._client = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key,
        )
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client_instance().embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client_instance().embed_query(text)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._client = None
        self._fallback = HashEmbeddingProvider()

    @property
    def dimension(self) -> int:
        return self._fallback.dimension

    def _client_instance(self):
        if self._client is not None:
            return self._client
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SentenceTransformer embeddings are unavailable. Install sentence-transformers."
            ) from exc

        try:
            self._client = SentenceTransformer(self.model_name)
        except Exception:
            self._client = self._fallback
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._client_instance()
        if isinstance(client, HashEmbeddingProvider):
            return client.embed_documents(texts)
        try:
            vectors = client.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.astype(np.float32).tolist()
        except Exception:
            self._client = self._fallback
            return self._fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        client = self._client_instance()
        if isinstance(client, HashEmbeddingProvider):
            return client.embed_query(text)
        try:
            vector = client.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vector.astype(np.float32).tolist()
        except Exception:
            self._client = self._fallback
            return self._fallback.embed_query(text)
