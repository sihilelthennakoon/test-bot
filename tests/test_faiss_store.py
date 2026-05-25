from pathlib import Path
from types import ModuleType
import sys

from ragbot.embeddings.providers import HashEmbeddingProvider, SentenceTransformerEmbeddingProvider
from ragbot.schemas import DocumentChunk
from ragbot.vectorstore.faiss_store import FaissVectorStore


def test_faiss_store_round_trip(tmp_path: Path) -> None:
    provider = HashEmbeddingProvider(dimension=64)
    store = FaissVectorStore.load_or_create(provider, tmp_path)
    store.add_chunks(
        [
            DocumentChunk(chunk_id="chunk_1", source_path="doc.txt", text="hotline is 555-123-4567", chunk_index=0),
            DocumentChunk(chunk_id="chunk_2", source_path="doc.txt", text="contact support@acme.example", chunk_index=1),
        ]
    )
    store.persist()

    loaded = FaissVectorStore.load_or_create(provider, tmp_path)
    hits = loaded.search("What is the hotline?", top_k=1)

    assert hits
    assert hits[0].chunk_id in {"chunk_1", "chunk_2"}


def test_sentence_transformer_provider_falls_back_to_hash(monkeypatch) -> None:
    fake_module = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("model download failed")

    fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    provider = SentenceTransformerEmbeddingProvider()

    vector = provider.embed_query("hello world")

    assert len(vector) == 384
    assert any(value != 0.0 for value in vector)
