from pathlib import Path

from ragbot.embeddings.providers import HashEmbeddingProvider
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
