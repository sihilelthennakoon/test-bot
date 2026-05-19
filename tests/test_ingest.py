from pathlib import Path

from ragbot.embeddings.providers import HashEmbeddingProvider
from ragbot.ingestion.ingest import IngestionService
from ragbot.safety.pii import PIIMasker
from ragbot.vectorstore.faiss_store import FaissVectorStore


def test_ingestion_builds_index(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Reach us at test@example.com for support.")
    store = FaissVectorStore.load_or_create(HashEmbeddingProvider(dimension=64), tmp_path / "index")
    service = IngestionService(store=store, pii_masker=PIIMasker())

    response = service.ingest_file(source)

    assert response.chunk_count >= 1
    assert Path(response.index_path).exists()
