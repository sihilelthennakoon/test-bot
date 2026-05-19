from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragbot.safety.pii import PIIMasker
from ragbot.schemas import DocumentChunk, IngestResponse, new_id
from ragbot.vectorstore.faiss_store import FaissVectorStore


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    if not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end].strip())
        if end >= text_length:
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


@dataclass(slots=True)
class IngestionService:
    store: FaissVectorStore
    pii_masker: PIIMasker

    def ingest_file(self, source_path: str | Path, *, rebuild: bool = True) -> IngestResponse:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        text = path.read_text(encoding="utf-8")
        raw_chunks = split_text(text)
        chunks: list[DocumentChunk] = []
        masked_chunk_count = 0
        for chunk_index, chunk_text in enumerate(raw_chunks):
            masked_result = self.pii_masker.mask(chunk_text)
            masked_chunk_count += int(masked_result.was_masked)
            chunks.append(
                DocumentChunk(
                    chunk_id=new_id("chunk"),
                    source_path=str(path),
                    text=masked_result.text,
                    chunk_index=chunk_index,
                    metadata={
                        "source_path": str(path),
                        "was_masked": masked_result.was_masked,
                        "pii_entities": masked_result.entities,
                    },
                )
            )

        if rebuild:
            self.store.clear()
        self.store.add_chunks(chunks)
        self.store.persist()
        return IngestResponse(
            source_path=str(path),
            chunk_count=len(chunks),
            index_path=str(self.store.index_path),
            masked_chunk_count=masked_chunk_count,
        )

    def ingest_directory(self, source_dir: str | Path, *, rebuild: bool = True) -> list[IngestResponse]:
        directory = Path(source_dir)
        responses: list[IngestResponse] = []
        for file_path in sorted(directory.glob("*.txt")):
            responses.append(self.ingest_file(file_path, rebuild=rebuild))
            rebuild = False
        return responses


def summarize_ingestion(responses: list[IngestResponse]) -> dict[str, int]:
    return {
        "files": len(responses),
        "chunks": sum(response.chunk_count for response in responses),
        "masked_chunks": sum(response.masked_chunk_count for response in responses),
    }
