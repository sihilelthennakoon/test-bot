from __future__ import annotations

from dataclasses import dataclass
import os

from ragbot.config import Settings, get_settings
from ragbot.embeddings.providers import GeminiEmbeddingProvider, HashEmbeddingProvider
from ragbot.graph.chat_graph import ChatRuntime, build_chat_app
from ragbot.ingestion.ingest import IngestionService
from ragbot.llm.gemini import GeminiAnswerer
from ragbot.observability.phoenix import create_trace_recorder
from ragbot.safety.guardrails import GuardrailEngine
from ragbot.safety.pii import PIIMasker
from ragbot.vectorstore.faiss_store import FaissVectorStore


def build_embedding_provider(settings: Settings):
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return GeminiEmbeddingProvider(settings.gemini_embedding_model, api_key=api_key)
    return HashEmbeddingProvider()


@dataclass(slots=True)
class ChatService:
    settings: Settings
    store: FaissVectorStore
    runtime: ChatRuntime
    ingestion: IngestionService

    @classmethod
    def create(cls, settings: Settings | None = None) -> "ChatService":
        settings = settings or get_settings()
        embedding_provider = build_embedding_provider(settings)
        store = FaissVectorStore.load_or_create(embedding_provider, settings.index_dir)
        tracer = create_trace_recorder(
            project_name=settings.phoenix_project_name,
            collector_endpoint=settings.phoenix_collector_endpoint,
            evaluation_path=settings.index_dir / "evaluations.jsonl",
        )
        import os

        runtime = ChatRuntime(
            store=store,
            answerer=GeminiAnswerer(settings.gemini_model, api_key=os.getenv("GOOGLE_API_KEY")),
            pii_masker=PIIMasker(),
            guardrails=GuardrailEngine(max_chars=settings.max_input_chars),
            tracer=tracer,
        )
        ingestion = IngestionService(store=store, pii_masker=PIIMasker())
        return cls(settings=settings, store=store, runtime=runtime, ingestion=ingestion)

    def build_app(self):
        return build_chat_app(self.runtime)
