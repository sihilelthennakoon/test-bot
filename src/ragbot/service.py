from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from ragbot.config import Settings, get_settings
from ragbot.embeddings.providers import (
    GeminiEmbeddingProvider,
    HashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from ragbot.evaluations.runner import EvaluationRunner
from ragbot.graph.chat_graph import ChatRuntime, build_chat_app
from ragbot.ingestion.ingest import IngestionService
from ragbot.llm.gemini import GeminiAnswerer
from ragbot.observability import bootstrap_phoenix
from ragbot.rca.service import RCATraceService
from ragbot.safety.guardrails import GuardrailEngine
from ragbot.safety.pii import PIIMasker
from ragbot.vectorstore.faiss_store import FaissVectorStore


def build_embedding_provider(settings: Settings):
    if settings.embedding_provider in {"sentence-transformer", "sentence_transformers", "sentence-transformer-model"}:
        try:
            return SentenceTransformerEmbeddingProvider(settings.sentence_transformer_model)
        except Exception:
            return HashEmbeddingProvider()
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
    rca: RCATraceService
    chat_app: Any

    @classmethod
    def create(cls, settings: Settings | None = None) -> "ChatService":
        settings = settings or get_settings()
        bootstrap_phoenix(
            project_name=settings.phoenix_project_name,
            collector_endpoint=settings.phoenix_collector_endpoint,
        )
        embedding_provider = build_embedding_provider(settings)
        store = FaissVectorStore.load_or_create(embedding_provider, settings.index_dir)

        # Initialize evaluator runner with default evaluators
        evaluator_runner = EvaluationRunner()

        runtime = ChatRuntime(
            store=store,
            answerer=GeminiAnswerer(settings.gemini_model, api_key=os.getenv("GOOGLE_API_KEY")),
            pii_masker=PIIMasker(),
            guardrails=GuardrailEngine(max_chars=settings.max_input_chars),
            evaluator_runner=evaluator_runner,
        )
        ingestion = IngestionService(store=store, pii_masker=PIIMasker())
        rca = RCATraceService.from_settings(settings)
        chat_app = build_chat_app(runtime, settings=settings)
        return cls(
            settings=settings,
            store=store,
            runtime=runtime,
            ingestion=ingestion,
            rca=rca,
            chat_app=chat_app,
        )

    def build_app(self):
        return self.chat_app

    def build_invocation_config(self, conversation_id: str | None = None) -> dict[str, Any]:
        return {
            "metadata": {
                "conversation_id": conversation_id,
                "session_id": conversation_id,
                "session_identifier": conversation_id,
                "environment": self.settings.environment,
                "app_version": self.settings.app_version,
                "use_case": self.settings.use_case,
            },
            "tags": [],
        }
