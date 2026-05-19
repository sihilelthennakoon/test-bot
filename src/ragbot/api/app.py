from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ragbot.config import get_settings
from ragbot.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from ragbot.service import ChatService


def create_app() -> FastAPI:
    settings = get_settings()
    service = ChatService.create(settings)
    app = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        description="Enterprise RAG chatbot with LangGraph, FAISS, and Phoenix observability",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/status")
    def status() -> dict[str, str | int]:
        """Service status endpoint with index information."""
        store = service.store
        chunk_count = len(store._records) if store._records else 0
        return {
            "status": "ready",
            "project": settings.project_name,
            "chunks_indexed": chunk_count,
            "index_path": str(store.index_path),
        }

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        """Ingest a text file and add it to the FAISS index.
        
        Args:
            source_path: Path to the text file to ingest
            rebuild: If true, clear the index before adding new chunks
        """
        return service.ingestion.ingest_file(request.source_path, rebuild=request.rebuild)

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """Send a chat message to the RAG chatbot.
        
        The message is processed through:
        1. Input guardrails check (prompt injection, length)
        2. PII masking
        3. Semantic retrieval from FAISS
        4. Generation via Gemini
        5. Output guardrails check
        6. Output PII masking
        
        Args:
            message: The user's question or statement
            conversation_id: Optional conversation tracking ID
            top_k: Number of retrieval results (1-12, default 4)
        """
        try:
            return service.runtime.run(
                request.message,
                top_k=request.top_k,
                conversation_id=request.conversation_id,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
