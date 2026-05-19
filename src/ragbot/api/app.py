from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ragbot.config import get_settings
from ragbot.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from ragbot.service import ChatService


def create_app() -> FastAPI:
    settings = get_settings()
    service = ChatService.create(settings)
    app = FastAPI(title=settings.project_name, version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        return service.ingestion.ingest_file(request.source_path, rebuild=request.rebuild)

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
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
