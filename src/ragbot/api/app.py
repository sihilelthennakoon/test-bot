from __future__ import annotations

from contextlib import asynccontextmanager
import json
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from data_fetch.fetch_from_phoenix import (
    build_default_adapter,
    fetch_phoenix_spans_dataframe,
    RawSpansDataFrameRequest,
)
from ragbot.config import get_settings
from ragbot.evaluations.batch_evaluation.evaluate_batch import (
    BatchEvaluationConfig,
    resolve_batch_evaluation_config,
    run_span_batch,
    write_last_eval_timestamp,
)
from ragbot.evaluations.batch_evaluation.scheduler import BatchEvaluationScheduler
from ragbot.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from ragbot.service import ChatService


class RawSpansDataFrameResponse(BaseModel):
    columns: list[str]
    row_count: int
    rows: list[dict[str, Any]]


class BatchEvaluationRunRequest(BaseModel):
    from_time: datetime | None = None
    to_time: datetime | None = None
    project_name: str | None = None
    span_kind: str | None = None
    limit: int = Field(default=1000, ge=1)
    root_spans_only: bool | None = None
    sync_annotations: bool = True
    save_annotations: bool = True


class BatchEvaluationRunResponse(BaseModel):
    span_count: int
    evaluated_count: int
    annotation_count: int
    annotations_saved: bool


RawSpansDataFrameResponse.model_rebuild()
BatchEvaluationRunRequest.model_rebuild()
BatchEvaluationRunResponse.model_rebuild()


def _dataframe_to_rows(dataframe) -> list[dict[str, Any]]:
    if dataframe.empty:
        return []
    return json.loads(dataframe.to_json(orient="records", date_format="iso"))


def create_app() -> FastAPI:
    settings = get_settings()
    service = ChatService.create(settings)
    scheduler = (
        BatchEvaluationScheduler(
            config=BatchEvaluationConfig(
                project_name=settings.phoenix_project_name,
                span_kind=settings.phoenix_fetch_span_kind,
                limit=settings.batch_evaluation_cron_limit,
                phoenix_base_url=settings.phoenix_query_endpoint,
                sync_annotations=settings.batch_evaluation_cron_sync_annotations,
                save_annotations=settings.batch_evaluation_cron_save_annotations,
                use_last_eval_timestamp=settings.batch_evaluation_use_last_eval_timestamp,
                last_eval_timestamp_file=settings.last_eval_timestamp_file,
            ),
            interval_seconds=settings.batch_evaluation_cron_interval_seconds,
        )
        if settings.batch_evaluation_cron_enabled
        else None
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if scheduler is not None:
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.stop()

    app = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        description="Enterprise RAG chatbot with LangGraph and FAISS",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
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
            result = service.chat_app.invoke(
                {
                    "message": request.message,
                    "top_k": request.top_k,
                    "conversation_id": request.conversation_id,
                },
                config=service.build_invocation_config(request.conversation_id),
            )
            return ChatResponse.model_validate(result)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/phoenix/spans/raw", response_model=RawSpansDataFrameResponse)
    def fetch_raw_phoenix_spans(request: RawSpansDataFrameRequest) -> RawSpansDataFrameResponse:
        """Fetch raw Phoenix spans as DataFrame-shaped JSON."""
        try:
            adapter = build_default_adapter(settings.phoenix_query_endpoint)
            spans_df = fetch_phoenix_spans_dataframe(
                RawSpansDataFrameRequest(
                    from_time=request.from_time,
                    to_time=request.to_time,
                    project_name=request.project_name or settings.phoenix_project_name,
                    limit=request.limit,
                    root_spans_only=request.root_spans_only,
                ),
                adapter,
            )
            return RawSpansDataFrameResponse(
                columns=[str(column) for column in spans_df.columns],
                row_count=int(len(spans_df)),
                rows=_dataframe_to_rows(spans_df),
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/evaluations/batch/run", response_model=BatchEvaluationRunResponse)
    def run_batch_evaluation(request: BatchEvaluationRunRequest) -> BatchEvaluationRunResponse:
        """Run Phoenix span evaluations and optionally write annotations back."""
        try:
            batch_config = resolve_batch_evaluation_config(
                BatchEvaluationConfig(
                    from_time=request.from_time,
                    to_time=request.to_time,
                    project_name=request.project_name or settings.phoenix_project_name,
                    span_kind=request.span_kind or settings.phoenix_fetch_span_kind,
                    limit=request.limit,
                    root_spans_only=request.root_spans_only,
                    phoenix_base_url=settings.phoenix_query_endpoint,
                    sync_annotations=request.sync_annotations,
                    save_annotations=request.save_annotations,
                    use_last_eval_timestamp=settings.batch_evaluation_use_last_eval_timestamp,
                    last_eval_timestamp_file=settings.last_eval_timestamp_file,
                )
            )
            result = run_span_batch(batch_config)
            if batch_config.use_last_eval_timestamp and batch_config.last_eval_timestamp_file is not None:
                write_last_eval_timestamp(batch_config.last_eval_timestamp_file, batch_config.to_time)
            return BatchEvaluationRunResponse(
                span_count=result.span_count,
                evaluated_count=int(len(result.evaluation_df)),
                annotation_count=result.annotation_count,
                annotations_saved=request.save_annotations and result.annotation_count > 0,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
