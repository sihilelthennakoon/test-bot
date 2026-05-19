from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    top_k: int = Field(default=4, ge=1, le=12)


class IngestRequest(BaseModel):
    source_path: str
    rebuild: bool = True


class DocumentChunk(BaseModel):
    chunk_id: str
    source_path: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk_id: str
    source_path: str
    text: str
    score: float
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SafetyDecision(BaseModel):
    allowed: bool
    reason: str
    warnings: list[str] = Field(default_factory=list)
    masked_text: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievalHit] = Field(default_factory=list)
    input_safety: SafetyDecision
    output_safety: SafetyDecision
    trace_id: str = Field(default_factory=new_id)
    conversation_id: str | None = None


class IngestResponse(BaseModel):
    source_path: str
    chunk_count: int
    index_path: str
    masked_chunk_count: int


class EvaluationExample(BaseModel):
    question: str
    expected_terms: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    question: str
    answer: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    contains_expected_terms: bool = False
    trace_id: str = Field(default_factory=new_id)
