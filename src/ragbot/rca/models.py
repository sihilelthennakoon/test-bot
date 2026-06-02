from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PhoenixPullFilters(BaseModel):
    project_name: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    trace_id: str | None = None
    conversation_id: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = None


class PhoenixSpanRecord(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    name: str
    span_kind: str | None = None
    status_code: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class PhoenixTraceRecord(BaseModel):
    trace_id: str
    project_name: str | None = None
    conversation_id: str | None = None
    root_span_id: str | None = None
    root_span_name: str | None = None
    status_code: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float | None = None
    span_count: int = 0
    spans: list[PhoenixSpanRecord] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class PhoenixTracePullResult(BaseModel):
    fetched_at: datetime = Field(default_factory=utc_now)
    source: str
    project_name: str
    filters: PhoenixPullFilters
    trace_count: int = 0
    traces: list[PhoenixTraceRecord] = Field(default_factory=list)
    next_cursor: str | None = None
    snapshot_path: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    def with_snapshot_path(self, path: Path) -> "PhoenixTracePullResult":
        return self.model_copy(update={"snapshot_path": str(path)})
