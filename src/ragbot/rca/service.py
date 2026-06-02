from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ragbot.config import Settings

from .models import PhoenixPullFilters, PhoenixSpanRecord, PhoenixTracePullResult, PhoenixTraceRecord
from .phoenix_client import PhoenixClient, PhoenixClientError
from .storage import RCAPullStorage


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def _duration_ms(start_time: datetime | None, end_time: datetime | None) -> float | None:
    if not start_time or not end_time:
        return None
    return max((end_time - start_time).total_seconds() * 1000, 0.0)


def _extract_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    attributes = payload.get("attributes")
    if isinstance(attributes, dict):
        return attributes
    return {}


def _extract_session_identifier(payload: dict[str, Any]) -> str | None:
    attributes = _extract_attributes(payload)
    for key in (
        "session.id",
        "session_id",
        "session_identifier",
        "metadata.conversation_id",
        "conversation_id",
    ):
        value = attributes.get(key) or payload.get(key)
        if value:
            return str(value)
    return None


def _normalize_span(payload: dict[str, Any], trace_id: str) -> PhoenixSpanRecord:
    start_time = _parse_datetime(payload.get("start_time") or payload.get("startTime"))
    end_time = _parse_datetime(payload.get("end_time") or payload.get("endTime"))
    span_id = str(payload.get("span_id") or payload.get("spanId") or payload.get("id") or "")
    parent_span_id = payload.get("parent_span_id") or payload.get("parentSpanId") or payload.get("parent_id")
    name = str(payload.get("name") or payload.get("span_name") or "unknown")
    status = payload.get("status_code") or payload.get("statusCode")
    if isinstance(status, dict):
        status = status.get("code") or status.get("status_code")
    return PhoenixSpanRecord(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=str(parent_span_id) if parent_span_id else None,
        name=name,
        span_kind=payload.get("span_kind") or payload.get("spanKind") or payload.get("kind"),
        status_code=str(status) if status is not None else None,
        start_time=start_time,
        end_time=end_time,
        duration_ms=_duration_ms(start_time, end_time),
        attributes=_extract_attributes(payload),
        raw=payload,
    )


def _pick_root_span(spans: list[PhoenixSpanRecord]) -> PhoenixSpanRecord | None:
    if not spans:
        return None
    for span in spans:
        if not span.parent_span_id:
            return span
    return min(
        spans,
        key=lambda span: (
            span.start_time is None,
            span.start_time or datetime.max,
            span.span_id,
        ),
    )


def _normalize_trace(payload: dict[str, Any], project_name: str, spans_payload: list[dict[str, Any]] | None = None) -> PhoenixTraceRecord:
    trace_id = str(payload.get("trace_id") or payload.get("traceId") or payload.get("id") or "")
    span_items = spans_payload if spans_payload is not None else payload.get("spans") or []
    spans = [_normalize_span(span_payload, trace_id) for span_payload in span_items]
    root_span = _pick_root_span(spans)
    start_time = _parse_datetime(payload.get("start_time") or payload.get("startTime"))
    end_time = _parse_datetime(payload.get("end_time") or payload.get("endTime"))
    if start_time is None and root_span is not None:
        start_time = root_span.start_time
    if end_time is None and spans:
        dated_spans = [span.end_time for span in spans if span.end_time is not None]
        end_time = max(dated_spans) if dated_spans else None
    attributes = _extract_attributes(payload)
    conversation_id = _extract_session_identifier(payload)
    if conversation_id is None and root_span is not None:
        conversation_id = _extract_session_identifier(root_span.raw)
    status = payload.get("status_code") or payload.get("statusCode")
    if isinstance(status, dict):
        status = status.get("code") or status.get("status_code")
    if status is None and root_span is not None:
        status = root_span.status_code
    return PhoenixTraceRecord(
        trace_id=trace_id,
        project_name=project_name,
        conversation_id=conversation_id,
        root_span_id=root_span.span_id if root_span else None,
        root_span_name=root_span.name if root_span else None,
        status_code=str(status) if status is not None else None,
        start_time=start_time,
        end_time=end_time,
        duration_ms=_duration_ms(start_time, end_time),
        span_count=len(spans),
        spans=spans,
        raw=payload,
    )


def _extract_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("data", "traces", "spans", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_next_cursor(payload: dict[str, Any]) -> str | None:
    for key in ("next_cursor", "nextCursor", "cursor"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    meta = payload.get("meta")
    if isinstance(meta, dict):
        for key in ("next_cursor", "nextCursor", "cursor"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
    return None


@dataclass(slots=True)
class RCATraceService:
    client: PhoenixClient
    storage: RCAPullStorage
    default_project_name: str

    @classmethod
    def from_settings(cls, settings: Settings) -> "RCATraceService":
        return cls(
            client=PhoenixClient(base_url=settings.phoenix_pull_base_url),
            storage=RCAPullStorage(settings.rca_pull_save_dir),
            default_project_name=settings.phoenix_project_name,
        )

    def build_filters(
        self,
        *,
        project_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        trace_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> PhoenixPullFilters:
        return PhoenixPullFilters(
            project_name=project_name or self.default_project_name,
            start_time=start_time,
            end_time=end_time,
            trace_id=trace_id,
            conversation_id=conversation_id,
            limit=limit,
            cursor=cursor,
        )

    def pull_traces(
        self,
        *,
        project_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        trace_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> PhoenixTracePullResult:
        filters = self.build_filters(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
            trace_id=trace_id,
            conversation_id=conversation_id,
            limit=limit,
            cursor=cursor,
        )

        source = "phoenix.traces"
        try:
            payload = self.client.fetch_traces(filters)
            traces = [
                _normalize_trace(trace_payload, filters.project_name)
                for trace_payload in _extract_records(payload)
            ]
            next_cursor = _extract_next_cursor(payload)
        except PhoenixClientError as exc:
            if exc.status_code not in {404, 405, 501}:
                raise
            source = "phoenix.spans_fallback"
            spans_payload = self.client.fetch_spans(filters)
            traces = self._group_spans_into_traces(spans_payload, filters.project_name)
            payload = {"trace_error": str(exc), "spans_response": spans_payload}
            next_cursor = _extract_next_cursor(spans_payload)
        except Exception as exc:
            source = "phoenix.spans_fallback"
            spans_payload = self.client.fetch_spans(filters)
            traces = self._group_spans_into_traces(spans_payload, filters.project_name)
            payload = {"trace_error": str(exc), "spans_response": spans_payload}
            next_cursor = _extract_next_cursor(spans_payload)

        result = PhoenixTracePullResult(
            source=source,
            project_name=filters.project_name,
            filters=filters,
            trace_count=len(traces),
            traces=traces,
            next_cursor=next_cursor,
            raw_payload=payload,
        )
        snapshot_path = self.storage.save(result)
        return result.with_snapshot_path(snapshot_path)

    def _group_spans_into_traces(self, payload: dict[str, Any], project_name: str) -> list[PhoenixTraceRecord]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        raw_span_records = _extract_records(payload)
        for span_payload in raw_span_records:
            trace_id = str(span_payload.get("trace_id") or span_payload.get("traceId") or "")
            if not trace_id:
                continue
            grouped[trace_id].append(span_payload)

        traces: list[PhoenixTraceRecord] = []
        for trace_id, span_payloads in grouped.items():
            traces.append(_normalize_trace({"trace_id": trace_id}, project_name, span_payloads))
        traces.sort(
            key=lambda trace: (
                trace.start_time is None,
                trace.start_time or datetime.max,
                trace.trace_id,
            )
        )
        return traces
