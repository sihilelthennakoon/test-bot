from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Protocol
from urllib.parse import urlparse

import pandas as pd
from pydantic import BaseModel, Field

from ragbot.schemas import new_id, utc_now


def parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _ensure_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        return parse_datetime(value)
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _extract_span_kind(span: dict[str, Any]) -> str | None:
    candidates = [
        span.get("span_kind"),
        span.get("kind"),
        span.get("spanKind"),
        span.get("span_kind_name"),
    ]
    attributes = _as_mapping(span.get("attributes"))
    candidates.extend(
        [
            attributes.get("openinference.span.kind"),
            attributes.get("span.kind"),
            attributes.get("span_kind"),
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, str):
            cleaned = candidate.strip()
            if cleaned:
                return cleaned.upper()
    return None


def _event_sort_key(timestamp: datetime, trace_id: str, span_id: str | None) -> tuple[str, str, str]:
    return (to_iso_utc(timestamp), trace_id, span_id or "")


class SpanRecord(BaseModel):
    span_id: str
    trace_id: str
    name: str | None = None
    span_kind: str | None = None
    start_time: datetime
    end_time: datetime | None = None
    status_code: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    trace_id: str
    project_name: str | None = None
    start_time: datetime
    end_time: datetime | None = None
    spans: list[SpanRecord] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class FetchCheckpoint(BaseModel):
    last_timestamp: datetime | None = None
    last_trace_id: str | None = None
    last_span_id: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class FetchSummary(BaseModel):
    run_id: str
    output_file: str
    traces_written: int
    spans_written: int
    from_time: datetime
    to_time: datetime
    delta_mode: bool
    checkpoint_updated: bool
    checkpoint: FetchCheckpoint


class RawSpansDataFrameRequest(BaseModel):
    from_time: datetime | None = None
    to_time: datetime | None = None
    project_name: str | None = None
    limit: int = Field(default=1000, ge=1)
    root_spans_only: bool | None = None


@dataclass(frozen=True, slots=True)
class FetchRequest:
    from_time: datetime | None
    to_time: datetime
    project_name: str | None
    span_kind: str
    batch_size: int
    output_dir: Path
    checkpoint_file: Path
    delta: bool = False
    update_checkpoint: bool = True


class PhoenixAdapter(Protocol):
    def fetch_traces(
        self,
        *,
        from_time: datetime,
        to_time: datetime,
        project_name: str | None,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        ...

    def fetch_spans(
        self,
        *,
        trace_id: str,
        from_time: datetime,
        to_time: datetime,
        project_name: str | None,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        ...

    def fetch_spans_dataframe(
        self,
        *,
        from_time: datetime | None,
        to_time: datetime | None,
        project_name: str | None,
        limit: int,
        root_spans_only: bool | None,
    ) -> pd.DataFrame:
        ...

    def log_span_annotations_dataframe(
        self,
        *,
        annotations_df: pd.DataFrame,
        sync: bool,
    ) -> None:
        ...


class PhoenixSdkAdapter:
    """Fetches traces and spans using Phoenix SDK client methods when available."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._client = self._build_client(endpoint)

    @staticmethod
    def _build_client(endpoint: str) -> Any:
        try:
            from phoenix.client import Client  # type: ignore
        except Exception as exc:  # pragma: no cover - import availability depends on env
            raise RuntimeError(
                "Phoenix SDK client is unavailable. Install arize-phoenix and verify imports."
            ) from exc

        return Client(base_url=endpoint)

    @staticmethod
    def _to_records(payload: Any) -> list[dict[str, Any]]:
        if payload is None:
            return []
        if hasattr(payload, "model_dump"):
            return [payload.model_dump()]
        if isinstance(payload, list):
            records: list[dict[str, Any]] = []
            for row in payload:
                if isinstance(row, dict):
                    records.append(row)
                elif hasattr(row, "model_dump"):
                    records.append(row.model_dump())
            return records
        if hasattr(payload, "to_dict"):
            try:
                converted = payload.to_dict(orient="records")
            except TypeError:
                converted = payload.to_dict()
            if isinstance(converted, list):
                return [row for row in converted if isinstance(row, dict)]
            if isinstance(converted, dict):
                return [converted]
        if isinstance(payload, dict):
            return [payload]
        return []

    @staticmethod
    def _call_with_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> Any:
        pending = dict(kwargs)
        while True:
            try:
                return fn(**pending)
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in list(pending.keys()):
                    if f"'{key}'" in message and ("unexpected" in message or "positional" in message):
                        pending.pop(key)
                        removed = True
                        break
                if not removed:
                    raise
                if not pending:
                    return fn()

    def fetch_traces(
        self,
        *,
        from_time: datetime,
        to_time: datetime,
        project_name: str | None,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        project_identifier = project_name or "default"
        payload = self._call_with_supported_kwargs(
            self._client.traces.get_traces,
            {
                "project_identifier": project_identifier,
                "start_time": from_time,
                "end_time": to_time,
                "include_spans": False,
                "limit": batch_size,
            },
        )
        return self._to_records(payload)

    def fetch_spans(
        self,
        *,
        trace_id: str,
        from_time: datetime,
        to_time: datetime,
        project_name: str | None,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        project_identifier = project_name or "default"
        payload = self._call_with_supported_kwargs(
            self._client.spans.get_spans,
            {
                "project_identifier": project_identifier,
                "start_time": from_time,
                "end_time": to_time,
                "trace_ids": [trace_id],
                "limit": batch_size,
            },
        )
        return self._to_records(payload)

    def fetch_spans_dataframe(
        self,
        *,
        from_time: datetime | None,
        to_time: datetime | None,
        project_name: str | None,
        limit: int,
        root_spans_only: bool | None,
    ) -> pd.DataFrame:
        payload = self._call_with_supported_kwargs(
            self._client.spans.get_spans_dataframe,
            {
                "start_time": from_time,
                "end_time": to_time,
                "limit": limit,
                "root_spans_only": root_spans_only,
                "project_name": project_name,
            },
        )
        if isinstance(payload, pd.DataFrame):
            return payload
        return pd.DataFrame(self._to_records(payload))

    def log_span_annotations_dataframe(
        self,
        *,
        annotations_df: pd.DataFrame,
        sync: bool,
    ) -> None:
        self._call_with_supported_kwargs(
            self._client.spans.log_span_annotations_dataframe,
            {
                "dataframe": annotations_df,
                "sync": sync,
            },
        )


def load_checkpoint(path: Path) -> FetchCheckpoint:
    if not path.exists():
        return FetchCheckpoint()
    payload = json.loads(path.read_text())
    return FetchCheckpoint.model_validate(payload)


def save_checkpoint(path: Path, checkpoint: FetchCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f"{path.name}.",
        suffix=".tmp",
    ) as temp_file:
        temp_file.write(checkpoint.model_dump_json(indent=2))
        temp_name = temp_file.name
    Path(temp_name).replace(path)


def _normalize_trace(
    record: dict[str, Any],
    default_project_name: str | None,
) -> tuple[str, datetime, datetime | None, dict[str, Any]]:
    trace_id = str(record.get("trace_id") or record.get("traceId") or record.get("id") or "").strip()
    if not trace_id:
        raise ValueError("Trace record is missing trace_id")
    start_time = _ensure_datetime(record.get("start_time") or record.get("startTime") or record.get("timestamp"))
    if start_time is None:
        raise ValueError(f"Trace {trace_id} is missing start_time")
    end_time = _ensure_datetime(record.get("end_time") or record.get("endTime"))
    attributes = _as_mapping(record.get("attributes"))
    project_name = record.get("project_name") or record.get("projectName") or default_project_name
    if project_name is not None:
        attributes = dict(attributes)
        attributes.setdefault("project_name", str(project_name))
    return trace_id, start_time, end_time, attributes


def _normalize_span(record: dict[str, Any], trace_id: str) -> SpanRecord:
    span_id = str(record.get("span_id") or record.get("spanId") or record.get("id") or "").strip()
    if not span_id:
        raise ValueError(f"Span for trace {trace_id} is missing span_id")
    start_time = _ensure_datetime(record.get("start_time") or record.get("startTime") or record.get("timestamp"))
    if start_time is None:
        raise ValueError(f"Span {span_id} is missing start_time")
    end_time = _ensure_datetime(record.get("end_time") or record.get("endTime"))
    attributes = _as_mapping(record.get("attributes"))
    return SpanRecord(
        span_id=span_id,
        trace_id=trace_id,
        name=(record.get("name") or record.get("span_name")),
        span_kind=_extract_span_kind(record),
        start_time=start_time,
        end_time=end_time,
        status_code=(record.get("status_code") or record.get("statusCode")),
        attributes=attributes,
    )


def _record_is_new(
    event_timestamp: datetime,
    trace_id: str,
    span_id: str | None,
    checkpoint: FetchCheckpoint,
) -> bool:
    if checkpoint.last_timestamp is None:
        return True
    if event_timestamp > checkpoint.last_timestamp:
        return True
    if event_timestamp < checkpoint.last_timestamp:
        return False
    checkpoint_key = (checkpoint.last_trace_id or "", checkpoint.last_span_id or "")
    event_key = (trace_id, span_id or "")
    return event_key > checkpoint_key


def _build_run_output_path(output_dir: Path, run_id: str) -> Path:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"phoenix_fetch_{timestamp}_{run_id}.jsonl"


def _write_jsonl(path: Path, trace_rows: list[TraceRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for trace in trace_rows:
            handle.write(trace.model_dump_json())
            handle.write("\n")


def fetch_phoenix_traces_and_spans(request: FetchRequest, adapter: PhoenixAdapter) -> FetchSummary:
    run_id = new_id("phoenix_fetch")
    checkpoint = load_checkpoint(request.checkpoint_file)

    effective_from = request.from_time
    if request.delta:
        if checkpoint.last_timestamp is not None and effective_from is not None:
            effective_from = max(effective_from, checkpoint.last_timestamp)
        elif checkpoint.last_timestamp is not None:
            effective_from = checkpoint.last_timestamp

    if effective_from is None:
        raise ValueError("from_time is required when no checkpoint timestamp exists")

    if effective_from >= request.to_time:
        raise ValueError("from_time must be earlier than to_time")

    normalized_span_kind = request.span_kind.strip().upper()
    traces_payload = adapter.fetch_traces(
        from_time=effective_from,
        to_time=request.to_time,
        project_name=request.project_name,
        batch_size=request.batch_size,
    )

    trace_rows: list[TraceRecord] = []
    latest_event: tuple[str, str, str] | None = None
    latest_event_data: tuple[datetime, str, str | None] | None = None
    spans_written = 0

    for trace_payload in traces_payload:
        trace_id, trace_start_time, trace_end_time, trace_attributes = _normalize_trace(
            trace_payload,
            request.project_name,
        )
        spans_payload = adapter.fetch_spans(
            trace_id=trace_id,
            from_time=effective_from,
            to_time=request.to_time,
            project_name=request.project_name,
            batch_size=request.batch_size,
        )

        normalized_spans: list[SpanRecord] = []
        for span_payload in spans_payload:
            span = _normalize_span(span_payload, trace_id)
            if normalized_span_kind and (span.span_kind or "").upper() != normalized_span_kind:
                continue
            if not _record_is_new(span.start_time, trace_id, span.span_id, checkpoint):
                continue

            normalized_spans.append(span)
            spans_written += 1

            event_key = _event_sort_key(span.start_time, trace_id, span.span_id)
            if latest_event is None or event_key > latest_event:
                latest_event = event_key
                latest_event_data = (span.start_time, trace_id, span.span_id)

        if not normalized_spans:
            continue

        trace_rows.append(
            TraceRecord(
                trace_id=trace_id,
                project_name=request.project_name,
                start_time=trace_start_time,
                end_time=trace_end_time,
                spans=sorted(normalized_spans, key=lambda row: row.start_time),
                attributes=trace_attributes,
            )
        )

    output_path = _build_run_output_path(request.output_dir, run_id)
    _write_jsonl(output_path, trace_rows)

    checkpoint_updated = False
    if request.update_checkpoint and latest_event_data is not None:
        event_ts, trace_id, span_id = latest_event_data
        checkpoint = FetchCheckpoint(
            last_timestamp=event_ts,
            last_trace_id=trace_id,
            last_span_id=span_id,
            updated_at=utc_now(),
        )
        save_checkpoint(request.checkpoint_file, checkpoint)
        checkpoint_updated = True

    return FetchSummary(
        run_id=run_id,
        output_file=str(output_path),
        traces_written=len(trace_rows),
        spans_written=spans_written,
        from_time=effective_from,
        to_time=request.to_time,
        delta_mode=request.delta,
        checkpoint_updated=checkpoint_updated,
        checkpoint=checkpoint,
    )


def make_fetch_request(
    *,
    from_time: str | None,
    to_time: str | None,
    project_name: str | None,
    span_kind: str,
    batch_size: int,
    output_dir: str | Path,
    checkpoint_file: str | Path,
    delta: bool,
    update_checkpoint: bool,
) -> FetchRequest:
    resolved_to = parse_datetime(to_time) if to_time else utc_now()
    resolved_from = parse_datetime(from_time) if from_time else None
    return FetchRequest(
        from_time=resolved_from,
        to_time=resolved_to,
        project_name=project_name,
        span_kind=span_kind,
        batch_size=batch_size,
        output_dir=Path(output_dir),
        checkpoint_file=Path(checkpoint_file),
        delta=delta,
        update_checkpoint=update_checkpoint,
    )


def make_raw_spans_dataframe_request(
    *,
    from_time: str | datetime | None,
    to_time: str | datetime | None,
    project_name: str | None,
    limit: int,
    root_spans_only: bool | None,
) -> RawSpansDataFrameRequest:
    return RawSpansDataFrameRequest(
        from_time=_ensure_datetime(from_time),
        to_time=_ensure_datetime(to_time),
        project_name=project_name,
        limit=limit,
        root_spans_only=root_spans_only,
    )


def fetch_phoenix_spans_dataframe(
    request: RawSpansDataFrameRequest,
    adapter: PhoenixAdapter,
) -> pd.DataFrame:
    return adapter.fetch_spans_dataframe(
        from_time=request.from_time,
        to_time=request.to_time,
        project_name=request.project_name,
        limit=request.limit,
        root_spans_only=request.root_spans_only,
    )


def log_phoenix_span_annotations(
    annotations_df: pd.DataFrame,
    adapter: PhoenixAdapter,
    *,
    sync: bool = True,
) -> None:
    if annotations_df.empty:
        return
    adapter.log_span_annotations_dataframe(annotations_df=annotations_df, sync=sync)


def build_default_adapter(phoenix_query_endpoint: str) -> PhoenixAdapter:
    parsed = urlparse(phoenix_query_endpoint)
    if not parsed.scheme:
        raise ValueError(f"Invalid PHOENIX_QUERY_ENDPOINT value: {phoenix_query_endpoint}")
    return PhoenixSdkAdapter(phoenix_query_endpoint)
