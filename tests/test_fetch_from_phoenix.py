from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_fetch.fetch_from_phoenix import (
    FetchCheckpoint,
    FetchRequest,
    FetchSummary,
    PhoenixAdapter,
    RawSpansDataFrameRequest,
    fetch_phoenix_traces_and_spans,
    fetch_phoenix_spans_dataframe,
    parse_datetime,
    save_checkpoint,
)


class FakePhoenixAdapter(PhoenixAdapter):
    def __init__(self, traces: list[dict], spans_by_trace: dict[str, list[dict]]) -> None:
        self._traces = traces
        self._spans_by_trace = spans_by_trace

    def fetch_traces(
        self,
        *,
        from_time,
        to_time,
        project_name,
        batch_size,
    ) -> list[dict]:
        return self._traces[:batch_size]

    def fetch_spans(
        self,
        *,
        trace_id,
        from_time,
        to_time,
        project_name,
        batch_size,
    ) -> list[dict]:
        return self._spans_by_trace.get(trace_id, [])[:batch_size]

    def fetch_spans_dataframe(
        self,
        *,
        from_time,
        to_time,
        project_name,
        limit,
        root_spans_only,
    ) -> pd.DataFrame:
        rows = [
            span
            for spans in self._spans_by_trace.values()
            for span in spans
        ]
        return pd.DataFrame(rows[:limit])

    def log_span_annotations_dataframe(
        self,
        *,
        annotations_df,
        sync,
    ) -> None:
        self.annotations_df = annotations_df
        self.sync_annotations = sync


def test_fetch_raw_spans_dataframe_uses_adapter() -> None:
    adapter = FakePhoenixAdapter(
        traces=[],
        spans_by_trace={
            "t1": [
                {
                    "span_id": "s1",
                    "trace_id": "t1",
                    "input": "What is Paris?",
                    "output": "A city.",
                }
            ]
        },
    )

    spans_df = fetch_phoenix_spans_dataframe(
        RawSpansDataFrameRequest(
            from_time=parse_datetime("2026-06-01T09:00:00Z"),
            to_time=parse_datetime("2026-06-01T10:00:00Z"),
            project_name="test-bot-with-eval",
            limit=10,
        ),
        adapter,
    )

    assert spans_df.to_dict(orient="records") == [
        {
            "span_id": "s1",
            "trace_id": "t1",
            "input": "What is Paris?",
            "output": "A city.",
        }
    ]

def test_fetch_filters_chain_spans_and_writes_jsonl(tmp_path: Path) -> None:
    request = FetchRequest(
        from_time=parse_datetime("2026-06-01T09:00:00Z"),
        to_time=parse_datetime("2026-06-01T10:00:00Z"),
        project_name="test-bot-with-eval",
        span_kind="CHAIN",
        batch_size=100,
        output_dir=tmp_path / "out",
        checkpoint_file=tmp_path / "checkpoint.json",
        delta=False,
        update_checkpoint=True,
    )

    adapter = FakePhoenixAdapter(
        traces=[
            {
                "trace_id": "t1",
                "start_time": "2026-06-01T09:10:00Z",
                "attributes": {"source": "phoenix"},
            }
        ],
        spans_by_trace={
            "t1": [
                {
                    "span_id": "s1",
                    "trace_id": "t1",
                    "start_time": "2026-06-01T09:10:00Z",
                    "name": "chain-run",
                    "attributes": {"openinference.span.kind": "CHAIN"},
                },
                {
                    "span_id": "s2",
                    "trace_id": "t1",
                    "start_time": "2026-06-01T09:11:00Z",
                    "name": "llm-run",
                    "attributes": {"openinference.span.kind": "LLM"},
                },
            ]
        },
    )

    summary = fetch_phoenix_traces_and_spans(request, adapter)

    assert isinstance(summary, FetchSummary)
    assert summary.traces_written == 1
    assert summary.spans_written == 1
    assert Path(summary.output_file).exists()

    lines = Path(summary.output_file).read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["trace_id"] == "t1"
    assert len(payload["spans"]) == 1
    assert payload["spans"][0]["span_id"] == "s1"


def test_delta_uses_checkpoint_and_tie_breaker(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    save_checkpoint(
        checkpoint_path,
        FetchCheckpoint(
            last_timestamp=parse_datetime("2026-06-01T09:10:00Z"),
            last_trace_id="t1",
            last_span_id="s1",
        ),
    )

    request = FetchRequest(
        from_time=parse_datetime("2026-06-01T09:00:00Z"),
        to_time=parse_datetime("2026-06-01T10:00:00Z"),
        project_name="test-bot-with-eval",
        span_kind="CHAIN",
        batch_size=100,
        output_dir=tmp_path / "out",
        checkpoint_file=checkpoint_path,
        delta=True,
        update_checkpoint=True,
    )

    adapter = FakePhoenixAdapter(
        traces=[{"trace_id": "t1", "start_time": "2026-06-01T09:10:00Z"}],
        spans_by_trace={
            "t1": [
                {
                    "span_id": "s1",
                    "trace_id": "t1",
                    "start_time": "2026-06-01T09:10:00Z",
                    "attributes": {"openinference.span.kind": "CHAIN"},
                },
                {
                    "span_id": "s2",
                    "trace_id": "t1",
                    "start_time": "2026-06-01T09:10:00Z",
                    "attributes": {"openinference.span.kind": "CHAIN"},
                },
            ]
        },
    )

    summary = fetch_phoenix_traces_and_spans(request, adapter)

    assert summary.traces_written == 1
    assert summary.spans_written == 1

    lines = Path(summary.output_file).read_text().strip().splitlines()
    payload = json.loads(lines[0])
    assert [span["span_id"] for span in payload["spans"]] == ["s2"]


def test_checkpoint_not_updated_when_no_new_spans(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    original = FetchCheckpoint(
        last_timestamp=parse_datetime("2026-06-01T09:10:00Z"),
        last_trace_id="t1",
        last_span_id="s1",
    )
    save_checkpoint(checkpoint_path, original)

    request = FetchRequest(
        from_time=parse_datetime("2026-06-01T09:00:00Z"),
        to_time=parse_datetime("2026-06-01T10:00:00Z"),
        project_name="test-bot-with-eval",
        span_kind="CHAIN",
        batch_size=100,
        output_dir=tmp_path / "out",
        checkpoint_file=checkpoint_path,
        delta=True,
        update_checkpoint=True,
    )

    adapter = FakePhoenixAdapter(
        traces=[{"trace_id": "t1", "start_time": "2026-06-01T09:10:00Z"}],
        spans_by_trace={
            "t1": [
                {
                    "span_id": "s1",
                    "trace_id": "t1",
                    "start_time": "2026-06-01T09:10:00Z",
                    "attributes": {"openinference.span.kind": "CHAIN"},
                }
            ]
        },
    )

    summary = fetch_phoenix_traces_and_spans(request, adapter)

    assert summary.traces_written == 0
    assert summary.spans_written == 0
    assert summary.checkpoint_updated is False

    saved = json.loads(checkpoint_path.read_text())
    assert saved["last_span_id"] == "s1"
