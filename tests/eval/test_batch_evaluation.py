from __future__ import annotations

import pandas as pd

from ragbot.evaluations.batch_evaluation import evaluate_batch
from ragbot.evaluations.batch_evaluation.evaluate_batch import (
    BatchEvaluationConfig,
    _build_evaluation_frame,
    run_span_batch,
)


class FakePhoenixAdapter:
    def __init__(self, spans_df: pd.DataFrame) -> None:
        self.spans_df = spans_df
        self.logged_annotations_df = pd.DataFrame()
        self.sync_annotations: bool | None = None

    def fetch_traces(self, *, from_time, to_time, project_name, batch_size) -> list[dict]:
        return []

    def fetch_spans(self, *, trace_id, from_time, to_time, project_name, batch_size) -> list[dict]:
        return []

    def fetch_spans_dataframe(
        self,
        *,
        from_time,
        to_time,
        project_name,
        limit,
        root_spans_only,
    ) -> pd.DataFrame:
        return self.spans_df.head(limit)

    def log_span_annotations_dataframe(self, *, annotations_df, sync) -> None:
        self.logged_annotations_df = annotations_df.copy()
        self.sync_annotations = sync


class FakeEvaluationRunner:
    async def evaluate_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        scored = dataframe.copy()
        scored["correctness_score"] = [
            {"score": 1.0, "label": "CORRECT", "explanation": "Answered directly."}
            for _ in range(len(scored))
        ]
        scored["relevance_score"] = [
            {"score": 1.0, "label": "RELEVANT", "explanation": "Relevant response."}
            for _ in range(len(scored))
        ]
        scored["groundedness_score"] = [
            {"score": 1.0, "label": "GROUNDED", "explanation": "Supported by context."}
            for _ in range(len(scored))
        ]
        scored["safety_score"] = [
            {"score": 1.0, "label": "SAFE", "explanation": "No safety issues."}
            for _ in range(len(scored))
        ]
        return scored


def test_run_span_batch_fetches_with_adapter_and_logs_annotations(monkeypatch) -> None:
    adapter = FakePhoenixAdapter(
        pd.DataFrame(
            [
                {
                    "span_id": "s1",
                    "trace_id": "t1",
                    "input": "What is the capital of France?",
                    "output": "Paris is the capital of France.",
                    "reference": "France's capital is Paris.",
                    "attributes": {"openinference.span.kind": "CHAIN"},
                }
            ]
        )
    )
    monkeypatch.setattr(evaluate_batch, "_build_runner", lambda: FakeEvaluationRunner())

    result = run_span_batch(
        BatchEvaluationConfig(
            adapter=adapter,
            project_name="test-bot-with-eval",
            span_kind="CHAIN",
            limit=10,
            sync_annotations=False,
        )
    )

    assert result.span_count == 1
    assert result.annotation_count == 4
    assert adapter.sync_annotations is False
    assert adapter.logged_annotations_df["annotation_name"].tolist() == [
        "correctness",
        "relevance",
        "faithfulness",
        "safety",
    ]


def test_run_span_batch_can_skip_annotation_logging(monkeypatch) -> None:
    adapter = FakePhoenixAdapter(
        pd.DataFrame(
            [
                {
                    "span_id": "s1",
                    "input": "Question",
                    "output": "Answer",
                    "attributes": {"openinference.span.kind": "CHAIN"},
                }
            ]
        )
    )
    monkeypatch.setattr(evaluate_batch, "_build_runner", lambda: FakeEvaluationRunner())

    result = run_span_batch(
        BatchEvaluationConfig(
            adapter=adapter,
            span_kind="CHAIN",
            save_annotations=False,
        )
    )

    assert result.annotation_count == 4
    assert adapter.logged_annotations_df.empty


def test_run_span_batch_only_counts_and_evaluates_requested_span_kind(monkeypatch) -> None:
    adapter = FakePhoenixAdapter(
        pd.DataFrame(
            [
                {
                    "span_id": "chain-1",
                    "input": "Question",
                    "output": "Answer",
                    "attributes": {"openinference.span.kind": "CHAIN"},
                },
                {
                    "span_id": "llm-1",
                    "input": "Question",
                    "output": "Answer",
                    "attributes": {"openinference.span.kind": "LLM"},
                },
                {
                    "span_id": "unknown-1",
                    "input": "Question",
                    "output": "Answer",
                },
            ]
        )
    )
    monkeypatch.setattr(evaluate_batch, "_build_runner", lambda: FakeEvaluationRunner())

    result = run_span_batch(
        BatchEvaluationConfig(
            adapter=adapter,
            span_kind="CHAIN",
        )
    )

    assert result.span_count == 1
    assert result.evaluation_df["span_id"].tolist() == ["chain-1"]
    assert result.annotation_count == 4


def test_build_evaluation_frame_handles_phoenix_context_span_id_index_collision() -> None:
    spans_df = pd.DataFrame(
        [
            {
                "context.span_id": "s1",
                "input": "Question",
                "output": "Answer",
                "attributes": {"openinference.span.kind": "CHAIN"},
            }
        ]
    )
    spans_df.index.name = "context.span_id"

    evaluation_df = _build_evaluation_frame(spans_df, span_kind="CHAIN")

    assert evaluation_df["span_id"].tolist() == ["s1"]


def test_build_evaluation_frame_extracts_openinference_flattened_columns() -> None:
    spans_df = pd.DataFrame(
        [
            {
                "context.span_id": "s1",
                "attributes.input.value": '{"question": "What is the capital of France?"}',
                "attributes.output.value": '{"answer": "Paris is the capital of France."}',
                "attributes.retrieval.documents": [
                    {"document": {"content": "France's capital is Paris."}}
                ],
                "attributes.openinference.span.kind": "CHAIN",
            }
        ]
    )

    evaluation_df = _build_evaluation_frame(spans_df, span_kind="CHAIN")

    row = evaluation_df.iloc[0]
    assert row["span_id"] == "s1"
    assert "What is the capital of France?" in row["input"]
    assert "Paris is the capital of France." in row["output"]
    assert "France's capital is Paris." in row["reference"]
