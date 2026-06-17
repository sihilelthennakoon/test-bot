from __future__ import annotations

import pandas as pd

from ragbot.evaluations.batch_evaluation import alerts
from ragbot.evaluations.batch_evaluation import evaluate_batch as batch_module
from ragbot.evaluations.batch_evaluation.evaluate_batch import (
	BatchEvaluationConfig,
	_build_evaluation_frame,
	_filter_main_application_spans,
	merge_final_rca_into_annotations,
)
from ragbot.rca.rca_judge import RCAJudgeError


def test_filter_main_application_spans_keeps_only_root_chain_span() -> None:
	spans_df = pd.DataFrame(
		[
			{
				"name": "LangGraph",
				"span_kind": "CHAIN",
				"parent_id": None,
				"context.span_id": "root-chain",
			},
			{
				"name": "retrieve",
				"span_kind": "CHAIN",
				"parent_id": "root-chain",
				"context.span_id": "child-chain",
			},
			{
				"name": "ChatGoogleGenerativeAI",
				"span_kind": "LLM",
				"parent_id": "child-chain",
				"context.span_id": "llm-child",
			},
			{
				"name": "safety.evaluate",
				"span_kind": "CHAIN",
				"parent_id": None,
				"context.span_id": "root-evaluator",
			},
			{
				"name": "unknown-root",
				"span_kind": "CHAIN",
				"context.span_id": "missing-parent-column",
			},
		]
	)

	filtered = _filter_main_application_spans(spans_df)

	assert filtered["context.span_id"].tolist() == ["root-chain"]


def test_build_evaluation_frame_maps_langgraph_payload_to_evaluator_fields() -> None:
	spans_df = pd.DataFrame(
		[
			{
				"name": "LangGraph",
				"span_kind": "CHAIN",
				"parent_id": None,
				"context.span_id": "root-chain",
				"attributes.input.value": '{"message": "How do I get emergency help?", "top_k": 4}',
				"attributes.output.value": (
					'{"message": "How do I get emergency help?", '
					'"context": "Emergency help is available by calling 555-123-4567.", '
					'"answer": "Call 555-123-4567 for emergency help.", '
					'"response": {"answer": "Call 555-123-4567 for emergency help."}}'
				),
			}
		]
	)

	evaluation_df = _build_evaluation_frame(spans_df, span_kind="CHAIN")

	row = evaluation_df.iloc[0]
	assert row["input"] == "How do I get emergency help?"
	assert row["output"] == "Call 555-123-4567 for emergency help."
	assert row["reference"] == "Emergency help is available by calling 555-123-4567."


def test_merge_final_rca_into_annotations_preserves_columns_and_adds_rca_metadata() -> None:
	annotation_df = pd.DataFrame(
		[
			{
				"span_id": "span-1",
				"annotation_name": "safety",
				"annotator_kind": "CODE",
				"label": "pass",
				"score": 1.0,
				"explanation": "safe",
				"metadata": {"source_evaluator": "safety"},
			}
		]
	)
	rca_features_df = pd.DataFrame(
		[
			{
				"span_id": "span-1",
				"context_trace_id": "trace-1",
			}
		]
	)
	final_rca_df = pd.DataFrame(
		[
			{
				"trace_id": "trace-1",
				"evaluator_name": "safety",
				"root_cause_category": "retrieval_gap",
				"confidence": 0.87,
				"recommended_action": "Improve retrieval coverage.",
			}
		]
	)

	merged = merge_final_rca_into_annotations(
		annotation_df=annotation_df,
		final_rca_df=final_rca_df,
		rca_features_df=rca_features_df,
		selected_features=[
			"root_cause_category",
			"confidence",
			"recommended_action",
		],
	)

	assert list(merged.columns) == list(annotation_df.columns)
	assert merged.loc[0, "metadata"] == {
		"source_evaluator": "safety",
		"rca_root_cause_category": "retrieval_gap",
		"rca_confidence": 0.87,
		"rca_recommended_action": "Improve retrieval coverage.",
		"rca_evaluator_name": "safety",
	}


def test_detect_degradation_alerts_when_low_score_percentage_meets_threshold() -> None:
	annotations_df = pd.DataFrame(
		[
			{"annotation_name": "correctness", "score": 0.0},
			{"annotation_name": "correctness", "score": 0.25},
			{"annotation_name": "correctness", "score": 1.0},
		]
	)
	config = {
		"enabled": True,
		"evaluators": {
			"correctness": {
				"score_threshold": 0.5,
				"percent_threshold": 60,
			}
		},
	}

	detected_alerts = alerts.detect_degradation_alerts(annotations_df, config)

	assert detected_alerts == [
		{
			"evaluator": "correctness",
			"low_score_count": 2,
			"total_count": 3,
			"low_score_percent": 66.66666666666666,
			"score_threshold": 0.5,
			"percent_threshold": 60.0,
		}
	]


def test_detect_degradation_alerts_does_not_alert_below_threshold() -> None:
	annotations_df = pd.DataFrame(
		[
			{"annotation_name": "safety", "score": 0.0},
			{"annotation_name": "safety", "score": 1.0},
		]
	)
	config = {
		"enabled": True,
		"evaluators": {
			"safety": {
				"score_threshold": 0.5,
				"percent_threshold": 75,
			}
		},
	}

	assert alerts.detect_degradation_alerts(annotations_df, config) == []


def test_detect_degradation_alerts_is_disabled_by_config() -> None:
	annotations_df = pd.DataFrame([{"annotation_name": "relevance", "score": 0.0}])
	config = {
		"enabled": False,
		"evaluators": {
			"relevance": {
				"score_threshold": 0.5,
				"percent_threshold": 1,
			}
		},
	}

	assert alerts.detect_degradation_alerts(annotations_df, config) == []


def test_detect_degradation_alerts_skips_unconfigured_evaluators() -> None:
	annotations_df = pd.DataFrame([{"annotation_name": "correctness", "score": 0.0}])
	config = {
		"enabled": True,
		"evaluators": {
			"safety": {
				"score_threshold": 0.5,
				"percent_threshold": 1,
			}
		},
	}

	assert alerts.detect_degradation_alerts(annotations_df, config) == []


def test_detect_degradation_alerts_skips_invalid_threshold_config(caplog) -> None:
	annotations_df = pd.DataFrame([{"annotation_name": "correctness", "score": 0.0}])
	config = {
		"enabled": True,
		"evaluators": {
			"correctness": {
				"score_threshold": 1.5,
				"percent_threshold": 1,
			}
		},
	}

	assert alerts.detect_degradation_alerts(annotations_df, config) == []
	assert "Skipping degradation alert detection" in caplog.text


def test_evaluate_span_batch_runs_degradation_alert_check_after_annotations(monkeypatch) -> None:
	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			return pd.DataFrame(
				[
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-1",
						"context.trace_id": "trace-1",
						"attributes.input.value": '{"message": "Question?"}',
						"attributes.output.value": '{"answer": "Answer.", "context": "Context."}',
					}
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			return pd.DataFrame()

		def log_span_annotations_dataframe(self, **kwargs):
			raise AssertionError("save_annotations=False should skip Phoenix logging")

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [
				{"score": 1.0, "label": "CORRECT", "explanation": "ok"}
			]
			return scored_df

	seen_annotations: list[pd.DataFrame] = []

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])
	monkeypatch.setattr(
		alerts,
		"log_degradation_alerts",
		lambda annotations_df: seen_annotations.append(annotations_df.copy()),
	)

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			save_annotations=False,
			from_time="2026-06-16T10:00:00+00:00",
			to_time="2026-06-16T10:05:00+00:00",
		)
	)

	assert result.annotation_count == 1
	assert len(seen_annotations) == 1
	assert seen_annotations[0]["annotation_name"].tolist() == ["correctness"]


def test_evaluate_span_batch_continues_when_rca_judge_fails(monkeypatch) -> None:
	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			return pd.DataFrame(
				[
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-1",
						"context.trace_id": "trace-1",
						"status_code": "OK",
						"status_message": "",
						"start_time": "2026-06-16T10:00:00+00:00",
						"end_time": "2026-06-16T10:00:01+00:00",
						"attributes.metadata": '{"ls_integration":"langgraph","langgraph_node":"finalize"}',
						"attributes.input.value": '{"message": "Question?"}',
						"attributes.output.value": (
							'{"answer": "Answer.", "context": "Context.", '
							'"response": {"answer": "Answer."}, '
							'"sources": [{"score": 0.2, "metadata": {}}]}'
						),
					}
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			return pd.DataFrame()

		def log_span_annotations_dataframe(self, **kwargs):
			raise AssertionError("save_annotations=False should skip Phoenix logging")

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [
				{"score": 0.0, "label": "INCORRECT", "explanation": "bad"}
			]
			scored_df["relevance_score"] = [
				{"score": 1.0, "label": "RELEVANT", "explanation": "ok"}
			]
			scored_df["groundedness_score"] = [
				{"score": 1.0, "label": "GROUNDED", "explanation": "ok"}
			]
			scored_df["safety_score"] = [
				{"score": 1.0, "label": "SAFE", "explanation": "ok"}
			]
			return scored_df

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(
		"ragbot.rca.rca_service.RCAJudge.judge",
		lambda self, evidence: (_ for _ in ()).throw(RCAJudgeError("RCA LLM judge returned invalid JSON.")),
	)

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			save_annotations=False,
			from_time="2026-06-16T10:00:00+00:00",
			to_time="2026-06-16T10:05:00+00:00",
		)
	)

	assert result.annotation_count == 4
	correctness_metadata = result.annotations_df.loc[
		result.annotations_df["annotation_name"] == "correctness", "metadata"
	].iloc[0]
	assert correctness_metadata["rca_root_cause_category"] == "UNKNOWN"
	assert correctness_metadata["rca_confidence"] == 0.0
	assert "fallback used" in correctness_metadata["rca_explanation"]
