from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ragbot.evaluations.batch_evaluation import alerts
from ragbot.evaluations.batch_evaluation import evaluate_batch as batch_module
from ragbot.evaluations.batch_evaluation.evaluate_batch import (
	BatchEvaluationCheckpoint,
	BatchEvaluationConfig,
	_build_evaluation_frame,
	_filter_main_application_spans,
	build_composite_checkpoint_key,
	merge_final_rca_into_annotations,
	read_checkpoint,
	resolve_batch_evaluation_config,
	update_checkpoint_after_success,
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


def test_resolve_batch_evaluation_config_uses_explicit_from_time_on_first_checkpoint_run(tmp_path: Path) -> None:
	config = resolve_batch_evaluation_config(
		BatchEvaluationConfig(
			from_time="2026-06-16T10:00:00+00:00",
			to_time="2026-06-16T10:05:00+00:00",
			use_checkpoint=True,
			checkpoint_file=tmp_path / "checkpoint.json",
		)
	)

	assert config.from_time == datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc)


def test_resolve_batch_evaluation_config_uses_full_history_when_checkpoint_missing(tmp_path: Path, monkeypatch) -> None:
	fixed_now = datetime(2026, 6, 16, 10, 5, tzinfo=timezone.utc)
	monkeypatch.setattr(batch_module, "_utc_now", lambda: fixed_now)

	config = resolve_batch_evaluation_config(
		BatchEvaluationConfig(
			use_checkpoint=True,
			checkpoint_file=tmp_path / "checkpoint.json",
		)
	)

	assert config.from_time is None
	assert config.to_time == fixed_now


def test_resolve_batch_evaluation_config_uses_full_history_when_checkpoint_empty(tmp_path: Path, monkeypatch) -> None:
	fixed_now = datetime(2026, 6, 16, 10, 5, tzinfo=timezone.utc)
	monkeypatch.setattr(batch_module, "_utc_now", lambda: fixed_now)
	checkpoint_file = tmp_path / "checkpoint.json"
	checkpoint_file.write_text("", encoding="utf-8")

	config = resolve_batch_evaluation_config(
		BatchEvaluationConfig(
			use_checkpoint=True,
			checkpoint_file=checkpoint_file,
		)
	)

	assert config.from_time is None
	assert config.to_time == fixed_now


def test_resolve_batch_evaluation_config_uses_high_watermark_exactly(tmp_path: Path) -> None:
	checkpoint_file = tmp_path / "checkpoint.json"
	batch_module.write_checkpoint(
		checkpoint_file,
		BatchEvaluationCheckpoint(
			high_watermark_time=datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc),
			processed_keys=(),
		),
	)

	config = resolve_batch_evaluation_config(
		BatchEvaluationConfig(
			use_checkpoint=True,
			checkpoint_file=checkpoint_file,
			to_time="2026-06-16T10:05:00+00:00",
		)
	)

	assert config.from_time == datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc)


def test_evaluate_span_batch_bootstrap_skips_only_fully_annotated_spans(monkeypatch, tmp_path: Path) -> None:
	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			assert kwargs["from_time"] is None
			assert kwargs["root_spans_only"] is True
			return pd.DataFrame(
				[
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-partial",
						"context.trace_id": "trace-partial",
						"start_time": "2026-06-16T09:00:00+00:00",
						"attributes.input.value": '{"message": "Partial?"}',
						"attributes.output.value": '{"answer": "Answer 1.", "context": "Context 1."}',
					},
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-full",
						"context.trace_id": "trace-full",
						"start_time": "2026-06-16T09:01:00+00:00",
						"attributes.input.value": '{"message": "Full?"}',
						"attributes.output.value": '{"answer": "Answer 2.", "context": "Context 2."}',
					},
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			return pd.DataFrame(
				[
					{"span_id": "span-partial", "annotation_name": "correctness"},
					{"span_id": "span-full", "annotation_name": "correctness"},
					{"span_id": "span-full", "annotation_name": "relevance"},
					{"span_id": "span-full", "annotation_name": "faithfulness"},
					{"span_id": "span-full", "annotation_name": "safety"},
				]
			)

		def log_span_annotations_dataframe(self, **kwargs):
			raise AssertionError("save_annotations=False should skip Phoenix logging")

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			assert evaluation_df["span_id"].tolist() == ["span-partial"]
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [{"score": 1.0, "label": "CORRECT", "explanation": "ok"}]
			return scored_df

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			save_annotations=False,
			use_checkpoint=True,
			checkpoint_file=tmp_path / "checkpoint.json",
			to_time="2026-06-16T10:05:00+00:00",
		)
	)

	assert result.span_count == 1
	assert result.evaluation_df["span_id"].tolist() == ["span-partial"]


def test_evaluate_span_batch_mixed_annotation_completeness_is_decided_per_span(monkeypatch, tmp_path: Path) -> None:
	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			assert kwargs["from_time"] is None
			assert kwargs["root_spans_only"] is True
			return pd.DataFrame(
				[
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-empty",
						"context.trace_id": "trace-empty",
						"start_time": "2026-06-16T09:00:00+00:00",
						"attributes.input.value": '{"message": "Empty?"}',
						"attributes.output.value": '{"answer": "Answer 1.", "context": "Context 1."}',
					},
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-partial",
						"context.trace_id": "trace-partial",
						"start_time": "2026-06-16T09:01:00+00:00",
						"attributes.input.value": '{"message": "Partial?"}',
						"attributes.output.value": '{"answer": "Answer 2.", "context": "Context 2."}',
					},
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-full",
						"context.trace_id": "trace-full",
						"start_time": "2026-06-16T09:02:00+00:00",
						"attributes.input.value": '{"message": "Full?"}',
						"attributes.output.value": '{"answer": "Answer 3.", "context": "Context 3."}',
					},
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			assert kwargs["span_ids"] == ["span-empty", "span-partial", "span-full"]
			return pd.DataFrame(
				[
					{"span_id": "span-partial", "annotation_name": "correctness"},
					{"span_id": "span-full", "annotation_name": "correctness"},
					{"span_id": "span-full", "annotation_name": "relevance"},
					{"span_id": "span-full", "annotation_name": "faithfulness"},
					{"span_id": "span-full", "annotation_name": "safety"},
				]
			)

		def log_span_annotations_dataframe(self, **kwargs):
			raise AssertionError("save_annotations=False should skip Phoenix logging")

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			assert evaluation_df["span_id"].tolist() == ["span-empty", "span-partial"]
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [
				{"score": 1.0, "label": "CORRECT", "explanation": "ok"},
				{"score": 1.0, "label": "CORRECT", "explanation": "ok"},
			]
			return scored_df

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			save_annotations=False,
			use_checkpoint=True,
			checkpoint_file=tmp_path / "checkpoint.json",
			to_time="2026-06-16T10:05:00+00:00",
		)
	)

	assert result.span_count == 2
	assert result.evaluation_df["span_id"].tolist() == ["span-empty", "span-partial"]


def test_evaluate_span_batch_recovers_span_id_from_annotation_index(monkeypatch, tmp_path: Path) -> None:
	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			assert kwargs["from_time"] is None
			assert kwargs["root_spans_only"] is True
			return pd.DataFrame(
				[
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-full",
						"context.trace_id": "trace-full",
						"start_time": "2026-06-16T09:00:00+00:00",
						"attributes.input.value": '{"message": "Full?"}',
						"attributes.output.value": '{"answer": "Answer 1.", "context": "Context 1."}',
					},
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-missing",
						"context.trace_id": "trace-missing",
						"start_time": "2026-06-16T09:01:00+00:00",
						"attributes.input.value": '{"message": "Missing?"}',
						"attributes.output.value": '{"answer": "Answer 2.", "context": "Context 2."}',
					},
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			assert kwargs["span_ids"] == ["span-full", "span-missing"]
			return pd.DataFrame(
				[
					{"annotation_name": "correctness"},
					{"annotation_name": "relevance"},
					{"annotation_name": "faithfulness"},
					{"annotation_name": "safety"},
				],
				index=pd.Index(["span-full"] * 4, name="span_id"),
			)

		def log_span_annotations_dataframe(self, **kwargs):
			raise AssertionError("save_annotations=False should skip Phoenix logging")

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			assert evaluation_df["span_id"].tolist() == ["span-missing"]
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [{"score": 1.0, "label": "CORRECT", "explanation": "ok"}]
			return scored_df

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			save_annotations=False,
			use_checkpoint=True,
			checkpoint_file=tmp_path / "checkpoint.json",
			to_time="2026-06-16T10:05:00+00:00",
		)
	)

	assert result.span_count == 1
	assert result.evaluation_df["span_id"].tolist() == ["span-missing"]


def test_evaluate_span_batch_checks_annotation_presence_only_on_root_langgraph_span(monkeypatch, tmp_path: Path) -> None:
	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			assert kwargs["from_time"] is None
			assert kwargs["root_spans_only"] is True
			return pd.DataFrame(
				[
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "root-span",
						"context.trace_id": "trace-root",
						"start_time": "2026-06-16T09:00:00+00:00",
						"attributes.input.value": '{"message": "Root question?"}',
						"attributes.output.value": '{"answer": "Root answer.", "context": "Root context."}',
					},
					{
						"name": "child-node",
						"span_kind": "CHAIN",
						"parent_id": "root-span",
						"span_id": "child-span",
						"context.trace_id": "trace-root",
						"start_time": "2026-06-16T09:00:01+00:00",
						"attributes.input.value": '{"message": "Child question?"}',
						"attributes.output.value": '{"answer": "Child answer.", "context": "Child context."}',
					},
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			assert kwargs["span_ids"] == ["root-span"]
			return pd.DataFrame(
				[
					{"span_id": "child-span", "annotation_name": "correctness"},
					{"span_id": "child-span", "annotation_name": "relevance"},
					{"span_id": "child-span", "annotation_name": "faithfulness"},
					{"span_id": "child-span", "annotation_name": "safety"},
				]
			)

		def log_span_annotations_dataframe(self, **kwargs):
			raise AssertionError("save_annotations=False should skip Phoenix logging")

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			assert evaluation_df["span_id"].tolist() == ["root-span"]
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [{"score": 1.0, "label": "CORRECT", "explanation": "ok"}]
			return scored_df

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			save_annotations=False,
			use_checkpoint=True,
			checkpoint_file=tmp_path / "checkpoint.json",
			to_time="2026-06-16T10:05:00+00:00",
		)
	)

	assert result.span_count == 1
	assert result.evaluation_df["span_id"].tolist() == ["root-span"]


def test_evaluate_span_batch_only_considers_root_chain_langgraph_candidates(monkeypatch, tmp_path: Path) -> None:
	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			assert kwargs["from_time"] is None
			assert kwargs["root_spans_only"] is True
			return pd.DataFrame(
				[
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "eligible-root",
						"context.trace_id": "trace-eligible",
						"start_time": "2026-06-16T09:00:00+00:00",
						"attributes.input.value": '{"message": "Eligible?"}',
						"attributes.output.value": '{"answer": "Eligible answer.", "context": "Eligible context."}',
					},
					{
						"name": "NotLangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "other-root",
						"context.trace_id": "trace-other",
						"start_time": "2026-06-16T09:01:00+00:00",
						"attributes.input.value": '{"message": "Other?"}',
						"attributes.output.value": '{"answer": "Other answer.", "context": "Other context."}',
					},
					{
						"name": "LangGraph",
						"span_kind": "LLM",
						"parent_id": None,
						"span_id": "llm-root",
						"context.trace_id": "trace-llm",
						"start_time": "2026-06-16T09:02:00+00:00",
						"attributes.input.value": '{"message": "LLM?"}',
						"attributes.output.value": '{"answer": "LLM answer.", "context": "LLM context."}',
					},
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			assert kwargs["span_ids"] == ["eligible-root"]
			return pd.DataFrame()

		def log_span_annotations_dataframe(self, **kwargs):
			raise AssertionError("save_annotations=False should skip Phoenix logging")

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			assert evaluation_df["span_id"].tolist() == ["eligible-root"]
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [{"score": 1.0, "label": "CORRECT", "explanation": "ok"}]
			return scored_df

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			save_annotations=False,
			use_checkpoint=True,
			checkpoint_file=tmp_path / "checkpoint.json",
			to_time="2026-06-16T10:05:00+00:00",
		)
	)

	assert result.span_count == 1
	assert result.evaluation_df["span_id"].tolist() == ["eligible-root"]


def test_evaluate_span_batch_skips_only_processed_composite_keys_with_duplicate_timestamps(monkeypatch, tmp_path: Path) -> None:
	checkpoint_file = tmp_path / "checkpoint.json"
	processed_span = {
		"name": "LangGraph",
		"span_kind": "CHAIN",
		"parent_id": None,
		"span_id": "span-1",
		"context.trace_id": "trace-1",
		"start_time": "2026-06-16T10:00:00+00:00",
	}
	batch_module.write_checkpoint(
		checkpoint_file,
		BatchEvaluationCheckpoint(
			high_watermark_time=datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc),
			processed_keys=(build_composite_checkpoint_key(processed_span),),
		),
	)

	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			return pd.DataFrame(
				[
					{
						**processed_span,
						"attributes.input.value": '{"message": "Question 1?"}',
						"attributes.output.value": '{"answer": "Answer 1.", "context": "Context 1."}',
					},
					{
						"name": "LangGraph",
						"span_kind": "CHAIN",
						"parent_id": None,
						"span_id": "span-2",
						"context.trace_id": "trace-2",
						"start_time": "2026-06-16T10:00:00+00:00",
						"attributes.input.value": '{"message": "Question 2?"}',
						"attributes.output.value": '{"answer": "Answer 2.", "context": "Context 2."}',
					},
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			return pd.DataFrame()

		def log_span_annotations_dataframe(self, **kwargs):
			self.logged_annotations = kwargs["annotations_df"].copy()

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			assert evaluation_df["span_id"].tolist() == ["span-2"]
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [{"score": 1.0, "label": "CORRECT", "explanation": "ok"}]
			return scored_df

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			to_time="2026-06-16T10:05:00+00:00",
			use_checkpoint=True,
			checkpoint_file=checkpoint_file,
		)
	)

	assert result.span_count == 1
	assert result.evaluation_df["span_id"].tolist() == ["span-2"]


def test_update_checkpoint_after_success_keeps_only_max_timestamp_keys(tmp_path: Path) -> None:
	checkpoint_file = tmp_path / "checkpoint.json"
	old_checkpoint = BatchEvaluationCheckpoint(
		high_watermark_time=datetime(2026, 6, 16, 10, 5, tzinfo=timezone.utc),
		processed_keys=("trace-old|2026-06-16T10:05:00+00:00|LangGraph",),
	)
	evaluated_spans_df = pd.DataFrame(
		[
			{
				"context.trace_id": "trace-new-1",
				"start_time": "2026-06-16T10:10:00+00:00",
				"name": "LangGraph",
			},
			{
				"context.trace_id": "trace-new-2",
				"start_time": "2026-06-16T10:10:00+00:00",
				"name": "LangGraph",
			},
		]
	)

	new_checkpoint = update_checkpoint_after_success(
		checkpoint_file,
		evaluated_spans_df,
		old_checkpoint,
	)

	saved_checkpoint = read_checkpoint(checkpoint_file)
	assert new_checkpoint.high_watermark_time == datetime(2026, 6, 16, 10, 10, tzinfo=timezone.utc)
	assert saved_checkpoint == new_checkpoint
	assert saved_checkpoint.processed_keys == (
		"trace-new-1|2026-06-16T10:10:00+00:00|LangGraph",
		"trace-new-2|2026-06-16T10:10:00+00:00|LangGraph",
	)


def test_evaluate_span_batch_does_not_update_checkpoint_when_annotation_write_fails(monkeypatch, tmp_path: Path) -> None:
	checkpoint_file = tmp_path / "checkpoint.json"
	original_payload = {
		"high_watermark_time": "2026-06-16T09:55:00+00:00",
		"processed_keys": ["trace-old|2026-06-16T09:55:00+00:00|LangGraph"],
	}
	checkpoint_file.write_text(json.dumps(original_payload), encoding="utf-8")

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
						"start_time": "2026-06-16T10:00:00+00:00",
						"attributes.input.value": '{"message": "Question?"}',
						"attributes.output.value": '{"answer": "Answer.", "context": "Context."}',
					}
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			return pd.DataFrame()

		def log_span_annotations_dataframe(self, **kwargs):
			raise RuntimeError("phoenix write failed")

	class FakeRunner:
		async def evaluate_dataframe(self, evaluation_df):
			scored_df = evaluation_df.copy()
			scored_df["correctness_score"] = [{"score": 1.0, "label": "CORRECT", "explanation": "ok"}]
			return scored_df

	monkeypatch.setattr(batch_module, "_build_runner", lambda: FakeRunner())
	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])

	try:
		batch_module.run_span_batch(
			BatchEvaluationConfig(
				adapter=FakeAdapter(),
				from_time="2026-06-16T09:55:00+00:00",
				to_time="2026-06-16T10:05:00+00:00",
				use_checkpoint=True,
				checkpoint_file=checkpoint_file,
			)
		)
	except RuntimeError as exc:
		assert str(exc) == "phoenix write failed"
	else:
		raise AssertionError("Expected Phoenix write failure")

	assert json.loads(checkpoint_file.read_text(encoding="utf-8")) == original_payload


def test_evaluate_span_batch_leaves_checkpoint_unchanged_when_all_spans_are_already_processed(monkeypatch, tmp_path: Path) -> None:
	checkpoint_file = tmp_path / "checkpoint.json"
	processed_span = {
		"name": "LangGraph",
		"span_kind": "CHAIN",
		"parent_id": None,
		"span_id": "span-1",
		"context.trace_id": "trace-1",
		"start_time": "2026-06-16T10:00:00+00:00",
	}
	batch_module.write_checkpoint(
		checkpoint_file,
		BatchEvaluationCheckpoint(
			high_watermark_time=datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc),
			processed_keys=(build_composite_checkpoint_key(processed_span),),
		),
	)

	class FakeAdapter:
		def fetch_spans_dataframe(self, **kwargs):
			return pd.DataFrame(
				[
					{
						**processed_span,
						"attributes.input.value": '{"message": "Question?"}',
						"attributes.output.value": '{"answer": "Answer.", "context": "Context."}',
					}
				]
			)

		def fetch_span_annotations_dataframe(self, **kwargs):
			return pd.DataFrame()

		def log_span_annotations_dataframe(self, **kwargs):
			raise AssertionError("No Phoenix write expected when nothing survives checkpoint filtering")

	monkeypatch.setattr(batch_module, "generate_rca", lambda rca_df: [])

	result = batch_module.run_span_batch(
		BatchEvaluationConfig(
			adapter=FakeAdapter(),
			from_time="2026-06-16T09:55:00+00:00",
			to_time="2026-06-16T10:05:00+00:00",
			use_checkpoint=True,
			checkpoint_file=checkpoint_file,
		)
	)

	assert result.span_count == 0
	assert read_checkpoint(checkpoint_file).processed_keys == (
		build_composite_checkpoint_key(processed_span),
	)
