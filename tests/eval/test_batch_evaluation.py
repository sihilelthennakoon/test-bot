from __future__ import annotations

import pandas as pd

from ragbot.evaluations.batch_evaluation.evaluate_batch import (
	_build_evaluation_frame,
	_filter_main_application_spans,
	merge_final_rca_into_annotations,
)


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
