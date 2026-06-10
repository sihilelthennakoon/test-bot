from __future__ import annotations

import pandas as pd

from ragbot.evaluations.batch_evaluation.evaluate_batch import merge_final_rca_into_annotations


def test_merge_final_rca_into_annotations_replaces_metadata_without_extra_columns() -> None:
	annotation_df = pd.DataFrame(
		[
			{
				"span_id": "span-1",
				"annotation_name": "safety",
				"annotator_kind": "CODE",
				"label": "pass",
				"score": 1.0,
				"explanation": "safe",
				"metadata": {"old": "value"},
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
		"rca_root_cause_category": "retrieval_gap",
		"rca_confidence": 0.87,
		"rca_recommended_action": "Improve retrieval coverage.",
	}
