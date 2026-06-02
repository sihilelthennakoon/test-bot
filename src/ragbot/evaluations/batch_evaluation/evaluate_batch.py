"""Batch Phoenix span evaluation workflow.

This module implements the production pattern described in the request:

Pull -> spans_df = client.spans.get_spans_dataframe()
Evaluate -> correctness, relevance, faithfulness, safety
Write Back -> client.spans.log_span_annotations_dataframe(...)

The implementation is intentionally minimal-friction: it reads spans from Phoenix,
runs the local evaluator factory functions already used by the project, and writes
span annotations back into Phoenix so they can be filtered directly in the UI.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from phoenix.client import Client

from ragbot.evaluations import correctness, groundedness, relevance, safety
from ragbot.evaluations.runner import EvaluationRunner


PASS_LABELS = {
	"CORRECT",
	"RELEVANT",
	"GROUNDED",
	"SAFE",
	"VALID",
	"PASS",
	"PASSING",
}

INPUT_CANDIDATES = (
	"input",
	"input_text",
	"question",
	"query",
	"prompt",
	"message",
	"user_message",
	"span_name",
	"name",
)

OUTPUT_CANDIDATES = (
	"output",
	"response",
	"response_text",
	"answer",
	"completion",
	"content",
)

REFERENCE_CANDIDATES = (
	"reference",
	"context",
	"retrieved_context",
	"source_context",
	"documents",
	"retrieved_docs",
)


def _default_phoenix_base_url() -> str:
	return (
		os.getenv("PHOENIX_QUERY_ENDPOINT")
		or os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
		or "http://localhost:6006"
	)


def _normalize_datetime(value: datetime | str | None) -> datetime | None:
	if value is None:
		return None
	if isinstance(value, datetime):
		if value.tzinfo is None:
			return value.replace(tzinfo=timezone.utc)
		return value.astimezone(timezone.utc)
	normalized = value.strip().replace("Z", "+00:00")
	parsed = datetime.fromisoformat(normalized)
	if parsed.tzinfo is None:
		return parsed.replace(tzinfo=timezone.utc)
	return parsed.astimezone(timezone.utc)


def _coerce_text(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, str):
		return value.strip()
	if isinstance(value, dict):
		return "\n\n".join(
			part
			for part in (_coerce_text(item) for item in value.values())
			if part
		)
	if isinstance(value, (list, tuple, set)):
		return "\n\n".join(part for part in (_coerce_text(item) for item in value) if part)
	return str(value).strip()


def _first_non_empty(record: dict[str, Any], keys: tuple[str, ...]) -> str:
	attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}

	def lookup(source: dict[str, Any], key: str) -> Any:
		if key in source:
			return source.get(key)
		dotted = key.replace("_", ".")
		if dotted in source:
			return source.get(dotted)
		underscored = key.replace(".", "_")
		if underscored in source:
			return source.get(underscored)
		return None

	for key in keys:
		candidate = _coerce_text(lookup(record, key))
		if candidate:
			return candidate

		if attributes:
			candidate = _coerce_text(lookup(attributes, key))
			if candidate:
				return candidate

	return ""


def _span_kind(record: dict[str, Any]) -> str:
	attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
	candidates = (
		record.get("span_kind"),
		record.get("kind"),
		record.get("spanKind"),
		record.get("span_kind_name"),
		attributes.get("openinference.span.kind") if attributes else None,
		attributes.get("span.kind") if attributes else None,
		attributes.get("span_kind") if attributes else None,
	)
	for candidate in candidates:
		text = _coerce_text(candidate)
		if text:
			return text.upper()
	return ""


def _ensure_span_id(record: dict[str, Any], fallback_index: int) -> str:
	span_id = _coerce_text(
		record.get("span_id")
		or record.get("spanId")
		or record.get("id")
		or record.get("span_uuid")
	)
	if span_id:
		return span_id
	return f"span_{fallback_index}"


def _build_evaluation_frame(spans_df: pd.DataFrame, span_kind: str | None = None) -> pd.DataFrame:
	normalized = spans_df.copy()

	if "span_id" not in normalized.columns:
		normalized = normalized.reset_index()
		if "span_id" not in normalized.columns and "index" in normalized.columns:
			normalized = normalized.rename(columns={"index": "span_id"})

	rows: list[dict[str, Any]] = []
	for index, (_, series) in enumerate(normalized.iterrows()):
		record = series.to_dict()
		record_span_kind = _span_kind(record)
		if span_kind and record_span_kind and record_span_kind != span_kind.upper():
			continue

		span_id = _coerce_text(record.get("span_id")) or _ensure_span_id(record, index)
		input_text = _first_non_empty(record, INPUT_CANDIDATES)
		output_text = _first_non_empty(record, OUTPUT_CANDIDATES)
		reference_text = _first_non_empty(record, REFERENCE_CANDIDATES)

		rows.append(
			{
				**record,
				"span_id": span_id,
				"input": input_text,
				"output": output_text,
				"reference": reference_text,
				"span_kind": record_span_kind or record.get("span_kind") or record.get("kind") or "",
			}
		)

	if not rows:
		return pd.DataFrame(columns=["span_id", "input", "output", "reference", "span_kind"])

	return pd.DataFrame(rows)


def _build_runner() -> EvaluationRunner:
	return EvaluationRunner(
		evaluators={
			"correctness": correctness.create_correctness_evaluator(),
			"relevance": relevance.create_relevance_evaluator(),
			"groundedness": groundedness.create_groundedness_evaluator(),
			"safety": safety.create_safety_evaluator(),
		}
	)


def _score_cell_to_annotation_payload(score_cell: Any) -> tuple[float, str, str, dict[str, Any]]:
	if isinstance(score_cell, dict):
		score = float(score_cell.get("score", 0.0) or 0.0)
		raw_label = _coerce_text(score_cell.get("label")).upper()
		explanation = _coerce_text(score_cell.get("explanation"))
		metadata = score_cell.get("metadata") if isinstance(score_cell.get("metadata"), dict) else {}
	else:
		score = float(score_cell or 0.0)
		raw_label = ""
		explanation = ""
		metadata = {}

	label = "pass" if raw_label in PASS_LABELS else "fail"
	if not raw_label:
		label = "pass" if score >= 0.5 else "fail"

	return score, label, explanation, dict(metadata)


def _build_annotations_frame(evaluated_df: pd.DataFrame) -> pd.DataFrame:
	annotation_rows: list[dict[str, Any]] = []
	evaluator_to_annotation_name = {
		"correctness": "correctness",
		"relevance": "relevance",
		"groundedness": "faithfulness",
		"safety": "safety",
	}

	for _, series in evaluated_df.iterrows():
		record = series.to_dict()
		span_id = _coerce_text(record.get("span_id"))
		if not span_id:
			continue

		for evaluator_name, annotation_name in evaluator_to_annotation_name.items():
			score_column = f"{evaluator_name}_score"
			if score_column not in record:
				continue

			score_cell = record[score_column]
			if score_cell is None:
				continue

			score, label, explanation, metadata = _score_cell_to_annotation_payload(score_cell)
			annotation_rows.append(
				{
					"span_id": span_id,
					"annotation_name": annotation_name,
					"annotator_kind": "CODE",
					"label": label,
					"score": score,
					"explanation": explanation,
					"metadata": {
						**metadata,
						"source_evaluator": evaluator_name,
						"source_label": _coerce_text(score_cell.get("label")).upper() if isinstance(score_cell, dict) else "",
					},
				}
			)

	return pd.DataFrame(annotation_rows)


@dataclass(frozen=True, slots=True)
class BatchEvaluationConfig:
	"""Configuration for a Phoenix span evaluation batch."""

	from_time: datetime | str | None = None
	to_time: datetime | str | None = None
	project_name: str | None = None
	span_kind: str | None = None
	limit: int = 1000
	root_spans_only: bool | None = None
	phoenix_base_url: str = _default_phoenix_base_url()
	sync_annotations: bool = True


@dataclass(frozen=True, slots=True)
class BatchEvaluationResult:
	"""Result payload for a Phoenix span evaluation batch."""

	spans_df: pd.DataFrame
	evaluation_df: pd.DataFrame
	annotations_df: pd.DataFrame

	@property
	def span_count(self) -> int:
		return int(len(self.spans_df))

	@property
	def annotation_count(self) -> int:
		return int(len(self.annotations_df))


async def evaluate_span_batch(config: BatchEvaluationConfig) -> BatchEvaluationResult:
	"""Pull Phoenix spans, evaluate them, and write the annotations back."""

	client = Client(base_url=config.phoenix_base_url)
	spans_df = client.spans.get_spans_dataframe(
		start_time=_normalize_datetime(config.from_time),
		end_time=_normalize_datetime(config.to_time),
		limit=config.limit,
		root_spans_only=config.root_spans_only,
		project_name=config.project_name,
	)

	if spans_df.empty:
		annotations_df = pd.DataFrame(
			columns=["span_id", "annotation_name", "annotator_kind", "label", "score", "explanation", "metadata"]
		)
		return BatchEvaluationResult(spans_df=spans_df, evaluation_df=spans_df.copy(), annotations_df=annotations_df)

	evaluation_df = _build_evaluation_frame(spans_df, span_kind=config.span_kind)
	if evaluation_df.empty:
		annotations_df = pd.DataFrame(
			columns=["span_id", "annotation_name", "annotator_kind", "label", "score", "explanation", "metadata"]
		)
		return BatchEvaluationResult(spans_df=spans_df, evaluation_df=evaluation_df, annotations_df=annotations_df)

	runner = _build_runner()
	scored_df = await runner.evaluate_dataframe(evaluation_df)
	annotations_df = _build_annotations_frame(scored_df)

	if not annotations_df.empty:
		client.spans.log_span_annotations_dataframe(
			dataframe=annotations_df,
			sync=config.sync_annotations,
		)

	return BatchEvaluationResult(
		spans_df=spans_df,
		evaluation_df=scored_df,
		annotations_df=annotations_df,
	)


def run_span_batch(config: BatchEvaluationConfig) -> BatchEvaluationResult:
	"""Synchronous wrapper for the batch evaluation job."""

	return asyncio.run(evaluate_span_batch(config))


__all__ = [
	"BatchEvaluationConfig",
	"BatchEvaluationResult",
	"evaluate_span_batch",
	"run_span_batch",
]
