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

import ast
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from data_fetch.fetch_from_phoenix import (
	build_default_adapter,
	fetch_phoenix_spans_dataframe,
	log_phoenix_span_annotations,
	RawSpansDataFrameRequest,
	PhoenixAdapter,
)
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
	"input.value",
	"input.text",
	"input.message",
	"input.messages",
	"input_text",
	"question",
	"query",
	"prompt",
	"message",
	"user_message",
	"llm.input_messages",
	"llm.prompts",
	"span_name",
	"name",
)

OUTPUT_CANDIDATES = (
	"output",
	"output.value",
	"output.text",
	"output.message",
	"output.messages",
	"response",
	"response_text",
	"answer",
	"completion",
	"content",
	"llm.output_messages",
	"llm.completions",
)

REFERENCE_CANDIDATES = (
	"reference",
	"context",
	"context.value",
	"retrieved_context",
	"source_context",
	"documents",
	"retrieved_docs",
	"retrieval.documents",
	"retrieval.documents.document.content",
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
		text = value.strip()
		if not text:
			return ""
		if text[0] in {"{", "["}:
			try:
				return _coerce_text(json.loads(text))
			except json.JSONDecodeError:
				return text
		return text
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
		candidates = (
			key,
			key.replace("_", "."),
			key.replace(".", "_"),
			f"attributes.{key}",
			f"attributes.{key.replace('_', '.')}",
			f"attributes.{key.replace('.', '_')}",
		)
		for candidate in candidates:
			if candidate in source:
				return source.get(candidate)
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
		record.get("openinference.span.kind"),
		record.get("attributes.openinference.span.kind"),
		record.get("attributes.span.kind"),
		record.get("attributes.span_kind"),
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
		or record.get("context.span_id")
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
		if "context.span_id" in normalized.columns:
			normalized["span_id"] = normalized["context.span_id"]
		else:
			index_name = normalized.index.name or "index"
			index_column = index_name if index_name not in normalized.columns else "__index_span_id"
			normalized = normalized.reset_index(names=index_column)
			if "span_id" not in normalized.columns:
				normalized = normalized.rename(columns={index_column: "span_id"})

	rows: list[dict[str, Any]] = []
	for index, (_, series) in enumerate(normalized.iterrows()):
		record = series.to_dict()
		record_span_kind = _span_kind(record)
		if span_kind and record_span_kind != span_kind.upper():
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


def _build_annotations_frame(evaluated_df: pd.DataFrame) -> pd.DataFrame: #tt
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


def _parse_structured_value(value: Any) -> Any:
	if value is None:
		return None
	if isinstance(value, (dict, list)):
		return value
	if isinstance(value, str):
		text = value.strip()
		if not text:
			return None
		try:
			return json.loads(text)
		except json.JSONDecodeError:
			try:
				return ast.literal_eval(text)
			except (ValueError, SyntaxError):
				return None
	return None


def _extract_message(payload: Any) -> str:
	if isinstance(payload, dict):
		return _coerce_text(
			payload.get("message")
			or payload.get("question")
			or payload.get("input")
			or payload.get("query")
		)
	return ""


def _extract_answer(payload: Any) -> str:
	if isinstance(payload, dict):
		response = payload.get("response")
		if isinstance(response, dict):
			answer = _coerce_text(response.get("answer"))
			if answer:
				return answer
		return _coerce_text(payload.get("answer") or payload.get("output") or payload.get("response"))
	return ""


def _extract_sources(payload: Any) -> list[dict[str, Any]]:
	if not isinstance(payload, dict):
		return []

	sources = payload.get("sources")
	if isinstance(sources, list):
		return [source for source in sources if isinstance(source, dict)]

	response = payload.get("response")
	if isinstance(response, dict) and isinstance(response.get("sources"), list):
		return [source for source in response["sources"] if isinstance(source, dict)]

	return []


def _extract_scores(payload: Any, sources: list[dict[str, Any]]) -> list[float]:
	scores: list[float] = []
	if isinstance(payload, dict):
		raw_scores = payload.get("scores")
		if isinstance(raw_scores, list):
			for raw_score in raw_scores:
				try:
					scores.append(float(raw_score))
				except (TypeError, ValueError):
					pass

	if scores:
		return scores

	for source in sources:
		try:
			scores.append(float(source.get("score")))
		except (TypeError, ValueError):
			continue
	return scores


def _extract_guardrail_allowed(payload: Any, kind: str) -> bool | None:
	if not isinstance(payload, dict):
		return None

	if kind == "input":
		candidates = (
			payload.get("input_decision"),
			payload.get("response", {}).get("input_safety") if isinstance(payload.get("response"), dict) else None,
		)
	else:
		candidates = (
			payload.get("output_decision"),
			payload.get("response", {}).get("output_safety") if isinstance(payload.get("response"), dict) else None,
		)

	for candidate in candidates:
		if isinstance(candidate, dict) and "allowed" in candidate:
			return bool(candidate.get("allowed"))
	return None


def _extract_pii_detected(payloads: list[Any], sources: list[dict[str, Any]]) -> bool:
	for payload in payloads:
		if not isinstance(payload, dict):
			continue

		for key in ("input_masked", "output_masked"):
			masked = payload.get(key)
			if isinstance(masked, dict) and masked.get("entities"):
				return True

		response = payload.get("response")
		if isinstance(response, dict):
			for key in ("input_safety", "output_safety"):
				safety_payload = response.get(key)
				if isinstance(safety_payload, dict) and safety_payload.get("entities"):
					return True

	for source in sources:
		metadata = source.get("metadata")
		if isinstance(metadata, dict) and metadata.get("pii_entities"):
			return True

	return False


def _contains_normalized_text(haystack: str, needle: str) -> bool | None:
	normalized_haystack = " ".join(haystack.lower().split())
	normalized_needle = " ".join(needle.lower().split())
	if not normalized_haystack or not normalized_needle:
		return None
	return normalized_needle in normalized_haystack


def build_rca_feature_frame(annotation_df: pd.DataFrame, span_df: pd.DataFrame) -> pd.DataFrame:
	"""Build an RCA-ready feature dataframe from annotation and span data."""

	if span_df.empty:
		return pd.DataFrame(
			columns=[
				"span_id",
				"eval_trace_id",
				"context_trace_id",
				"question",
				"answer",
				"correctness_score",
				"relevance_score",
				"faithfulness_score",
				"safety_score",
				"correctness_explanation",
				"relevance_explanation",
				"faithfulness_explanation",
				"safety_explanation",
				"retrieved_doc_count",
				"retrieval_scores",
				"avg_retrieval_score",
				"retrieved_context",
				"context_has_answer",
				"answer_supported_by_context",
				"input_guardrail_allowed",
				"output_guardrail_allowed",
				"pii_detected",
				"span_status",
				"span_error",
				"latency_ms",
				"app_version",
				"environment",
				"langgraph_node",
			]
		)

	working_spans_df = span_df.copy()
	if "span_id" not in working_spans_df.columns or "input" not in working_spans_df.columns or "output" not in working_spans_df.columns:
		working_spans_df = _build_evaluation_frame(working_spans_df)

	annotation_features: dict[str, dict[str, Any]] = {}
	if not annotation_df.empty:
		for _, series in annotation_df.iterrows():
			record = series.to_dict()
			span_id = _coerce_text(record.get("span_id"))
			annotation_name = _coerce_text(record.get("annotation_name"))
			if not span_id or not annotation_name:
				continue

			metadata = _parse_structured_value(record.get("metadata"))
			if not isinstance(metadata, dict):
				metadata = {}

			features = annotation_features.setdefault(span_id, {})
			features[f"{annotation_name}_score"] = record.get("score")
			features[f"{annotation_name}_explanation"] = _coerce_text(record.get("explanation"))
			if not features.get("trace_id"):
				features["trace_id"] = _coerce_text(metadata.get("trace_id"))

	rows: list[dict[str, Any]] = []
	for _, series in working_spans_df.iterrows():
		record = series.to_dict()
		span_id = _coerce_text(record.get("span_id"))
		if not span_id:
			continue

		annotation_values = annotation_features.get(span_id, {})
		metadata = _parse_structured_value(record.get("attributes.metadata"))
		if not isinstance(metadata, dict):
			metadata = {}

		input_payload = _parse_structured_value(record.get("attributes.input.value"))
		output_payload = _parse_structured_value(record.get("attributes.output.value"))
		payload_sources = _extract_sources(output_payload) or _extract_sources(input_payload)
		retrieval_scores = _extract_scores(output_payload, payload_sources) or _extract_scores(input_payload, payload_sources)
		retrieved_context = _coerce_text(
			(output_payload or {}).get("context") if isinstance(output_payload, dict) else ""
		) or _coerce_text(
			(input_payload or {}).get("context") if isinstance(input_payload, dict) else ""
		) or _coerce_text(record.get("reference"))

		question = _extract_message(input_payload) or _coerce_text(record.get("input"))
		answer = _extract_answer(output_payload) or _extract_answer(input_payload) or _coerce_text(record.get("output"))

		start_time = _normalize_datetime(record.get("start_time"))
		end_time = _normalize_datetime(record.get("end_time"))
		latency_ms: float | None = None
		if start_time and end_time:
			latency_ms = round((end_time - start_time).total_seconds() * 1000, 3)

		context_has_answer = _contains_normalized_text(retrieved_context, answer)
		faithfulness_score = annotation_values.get("faithfulness_score")
		answer_supported_by_context = None
		if faithfulness_score is not None:
			try:
				answer_supported_by_context = float(faithfulness_score) >= 0.5
			except (TypeError, ValueError):
				answer_supported_by_context = context_has_answer
		else:
			answer_supported_by_context = context_has_answer

		rows.append(
			{
				"span_id": span_id,
				"eval_trace_id": annotation_values.get("trace_id"),
				"context_trace_id": _coerce_text(record.get("context.trace_id")),
				"question": question,
				"answer": answer,
				"correctness_score": annotation_values.get("correctness_score"),
				"relevance_score": annotation_values.get("relevance_score"),
				"faithfulness_score": annotation_values.get("faithfulness_score"),
				"safety_score": annotation_values.get("safety_score"),
				"correctness_explanation": annotation_values.get("correctness_explanation", ""),
				"relevance_explanation": annotation_values.get("relevance_explanation", ""),
				"faithfulness_explanation": annotation_values.get("faithfulness_explanation", ""),
				"safety_explanation": annotation_values.get("safety_explanation", ""),
				"retrieved_doc_count": len(payload_sources),
				"retrieval_scores": retrieval_scores,
				"avg_retrieval_score": round(sum(retrieval_scores) / len(retrieval_scores), 6) if retrieval_scores else None,
				"retrieved_context": retrieved_context,
				"context_has_answer": context_has_answer,
				"answer_supported_by_context": answer_supported_by_context,
				"input_guardrail_allowed": _extract_guardrail_allowed(input_payload, "input") if input_payload else None,
				"output_guardrail_allowed": _extract_guardrail_allowed(output_payload, "output") if output_payload else None,
				"pii_detected": _extract_pii_detected([input_payload, output_payload], payload_sources),
				"span_status": _coerce_text(record.get("status_code")),
				"span_error": _coerce_text(record.get("status_message")),
				"latency_ms": latency_ms,
				"app_version": _coerce_text(metadata.get("app_version")),
				"environment": _coerce_text(metadata.get("environment")),
				"langgraph_node": _coerce_text(metadata.get("langgraph_node")),
			}
		)

	df = pd.DataFrame(
		rows,
		columns=[
			"span_id",
			"eval_trace_id",
			"context_trace_id",
			"question",
			"answer",
			"correctness_score",
			"relevance_score",
			"faithfulness_score",
			"safety_score",
			"correctness_explanation",
			"relevance_explanation",
			"faithfulness_explanation",
			"safety_explanation",
			"retrieved_doc_count",
			"retrieval_scores",
			"avg_retrieval_score",
			"retrieved_context",
			"context_has_answer",
			"answer_supported_by_context",
			"input_guardrail_allowed",
			"output_guardrail_allowed",
			"pii_detected",
			"span_status",
			"span_error",
			"latency_ms",
			"app_version",
			"environment",
			"langgraph_node",
		],
	)
	return df
	


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
	save_annotations: bool = True
	adapter: PhoenixAdapter | None = None


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

	adapter = config.adapter or build_default_adapter(config.phoenix_base_url)
	spans_df = fetch_phoenix_spans_dataframe(
		RawSpansDataFrameRequest(
			from_time=_normalize_datetime(config.from_time),
			to_time=_normalize_datetime(config.to_time),
			limit=config.limit,
			root_spans_only=config.root_spans_only,
			project_name=config.project_name,
		),
		adapter,
	)

	if spans_df.empty:
		annotations_df = pd.DataFrame(
			columns=["span_id", "annotation_name", "annotator_kind", "label", "score", "explanation", "metadata"]
		)
		return BatchEvaluationResult(spans_df=spans_df, evaluation_df=spans_df.copy(), annotations_df=annotations_df)

	evaluation_df = _build_evaluation_frame(spans_df, span_kind=config.span_kind)
	evaluation_df.to_csv("csv/evaluation_input.csv", index=False)
	if evaluation_df.empty:
		annotations_df = pd.DataFrame(
			columns=["span_id", "annotation_name", "annotator_kind", "label", "score", "explanation", "metadata"]
		)
		return BatchEvaluationResult(spans_df=evaluation_df, evaluation_df=evaluation_df, annotations_df=annotations_df)

	runner = _build_runner()
	scored_df = await runner.evaluate_dataframe(evaluation_df)
	scored_df.to_csv("csv/evaluation_scored.csv", index=False)
	annotations_df = _build_annotations_frame(scored_df)#tt

	annotations_df.to_csv("csv/annotations.csv", index=False)

	rca_df = build_rca_feature_frame(annotations_df, evaluation_df)
	rca_df.to_csv("csv/rca_features.csv", index=False)

	if config.save_annotations: #put call here
		log_phoenix_span_annotations(
			annotations_df,
			adapter,
			sync=config.sync_annotations,
		)

	return BatchEvaluationResult(
		spans_df=evaluation_df,
		evaluation_df=scored_df,
		annotations_df=annotations_df,
	)


def run_span_batch(config: BatchEvaluationConfig) -> BatchEvaluationResult:
	"""Synchronous wrapper for the batch evaluation job."""

	return asyncio.run(evaluate_span_batch(config))


__all__ = [
	"BatchEvaluationConfig",
	"BatchEvaluationResult",
	"build_rca_feature_frame",
	"evaluate_span_batch",
	"run_span_batch",
]
