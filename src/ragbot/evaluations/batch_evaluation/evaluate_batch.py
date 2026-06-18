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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Iterable
from ragbot.rca.rca_service import generate_rca

import pandas as pd

from data_fetch.fetch_from_phoenix import (
	build_default_adapter,
	fetch_existing_span_annotations,
	log_phoenix_span_annotations,
	RawSpansDataFrameRequest,
	PhoenixAdapter,
)
from ragbot.evaluations import correctness, groundedness, relevance, safety
from ragbot.evaluations.batch_evaluation import alerts
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

RCA_EVALUATOR_TO_SOURCE_EVALUATOR = {
	"correctness": "correctness",
	"relevance": "relevance",
	"faithfulness": "groundedness",
	"safety": "safety",
}

BATCH_EVALUATION_REQUIRED_ANNOTATIONS = frozenset(
	{"correctness", "relevance", "faithfulness", "safety"}
)

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


def _utc_now() -> datetime:
	return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class BatchEvaluationCheckpoint:
	high_watermark_time: datetime | None = None
	processed_keys: tuple[str, ...] = ()


def read_checkpoint(path: Path) -> BatchEvaluationCheckpoint:
	try:
		raw_value = path.read_text(encoding="utf-8").strip()
	except FileNotFoundError:
		return BatchEvaluationCheckpoint()
	except OSError as exc:
		raise ValueError(f"Unable to read batch evaluation checkpoint file: {path}") from exc

	if not raw_value:
		return BatchEvaluationCheckpoint()

	try:
		payload = json.loads(raw_value)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Invalid batch evaluation checkpoint JSON in {path}") from exc

	if not isinstance(payload, dict):
		raise ValueError(f"Batch evaluation checkpoint must be a JSON object: {path}")

	try:
		high_watermark_time = _normalize_datetime(payload.get("high_watermark_time"))
	except ValueError as exc:
		raise ValueError(f"Invalid batch evaluation checkpoint high_watermark_time in {path}") from exc
	raw_processed_keys = payload.get("processed_keys", [])
	if not isinstance(raw_processed_keys, list) or any(not isinstance(item, str) for item in raw_processed_keys):
		raise ValueError(f"Batch evaluation checkpoint processed_keys must be a list of strings: {path}")

	return BatchEvaluationCheckpoint(
		high_watermark_time=high_watermark_time,
		processed_keys=tuple(raw_processed_keys),
	)


def write_checkpoint(path: Path, checkpoint: BatchEvaluationCheckpoint) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"high_watermark_time": (
			checkpoint.high_watermark_time.astimezone(timezone.utc).isoformat()
			if checkpoint.high_watermark_time is not None
			else None
		),
		"processed_keys": list(checkpoint.processed_keys),
	}

	with tempfile.NamedTemporaryFile(
		"w",
		encoding="utf-8",
		dir=path.parent,
		prefix=f"{path.name}.",
		suffix=".tmp",
		delete=False,
	) as temp_file:
		json.dump(payload, temp_file, indent=2, ensure_ascii=False)
		temp_file.write("\n")
		temp_path = Path(temp_file.name)

	temp_path.replace(path)


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


def _find_nested_value(value: Any, keys: tuple[str, ...]) -> Any:
	if value is None:
		return None
	if isinstance(value, str):
		text = value.strip()
		if not text or text[0] not in {"{", "["}:
			return None
		try:
			return _find_nested_value(json.loads(text), keys)
		except json.JSONDecodeError:
			return None
	if isinstance(value, dict):
		for key in keys:
			if key in value:
				return value[key]
			alternate = key.replace(".", "_") if "." in key else key.replace("_", ".")
			if alternate in value:
				return value[alternate]
		for item in value.values():
			found = _find_nested_value(item, keys)
			if found is not None:
				return found
	if isinstance(value, (list, tuple, set)):
		for item in value:
			found = _find_nested_value(item, keys)
			if found is not None:
				return found
	return None


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

	for source in (record, attributes):
		for value in source.values():
			candidate = _coerce_text(_find_nested_value(value, keys))
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


def _span_id_value(record: dict[str, Any]) -> str:
	return _coerce_text(
		record.get("span_id")
		or record.get("context.span_id")
		or record.get("spanId")
		or record.get("id")
		or record.get("span_uuid")
	)


def _trace_id_value(record: dict[str, Any]) -> str:
	return _coerce_text(record.get("context.trace_id") or record.get("trace_id"))


def _span_name_value(record: dict[str, Any]) -> str:
	return _coerce_text(record.get("name") or record.get("span_name"))


def _start_time_value(record: dict[str, Any]) -> datetime | None:
	return _normalize_datetime(record.get("start_time"))


def build_composite_checkpoint_key(record: dict[str, Any]) -> str:
	trace_id = _trace_id_value(record)
	start_time = _start_time_value(record)
	span_name = _span_name_value(record)
	start_time_text = start_time.isoformat() if start_time is not None else ""
	return f"{trace_id}|{start_time_text}|{span_name}"


PARENT_ID_CANDIDATES = (
	"parent_id",
	"parent.span_id",
	"parent_span_id",
	"parentSpanId",
	"context.parent_id",
	"context.parent_span_id",
)


def _parent_id_value(record: dict[str, Any]) -> Any:
	for key in PARENT_ID_CANDIDATES:
		if key in record:
			return record.get(key)
	return None


def _has_parent_id_field(record: dict[str, Any]) -> bool:
	return any(key in record for key in PARENT_ID_CANDIDATES)


def _is_missing_parent_id(value: Any) -> bool:
	if value is None:
		return True
	try:
		if pd.isna(value):
			return True
	except (TypeError, ValueError):
		pass
	return not _coerce_text(value)


def _is_langgraph_application_span(record: dict[str, Any]) -> bool:
	name = _coerce_text(record.get("name") or record.get("span_name"))
	if name == "LangGraph":
		return True

	metadata = _parse_structured_value(record.get("attributes.metadata"))
	if isinstance(metadata, dict) and _coerce_text(metadata.get("ls_integration")) == "langgraph":
		return True

	return False


def _is_main_application_span(record: dict[str, Any]) -> bool:
	if not _has_parent_id_field(record):
		return False
	return (
		_is_missing_parent_id(_parent_id_value(record))
		and _span_kind(record) == "CHAIN"
		and _is_langgraph_application_span(record)
	)


def _filter_main_application_spans(spans_df: pd.DataFrame) -> pd.DataFrame:
	if spans_df.empty:
		return spans_df.copy()
	mask = [
		_is_main_application_span(series.to_dict())
		for _, series in spans_df.iterrows()
	]
	return spans_df.loc[mask].copy()


def _filter_batch_evaluation_candidate_spans(spans_df: pd.DataFrame) -> pd.DataFrame:
	"""Keep only root LangGraph CHAIN spans for batch evaluation."""
	return _filter_main_application_spans(spans_df)


def filter_checkpointed_spans(
	spans_df: pd.DataFrame,
	checkpoint: BatchEvaluationCheckpoint,
) -> pd.DataFrame:
	if spans_df.empty or checkpoint.high_watermark_time is None:
		return spans_df.copy()

	processed_keys = set(checkpoint.processed_keys)
	high_watermark_time = checkpoint.high_watermark_time
	mask = []
	for _, series in spans_df.iterrows():
		record = series.to_dict()
		start_time = _start_time_value(record)
		if start_time is None:
			mask.append(True)
			continue
		if start_time < high_watermark_time:
			mask.append(False)
			continue
		if start_time > high_watermark_time:
			mask.append(True)
			continue
		mask.append(build_composite_checkpoint_key(record) not in processed_keys)
	return spans_df.loc[mask].copy()


def update_checkpoint_after_success(
	path: Path,
	evaluated_spans_df: pd.DataFrame,
	old_checkpoint: BatchEvaluationCheckpoint,
) -> BatchEvaluationCheckpoint:
	if evaluated_spans_df.empty:
		return old_checkpoint

	processed_records: list[tuple[datetime, str]] = []
	for _, series in evaluated_spans_df.iterrows():
		record = series.to_dict()
		start_time = _start_time_value(record)
		if start_time is None:
			continue
		processed_records.append((start_time, build_composite_checkpoint_key(record)))

	if not processed_records:
		return old_checkpoint

	new_high_watermark_time = max(start_time for start_time, _ in processed_records)
	watermark_keys = {
		key
		for start_time, key in processed_records
		if start_time == new_high_watermark_time
	}

	new_checkpoint = BatchEvaluationCheckpoint(
		high_watermark_time=new_high_watermark_time,
		processed_keys=tuple(sorted(watermark_keys)),
	)
	write_checkpoint(path, new_checkpoint)
	return new_checkpoint


def _fully_batch_annotated_span_ids(annotations_df: pd.DataFrame) -> set[str]:
	if (
		annotations_df.empty
		or "span_id" not in annotations_df.columns
		or "annotation_name" not in annotations_df.columns
	):
		return set()

	annotation_names_by_span_id: dict[str, set[str]] = {}
	for _, series in annotations_df.iterrows():
		span_id = _coerce_text(series.get("span_id"))
		annotation_name = _coerce_text(series.get("annotation_name"))
		if not span_id or not annotation_name:
			continue
		annotation_names_by_span_id.setdefault(span_id, set()).add(annotation_name)

	return {
		span_id
		for span_id, annotation_names in annotation_names_by_span_id.items()
		if BATCH_EVALUATION_REQUIRED_ANNOTATIONS.issubset(annotation_names)
	}


def _filter_spans_missing_batch_annotations(
	spans_df: pd.DataFrame,
	annotations_df: pd.DataFrame,
) -> pd.DataFrame:
	if spans_df.empty:
		return spans_df.copy()

	fully_annotated_span_ids = _fully_batch_annotated_span_ids(annotations_df)
	mask = []
	for _, series in spans_df.iterrows():
		span_id = _span_id_value(series.to_dict())
		mask.append(span_id not in fully_annotated_span_ids)
	return spans_df.loc[mask].copy()


def _evaluation_input(record: dict[str, Any], input_payload: Any, output_payload: Any) -> str:
	return (
		_extract_message(input_payload)
		or _extract_message(output_payload)
		or _first_non_empty(record, INPUT_CANDIDATES)
	)


def _evaluation_output(record: dict[str, Any], input_payload: Any, output_payload: Any) -> str:
	return (
		_extract_answer(output_payload)
		or _extract_answer(input_payload)
		or _first_non_empty(record, OUTPUT_CANDIDATES)
	)


def _evaluation_reference(record: dict[str, Any], input_payload: Any, output_payload: Any) -> str:
	return (
		_coerce_text((output_payload or {}).get("context") if isinstance(output_payload, dict) else "")
		or _coerce_text((input_payload or {}).get("context") if isinstance(input_payload, dict) else "")
		or _first_non_empty(record, REFERENCE_CANDIDATES)
	)


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
		input_payload = _parse_structured_value(record.get("attributes.input.value"))
		output_payload = _parse_structured_value(record.get("attributes.output.value"))
		input_text = _evaluation_input(record, input_payload, output_payload)
		output_text = _evaluation_output(record, input_payload, output_payload)
		reference_text = _evaluation_reference(record, input_payload, output_payload)

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
	elif raw_label in PASS_LABELS:
		score = 1.0
	else:
		score = 0.0

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


def merge_final_rca_into_annotations(
	annotation_df: pd.DataFrame,
	final_rca_df: pd.DataFrame,
	rca_features_df: pd.DataFrame,
	selected_features: Iterable[str],
	*,
	prefix: str = "rca_",
) -> pd.DataFrame:
	"""Add selected final RCA fields to annotation metadata via span_id -> trace_id mapping."""

	selected_columns = [column for column in selected_features]
	if annotation_df.empty or final_rca_df.empty or not selected_columns:
		return annotation_df.copy()

	missing_rca_columns = sorted(column for column in selected_columns if column not in final_rca_df.columns)
	if missing_rca_columns:
		raise ValueError(
			f"Selected final RCA columns are missing from final_rca_df: {', '.join(missing_rca_columns)}"
		)

	if "span_id" not in annotation_df.columns:
		raise ValueError("annotation_df must contain a 'span_id' column.")
	if "span_id" not in rca_features_df.columns or "context_trace_id" not in rca_features_df.columns:
		raise ValueError("rca_features_df must contain 'span_id' and 'context_trace_id' columns.")
	if "trace_id" not in final_rca_df.columns or "evaluator_name" not in final_rca_df.columns:
		raise ValueError("final_rca_df must contain 'trace_id' and 'evaluator_name' columns.")

	span_trace_df = (
		rca_features_df.loc[:, ["span_id", "context_trace_id"]]
		.dropna(subset=["span_id", "context_trace_id"])
		.drop_duplicates(subset=["span_id"])
	)
	span_trace_map = {
		_coerce_text(record["span_id"]): _coerce_text(record["context_trace_id"])
		for record in span_trace_df.to_dict("records")
	}

	rca_columns = ["trace_id", "evaluator_name", *selected_columns]
	if "evaluator_score" in final_rca_df.columns and "evaluator_score" not in rca_columns:
		rca_columns.append("evaluator_score")
	rca_subset = final_rca_df.loc[:, rca_columns].copy()
	rca_by_trace_and_evaluator = {
		(
			_coerce_text(record["trace_id"]),
			RCA_EVALUATOR_TO_SOURCE_EVALUATOR.get(
				_coerce_text(record["evaluator_name"]),
				_coerce_text(record["evaluator_name"]),
			),
		): record
		for record in rca_subset.to_dict("records")
	}

	merged = annotation_df.copy()
	metadata_values: list[dict[str, Any]] = []
	for _, series in merged.iterrows():
		span_id = _coerce_text(series.get("span_id"))
		trace_id = span_trace_map.get(span_id, "")
		existing_metadata = series.get("metadata")
		metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
		source_evaluator = _coerce_text(metadata.get("source_evaluator"))
		rca_record = rca_by_trace_and_evaluator.get((trace_id, source_evaluator), {})
		for column in selected_columns:
			value = rca_record.get(column)
			if value is not None and not pd.isna(value):
				metadata[f"{prefix}{column}"] = value
		evaluator_name = _coerce_text(rca_record.get("evaluator_name"))
		if evaluator_name:
			metadata[f"{prefix}evaluator_name"] = evaluator_name
		evaluator_score = rca_record.get("evaluator_score")
		if evaluator_score is not None and not pd.isna(evaluator_score):
			metadata[f"{prefix}evaluator_score"] = evaluator_score
		metadata_values.append(metadata)

	merged["metadata"] = metadata_values
	return merged
	


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
	checkpoint_file: Path | None = None
	use_checkpoint: bool = False
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


def resolve_batch_evaluation_config(
	config: BatchEvaluationConfig,
	*,
	now: datetime | None = None,
) -> BatchEvaluationConfig:
	resolved_to = _normalize_datetime(config.to_time) or _normalize_datetime(now or _utc_now())
	resolved_from = _normalize_datetime(config.from_time)

	if config.use_checkpoint:
		if config.checkpoint_file is None:
			raise ValueError("checkpoint_file is required when checkpoint mode is enabled")
		checkpoint = read_checkpoint(config.checkpoint_file)
		if checkpoint.high_watermark_time is not None:
			resolved_from = checkpoint.high_watermark_time
	elif resolved_from is None:
		raise ValueError("from_time is required when checkpoint mode is disabled and no explicit from_time is provided")

	if resolved_from is not None and resolved_from >= resolved_to:
		raise ValueError("from_time must be earlier than to_time")

	return replace(
		config,
		from_time=resolved_from,
		to_time=resolved_to,
	)


async def evaluate_span_batch(config: BatchEvaluationConfig) -> BatchEvaluationResult:
	"""Pull Phoenix spans, evaluate them, and write the annotations back."""

	config = resolve_batch_evaluation_config(config)
	adapter = config.adapter or build_default_adapter(config.phoenix_base_url)
	checkpoint = read_checkpoint(config.checkpoint_file) if config.use_checkpoint and config.checkpoint_file is not None else BatchEvaluationCheckpoint()
	request = RawSpansDataFrameRequest(
		from_time=_normalize_datetime(config.from_time),
		to_time=_normalize_datetime(config.to_time),
		limit=config.limit,
		root_spans_only=True,
		project_name=config.project_name,
	)
	spans_df = adapter.fetch_spans_dataframe(
		from_time=request.from_time,
		to_time=request.to_time,
		project_name=request.project_name,
		limit=request.limit,
		root_spans_only=request.root_spans_only,
	)
	with open("csv/spans_df_prev.json", "a", encoding="utf-8") as f:
		f.write(spans_df.to_json(orient="records", date_format="iso", indent=2))
		f.write("\n")

	spans_df = _filter_batch_evaluation_candidate_spans(spans_df)
	# with open("csv/spans_df.json", "a", encoding="utf-8") as f:
	# 	f.write(spans_df.to_json(orient="records", date_format="iso", indent=2))
	# 	f.write("\n")
	existing_annotations_df = fetch_existing_span_annotations(spans_df, request, adapter)
	# with open("csv/existing_annotations_df.json", "a", encoding="utf-8") as f:
	# 	f.write(existing_annotations_df.to_json(orient="records", date_format="iso", indent=2))
	# 	f.write("\n")
	spans_df = _filter_spans_missing_batch_annotations(spans_df, existing_annotations_df)
	if config.use_checkpoint:
		spans_df = filter_checkpointed_spans(spans_df, checkpoint)
	
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
		return BatchEvaluationResult(spans_df=evaluation_df, evaluation_df=evaluation_df, annotations_df=annotations_df)

	runner = _build_runner()
	scored_df = await runner.evaluate_dataframe(evaluation_df)
	annotations_df = _build_annotations_frame(scored_df)
	alerts.log_degradation_alerts(annotations_df)

	rca_df = build_rca_feature_frame(annotations_df, evaluation_df)

	final_rca = generate_rca(rca_df)

	merged_annotations_df = merge_final_rca_into_annotations(
		annotation_df=annotations_df,
		final_rca_df=pd.DataFrame(final_rca),
		rca_features_df=rca_df,
		selected_features=[
			"root_cause_category",
			"confidence",
			"explanation",
			"recommended_action",
		],
	)

	if config.save_annotations:
		log_phoenix_span_annotations(
			merged_annotations_df,
			adapter,
			sync=config.sync_annotations,
		)
		if config.use_checkpoint and config.checkpoint_file is not None and not merged_annotations_df.empty:
			update_checkpoint_after_success(
				config.checkpoint_file,
				evaluation_df,
				checkpoint,
			)

	return BatchEvaluationResult(
		spans_df=evaluation_df,
		evaluation_df=scored_df,
		annotations_df=merged_annotations_df,
	)


def run_span_batch(config: BatchEvaluationConfig) -> BatchEvaluationResult:
	"""Synchronous wrapper for the batch evaluation job."""

	return asyncio.run(evaluate_span_batch(config))


__all__ = [
	"BatchEvaluationCheckpoint",
	"BatchEvaluationConfig",
	"BatchEvaluationResult",
	"build_rca_feature_frame",
	"build_composite_checkpoint_key",
	"evaluate_span_batch",
	"filter_checkpointed_spans",
	"merge_final_rca_into_annotations",
	"read_checkpoint",
	"resolve_batch_evaluation_config",
	"run_span_batch",
	"update_checkpoint_after_success",
	"write_checkpoint",
]
