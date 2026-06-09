from __future__ import annotations

import ast
import math
from collections import Counter
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "span_id",
    "question",
    "answer",
    "correctness_score",
    "relevance_score",
    "faithfulness_score",
    "safety_score",
    "retrieved_doc_count",
    "retrieval_scores",
    "retrieved_context",
    "context_has_answer",
    "answer_supported_by_context",
    "pii_detected",
    "span_status",
    "latency_ms",
}

TRACE_ID_CANDIDATES = ("trace_id", "context_trace_id", "eval_trace_id")
FINAL_NODE_ALIASES = {"finalize", "finalized"}


def _coerce_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _coerce_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = _coerce_text(value).lower()
    if normalized in {"true", "1", "yes", "y", "pass", "allowed", "ok"}:
        return True
    if normalized in {"false", "0", "no", "n", "fail", "blocked"}:
        return False
    return None


def _coerce_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_list(value: Any) -> list[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        return [parsed]
    return [value]


def _pick_best_text(series: pd.Series) -> str:
    candidates = []
    for value in series:
        text = _coerce_text(value)
        if text:
            candidates.append(text)
    if not candidates:
        return ""
    counts = Counter(candidates)
    return max(candidates, key=lambda item: (counts[item], len(item)))


def _aggregate_score(series: pd.Series) -> float | None:
    scores = [_coerce_float(value) for value in series]
    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return None
    return round(min(valid_scores), 6)


def _aggregate_bool(series: pd.Series) -> bool | None:
    values = [_coerce_bool(value) for value in series]
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return all(valid_values)


def _aggregate_retrieval_scores(series: pd.Series) -> list[float]:
    scores: list[float] = []
    for value in series:
        for item in _coerce_list(value):
            score = _coerce_float(item)
            if score is not None:
                scores.append(score)
    return scores


def _aggregate_errors(series: pd.Series) -> list[str]:
    seen: list[str] = []
    for value in series:
        text = _coerce_text(value)
        if text and text not in seen:
            seen.append(text)
    return seen


def _normalize_langgraph_node(value: Any) -> str:
    normalized = _coerce_text(value).lower()
    if normalized in FINAL_NODE_ALIASES:
        return "finalize"
    return normalized


def _score_source_group(group: pd.DataFrame) -> pd.DataFrame:
    if "langgraph_node" not in group.columns:
        return group

    normalized_nodes = group["langgraph_node"].apply(_normalize_langgraph_node)
    finalize_group = group.loc[normalized_nodes == "finalize"]
    if not finalize_group.empty:
        return finalize_group
    return group


def _resolve_trace_id_column(df: pd.DataFrame) -> str:
    for candidate in TRACE_ID_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "RCA feature dataframe must contain one of: "
        + ", ".join(TRACE_ID_CANDIDATES)
    )


def _validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(column for column in REQUIRED_COLUMNS if column not in df.columns)
    if missing:
        raise ValueError(f"RCA feature dataframe is missing required columns: {', '.join(missing)}")


def build_evidence_packages(rca_features_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Collapse RCA feature rows into one evidence package per trace."""

    if rca_features_df.empty:
        return []

    _validate_columns(rca_features_df)
    trace_id_column = _resolve_trace_id_column(rca_features_df)

    evidence_packages: list[dict[str, Any]] = []
    for trace_id, group in rca_features_df.groupby(trace_id_column, dropna=False, sort=False):
        normalized_trace_id = _coerce_text(trace_id)
        if not normalized_trace_id:
            continue

        score_group = _score_source_group(group)
        retrieval_scores = _aggregate_retrieval_scores(group.get("retrieval_scores", pd.Series(dtype=object)))
        latency_values = [_coerce_float(value) for value in group.get("latency_ms", pd.Series(dtype=float))]
        valid_latency_values = [value for value in latency_values if value is not None]

        status_values = [
            _coerce_text(value).upper()
            for value in group.get("span_status", pd.Series(dtype=object))
            if _coerce_text(value)
        ]
        overall_status = "OK" if status_values and all(value == "OK" for value in status_values) else "ERROR"
        if not status_values:
            overall_status = "UNKNOWN"

        evidence_packages.append(
            {
                "trace_id": normalized_trace_id,
                "question": _pick_best_text(group["question"]),
                "answer": _pick_best_text(group["answer"]),
                "scores": {
                    "correctness": _aggregate_score(score_group["correctness_score"]),
                    "relevance": _aggregate_score(score_group["relevance_score"]),
                    "faithfulness": _aggregate_score(score_group["faithfulness_score"]),
                    "safety": _aggregate_score(score_group["safety_score"]),
                },
                "retrieval": {
                    "doc_count": int(max((_coerce_float(value) or 0) for value in group["retrieved_doc_count"])),
                    "avg_score": round(sum(retrieval_scores) / len(retrieval_scores), 6) if retrieval_scores else None,
                    "scores": retrieval_scores,
                    "context": _pick_best_text(group["retrieved_context"]),
                    "context_has_answer": _aggregate_bool(group["context_has_answer"]),
                    "answer_supported_by_context": _aggregate_bool(group["answer_supported_by_context"]),
                },
                "guardrails": {
                    "input_allowed": _aggregate_bool(
                        group["input_guardrail_allowed"] if "input_guardrail_allowed" in group else pd.Series(dtype=object)
                    ),
                    "output_allowed": _aggregate_bool(
                        group["output_guardrail_allowed"] if "output_guardrail_allowed" in group else pd.Series(dtype=object)
                    ),
                    "pii_detected": _aggregate_bool(
                        group["pii_detected"] if "pii_detected" in group else pd.Series(dtype=object)
                    ),
                },
                "spans": {
                    "status": overall_status,
                    "errors": _aggregate_errors(
                        group["span_error"] if "span_error" in group else pd.Series(dtype=object)
                    ),
                    "latency_ms": round(sum(valid_latency_values), 3) if valid_latency_values else None,
                    "span_count": int(len(group)),
                },
                "metadata": {
                    "app_version": _pick_best_text(
                        group["app_version"] if "app_version" in group else pd.Series(dtype=object)
                    ),
                    "environment": _pick_best_text(
                        group["environment"] if "environment" in group else pd.Series(dtype=object)
                    ),
                    "langgraph_nodes": sorted(
                        {
                            _normalize_langgraph_node(value)
                            for value in (
                                group["langgraph_node"] if "langgraph_node" in group else pd.Series(dtype=object)
                            )
                            if _normalize_langgraph_node(value)
                        }
                    ),
                    "span_ids": [
                        _coerce_text(value)
                        for value in group["span_id"]
                        if _coerce_text(value)
                    ],
                },
            }
        )

    return evidence_packages
