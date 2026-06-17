from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from ragbot.config import get_settings
from ragbot.rca.evidence_builder import build_evidence_packages
from ragbot.rca.rca_judge import RCAJudge, RCAJudgeError


def _extract_failed_scores(scores: dict[str, Any], threshold: float) -> dict[str, float]:
    failed_scores: dict[str, float] = {}
    for score_name, score_value in scores.items():
        if not isinstance(score_value, (int, float)):
            continue
        if float(score_value) < threshold:
            failed_scores[score_name] = round(float(score_value), 6)
    return failed_scores


def _extract_all_scores(scores: dict[str, Any]) -> dict[str, float | None]:
    all_scores: dict[str, float | None] = {}
    for score_name, score_value in scores.items():
        if score_value is None:
            all_scores[score_name] = None
            continue
        if isinstance(score_value, (int, float)):
            all_scores[score_name] = round(float(score_value), 6)
    return all_scores


def _build_evaluator_evidence(
    evidence: dict[str, Any],
    evaluator_name: str,
    evaluator_score: float,
    threshold: float,
) -> dict[str, Any]:
    evaluator_evidence = deepcopy(evidence)
    evaluator_evidence["evaluator_name"] = evaluator_name
    evaluator_evidence["evaluator_score"] = round(float(evaluator_score), 6)
    evaluator_evidence["failed_scores"] = _extract_failed_scores(
        {evaluator_name: evaluator_score},
        threshold,
    )
    return evaluator_evidence


def _fallback_judgement(error: RCAJudgeError) -> dict[str, Any]:
    error_message = str(error).strip() or "Unknown RCA judge failure."
    return {
        "root_cause_category": "UNKNOWN",
        "confidence": 0.0,
        "evidence": [f"RCA judge fallback triggered: {error_message}"],
        "explanation": f"RCA judge fallback used because the LLM response could not be normalized: {error_message}",
        "recommended_action": "Review the failing trace and the raw RCA judge response in application logs.",
        "judge_error": error_message,
    }


def generate_rca(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate one structured RCA result per failed evaluator in the dataframe."""

    settings = get_settings()
    judge = RCAJudge(model_name=settings.gemini_model)
    threshold = settings.rca_threshold

    results: list[dict[str, Any]] = []
    for evidence in build_evidence_packages(df):
        scores = _extract_all_scores(evidence.get("scores", {}))
        failed_scores = _extract_failed_scores(evidence.get("scores", {}), threshold)
        for evaluator_name, evaluator_score in failed_scores.items():
            evaluator_evidence = _build_evaluator_evidence(
                evidence,
                evaluator_name,
                evaluator_score,
                threshold,
            )
            try:
                judgement = judge.judge(evaluator_evidence)
            except RCAJudgeError as exc:
                judgement = _fallback_judgement(exc)
            results.append(
                {
                    "trace_id": evidence["trace_id"],
                    "evaluator_name": evaluator_name,
                    "evaluator_score": evaluator_score,
                    "evaluator_scores": scores,
                    "failed_scores": {evaluator_name: evaluator_score},
                    "root_cause_category": judgement["root_cause_category"],
                    "confidence": judgement["confidence"],
                    "evidence": judgement["evidence"],
                    "explanation": judgement["explanation"],
                    "recommended_action": judgement["recommended_action"],
                    "judge_error": judgement.get("judge_error", ""),
                }
            )
    return results
