from __future__ import annotations

import os
from typing import Any

import pandas as pd

from ragbot.config import get_settings
from ragbot.rca.evidence_builder import build_evidence_packages
from ragbot.rca.rca_judge import RCAJudge


def _extract_failed_scores(scores: dict[str, Any], threshold: float = 0.5) -> dict[str, float]:
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


def generate_rca(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate one structured RCA result per trace in the dataframe."""

    settings = get_settings()
    enable_llm = os.getenv("RCA_USE_LLM_JUDGE", "true").lower() not in {"false", "0", "no"}
    judge = RCAJudge(model_name=settings.gemini_model, enable_llm=enable_llm)

    results: list[dict[str, Any]] = []
    for evidence in build_evidence_packages(df):
        judgement = judge.judge(evidence)
        results.append(
            {
                "trace_id": evidence["trace_id"],
                "root_cause_category": judgement["root_cause_category"],
                "confidence": judgement["confidence"],
                "evaluator_scores": _extract_all_scores(evidence.get("scores", {})),
                "failed_scores": _extract_failed_scores(evidence.get("scores", {})),
                "evidence": judgement["evidence"],
                "explanation": judgement["explanation"],
                "recommended_action": judgement["recommended_action"],
            }
        )
    return results
