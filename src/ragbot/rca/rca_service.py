from __future__ import annotations

from copy import deepcopy
import sys
import time
from typing import Any

import pandas as pd

from ragbot.config import get_settings
from ragbot.rca.evidence_builder import build_evidence_packages
from ragbot.rca.rca_judge import RCAJudge, RCAJudgeError


class _RCAProgressBar:
    def __init__(self, total: int, *, label: str = "RCA") -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.current = 0
        self._closed = False
        self._started_at = time.monotonic()
        if self.total > 0:
            self._render()

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total_seconds = max(int(seconds), 0)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _render(self) -> None:
        total = max(self.total, 1)
        width = 30
        filled = int(width * self.current / total)
        bar = "█" * filled + "░" * (width - filled)
        percent = int((self.current / total) * 100)
        elapsed = self._format_elapsed(time.monotonic() - self._started_at)
        sys.stderr.write(
            f"\r{self.label} [{bar}] {self.current}/{self.total} {percent:3d}% elapsed {elapsed}"
        )
        sys.stderr.flush()

    def update(self, step: int = 1) -> None:
        if self.total <= 0 or self._closed:
            return
        self.current = min(self.total, self.current + step)
        self._render()

    def close(self) -> None:
        if self.total <= 0 or self._closed:
            return
        self.current = self.total
        self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()
        self._closed = True


def _create_rca_progress_bar(total: int, *, label: str = "RCA") -> _RCAProgressBar | None:
    if total <= 0:
        return None
    return _RCAProgressBar(total, label=label)


def _compact_trace_label(trace_id: str, *, prefix: str = "RCA") -> str:
    cleaned = str(trace_id).strip() or "unknown-trace"
    if len(cleaned) > 12:
        cleaned = f"{cleaned[:12]}..."
    return f"{prefix} {cleaned}"


def _plan_rca_work(
    df: pd.DataFrame,
    threshold: float,
) -> list[tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, float | None], str, float]]]]:
    grouped_work: list[tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, float | None], str, float]]]] = []
    for evidence in build_evidence_packages(df):
        scores = _extract_all_scores(evidence.get("scores", {}))
        failed_scores = _extract_failed_scores(evidence.get("scores", {}), threshold)
        trace_work = [
            (evidence, scores, evaluator_name, evaluator_score)
            for evaluator_name, evaluator_score in failed_scores.items()
        ]
        if trace_work:
            grouped_work.append((evidence, trace_work))
    return grouped_work


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

    grouped_work = _plan_rca_work(df, threshold)

    results: list[dict[str, Any]] = []
    for evidence, trace_work in grouped_work:
        progress = _create_rca_progress_bar(
            len(trace_work),
            label=_compact_trace_label(str(evidence.get("trace_id", ""))),
        )
        try:
            for _, scores, evaluator_name, evaluator_score in trace_work:
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
                finally:
                    if progress is not None:
                        progress.update()
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
        finally:
            if progress is not None:
                progress.close()
    return results
