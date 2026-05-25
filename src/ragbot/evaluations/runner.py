"""Evaluation runner - orchestrates Phoenix evaluators using evaluate_dataframe()."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd

from .results import EvaluationResult, TraceEvaluationResults

if TYPE_CHECKING:
    from phoenix.evals import Evaluator


class EvaluationRunner:
    """Orchestrates evaluation using Phoenix SDK."""

    PASS_LABELS = {
        "CORRECT",
        "SAFE",
        "VALID",
        "PASSING",
        "PASS",
        "RELEVANT",
        "GROUNDED",
    }
    WARNING_LABELS = {"WARNING", "WARN", "SKIPPED", "SKIP"}
    FAIL_LABELS = {
        "ERROR",
        "FAIL",
        "FAILING",
        "FAILED",
        "INVALID",
        "INCORRECT",
        "UNSAFE",
        "IRRELEVANT",
        "HALLUCINATED",
    }

    def __init__(
        self,
        evaluators: dict[str, Evaluator] | None = None,
    ):
        """Initialize runner with Phoenix evaluators.
        
        Args:
            evaluators: Dict mapping evaluator names to Phoenix Evaluator instances.
                       If None, creates default evaluators using factory functions.
        """
        if evaluators is None:
            evaluators = self._create_default_evaluators()
        
        self.evaluators = evaluators
        self._enabled = set(evaluators.keys())  # Track enabled evaluators

    @staticmethod
    def _create_default_evaluators() -> dict[str, Evaluator]:
        """Create default Phoenix evaluator instances."""
        from .config import EvaluationConfig
        from . import (
            correctness,
            format as format_eval,
            groundedness,
            latency,
            relevance,
            routing,
            safety,
        )
        config = EvaluationConfig()
        
        return {
            "correctness": correctness.create_correctness_evaluator(),
            "relevance": relevance.create_relevance_evaluator(),
            "groundedness": groundedness.create_groundedness_evaluator(),
            "routing": routing.create_routing_evaluator(),
            "safety": safety.create_safety_evaluator(
                check_pii=config.safety_check_pii,
                check_injection=config.safety_check_injection,
                check_profanity=config.safety_check_profanity,
            ),
            "latency": latency.create_latency_evaluator(
                pass_threshold_ms=int(config.latency_thresholds["pass"]),
                warning_threshold_ms=int(config.latency_thresholds["warning"]),
            ),
            "format": format_eval.create_format_evaluator(),
        }

    @staticmethod
    def _normalize_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _build_row(
        input_text: str,
        response_text: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "input": input_text or kwargs.get("question", "") or kwargs.get("query", ""),
            "output": response_text or kwargs.get("response", "") or kwargs.get("answer", ""),
        }

        context = kwargs.get("context")
        reference = kwargs.get("reference")
        if reference is None and context is not None:
            if isinstance(context, str):
                reference = context
            else:
                reference = "\n\n".join(str(item) for item in EvaluationRunner._normalize_list(context))

        retrieved_docs = kwargs.get("retrieved_docs")
        if retrieved_docs is None and context is not None:
            retrieved_docs = EvaluationRunner._normalize_list(context)

        scores = kwargs.get("scores")

        row.update(kwargs)
        if reference is not None:
            row["reference"] = reference
            row.setdefault("context", reference)
        if retrieved_docs is not None:
            row["retrieved_docs"] = EvaluationRunner._normalize_list(retrieved_docs)
        if scores is not None:
            row["scores"] = EvaluationRunner._normalize_list(scores)

        return row

    @staticmethod
    def _safe_float(value: Any, default: float = 0.5) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _label_score(label: str) -> tuple[float, bool]:
        normalized = label.upper()
        if normalized in EvaluationRunner.PASS_LABELS:
            return 1.0, True
        if normalized in EvaluationRunner.WARNING_LABELS:
            return 0.5, False
        if normalized in EvaluationRunner.FAIL_LABELS:
            return 0.0, False
        return 0.0, False

    @staticmethod
    def _scores_stats(scores: Any) -> dict[str, Any]:
        score_values = [
            EvaluationRunner._safe_float(score, 0.0)
            for score in EvaluationRunner._normalize_list(scores)
            if score is not None
        ]
        if not score_values:
            return {
                "top_score": None,
                "min_score": None,
                "avg_score": None,
                "score_spread": None,
            }
        return {
            "top_score": max(score_values),
            "min_score": min(score_values),
            "avg_score": sum(score_values) / len(score_values),
            "score_spread": max(score_values) - min(score_values),
        }

    @staticmethod
    def _metadata_for_evaluator(
        evaluator_name: str,
        row: dict[str, Any],
        raw_value: Any,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "raw_value": raw_value,
        }
        execution_details = row.get(f"{evaluator_name}_execution_details")
        if execution_details is not None:
            metadata["execution_details"] = execution_details

        if evaluator_name == "routing":
            retrieved_docs = EvaluationRunner._normalize_list(row.get("retrieved_docs"))
            scores = row.get("scores")
            metadata.update(
                {
                    "doc_count": len(retrieved_docs),
                    "threshold": row.get("min_score_threshold", row.get("routing_min_score_threshold")),
                    **EvaluationRunner._scores_stats(scores),
                }
            )
        elif evaluator_name == "latency":
            metadata["duration_ms"] = row.get("duration_ms")
            metadata["thresholds"] = {
                "pass": row.get("latency_pass_threshold_ms"),
                "warning": row.get("latency_warning_threshold_ms"),
            }
        elif evaluator_name == "safety":
            text = str(row.get("output", ""))
            pii_patterns: dict[str, re.Pattern[str]] = {
                "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                "phone": re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
                "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
                "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
            }
            normalized = text.lower()
            injection_patterns = (
                "ignore previous instructions",
                "ignore all previous instructions",
                "system prompt",
                "developer message",
                "show hidden instructions",
                "reveal your chain of thought",
            )
            profanity_patterns = ("fuck", "shit", "bitch", "asshole", "bastard")
            pii_categories = [
                label
                for label, pattern in pii_patterns.items()
                if pattern.search(text)
            ]
            metadata.update(
                {
                    "input_length": len(text),
                    "pii_detected": bool(pii_categories),
                    "pii_categories": sorted(pii_categories),
                    "injection_detected": any(pattern in normalized for pattern in injection_patterns),
                    "profanity_detected": any(
                        re.search(rf"\b{re.escape(pattern)}\b", normalized)
                        for pattern in profanity_patterns
                    ),
                }
            )
        elif evaluator_name == "groundedness":
            metadata["has_reference"] = bool(row.get("reference") or row.get("context"))
            metadata["reference_length"] = len(str(row.get("reference") or row.get("context") or ""))
        elif evaluator_name in {"correctness", "relevance", "format"}:
            metadata["input_length"] = len(str(row.get("input", "")))
            metadata["output_length"] = len(str(row.get("output", "")))

        return metadata

    @staticmethod
    def _normalise_evaluation_result(
        evaluator_name: str,
        row: dict[str, Any],
    ) -> EvaluationResult:
        error = row.get(f"{evaluator_name}_error")
        if error:
            return EvaluationResult(
                evaluator_name=evaluator_name,
                score=0.0,
                passed=False,
                reason=f"Evaluator error: {error}",
                metadata={"label": "ERROR", "error": str(error)},
            )

        score_key = f"{evaluator_name}_score"
        raw_score = row.get(score_key, row.get(evaluator_name))
        if raw_score is None:
            return EvaluationResult(
                evaluator_name=evaluator_name,
                score=0.0,
                passed=False,
                reason="Evaluator produced no score.",
                metadata={
                    "label": "ERROR",
                    **EvaluationRunner._metadata_for_evaluator(evaluator_name, row, raw_score),
                },
            )

        if isinstance(raw_score, dict):
            label = str(raw_score.get("label", raw_score.get("name", "UNKNOWN"))).upper()
            score_value = raw_score.get("score", raw_score.get("value"))
            mapped_score, passed = EvaluationRunner._label_score(label)
            score = EvaluationRunner._safe_float(score_value, mapped_score) if score_value is not None else mapped_score
            reason = raw_score.get("explanation", f"Classification: {label}")
            metadata = EvaluationRunner._metadata_for_evaluator(evaluator_name, row, raw_score)
            raw_metadata = raw_score.get("metadata")
            if isinstance(raw_metadata, dict):
                metadata.update(raw_metadata)
        elif isinstance(raw_score, str):
            label = raw_score.upper()
            score, passed = EvaluationRunner._label_score(label)
            reason = f"Classification: {label}"
            metadata = EvaluationRunner._metadata_for_evaluator(evaluator_name, row, raw_score)
        else:
            label = "SCORE"
            score = EvaluationRunner._safe_float(raw_score)
            passed = score >= 0.5
            reason = f"Score: {score:.2f}"
            metadata = EvaluationRunner._metadata_for_evaluator(evaluator_name, row, raw_score)

        metadata.setdefault("label", label)
        return EvaluationResult(
            evaluator_name=evaluator_name,
            score=max(0.0, min(1.0, score)),
            passed=passed,
            reason=reason,
            metadata=metadata,
        )

    def enable_evaluator(self, name: str) -> None:
        """Enable a specific evaluator."""
        if name in self.evaluators:
            self._enabled.add(name)

    def disable_evaluator(self, name: str) -> None:
        """Disable a specific evaluator."""
        if name in self.evaluators:
            self._enabled.discard(name)

    def get_enabled_evaluators(self) -> list[str]:
        """Get names of all enabled evaluators."""
        return sorted(self._enabled)

    def get_disabled_evaluators(self) -> list[str]:
        """Get names of all disabled evaluators."""
        all_names = set(self.evaluators.keys())
        return sorted(all_names - self._enabled)

    async def evaluate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Evaluate a pandas DataFrame using enabled Phoenix evaluators.
        
        Args:
            df: DataFrame with columns for each evaluator's inputs.
               Typically includes: input, output, reference (for groundedness)
        
        Returns:
            DataFrame with evaluation results appended as new columns.
        
        Example:
            >>> df = pd.DataFrame({
            ...     "input": ["What is AI?"],
            ...     "output": ["AI is artificial intelligence..."],
            ...     "reference": ["AI is the simulation of human intelligence..."]
            ... })
            >>> results_df = await runner.evaluate_dataframe(df)
        """
        try:
            from phoenix.evals import evaluate_dataframe as phoenix_evaluate_dataframe
        except ImportError:
            raise ImportError(
                "Phoenix SDK not installed. Install with: pip install arize-phoenix[evals]"
            )
        
        results_df = df.copy()
        
        # Run each enabled evaluator over the dataframe
        for evaluator_name in self.get_enabled_evaluators():
            if evaluator_name not in self.evaluators:
                continue
            
            evaluator = self.evaluators[evaluator_name]
            
            try:
                # Phoenix's evaluate_dataframe will add new column(s) with results
                results_df = phoenix_evaluate_dataframe(
                    dataframe=results_df,
                    evaluators=[evaluator],
                )
            except Exception as e:
                # Log error but continue with other evaluators
                results_df[f"{evaluator_name}_error"] = str(e)
        
        return results_df

    async def evaluate(
        self,
        trace_id: str | None = None,
        input_text: str = "",
        response_text: str = "",
        **kwargs: Any
    ) -> TraceEvaluationResults:
        """Evaluate a single request (convenience method).
        
        Args:
            trace_id: Unique identifier for this trace. Auto-generated if None.
            input_text: User's input question
            response_text: Bot's response
            **kwargs: Additional context (question, context, retrieved_docs, etc.)
        
        Returns:
            TraceEvaluationResults with evaluation scores from all enabled evaluators.
        """
        trace_id = trace_id or f"trace_{uuid4().hex[:12]}"
        
        # Build dataframe with single row
        row = self._build_row(input_text, response_text, kwargs)
        
        df = pd.DataFrame([row])
        
        # Evaluate the dataframe
        results_df = await self.evaluate_dataframe(df)
        result_row = results_df.iloc[0].to_dict()
        
        evaluations: dict[str, EvaluationResult] = {}
        for evaluator_name in self.get_enabled_evaluators():
            evaluations[evaluator_name] = self._normalise_evaluation_result(
                evaluator_name,
                result_row,
            )
        
        return TraceEvaluationResults(
            trace_id=trace_id,
            input_text=input_text,
            response_text=response_text,
            evaluations=evaluations,
        )

    async def evaluate_batch(
        self,
        requests: list[dict[str, Any]],
        trace_id_prefix: str = "batch",
    ) -> list[TraceEvaluationResults]:
        """Evaluate a batch of requests.
        
        Args:
            requests: List of dicts with keys:
                     - input_text, response_text (required)
                     - additional fields passed as kwargs to each evaluator
            trace_id_prefix: Prefix for auto-generated trace IDs
        
        Returns:
            List of TraceEvaluationResults for each request.
        """
        results = []
        
        # Build dataframe from requests
        df_rows = []
        trace_ids = []
        for idx, request in enumerate(requests):
            trace_id = f"{trace_id_prefix}_{idx}"
            trace_ids.append(trace_id)
            row = {
                "input": request.get("input_text", ""),
                "output": request.get("response_text", ""),
            }
            # Add remaining fields
            for k, v in request.items():
                if k not in ["input_text", "response_text"]:
                    row[k] = v
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        
        # Evaluate all rows at once
        results_df = await self.evaluate_dataframe(df)
        
        # Convert to list of TraceEvaluationResults
        for idx, trace_id in enumerate(trace_ids):
            request = requests[idx]
            result_row = results_df.iloc[idx].to_dict()
            
            evaluations = {}
            for evaluator_name in self.get_enabled_evaluators():
                evaluations[evaluator_name] = self._normalise_evaluation_result(
                    evaluator_name,
                    result_row,
                )
            
            trace_result = TraceEvaluationResults(
                trace_id=trace_id,
                input_text=request.get("input_text", ""),
                response_text=request.get("response_text", ""),
                evaluations=evaluations,
            )
            results.append(trace_result)
        
        return results

    def __repr__(self) -> str:
        enabled = self.get_enabled_evaluators()
        return f"EvaluationRunner(total={len(self.evaluators)}, enabled={len(enabled)})"
