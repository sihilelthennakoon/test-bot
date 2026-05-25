"""Evaluation results schema and storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from phoenix.evals import Score


class EvaluationResult(BaseModel):
    """Result from a single evaluator on a single request."""

    evaluator_name: str = Field(description="Name of the evaluator")
    score: float = Field(ge=0.0, le=1.0, description="Evaluation score (0-1)")
    passed: bool = Field(description="Whether evaluation passed")
    reason: str = Field(description="Explanation of result")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def from_phoenix_score(cls, evaluator_name: str, score: Score) -> EvaluationResult:
        """Create EvaluationResult from a Phoenix Score object.
        
        Args:
            evaluator_name: Name of the evaluator that produced this score
            score: Phoenix Score object from phoenix.evals
            
        Returns:
            EvaluationResult wrapping the Phoenix score
            
        Example:
            >>> from phoenix.evals import Score
            >>> phoenix_score = Score(value=0.95, label="CORRECT", ...)
            >>> result = EvaluationResult.from_phoenix_score("correctness", phoenix_score)
        """
        # Extract relevant fields from Phoenix Score or a dict-like Phoenix result.
        if isinstance(score, dict):
            score_value = score.get("value", score.get("score", 0.5))
            label = score.get("label", "UNKNOWN")
            reason = score.get("explanation", f"Classification: {label}")
        else:
            score_value = getattr(score, "value", 0.5)
            label = getattr(score, "label", "UNKNOWN")
            reason = getattr(score, "explanation", f"Classification: {label}")
        
        # Convert label to boolean if possible
        passed = label in ['CORRECT', 'SAFE', 'VALID', 'PASSING', 'PASS', 'RELEVANT', 'GROUNDED']
        
        # Ensure score is float between 0-1
        if isinstance(score_value, str):
            score_value = 1.0 if passed else 0.0
        else:
            score_value = float(score_value) if score_value is not None else 0.5
        
        return cls(
            evaluator_name=evaluator_name,
            score=max(0.0, min(1.0, score_value)),
            passed=passed,
            reason=reason,
            metadata={"phoenix_label": label},
        )


class TraceEvaluationResults(BaseModel):
    """Results from all evaluators for a single request trace."""

    trace_id: str = Field(description="Unique trace/request ID")
    input_text: str = Field(description="User input")
    response_text: str = Field(description="Bot response")
    evaluations: dict[str, EvaluationResult] = Field(
        default_factory=dict,
        description="Results keyed by evaluator name",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def aggregate_score(self) -> float:
        """Calculate mean score across all evaluators."""
        if not self.evaluations:
            return 0.5
        scores = [e.score for e in self.evaluations.values()]
        return sum(scores) / len(scores)

    def pass_rate(self) -> float:
        """Calculate percentage of evaluators that passed."""
        if not self.evaluations:
            return 0.0
        passed = sum(1 for e in self.evaluations.values() if e.passed)
        return passed / len(self.evaluations)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "input_text": self.input_text,
            "response_text": self.response_text,
            "evaluations": {
                name: {
                    "score": result.score,
                    "passed": result.passed,
                    "reason": result.reason,
                    "metadata": result.metadata,
                }
                for name, result in self.evaluations.items()
            },
            "aggregate_score": self.aggregate_score(),
            "pass_rate": self.pass_rate(),
            "timestamp": self.timestamp.isoformat(),
        }

    def items(self):
        """Expose evaluator results like a mapping for convenient iteration."""
        return self.evaluations.items()

    def keys(self):
        return self.evaluations.keys()

    def values(self):
        return self.evaluations.values()

    def __getitem__(self, key: str):
        return self.evaluations[key]


class EvaluationRunResults(BaseModel):
    """Aggregated results from evaluating a batch of requests."""

    run_id: str = Field(description="Unique run ID")
    run_name: str = Field(description="Human-readable run name")
    total_requests: int = Field(description="Total requests evaluated")
    traces: list[TraceEvaluationResults] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        if not self.traces:
            return {
                "total_requests": 0,
                "evaluators": {},
            }

        # Collect all evaluator scores
        evaluator_scores: dict[str, list[float]] = {}
        evaluator_passes: dict[str, int] = {}

        for trace in self.traces:
            for evaluator_name, result in trace.evaluations.items():
                if evaluator_name not in evaluator_scores:
                    evaluator_scores[evaluator_name] = []
                    evaluator_passes[evaluator_name] = 0
                evaluator_scores[evaluator_name].append(result.score)
                if result.passed:
                    evaluator_passes[evaluator_name] += 1

        # Compute statistics
        evaluator_stats = {}
        for evaluator_name in evaluator_scores:
            scores = evaluator_scores[evaluator_name]
            evaluator_stats[evaluator_name] = {
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "pass_count": evaluator_passes[evaluator_name],
                "pass_rate": evaluator_passes[evaluator_name] / len(scores),
            }

        return {
            "total_requests": self.total_requests,
            "evaluators": evaluator_stats,
            "mean_aggregate_score": sum(t.aggregate_score() for t in self.traces) / len(self.traces),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "total_requests": self.total_requests,
            "summary": self.to_summary(),
            "traces": [t.to_dict() for t in self.traces],
            "timestamp": self.timestamp.isoformat(),
        }

    def save_json(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: str | Path) -> None:
        """Save results to CSV format (one row per trace)."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.traces:
            return

        # Collect all unique evaluator names
        all_evaluators = set()
        for trace in self.traces:
            all_evaluators.update(trace.evaluations.keys())

        fieldnames = [
            "trace_id",
            "input_length",
            "response_length",
            "aggregate_score",
            "pass_rate",
            *sorted(all_evaluators),
            f"{sorted(all_evaluators)[0]}_reason" if all_evaluators else "",
        ]
        fieldnames = [f for f in fieldnames if f]  # Remove empty strings

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trace in self.traces:
                row = {
                    "trace_id": trace.trace_id,
                    "input_length": len(trace.input_text),
                    "response_length": len(trace.response_text),
                    "aggregate_score": trace.aggregate_score(),
                    "pass_rate": trace.pass_rate(),
                }

                for evaluator_name in sorted(all_evaluators):
                    if evaluator_name in trace.evaluations:
                        row[evaluator_name] = trace.evaluations[
                            evaluator_name
                        ].score

                writer.writerow(row)
