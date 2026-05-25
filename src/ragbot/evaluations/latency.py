"""Latency evaluator using Phoenix SDK create_evaluator()."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phoenix.evals import Evaluator


def create_latency_evaluator(
    pass_threshold_ms: int = 3000,
    warning_threshold_ms: int = 5000,
) -> Evaluator:
    """Create a Phoenix latency evaluator using create_evaluator().
    
    Monitors response time against SLA thresholds:
    - Pass: <= pass_threshold_ms
    - Warning: > pass_threshold_ms and <= warning_threshold_ms
    - Fail: > warning_threshold_ms
    
    Args:
        pass_threshold_ms: Pass threshold in milliseconds (default: 3000)
        warning_threshold_ms: Warning threshold in milliseconds (default: 5000)
    
    Returns:
        Phoenix Evaluator for latency validation.
    
    Example:
        >>> from ragbot.evaluations.latency import create_latency_evaluator
        >>> evaluator = create_latency_evaluator(pass_threshold_ms=3000)
        >>> results = await evaluator.evaluate_dataframe(df)
        >>> # df should have column: duration_ms
    """
    try:
        from phoenix.evals import create_evaluator
    except ImportError:
        raise ImportError(
            "Phoenix SDK not installed. Install with: pip install arize-phoenix[evals]"
        )
    
    @create_evaluator(name="latency", kind="code")
    def latency_function(duration_ms: float | None = None) -> dict[str, object]:
        """Evaluate latency against thresholds.
        
        Args:
            duration_ms: Response time in milliseconds
            
        Returns:
            "PASS", "WARNING", or "FAIL"
        """
        if duration_ms is None or duration_ms < 0:
            return {
                "label": "FAIL",
                "score": 0.0,
                "explanation": "Duration is missing or invalid.",
            }
        
        if duration_ms <= pass_threshold_ms:
            return {
                "label": "PASS",
                "score": 1.0,
                "explanation": f"Latency {duration_ms:.0f}ms is within pass threshold {pass_threshold_ms}ms.",
            }
        if duration_ms <= warning_threshold_ms:
            return {
                "label": "WARNING",
                "score": 0.5,
                "explanation": f"Latency {duration_ms:.0f}ms exceeded pass threshold {pass_threshold_ms}ms.",
            }
        return {
            "label": "FAIL",
            "score": 0.0,
            "explanation": f"Latency {duration_ms:.0f}ms exceeded warning threshold {warning_threshold_ms}ms.",
        }
    
    return latency_function
