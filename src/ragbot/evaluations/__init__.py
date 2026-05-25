"""Phoenix evaluation module for RAG chatbot.

This module provides evaluators for assessing chatbot responses across multiple dimensions:
- Correctness: Is the answer correct? (LLM-as-judge)
- Relevance: Is the response relevant to the query? (LLM-as-judge)
- Groundedness: Is the answer supported by retrieved context? (hallucination detection)
- Routing: Did the retriever find relevant documents? (deterministic)
- Safety: Is input/output safe from PII, injection, toxicity? (hybrid: LLM + rules)
- Latency: Does response time meet SLA? (deterministic, threshold-based)
- Format: Does response conform to schema? (deterministic, Pydantic)

All evaluators use Phoenix SDK APIs directly (create_classifier, create_evaluator, etc.).
"""

from .config import EvaluationConfig, get_phoenix_llm
from .offline import OfflineEvaluationPipeline, run_offline_evaluation
from .results import EvaluationResult, EvaluationRunResults, TraceEvaluationResults
from .runner import EvaluationRunner

# Factory functions for evaluators (function references, not class instances)
# These are imported lazily to avoid circular dependencies
from . import (
    correctness,
    format as format_eval,
    groundedness,
    latency,
    relevance,
    routing,
    safety,
)

__all__ = [
    # Configuration
    "EvaluationConfig",
    "get_phoenix_llm",
    # Results & Orchestration
    "EvaluationResult",
    "TraceEvaluationResults",
    "EvaluationRunResults",
    "EvaluationRunner",
    # Offline evaluation
    "OfflineEvaluationPipeline",
    "run_offline_evaluation",
    # Evaluator modules (factory functions within)
    "correctness",
    "relevance",
    "groundedness",
    "routing",
    "safety",
    "latency",
    "format_eval",
]

