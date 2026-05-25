"""Routing evaluator using Phoenix SDK create_evaluator()."""

from __future__ import annotations

from typing import TYPE_CHECKING
import re

if TYPE_CHECKING:
    from phoenix.evals import Evaluator


def create_routing_evaluator(
    min_doc_count: int = 1,
    min_score_threshold: float = 0.3,
) -> Evaluator:
    """Create a Phoenix routing evaluator using create_evaluator().
    
    Validates retriever effectiveness by checking:
    - Documents were retrieved (count >= min_doc_count)
    - Similarity scores are above threshold
    - Retrieved docs contain text and have cheap lexical overlap with the query
    
    Args:
        min_doc_count: Minimum number of docs to retrieve
        min_score_threshold: Minimum similarity score (0-1)
    
    Returns:
        Phoenix Evaluator for routing validation.
    
    Example:
        >>> from ragbot.evaluations.routing import create_routing_evaluator
        >>> evaluator = create_routing_evaluator(min_doc_count=1, min_score_threshold=0.3)
        >>> results = await evaluator.evaluate_dataframe(df)
        >>> # df should have columns: retrieved_docs, scores
    """
    try:
        from phoenix.evals import create_evaluator
    except ImportError:
        raise ImportError(
            "Phoenix SDK not installed. Install with: pip install arize-phoenix[evals]"
        )
    
    def doc_text(doc: object) -> str:
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            return str(doc.get("text", ""))
        return str(getattr(doc, "text", ""))

    def tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    @create_evaluator(name="routing", kind="code")
    def routing_function(input: str | None = None, retrieved_docs: list | None = None, scores: list | None = None) -> dict[str, object]:
        """Validate routing quality.
        
        Args:
            retrieved_docs: List of retrieved documents
            scores: List of similarity scores
            
        Returns:
            "PASSING" or "FAILING"
        """
        retrieved_docs = retrieved_docs or []
        scores = scores or []
        
        # Check minimum document count
        doc_count = len(retrieved_docs) if isinstance(retrieved_docs, list) else 0

        if doc_count < min_doc_count:
            return {
                "label": "FAILING",
                "score": 0.0,
                "explanation": f"Retrieved {doc_count} docs; expected at least {min_doc_count}.",
            }

        texts = [doc_text(doc).strip() for doc in retrieved_docs]
        if not any(texts):
            return {
                "label": "FAILING",
                "score": 0.0,
                "explanation": "Retrieved docs do not contain usable text.",
            }

        # Check minimum scores if provided
        if scores:
            scores_list = [float(score) for score in (scores if isinstance(scores, list) else [scores])]
            min_score = min(scores_list) if scores_list else 0.0
            top_score = max(scores_list) if scores_list else 0.0

            if min_score < min_score_threshold:
                return {
                    "label": "FAILING",
                    "score": max(0.0, min(1.0, top_score)),
                    "explanation": f"Minimum retrieval score {min_score:.3f} is below threshold {min_score_threshold:.3f}.",
                }

        query_tokens = tokenize(input or "")
        doc_tokens = set().union(*(tokenize(text) for text in texts)) if texts else set()
        if query_tokens and doc_tokens:
            overlap = len(query_tokens & doc_tokens) / len(query_tokens)
            if overlap == 0.0:
                return {
                    "label": "FAILING",
                    "score": 0.0,
                    "explanation": "Retrieved docs have no lexical overlap with the query.",
                }

        return {
            "label": "PASSING",
            "score": 1.0,
            "explanation": "Retrieved docs passed count, text, score, and lexical checks.",
        }
    
    return routing_function
