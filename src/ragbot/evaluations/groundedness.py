"""Groundedness evaluator using Phoenix SDK create_classifier()."""

from __future__ import annotations

from typing import TYPE_CHECKING
import re

if TYPE_CHECKING:
    from phoenix.evals import Evaluator


def create_groundedness_evaluator(llm: Evaluator | None = None) -> Evaluator:
    """Create a Phoenix groundedness evaluator using LLM-as-judge.
    
    Detects hallucinations by checking if the response is faithful to the retrieved context.
    Uses Phoenix's create_classifier with LLM template for faithfulness evaluation.
    
    Args:
        llm: Phoenix LLM instance. If None, uses default Google Gemini.
    
    Returns:
        Phoenix Evaluator for groundedness/faithfulness classification.
    
    Example:
        >>> from ragbot.evaluations.groundedness import create_groundedness_evaluator
        >>> evaluator = create_groundedness_evaluator()
        >>> results = await evaluator.evaluate_dataframe(df)
        >>> # df should have columns: input, output, reference (context)
    """
    try:
        from phoenix.evals import create_classifier, create_evaluator
    except ImportError:
        raise ImportError(
            "Phoenix SDK not installed. Install with: pip install arize-phoenix[evals]"
        )
    if llm is not None:
        template = """You are an expert evaluator assessing whether responses are faithful to source material.

Question: {input}

Source Context: {reference}

Response: {output}

Evaluate if the response is grounded in and faithful to the source context.
Consider:
1. Does the response only use information from the source?
2. Are there any hallucinations or made-up facts?
3. Is the response consistent with the source material?

Classify as:
- GROUNDED: Response is faithful to the source context
- HALLUCINATED: Response contains information not in the source or contradicts it

Respond with only one word: GROUNDED or HALLUCINATED"""

        return create_classifier(
            name="groundedness",
            prompt_template=template,
            llm=llm,
            choices=["GROUNDED", "HALLUCINATED"],
            direction="maximize",
        )

    def tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    @create_evaluator(name="groundedness", kind="code")
    def groundedness_function(
        input: str | None = None,
        output: str | None = None,
        reference: str | None = None,
        context: str | None = None,
    ) -> dict[str, object]:
        del input
        response = str(output or "")
        source = str(reference or context or "")
        if not response.strip():
            return {
                "label": "HALLUCINATED",
                "score": 0.0,
                "explanation": "Response is empty.",
            }
        if not source.strip():
            return {
                "label": "SKIPPED",
                "score": 0.5,
                "explanation": "No context/reference was provided for groundedness evaluation.",
            }

        response_tokens = tokens(response)
        source_tokens = tokens(source)
        if not response_tokens:
            return {
                "label": "HALLUCINATED",
                "score": 0.0,
                "explanation": "Response contains no evaluable content.",
            }
        overlap = len(response_tokens & source_tokens) / len(response_tokens)
        if overlap < 0.30:
            return {
                "label": "HALLUCINATED",
                "score": max(0.0, min(1.0, overlap)),
                "explanation": "Response has low support in the provided context.",
            }

        return {
            "label": "GROUNDED",
            "score": 1.0,
            "explanation": "Response content is supported by the provided context.",
        }

    return groundedness_function

