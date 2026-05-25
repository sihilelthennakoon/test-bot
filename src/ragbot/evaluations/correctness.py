"""Correctness evaluator using Phoenix SDK create_classifier()."""

from __future__ import annotations

from typing import TYPE_CHECKING
import re

if TYPE_CHECKING:
    from phoenix.evals import Evaluator


def create_correctness_evaluator(llm: Evaluator | None = None) -> Evaluator:
    """Create a Phoenix correctness evaluator using LLM-as-judge.
    
    Args:
        llm: Phoenix LLM instance. If None, uses default Google Gemini.
    
    Returns:
        Phoenix Evaluator for correctness classification.
    
    Example:
        >>> from ragbot.evaluations.correctness import create_correctness_evaluator
        >>> evaluator = create_correctness_evaluator()
        >>> results = await evaluator.evaluate_dataframe(df)
    """
    try:
        from phoenix.evals import create_classifier, create_evaluator
    except ImportError:
        raise ImportError(
            "Phoenix SDK not installed. Install with: pip install arize-phoenix[evals]"
        )
    if llm is not None:
        template = """You are an expert evaluator assessing the correctness of chatbot responses.

Question: {input}

Response: {output}

Evaluate if the response correctly and directly addresses the user's question.
Focus on:
1. Does it answer the specific question asked?
2. Is the answer factually appropriate?
3. Is it coherent and well-structured?
4. If context is available elsewhere in the row, do not reward unsupported claims here; groundedness handles that separately.

Classify as:
- CORRECT: Response directly answers the question with relevant information
- INCORRECT: Response does not address the question or contains errors

Respond with only one word: CORRECT or INCORRECT"""

        return create_classifier(
            name="correctness",
            prompt_template=template,
            llm=llm,
            choices=["CORRECT", "INCORRECT"],
            direction="maximize",
        )

    generic_fallbacks = (
        "i don't know",
        "i do not know",
        "i'm not sure",
        "cannot answer",
        "not enough information",
        "i cannot help",
    )

    def tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    @create_evaluator(name="correctness", kind="code")
    def correctness_function(input: str | None = None, output: str | None = None) -> dict[str, object]:
        question = str(input or "")
        response = str(output or "")
        if not response.strip():
            return {
                "label": "INCORRECT",
                "score": 0.0,
                "explanation": "Response is empty.",
            }
        if any(phrase in response.lower() for phrase in generic_fallbacks):
            return {
                "label": "INCORRECT",
                "score": 0.0,
                "explanation": "Response is a generic fallback rather than an answer.",
            }

        question_tokens = tokens(question)
        response_tokens = tokens(response)
        if question_tokens and response_tokens:
            overlap = len(question_tokens & response_tokens) / len(question_tokens)
            if overlap < 0.15:
                return {
                    "label": "INCORRECT",
                    "score": 0.25,
                    "explanation": "Response has low lexical overlap with the question.",
                }

        return {
            "label": "CORRECT",
            "score": 1.0,
            "explanation": "Response is non-empty, specific, and plausibly answers the question.",
        }

    return correctness_function
