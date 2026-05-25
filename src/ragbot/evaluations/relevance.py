"""Relevance evaluator using Phoenix SDK create_classifier()."""

from __future__ import annotations

from typing import TYPE_CHECKING
import re

if TYPE_CHECKING:
    from phoenix.evals import Evaluator


def create_relevance_evaluator(llm: Evaluator | None = None) -> Evaluator:
    """Create a Phoenix relevance evaluator using LLM-as-judge.
    
    Args:
        llm: Phoenix LLM instance. If None, uses default Google Gemini.
    
    Returns:
        Phoenix Evaluator for relevance classification.
    
    Example:
        >>> from ragbot.evaluations.relevance import create_relevance_evaluator
        >>> evaluator = create_relevance_evaluator()
        >>> results = await evaluator.evaluate_dataframe(df)
    """
    try:
        from phoenix.evals import create_classifier, create_evaluator
    except ImportError:
        raise ImportError(
            "Phoenix SDK not installed. Install with: pip install arize-phoenix[evals]"
        )
    if llm is not None:
        template = """You are an expert evaluator assessing the relevance of chatbot responses.

Question: {input}

Response: {output}

Evaluate if the response is relevant and helpful to the user's question.
A response is relevant if:
1. It addresses the topic of the question
2. It provides information related to what was asked
3. It honors the user's explicit constraints
4. It is cohesive and on-topic

Classify as:
- RELEVANT: Response addresses the question's topic and provides helpful information
- IRRELEVANT: Response does not address the question or is off-topic

Respond with only one word: RELEVANT or IRRELEVANT"""

        return create_classifier(
            name="relevance",
            prompt_template=template,
            llm=llm,
            choices=["RELEVANT", "IRRELEVANT"],
            direction="maximize",
        )

    generic_fallbacks = (
        "i don't know",
        "i do not know",
        "i'm not sure",
        "cannot answer",
        "not enough information",
    )

    def tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    @create_evaluator(name="relevance", kind="code")
    def relevance_function(input: str | None = None, output: str | None = None) -> dict[str, object]:
        question = str(input or "")
        response = str(output or "")
        if not question.strip() or not response.strip():
            return {
                "label": "IRRELEVANT",
                "score": 0.0,
                "explanation": "Question or response is empty.",
            }
        if any(phrase in response.lower() for phrase in generic_fallbacks):
            return {
                "label": "IRRELEVANT",
                "score": 0.0,
                "explanation": "Response is a generic fallback.",
            }

        question_tokens = tokens(question)
        response_tokens = tokens(response)
        overlap = len(question_tokens & response_tokens) / len(question_tokens) if question_tokens else 0.0
        if overlap < 0.10:
            return {
                "label": "IRRELEVANT",
                "score": 0.25,
                "explanation": "Response appears off-topic for the question.",
            }

        return {
            "label": "RELEVANT",
            "score": 1.0,
            "explanation": "Response is on-topic and non-generic.",
        }

    return relevance_function
