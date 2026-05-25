"""Format evaluator using Phoenix SDK create_evaluator()."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Type

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from phoenix.evals import Evaluator


def create_format_evaluator(schema: Type[BaseModel] | None = None) -> Evaluator:
    """Create a Phoenix format evaluator using create_evaluator().
    
    Validates response structure against optional Pydantic schema.
    Checks: JSON validity, schema compliance, required fields, data types.
    
    Args:
        schema: Optional Pydantic BaseModel for strict validation
    
    Returns:
        Phoenix Evaluator for format validation.
    
    Example:
        >>> from ragbot.evaluations.format import create_format_evaluator
        >>> from pydantic import BaseModel, Field
        >>> class ChatResponse(BaseModel):
        ...     response: str = Field(..., min_length=1)
        ...     source_docs: list[str] = Field(default_factory=list)
        >>> evaluator = create_format_evaluator(schema=ChatResponse)
        >>> results = await evaluator.evaluate_dataframe(df)
    """
    try:
        from phoenix.evals import create_evaluator
    except ImportError:
        raise ImportError(
            "Phoenix SDK not installed. Install with: pip install arize-phoenix[evals]"
        )
    
    @create_evaluator(name="format", kind="code")
    def format_function(output: str | dict | list | None = None) -> dict[str, object]:
        """Validate response format.
        
        Args:
            output: Response to validate (string or dict)
            
        Returns:
            "VALID" or "INVALID"
        """
        if output is None:
            return {
                "label": "INVALID",
                "score": 0.0,
                "explanation": "Response is missing.",
            }

        if schema is None:
            if isinstance(output, str):
                stripped = output.strip()
                if not stripped:
                    return {
                        "label": "INVALID",
                        "score": 0.0,
                        "explanation": "Response is empty.",
                    }
                if stripped[0] in "[{":
                    try:
                        json.loads(stripped)
                    except json.JSONDecodeError:
                        return {
                            "label": "INVALID",
                            "score": 0.0,
                            "explanation": "Response looks like malformed JSON.",
                        }
                if len(stripped) > 12000:
                    return {
                        "label": "WARNING",
                        "score": 0.5,
                        "explanation": "Response is unusually long.",
                    }
                return {
                    "label": "VALID",
                    "score": 1.0,
                    "explanation": "Plain-text response is non-empty and well formed.",
                }

            return {
                "label": "VALID",
                "score": 1.0,
                "explanation": "Structured response is present.",
            }
        
        # Try to parse as JSON if string
        if isinstance(output, str):
            try:
                response_dict = json.loads(output)
            except json.JSONDecodeError:
                return {
                    "label": "INVALID",
                    "score": 0.0,
                    "explanation": "Schema validation requires valid JSON.",
                }
        else:
            response_dict = output
        
        # Validate against schema if provided
        try:
            schema.model_validate(response_dict, from_attributes=True)
            return {
                "label": "VALID",
                "score": 1.0,
                "explanation": "Response matches the configured schema.",
            }
        except ValidationError as exc:
            return {
                "label": "INVALID",
                "score": 0.0,
                "explanation": f"Response failed schema validation: {exc.errors()[0]['msg']}",
            }
    
    return format_function
