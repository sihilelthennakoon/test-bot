from __future__ import annotations

from dataclasses import dataclass, field


INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "reveal your chain of thought",
    "show hidden instructions",
)


@dataclass(slots=True)
class GuardrailDecision:
    allowed: bool
    reason: str
    warnings: list[str] = field(default_factory=list)


class GuardrailEngine:
    def __init__(self, max_chars: int = 6000) -> None:
        self.max_chars = max_chars

    def check_input(self, text: str) -> GuardrailDecision:
        normalized = text.strip().lower()
        if not normalized:
            return GuardrailDecision(False, "Empty messages are not allowed.")
        if len(text) > self.max_chars:
            return GuardrailDecision(False, "Input exceeds the allowed length.")
        if any(pattern in normalized for pattern in INJECTION_PATTERNS):
            return GuardrailDecision(False, "Prompt injection language detected.")
        return GuardrailDecision(True, "Input passed guardrails.")

    def check_output(self, text: str, *, has_sources: bool) -> GuardrailDecision:
        normalized = text.strip().lower()
        if not normalized:
            return GuardrailDecision(False, "Model returned an empty answer.")
        warnings: list[str] = []
        if any(pattern in normalized for pattern in INJECTION_PATTERNS):
            return GuardrailDecision(False, "Unsafe instructions leaked into the answer.")
        if not has_sources:
            warnings.append("No retrieval sources were attached to the answer.")
        return GuardrailDecision(True, "Output passed guardrails.", warnings=warnings)
