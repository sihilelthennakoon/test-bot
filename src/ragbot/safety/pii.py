from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(slots=True)
class MaskResult:
    text: str
    was_masked: bool
    entities: list[dict[str, str]] = field(default_factory=list)


class PIIMasker:
    def __init__(self) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
            ("phone", re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")),
            ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
            ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
        ]

    def mask(self, text: str) -> MaskResult:
        masked = text
        entities: list[dict[str, str]] = []
        for label, pattern in self._patterns:
            matches = list(pattern.finditer(masked))
            if not matches:
                continue
            for match_index, match in enumerate(matches, start=1):
                raw_value = match.group(0)
                replacement = f"[{label.upper()}_{match_index}]"
                masked = masked.replace(raw_value, replacement)
                entities.append({"label": label, "value": raw_value, "replacement": replacement})
        return MaskResult(text=masked, was_masked=masked != text, entities=entities)
