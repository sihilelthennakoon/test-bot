from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


RCA_CATEGORIES = (
    "MODEL_DRIFT",
    "PROMPT_REGRESSION",
    "RETRIEVAL_FAILURE",
    "TOOL_INVOCATION_ERROR",
    "UNKNOWN",
)


class RCAJudgeError(RuntimeError):
    """Raised when the LLM judge cannot produce a valid RCA result."""


def _json_only_prompt(evidence: dict[str, Any]) -> str:
    return (
        "You are an expert Root Cause Analysis judge for an enterprise RAG chatbot.\n"
        "Epic goal: Provide root-cause signal distinguishing model drift, prompt regression, "
        "retrieval failure, and tool invocation errors.\n"
        "Your primary task is to choose exactly one root cause category from this list:\n"
        f"{', '.join(RCA_CATEGORIES)}\n\n"
        "Important guidance:\n"
        "- Evaluation scores are symptoms, not root causes.\n"
        "- Reason from retrieved context, retrieval statistics, guardrail outcomes, span status/errors, and answer behavior.\n"
        "- Use UNKNOWN only when the evidence is inconclusive.\n"
        "- Provide a short RCA description in the explanation field.\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "root_cause_category": "",\n'
        '  "confidence": 0.0,\n'
        '  "evidence": ["short evidence string"],\n'
        '  "explanation": "",\n'
        '  "recommended_action": ""\n'
        "}\n"
        'The "evidence" field must be an array of strings only.\n'
        'Valid evidence example: ["retrieval.avg_score: 0.14", "context_has_answer: false"]\n'
        'Invalid evidence example: ["question": "What is your name?"]\n\n'
        f"Evidence:\n{json.dumps(evidence, ensure_ascii=True, indent=2)}"
    )


def _extract_outermost_json_object(payload: str) -> str:
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return payload
    return payload[start : end + 1]


def _repair_malformed_evidence_array(payload: str) -> str:
    pattern = re.compile(r'("evidence"\s*:\s*\[\s*)(.*?)(\s*\])', flags=re.DOTALL)

    def replace(match: re.Match[str]) -> str:
        prefix, body, suffix = match.groups()
        if not re.search(r'"\s*:\s*', body):
            return match.group(0)

        repaired_body = re.sub(
            r'"([^"\\]+)"\s*:\s*(".*?"|\[[^\]]*\]|-?\d+(?:\.\d+)?|true|false|null)',
            lambda item: json.dumps(
                f"{item.group(1)}: {json.loads(item.group(2)) if item.group(2).startswith(chr(34)) else item.group(2)}"
            ),
            body,
            flags=re.DOTALL,
        )
        repaired_body = repaired_body.replace('",\n    "', '",\n    "')
        return f"{prefix}{repaired_body}{suffix}"

    return pattern.sub(replace, payload, count=1)


def _prepare_json_payload(payload: str) -> str:
    return _repair_malformed_evidence_array(_extract_outermost_json_object(payload))


def _safe_json_loads(payload: str) -> dict[str, Any]:
    prepared_payload = _prepare_json_payload(payload)
    try:
        return json.loads(prepared_payload)
    except json.JSONDecodeError:
        extracted_payload = _extract_outermost_json_object(prepared_payload)
        if extracted_payload == prepared_payload:
            raise
        repaired_payload = _repair_malformed_evidence_array(extracted_payload)
        return json.loads(repaired_payload)


def _strip_json_fence_markers(content: str) -> str:
    cleaned = re.sub(r"^\s*```json\s*", "", content, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _stringify_evidence_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        parts = []
        for key, value in item.items():
            text = _stringify_evidence_item(value)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts).strip()
    if isinstance(item, (list, tuple, set)):
        parts = [_stringify_evidence_item(value) for value in item]
        return "; ".join(part for part in parts if part).strip()
    return str(item).strip()


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    category = str(result.get("root_cause_category", "UNKNOWN")).strip().upper() or "UNKNOWN"
    if category not in RCA_CATEGORIES:
        category = "UNKNOWN"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    evidence = result.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [evidence]

    normalized = {
        "root_cause_category": category,
        "confidence": round(confidence, 3),
        "evidence": [text for item in evidence if (text := _stringify_evidence_item(item))],
        "explanation": str(result.get("explanation", "")).strip(),
        "recommended_action": str(result.get("recommended_action", "")).strip(),
    }
    if not normalized["explanation"]:
        raise RCAJudgeError("LLM judge returned an empty RCA explanation.")
    return normalized


@dataclass(slots=True)
class RCAJudge:
    model_name: str = "gemini-2.5-flash"
    api_key: str | None = None

    def judge(self, evidence: dict[str, Any]) -> dict[str, Any]:
        return self._judge_with_llm(evidence)

    def _judge_with_llm(self, evidence: dict[str, Any]) -> dict[str, Any]:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            raise RCAJudgeError("langchain_google_genai is required for RCA LLM judging.") from exc

        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RCAJudgeError("GOOGLE_API_KEY must be set for RCA LLM judging.")

        prompt = _json_only_prompt(evidence)
        try:
            model = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=api_key, temperature=0)
            response = model.invoke(prompt)
        except Exception as exc:
            raise RCAJudgeError("RCA LLM judge invocation failed.") from exc
        content = getattr(response, "content", None)
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise RCAJudgeError("RCA LLM judge returned empty content.")
        content = _strip_json_fence_markers(content)

        try:
            print (f"RCA Judge raw response content:\n{content}\n--- End of content ---")  # Debug log
            parsed = _safe_json_loads(content)
        except json.JSONDecodeError as exc:
            print(f"Failed to parse RCA LLM judge response as JSON. Content was:\n{content}\n--- End of content ---")  # Debug log
            raise RCAJudgeError("RCA LLM judge returned invalid JSON.") from exc

        return _normalize_result(parsed)
