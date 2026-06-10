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
        '  "evidence": [],\n'
        '  "explanation": "",\n'
        '  "recommended_action": ""\n'
        "}\n\n"
        f"Evidence:\n{json.dumps(evidence, ensure_ascii=True, indent=2)}"
    )


def _safe_json_loads(payload: str) -> dict[str, Any]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        return json.loads(payload[start : end + 1])


def _strip_json_fence_markers(content: str) -> str:
    cleaned = re.sub(r"^\s*```json\s*", "", content, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


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
        evidence = [str(evidence)]

    normalized = {
        "root_cause_category": category,
        "confidence": round(confidence, 3),
        "evidence": [str(item).strip() for item in evidence if str(item).strip()],
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
            parsed = _safe_json_loads(content)
        except json.JSONDecodeError as exc:
            raise RCAJudgeError("RCA LLM judge returned invalid JSON.") from exc

        return _normalize_result(parsed)
