from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


RCA_CATEGORIES = (
    "RETRIEVAL_FAILURE",
    "MODEL_IGNORED_CONTEXT",
    "GENERATION_OR_PROMPT_ISSUE",
    "SAFETY_GUARDRAIL_ISSUE",
    "RUNTIME_OR_SPAN_ERROR",
    "EVALUATION_DATA_GAP",
    "NO_ISSUE_DETECTED",
    "UNKNOWN",
)


def _json_only_prompt(evidence: dict[str, Any]) -> str:
    return (
        "You are an expert Root Cause Analysis judge for an enterprise RAG chatbot.\n"
        "Evaluation scores are symptoms, not root causes.\n"
        "Choose exactly one root cause category from this list:\n"
        f"{', '.join(RCA_CATEGORIES)}\n\n"
        "Use the evidence below to determine:\n"
        "1. most likely root cause\n"
        "2. supporting evidence\n"
        "3. confidence from 0.0 to 1.0\n"
        "4. recommended action\n\n"
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

    return {
        "root_cause_category": category,
        "confidence": round(confidence, 3),
        "evidence": [str(item).strip() for item in evidence if str(item).strip()],
        "explanation": str(result.get("explanation", "")).strip(),
        "recommended_action": str(result.get("recommended_action", "")).strip(),
    }


def _heuristic_judgement(evidence: dict[str, Any]) -> dict[str, Any]:
    scores = evidence.get("scores", {})
    retrieval = evidence.get("retrieval", {})
    guardrails = evidence.get("guardrails", {})
    spans = evidence.get("spans", {})

    correctness = scores.get("correctness")
    relevance = scores.get("relevance")
    faithfulness = scores.get("faithfulness")
    safety = scores.get("safety")
    doc_count = retrieval.get("doc_count") or 0
    context_has_answer = retrieval.get("context_has_answer")
    answer_supported = retrieval.get("answer_supported_by_context")
    span_status = spans.get("status")
    pii_detected = guardrails.get("pii_detected")
    input_allowed = guardrails.get("input_allowed")
    output_allowed = guardrails.get("output_allowed")

    if span_status not in {None, "", "OK"} or spans.get("errors"):
        return {
            "root_cause_category": "RUNTIME_OR_SPAN_ERROR",
            "confidence": 0.97,
            "evidence": [
                f"Span status is {span_status or 'UNKNOWN'}",
                f"Span errors: {', '.join(spans.get('errors', [])) or 'none recorded'}",
            ],
            "explanation": "The interaction shows runtime or span failures that can directly explain degraded output quality.",
            "recommended_action": "Inspect the failing node/span logs, fix the underlying exception or timeout, and rerun the trace.",
        }

    if safety is not None and safety < 0.5 and (pii_detected is True or input_allowed is False or output_allowed is False):
        return {
            "root_cause_category": "SAFETY_GUARDRAIL_ISSUE",
            "confidence": 0.96,
            "evidence": [
                f"Safety score is {safety}",
                f"PII detected={pii_detected}, input_allowed={input_allowed}, output_allowed={output_allowed}",
            ],
            "explanation": "Safety or guardrail signals indicate the response was blocked, redacted, or otherwise impacted by policy enforcement.",
            "recommended_action": "Review guardrail thresholds and masking behavior, and separate policy-triggered failures from answer quality regressions.",
        }

    if (correctness is not None and correctness < 0.5) or (relevance is not None and relevance < 0.5):
        if doc_count == 0 or not retrieval.get("context"):
            return {
                "root_cause_category": "RETRIEVAL_FAILURE",
                "confidence": 0.95,
                "evidence": [
                    f"Correctness={correctness}, relevance={relevance}",
                    f"Retrieved document count is {doc_count}",
                ],
                "explanation": "The answer quality is poor and retrieval produced no usable supporting context.",
                "recommended_action": "Audit query construction, indexing, and retriever settings; add retrieval diagnostics for zero-hit traces.",
            }

        if context_has_answer is True and answer_supported is False:
            return {
                "root_cause_category": "MODEL_IGNORED_CONTEXT",
                "confidence": 0.91,
                "evidence": [
                    "Retrieved context appears to contain the answer",
                    "Answer is not supported by the retrieved context",
                ],
                "explanation": "The retriever found relevant evidence, but the model response did not follow it.",
                "recommended_action": "Tighten grounding instructions, reduce prompt ambiguity, and inspect answer synthesis behavior for context adherence.",
            }

        return {
            "root_cause_category": "GENERATION_OR_PROMPT_ISSUE",
            "confidence": 0.82,
            "evidence": [
                f"Correctness={correctness}, relevance={relevance}, faithfulness={faithfulness}",
                f"Retrieved document count={doc_count}, context_has_answer={context_has_answer}",
            ],
            "explanation": "The trace shows answer-quality degradation without a clear retrieval outage, which points to prompt or generation behavior.",
            "recommended_action": "Review prompt instructions, answer formatting constraints, and model settings for this interaction path.",
        }

    if all(score is None for score in (correctness, relevance, faithfulness, safety)):
        return {
            "root_cause_category": "EVALUATION_DATA_GAP",
            "confidence": 0.88,
            "evidence": ["No evaluation scores were available for this trace."],
            "explanation": "There is not enough evaluation data to infer a reliable root cause.",
            "recommended_action": "Backfill missing evaluator outputs and ensure the RCA dataframe includes complete annotations.",
        }

    if (
        correctness is not None
        and correctness >= 0.5
        and relevance is not None
        and relevance >= 0.5
        and (safety is None or safety >= 0.5)
        and span_status in {None, "", "OK"}
    ):
        return {
            "root_cause_category": "NO_ISSUE_DETECTED",
            "confidence": 0.78,
            "evidence": [
                f"Correctness={correctness}, relevance={relevance}, faithfulness={faithfulness}, safety={safety}",
                f"Span status={span_status or 'UNKNOWN'}",
            ],
            "explanation": "The available evidence does not show a clear production issue for this trace.",
            "recommended_action": "No RCA action is required unless external user feedback indicates a hidden issue.",
        }

    return {
        "root_cause_category": "UNKNOWN",
        "confidence": 0.5,
        "evidence": ["Signals were mixed and did not match a single RCA pattern cleanly."],
        "explanation": "The trace requires manual review because the evidence is inconclusive.",
        "recommended_action": "Inspect the raw trace, retrieved context, and evaluator explanations together for a deeper diagnosis.",
    }


@dataclass(slots=True)
class RCAJudge:
    model_name: str = "gemini-2.5-flash"
    api_key: str | None = None
    enable_llm: bool = True

    def judge(self, evidence: dict[str, Any]) -> dict[str, Any]:
        llm_result = self._judge_with_llm(evidence)
        if llm_result is not None:
            return llm_result
        return _normalize_result(_heuristic_judgement(evidence))

    def _judge_with_llm(self, evidence: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enable_llm:
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception:
            return None

        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None

        prompt = _json_only_prompt(evidence)
        try:
            model = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=api_key, temperature=0)
            response = model.invoke(prompt)
        except Exception:
            return None
        content = getattr(response, "content", None)

        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str) or not content.strip():
            return None

        try:
            return _normalize_result(_safe_json_loads(content))
        except json.JSONDecodeError:
            return None
