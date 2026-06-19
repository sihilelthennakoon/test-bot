from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest

from ragbot.rca import rca_judge
from ragbot.rca.rca_judge import RCAJudge, RCAJudgeError
from ragbot.rca.rca_service import generate_rca


def _sample_evidence() -> dict:
    return {
        "trace_id": "trace-1",
        "question": "What changed?",
        "answer": "The model returned an unexpected answer.",
        "scores": {
            "correctness": 0.1,
            "relevance": 0.2,
            "faithfulness": 0.7,
            "safety": 1.0,
        },
        "retrieval": {
            "doc_count": 1,
            "avg_score": 0.8,
            "scores": [0.8],
            "context": "Some retrieved context",
            "context_has_answer": True,
            "answer_supported_by_context": False,
        },
        "guardrails": {
            "input_allowed": True,
            "output_allowed": True,
            "pii_detected": False,
        },
        "spans": {
            "status": "OK",
            "errors": [],
            "latency_ms": 42.0,
            "span_count": 1,
        },
        "metadata": {
            "app_version": "v1",
            "environment": "test",
            "langgraph_nodes": ["generate"],
            "span_ids": ["span-1"],
        },
    }


class _FakeResponse:
    def __init__(self, content):
        self.content = content


def _install_fake_llm(monkeypatch, response_content=None, invoke_error: Exception | None = None):
    class FakeChatGoogleGenerativeAI:
        def __init__(self, model: str, google_api_key: str, temperature: int) -> None:
            self.model = model
            self.google_api_key = google_api_key
            self.temperature = temperature

        def invoke(self, prompt: str):
            if invoke_error is not None:
                raise invoke_error
            return _FakeResponse(response_content)

    fake_module = types.SimpleNamespace(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)


@pytest.mark.parametrize("category", rca_judge.RCA_CATEGORIES)
def test_normalize_result_accepts_allowed_categories(category: str):
    normalized = rca_judge._normalize_result(
        {
            "root_cause_category": category,
            "confidence": 0.8765,
            "evidence": ["one", "two"],
            "explanation": "Short RCA description.",
            "recommended_action": "Review the issue.",
        }
    )

    assert normalized["root_cause_category"] == category
    assert normalized["confidence"] == 0.876


def test_normalize_result_maps_unsupported_category_to_unknown():
    normalized = rca_judge._normalize_result(
        {
            "root_cause_category": "SOMETHING_ELSE",
            "confidence": 0.4,
            "evidence": ["signal"],
            "explanation": "Short RCA description.",
            "recommended_action": "Review the issue.",
        }
    )

    assert normalized["root_cause_category"] == "UNKNOWN"


def test_prompt_is_epic_aligned():
    prompt = rca_judge._json_only_prompt(_sample_evidence())

    assert "Provide root-cause signal distinguishing model drift, prompt regression, retrieval failure, and tool invocation errors." in prompt
    assert "MODEL_DRIFT, PROMPT_REGRESSION, RETRIEVAL_FAILURE, TOOL_INVOCATION_ERROR, UNKNOWN" in prompt
    assert "Evaluation scores are symptoms, not root causes." in prompt
    assert "Provide a short RCA description in the explanation field." in prompt
    assert 'The "evidence" field must be an array of strings only.' in prompt
    assert 'Invalid evidence example: ["question": "What is your name?"]' in prompt


def test_strip_json_fence_markers_removes_markdown_wrapper():
    content = '```json\n{"root_cause_category":"UNKNOWN"}\n```'

    assert rca_judge._strip_json_fence_markers(content) == '{"root_cause_category":"UNKNOWN"}'


def test_safe_json_loads_extracts_json_from_surrounding_prose():
    payload = (
        "Here is the RCA result:\n"
        "{"
        '"root_cause_category":"UNKNOWN",'
        '"confidence":0.7,'
        '"evidence":["Signal limited."],'
        '"explanation":"Short RCA description.",'
        '"recommended_action":"Review manually."'
        "}\n"
        "Thanks."
    )

    parsed = rca_judge._safe_json_loads(payload)

    assert parsed["root_cause_category"] == "UNKNOWN"
    assert parsed["evidence"] == ["Signal limited."]


def test_safe_json_loads_repairs_malformed_evidence_pairs():
    payload = """
    {
      "root_cause_category": "RETRIEVAL_FAILURE",
      "confidence": 0.95,
      "evidence": [
        "question": "What is the capital of England?",
        "answer": "I do not know.",
        "retrieval.context_has_answer": false
      ],
      "explanation": "The retrieval system failed to find relevant documents.",
      "recommended_action": "Review the retriever."
    }
    """

    parsed = rca_judge._safe_json_loads(payload)

    assert parsed["evidence"] == [
        "question: What is the capital of England?",
        "answer: I do not know.",
        "retrieval.context_has_answer: false",
    ]


def test_normalize_result_stringifies_mixed_evidence_items():
    normalized = rca_judge._normalize_result(
        {
            "root_cause_category": "UNKNOWN",
            "confidence": 0.4,
            "evidence": [
                {"question": "What changed?"},
                ["scores.correctness: 0.1", "scores.relevance: 0.2"],
                7,
            ],
            "explanation": "Short RCA description.",
            "recommended_action": "Review the issue.",
        }
    )

    assert normalized["evidence"] == [
        "question: What changed?",
        "scores.correctness: 0.1; scores.relevance: 0.2",
        "7",
    ]


def test_judge_raises_when_google_api_key_missing(monkeypatch: pytest.MonkeyPatch):
    _install_fake_llm(monkeypatch, response_content="{}")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RCAJudgeError, match="GOOGLE_API_KEY"):
        RCAJudge().judge(_sample_evidence())


def test_judge_raises_when_llm_package_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_google_genai":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RCAJudgeError, match="langchain_google_genai"):
        RCAJudge(api_key="test-key").judge(_sample_evidence())


def test_judge_raises_when_model_invocation_fails(monkeypatch: pytest.MonkeyPatch):
    _install_fake_llm(monkeypatch, invoke_error=RuntimeError("boom"))

    with pytest.raises(RCAJudgeError, match="invocation failed"):
        RCAJudge(api_key="test-key").judge(_sample_evidence())


def test_judge_raises_when_response_content_is_empty(monkeypatch: pytest.MonkeyPatch):
    _install_fake_llm(monkeypatch, response_content="   ")

    with pytest.raises(RCAJudgeError, match="empty content"):
        RCAJudge(api_key="test-key").judge(_sample_evidence())


def test_judge_raises_when_response_json_is_invalid(monkeypatch: pytest.MonkeyPatch):
    _install_fake_llm(monkeypatch, response_content="not-json")

    with pytest.raises(RCAJudgeError, match="invalid JSON"):
        RCAJudge(api_key="test-key").judge(_sample_evidence())


def test_judge_accepts_valid_llm_json(monkeypatch: pytest.MonkeyPatch):
    _install_fake_llm(
        monkeypatch,
        response_content=json.dumps(
            {
                "root_cause_category": "PROMPT_REGRESSION",
                "confidence": 0.92,
                "evidence": ["The answer style changed."],
                "explanation": "Short RCA description about a prompt regression.",
                "recommended_action": "Compare the latest prompt template with the prior version.",
            }
        ),
    )

    result = RCAJudge(api_key="test-key").judge(_sample_evidence())

    assert result["root_cause_category"] == "PROMPT_REGRESSION"
    assert result["confidence"] == 0.92
    assert result["explanation"] == "Short RCA description about a prompt regression."


def test_judge_accepts_valid_llm_json_wrapped_in_markdown_fence(monkeypatch: pytest.MonkeyPatch):
    _install_fake_llm(
        monkeypatch,
        response_content=(
            "```json\n"
            "{\n"
            '  "root_cause_category": "UNKNOWN",\n'
            '  "confidence": 0.61,\n'
            '  "evidence": ["The signal is mixed."],\n'
            '  "explanation": "Short RCA description with mixed evidence.",\n'
            '  "recommended_action": "Review the trace manually."\n'
            "}\n"
            "```"
        ),
    )

    result = RCAJudge(api_key="test-key").judge(_sample_evidence())

    assert result["root_cause_category"] == "UNKNOWN"
    assert result["confidence"] == 0.61


def test_judge_repairs_malformed_evidence_pairs_from_llm_response(monkeypatch: pytest.MonkeyPatch):
    _install_fake_llm(
        monkeypatch,
        response_content=(
            "{\n"
            '  "root_cause_category": "RETRIEVAL_FAILURE",\n'
            '  "confidence": 0.95,\n'
            '  "evidence": [\n'
            '    "question": "What is the capital of England?",\n'
            '    "answer": "I do not know."\n'
            "  ],\n"
            '  "explanation": "The retrieval system failed to find any relevant documents.",\n'
            '  "recommended_action": "Review the retriever."\n'
            "}"
        ),
    )

    result = RCAJudge(api_key="test-key").judge(_sample_evidence())

    assert result["root_cause_category"] == "RETRIEVAL_FAILURE"
    assert result["evidence"] == [
        "question: What is the capital of England?",
        "answer: I do not know.",
    ]


def test_generate_rca_returns_scores_and_judgement_fields(monkeypatch: pytest.MonkeyPatch):
    seen_evidence: list[dict] = []

    monkeypatch.setattr(
        "ragbot.rca.rca_service.build_evidence_packages",
        lambda df: [_sample_evidence()],
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service.get_settings",
        lambda: types.SimpleNamespace(gemini_model="gemini-2.5-flash", rca_threshold=0.4),
    )
    monkeypatch.setattr(
        RCAJudge,
        "judge",
        lambda self, evidence: seen_evidence.append(evidence) or {
            "root_cause_category": "RETRIEVAL_FAILURE",
            "confidence": 0.95,
            "evidence": ["The retriever returned irrelevant context."],
            "explanation": "Short RCA description about retrieval failure.",
            "recommended_action": "Inspect retriever configuration.",
        },
    )

    results = generate_rca(pd.DataFrame([{"trace_id": "unused"}]))

    assert results == [
        {
            "trace_id": "trace-1",
            "evaluator_name": "correctness",
            "evaluator_score": 0.1,
            "root_cause_category": "RETRIEVAL_FAILURE",
            "confidence": 0.95,
            "evaluator_scores": {
                "correctness": 0.1,
                "relevance": 0.2,
                "faithfulness": 0.7,
                "safety": 1.0,
            },
            "failed_scores": {
                "correctness": 0.1,
            },
            "evidence": ["The retriever returned irrelevant context."],
            "explanation": "Short RCA description about retrieval failure.",
            "recommended_action": "Inspect retriever configuration.",
            "judge_error": "",
        },
        {
            "trace_id": "trace-1",
            "evaluator_name": "relevance",
            "evaluator_score": 0.2,
            "root_cause_category": "RETRIEVAL_FAILURE",
            "confidence": 0.95,
            "evaluator_scores": {
                "correctness": 0.1,
                "relevance": 0.2,
                "faithfulness": 0.7,
                "safety": 1.0,
            },
            "failed_scores": {
                "relevance": 0.2,
            },
            "evidence": ["The retriever returned irrelevant context."],
            "explanation": "Short RCA description about retrieval failure.",
            "recommended_action": "Inspect retriever configuration.",
            "judge_error": "",
        }
    ]
    assert [item["evaluator_name"] for item in seen_evidence] == ["correctness", "relevance"]
    assert seen_evidence[0]["failed_scores"] == {"correctness": 0.1}
    assert seen_evidence[1]["failed_scores"] == {"relevance": 0.2}


def test_generate_rca_scores_reflect_finalize_only_evidence(monkeypatch: pytest.MonkeyPatch):
    finalize_only_evidence = _sample_evidence()
    finalize_only_evidence["scores"] = {
        "correctness": 0.95,
        "relevance": 0.9,
        "faithfulness": 0.85,
        "safety": 0.8,
    }

    monkeypatch.setattr(
        "ragbot.rca.rca_service.build_evidence_packages",
        lambda df: [finalize_only_evidence],
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service.get_settings",
        lambda: types.SimpleNamespace(gemini_model="gemini-2.5-flash", rca_threshold=0.4),
    )
    monkeypatch.setattr(
        RCAJudge,
        "judge",
        lambda self, evidence: {
            "root_cause_category": "UNKNOWN",
            "confidence": 0.6,
            "evidence": ["The RCA signal is inconclusive."],
            "explanation": "Short RCA description with inconclusive evidence.",
            "recommended_action": "Review the trace manually.",
        },
    )

    results = generate_rca(pd.DataFrame([{"trace_id": "unused"}]))

    assert results == []


def test_generate_rca_uses_env_threshold_to_filter_failing_scores(monkeypatch: pytest.MonkeyPatch):
    seen_evidence: list[dict] = []

    monkeypatch.setattr(
        "ragbot.rca.rca_service.build_evidence_packages",
        lambda df: [_sample_evidence()],
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service.get_settings",
        lambda: types.SimpleNamespace(gemini_model="gemini-2.5-flash", rca_threshold=0.15),
    )
    monkeypatch.setattr(
        RCAJudge,
        "judge",
        lambda self, evidence: seen_evidence.append(evidence) or {
            "root_cause_category": "UNKNOWN",
            "confidence": 0.6,
            "evidence": ["Signal limited to one failing evaluator."],
            "explanation": "Short RCA description with threshold filtering.",
            "recommended_action": "Review the failed evaluator only.",
        },
    )

    results = generate_rca(pd.DataFrame([{"trace_id": "unused"}]))

    assert len(results) == 1
    assert results[0]["evaluator_name"] == "correctness"
    assert results[0]["failed_scores"] == {"correctness": 0.1}
    assert seen_evidence[0]["failed_scores"] == {"correctness": 0.1}


def test_generate_rca_returns_fallback_when_judge_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "ragbot.rca.rca_service.build_evidence_packages",
        lambda df: [_sample_evidence()],
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service.get_settings",
        lambda: types.SimpleNamespace(gemini_model="gemini-2.5-flash", rca_threshold=0.15),
    )
    monkeypatch.setattr(
        RCAJudge,
        "judge",
        lambda self, evidence: (_ for _ in ()).throw(RCAJudgeError("RCA LLM judge returned invalid JSON.")),
    )

    results = generate_rca(pd.DataFrame([{"trace_id": "unused"}]))

    assert len(results) == 1
    assert results[0]["root_cause_category"] == "UNKNOWN"
    assert results[0]["confidence"] == 0.0
    assert "fallback used" in results[0]["explanation"]
    assert results[0]["judge_error"] == "RCA LLM judge returned invalid JSON."


def test_generate_rca_skips_progress_bar_when_no_failed_scores(monkeypatch: pytest.MonkeyPatch):
    finalize_only_evidence = _sample_evidence()
    finalize_only_evidence["scores"] = {
        "correctness": 0.95,
        "relevance": 0.9,
        "faithfulness": 0.85,
        "safety": 0.8,
    }
    created_totals: list[int] = []

    monkeypatch.setattr(
        "ragbot.rca.rca_service.build_evidence_packages",
        lambda df: [finalize_only_evidence],
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service.get_settings",
        lambda: types.SimpleNamespace(gemini_model="gemini-2.5-flash", rca_threshold=0.4),
    )
    monkeypatch.setattr("ragbot.rca.rca_service._create_rca_progress_bar", lambda total, label="RCA": created_totals.append(total) or None)

    results = generate_rca(pd.DataFrame([{"trace_id": "unused"}]))

    assert results == []
    assert created_totals == []


def test_generate_rca_updates_progress_once_per_trace_judgement(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, str, int | None]] = []

    class FakeProgressBar:
        def __init__(self, total: int, label: str) -> None:
            self.label = label
            events.append(("create", label, total))

        def update(self, step: int = 1) -> None:
            events.append(("update", self.label, step))

        def close(self) -> None:
            events.append(("close", self.label, None))

    monkeypatch.setattr(
        "ragbot.rca.rca_service.build_evidence_packages",
        lambda df: [_sample_evidence()],
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service.get_settings",
        lambda: types.SimpleNamespace(gemini_model="gemini-2.5-flash", rca_threshold=0.4),
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service._create_rca_progress_bar",
        lambda total, label="RCA": FakeProgressBar(total, label),
    )
    monkeypatch.setattr(
        RCAJudge,
        "judge",
        lambda self, evidence: {
            "root_cause_category": "RETRIEVAL_FAILURE",
            "confidence": 0.95,
            "evidence": ["The retriever returned irrelevant context."],
            "explanation": "Short RCA description about retrieval failure.",
            "recommended_action": "Inspect retriever configuration.",
        },
    )

    results = generate_rca(pd.DataFrame([{"trace_id": "unused"}]))

    assert [result["evaluator_name"] for result in results] == ["correctness", "relevance"]
    assert events == [
        ("create", "RCA trace-1", 2),
        ("update", "RCA trace-1", 1),
        ("update", "RCA trace-1", 1),
        ("close", "RCA trace-1", None),
    ]


def test_generate_rca_creates_separate_progress_bar_for_each_trace(monkeypatch: pytest.MonkeyPatch):
    events: list[tuple[str, str, int | None]] = []

    class FakeProgressBar:
        def __init__(self, total: int, label: str) -> None:
            self.label = label
            events.append(("create", label, total))

        def update(self, step: int = 1) -> None:
            events.append(("update", self.label, step))

        def close(self) -> None:
            events.append(("close", self.label, None))

    second_evidence = _sample_evidence()
    second_evidence["trace_id"] = "trace-2-abcdef123456"
    second_evidence["scores"] = {
        "correctness": 0.95,
        "relevance": 0.2,
        "faithfulness": 0.9,
        "safety": 1.0,
    }

    monkeypatch.setattr(
        "ragbot.rca.rca_service.build_evidence_packages",
        lambda df: [_sample_evidence(), second_evidence],
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service.get_settings",
        lambda: types.SimpleNamespace(gemini_model="gemini-2.5-flash", rca_threshold=0.4),
    )
    monkeypatch.setattr(
        "ragbot.rca.rca_service._create_rca_progress_bar",
        lambda total, label="RCA": FakeProgressBar(total, label),
    )
    monkeypatch.setattr(
        RCAJudge,
        "judge",
        lambda self, evidence: {
            "root_cause_category": "RETRIEVAL_FAILURE",
            "confidence": 0.95,
            "evidence": ["The retriever returned irrelevant context."],
            "explanation": "Short RCA description about retrieval failure.",
            "recommended_action": "Inspect retriever configuration.",
        },
    )

    results = generate_rca(pd.DataFrame([{"trace_id": "unused"}]))

    assert [result["trace_id"] for result in results] == ["trace-1", "trace-1", "trace-2-abcdef123456"]
    assert events == [
        ("create", "RCA trace-1", 2),
        ("update", "RCA trace-1", 1),
        ("update", "RCA trace-1", 1),
        ("close", "RCA trace-1", None),
        ("create", "RCA trace-2-abcd...", 1),
        ("update", "RCA trace-2-abcd...", 1),
        ("close", "RCA trace-2-abcd...", None),
    ]
