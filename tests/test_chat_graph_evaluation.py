from __future__ import annotations

import asyncio
from typing import Any

from ragbot.evaluations.results import EvaluationResult, TraceEvaluationResults
from ragbot.graph.chat_graph import ChatGraphApp, ChatRuntime
from ragbot.safety.guardrails import GuardrailEngine
from ragbot.safety.pii import PIIMasker
from ragbot.schemas import ChatResponse, RetrievalHit, SafetyDecision


class FakeEvaluationRunner:
    def __init__(self, *, fail: bool = False) -> None:
        self.evaluators = {
            "correctness": object(),
            "format": object(),
            "groundedness": object(),
            "latency": object(),
            "relevance": object(),
            "routing": object(),
            "safety": object(),
        }
        self._enabled = set(self.evaluators)
        self.fail = fail
        self.enabled_seen_by_evaluate: list[set[str]] = []

    def enable_evaluator(self, name: str) -> None:
        if name in self.evaluators:
            self._enabled.add(name)

    def disable_evaluator(self, name: str) -> None:
        self._enabled.discard(name)

    def get_enabled_evaluators(self) -> list[str]:
        return sorted(self._enabled)

    async def evaluate(self, **kwargs: Any) -> TraceEvaluationResults:
        self.enabled_seen_by_evaluate.append(set(self._enabled))
        if self.fail:
            raise RuntimeError("evaluation failed")

        evaluations = {
            name: EvaluationResult(
                evaluator_name=name,
                score=1.0,
                passed=True,
                reason="ok",
            )
            for name in self.get_enabled_evaluators()
        }
        return TraceEvaluationResults(
            trace_id="trace_test",
            input_text=kwargs.get("input_text", ""),
            response_text=kwargs.get("response_text", ""),
            evaluations=evaluations,
        )


def build_runtime(runner: FakeEvaluationRunner) -> ChatRuntime:
    return ChatRuntime(
        store=object(),
        answerer=object(),
        pii_masker=object(),
        guardrails=object(),
        evaluator_runner=runner,
    )


def test_evaluate_specific_step_preserves_disabled_evaluators() -> None:
    runner = FakeEvaluationRunner()
    runner.disable_evaluator("safety")
    runtime = build_runtime(runner)

    result = asyncio.run(
        runtime.evaluate_specific_step(
            "retrieve",
            evaluators=["routing", "safety"],
            input_text="hello",
        )
    )

    assert runner.enabled_seen_by_evaluate == [{"routing"}]
    assert "routing" in runner.get_enabled_evaluators()
    assert "safety" not in runner.get_enabled_evaluators()
    assert result is not None
    assert result.trace_id == "trace_test"
    assert set(result.evaluations) == {"routing"}


def test_evaluate_specific_step_restores_state_after_error() -> None:
    runner = FakeEvaluationRunner(fail=True)
    runner.disable_evaluator("safety")
    runtime = build_runtime(runner)

    result = asyncio.run(
        runtime.evaluate_specific_step(
            "retrieve",
            evaluators=["routing"],
            input_text="hello",
        )
    )

    assert "routing" in runner.get_enabled_evaluators()
    assert "safety" not in runner.get_enabled_evaluators()
    assert isinstance(result, RuntimeError)
    assert str(result) == "evaluation failed"


class FakeStore:
    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id="chunk_1",
                source_path="docs/example.txt",
                text="Reset your password from the account settings page.",
                score=0.91,
                chunk_index=0,
            )
        ]


class FakeAnswerer:
    def generate(self, question: str, context: str) -> str:
        return "You can reset your password from the account settings page."


def test_run_populates_pipeline_response_attributes() -> None:
    runtime = ChatRuntime(
        store=FakeStore(),
        answerer=FakeAnswerer(),
        pii_masker=PIIMasker(),
        guardrails=GuardrailEngine(),
        evaluator_runner=FakeEvaluationRunner(),
    )

    response = asyncio.run(
        runtime.run(
            "How do I reset my password if my email is user@example.com?",
            conversation_id="conv_1",
        )
    )

    assert response.answer == "You can reset your password from the account settings page."
    assert response.conversation_id == "conv_1"
    assert response.input_safety.allowed is True
    assert response.output_safety.allowed is True
    assert response.input_safety.warnings == ["PII was masked before retrieval."]
    assert response.output_safety.warnings == []
    assert response.sources and response.sources[0].source_path == "docs/example.txt"


def test_chat_graph_app_forwards_config_and_wraps_response() -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

        def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
            self.calls.append((state, config))
            return {
                "response": ChatResponse(
                    answer="ok",
                    sources=[],
                    input_safety=SafetyDecision(allowed=True, reason="input ok"),
                    output_safety=SafetyDecision(allowed=True, reason="output ok"),
                    conversation_id=state.get("conversation_id"),
                )
            }

    runtime = ChatRuntime(
        store=FakeStore(),
        answerer=FakeAnswerer(),
        pii_masker=PIIMasker(),
        guardrails=GuardrailEngine(),
        evaluator_runner=FakeEvaluationRunner(),
    )
    fake_graph = FakeGraph()
    app = ChatGraphApp(runtime=runtime, graph=fake_graph)
    config = {
        "metadata": {
            "conversation_id": "conv_99",
            "environment": "dev",
            "app_version": "v1",
            "use_case": "rag_chatbot",
        },
        "tags": [],
    }

    result = app.invoke({"message": "hello", "top_k": 2, "conversation_id": "conv_99"}, config=config)

    assert fake_graph.calls == [
        (
            {"message": "hello", "top_k": 2, "conversation_id": "conv_99"},
            config,
        )
    ]
    assert result["answer"] == "ok"
    assert result["conversation_id"] == "conv_99"
