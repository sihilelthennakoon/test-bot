from __future__ import annotations

import asyncio
import json
from typing import Any

from ragbot.evaluations.results import EvaluationResult, TraceEvaluationResults
from ragbot.graph.chat_graph import ChatGraphApp, ChatRuntime
from ragbot.safety.guardrails import GuardrailEngine
from ragbot.safety.pii import PIIMasker
from ragbot.schemas import ChatResponse, RetrievalHit, SafetyDecision


class RecordingSpan:
    def __init__(self) -> None:
        self.name = ""
        self.attributes: dict[str, Any] = {}

    def __enter__(self) -> "RecordingSpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


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
    span = RecordingSpan()

    asyncio.run(
        runtime.evaluate_specific_step(
            span,
            "retrieve",
            evaluators=["routing", "safety"],
            input_text="hello",
        )
    )

    assert runner.enabled_seen_by_evaluate == [{"routing"}]
    assert "routing" in runner.get_enabled_evaluators()
    assert "safety" not in runner.get_enabled_evaluators()
    assert "ragbot.eval.retrieve.routing.score" in span.attributes
    assert "ragbot.eval.retrieve.safety.score" not in span.attributes


def test_evaluate_specific_step_restores_state_after_error() -> None:
    runner = FakeEvaluationRunner(fail=True)
    runner.disable_evaluator("safety")
    runtime = build_runtime(runner)
    span = RecordingSpan()

    asyncio.run(
        runtime.evaluate_specific_step(
            span,
            "retrieve",
            evaluators=["routing"],
            input_text="hello",
        )
    )

    assert "routing" in runner.get_enabled_evaluators()
    assert "safety" not in runner.get_enabled_evaluators()
    assert span.attributes["ragbot.eval.retrieve.error"] == "evaluation failed"


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    def start_as_current_span(self, name: str) -> RecordingSpan:
        span = RecordingSpan()
        span.name = name
        self.spans.append(span)
        return span


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


def test_run_populates_pipeline_trace_attributes(monkeypatch: Any) -> None:
    from ragbot.graph import chat_graph

    tracer = FakeTracer()
    monkeypatch.setattr(chat_graph.otel_trace, "get_tracer", lambda name: tracer)

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

    spans = {span.name: span for span in tracer.spans}
    request_io = json.loads(spans["chat.request"].attributes["ragbot.trace.io"])
    request_inputs = json.loads(spans["chat.request"].attributes["ragbot.trace.input_messages"])
    request_outputs = json.loads(spans["chat.request"].attributes["ragbot.trace.output_messages"])
    retrieve_inputs = json.loads(spans["chat.retrieve"].attributes["ragbot.node.retrieve.input_messages"])
    retrieve_outputs = json.loads(spans["chat.retrieve"].attributes["ragbot.node.retrieve.output_messages"])
    guardrail_output = json.loads(spans["chat.input_guardrails"].attributes["ragbot.node.input_guardrails.output"])
    generate_input = json.loads(spans["chat.generate"].attributes["ragbot.node.generate.input"])
    output_guardrails_output = json.loads(spans["chat.output_guardrails"].attributes["ragbot.node.output_guardrails.output"])

    assert response.answer == "You can reset your password from the account settings page."
    assert spans["chat.request"].attributes["kind"] == "chat"
    assert spans["chat.request"].attributes["ragbot.conversation_id"] == "conv_1"
    assert "[EMAIL_1]" in spans["chat.request"].attributes["ragbot.input.text"]
    assert "user@example.com" not in spans["chat.request"].attributes["ragbot.input.text"]
    assert spans["chat.request"].attributes["ragbot.trace.input"] == spans["chat.request"].attributes["ragbot.input.text"]
    assert spans["chat.request"].attributes["ragbot.trace.output"] == response.answer
    assert request_io["kind"] == "chat"
    assert request_io["input"]["role"] == "user"
    assert request_io["input"]["content"] == spans["chat.request"].attributes["ragbot.input.text"]
    assert request_io["output"]["role"] == "assistant"
    assert request_io["output"]["content"] == response.answer
    assert request_inputs[0]["role"] == "user"
    assert request_inputs[0]["content"] == spans["chat.request"].attributes["ragbot.input.text"]
    assert request_outputs[0]["content"] == response.answer
    assert spans["chat.request"].attributes["ragbot.eval.request.latency.score"] == 1.0

    assert spans["chat.input_guardrails"].attributes["kind"] == "chat"
    assert spans["chat.input_guardrails"].attributes["ragbot.node.name"] == "input_guardrails"
    assert json.loads(spans["chat.input_guardrails"].attributes["ragbot.node.input_guardrails.input_messages"])[0]["content"] == spans["chat.input_guardrails"].attributes["ragbot.node.input_guardrails.input"]
    assert guardrail_output["allowed"] is True
    assert guardrail_output["reason"] == "Input passed guardrails."
    assert spans["chat.input_guardrails"].attributes["ragbot.eval.input_guardrails.safety.score"] == 1.0

    assert spans["chat.retrieve"].attributes["kind"] == "chat"
    assert spans["chat.retrieve"].attributes["ragbot.node.name"] == "retrieve"
    assert "docs/example.txt" in spans["chat.retrieve"].attributes["ragbot.node.retrieve.output.sources"]
    assert json.loads(spans["chat.retrieve"].attributes["ragbot.node.retrieve.input_messages"])[0]["content"] == spans["chat.retrieve"].attributes["ragbot.node.retrieve.input"]
    assert retrieve_inputs[0]["role"] == "user"
    assert retrieve_inputs[0]["content"] == spans["chat.retrieve"].attributes["ragbot.node.retrieve.input"]
    assert retrieve_outputs[0]["role"] == "tool"
    assert spans["chat.retrieve"].attributes["ragbot.eval.retrieve.routing.passed"] is True

    assert spans["chat.generate"].attributes["kind"] == "chat"
    assert spans["chat.generate"].attributes["ragbot.node.name"] == "generate"
    assert "account settings" in spans["chat.generate"].attributes["ragbot.node.generate.output.answer"]
    assert generate_input["query"] == spans["chat.generate"].attributes["ragbot.node.generate.input.query"]
    assert json.loads(spans["chat.generate"].attributes["ragbot.node.generate.output_messages"])[0]["content"] == spans["chat.generate"].attributes["ragbot.node.generate.output"]
    assert spans["chat.generate"].attributes["ragbot.eval.generate.correctness.score"] == 1.0

    assert spans["chat.output_guardrails"].attributes["kind"] == "chat"
    assert spans["chat.output_guardrails"].attributes["ragbot.node.name"] == "output_guardrails"
    assert output_guardrails_output["allowed"] is True
    assert output_guardrails_output["reason"] == "Output passed guardrails."
    assert json.loads(spans["chat.output_guardrails"].attributes["ragbot.node.output_guardrails.input_messages"])[0]["content"] == spans["chat.output_guardrails"].attributes["ragbot.node.output_guardrails.input"]
    assert spans["chat.output_guardrails"].attributes["ragbot.eval.output_guardrails.safety.score"] == 1.0

    assert spans["chat.mask_output"].attributes["kind"] == "chat"
    assert spans["chat.mask_output"].attributes["ragbot.node.name"] == "mask_output"
    assert json.loads(spans["chat.mask_output"].attributes["ragbot.node.mask_output.output_messages"])[0]["content"] == spans["chat.mask_output"].attributes["ragbot.node.mask_output.output"]
    assert spans["chat.mask_output"].attributes["ragbot.node.mask_output.output"] == response.answer


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
