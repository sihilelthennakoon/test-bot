from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING, Any

from opentelemetry import trace as otel_trace
import asyncio
import time

from ragbot.llm.gemini import GeminiAnswerer
from ragbot.safety.guardrails import GuardrailDecision, GuardrailEngine
from ragbot.safety.pii import PIIMasker
from ragbot.schemas import ChatResponse, RetrievalHit, SafetyDecision
from ragbot.vectorstore.faiss_store import FaissVectorStore

if TYPE_CHECKING:
    from ragbot.evaluations.runner import EvaluationRunner


MAX_SPAN_TEXT_LENGTH = 4000
CHAT_KIND = "chat"


class _NullSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None


@dataclass(slots=True)
class ChatGraphApp:
    runtime: ChatRuntime
    graph: Any

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        state = {
            "message": payload["message"],
            "top_k": payload.get("top_k", 4),
            "conversation_id": payload.get("conversation_id"),
        }
        result = self.graph.invoke(state, config=config)
        response = result.get("response")
        if isinstance(response, ChatResponse):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        return result


@dataclass(slots=True)
class ChatRuntime:
    store: FaissVectorStore
    answerer: GeminiAnswerer
    pii_masker: PIIMasker
    guardrails: GuardrailEngine
    evaluator_runner: Any = None  # Optional EvaluationRunner for evaluations
    _evaluation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def run(self, message: str, *, top_k: int = 4, conversation_id: str | None = None) -> ChatResponse:
        """Execute RAG pipeline."""
        tracer = otel_trace.get_tracer(__name__)
        started_at = time.perf_counter()
        with tracer.start_as_current_span("chat.request") as span:
            self._set_chat_kind(span)
            span.set_attribute("ragbot.top_k", top_k)
            span.set_attribute("ragbot.input.length", len(message))
            if conversation_id:
                span.set_attribute("ragbot.conversation_id", conversation_id)
            span.set_attribute("ragbot.message_length", len(message))

            with tracer.start_as_current_span("chat.input_guardrails") as guardrail_span:
                input_decision = self.guardrails.check_input(message)
                self._set_chat_kind(guardrail_span)
                masked_message = self.pii_masker.mask(message).text
                self._set_text_attribute(
                    guardrail_span,
                    "ragbot.node.input_guardrails.input",
                    masked_message,
                )
                guardrail_span.set_attribute("ragbot.allowed", input_decision.allowed)
                guardrail_span.set_attribute("ragbot.reason", input_decision.reason)
                guardrail_span.set_attribute("ragbot.node.name", "input_guardrails")
                guardrail_span.set_attribute("ragbot.node.input_guardrails.output.allowed", input_decision.allowed)
                guardrail_span.set_attribute("ragbot.node.input_guardrails.output.reason", input_decision.reason)
                self._set_json_attribute(
                    guardrail_span,
                    "ragbot.node.input_guardrails.output.warnings",
                    input_decision.warnings,
                )
                self._set_chat_trace_fields(
                    guardrail_span,
                    prefix="ragbot.node.input_guardrails",
                    input_value=masked_message,
                    output_value={
                        "allowed": input_decision.allowed,
                        "reason": input_decision.reason,
                        "warnings": input_decision.warnings,
                    },
                    output_role="system",
                )
                if self.evaluator_runner:
                    await self.evaluate_specific_step(
                        guardrail_span,
                        "input_guardrails",
                        evaluators=["safety"],
                        response_text=masked_message,
                    )
            if not input_decision.allowed:
                safety = self._to_safety(input_decision)
                span.set_attribute("ragbot.allowed", False)
                span.set_attribute("ragbot.block_reason", input_decision.reason)
                self._set_chat_trace_fields(
                    span,
                    prefix="ragbot.trace",
                    input_value=self.pii_masker.mask(message).text,
                    output_value=input_decision.reason,
                    output_role="system",
                )
                self._set_text_attribute(span, "ragbot.output.answer", input_decision.reason)
                span.set_attribute("ragbot.output_safety.allowed", safety.allowed)
                span.set_attribute("ragbot.output_safety.reason", safety.reason)
                return ChatResponse(
                    answer=input_decision.reason,
                    sources=[],
                    input_safety=safety,
                    output_safety=safety,
                    conversation_id=conversation_id,
                )

            with tracer.start_as_current_span("chat.mask_input") as mask_span:
                self._set_chat_kind(mask_span)
                mask_span.set_attribute("ragbot.node.name", "mask_input")
                self._set_text_attribute(
                    mask_span,
                    "ragbot.node.mask_input.input",
                    self.pii_masker.mask(message).text,
                )
                masked_input = self.pii_masker.mask(message)
                mask_span.set_attribute("ragbot.was_masked", masked_input.was_masked)
                mask_span.set_attribute("ragbot.entities_count", len(masked_input.entities))
                self._set_text_attribute(mask_span, "ragbot.node.mask_input.output", masked_input.text)
                self._set_json_attribute(mask_span, "ragbot.node.mask_input.entities", masked_input.entities)
                self._set_chat_trace_fields(
                    mask_span,
                    prefix="ragbot.node.mask_input",
                    input_value=self.pii_masker.mask(message).text,
                    output_value=masked_input.text,
                )
            input_safety = SafetyDecision(
                allowed=True,
                reason=input_decision.reason,
                warnings=["PII was masked before retrieval."] if masked_input.was_masked else [],
                masked_text=masked_input.text,
            )
            self._set_chat_trace_fields(
                span,
                prefix="ragbot.trace",
                input_value=masked_input.text,
            )
            self._set_text_attribute(span, "ragbot.input.text", masked_input.text)
            span.set_attribute("ragbot.input_safety.allowed", input_safety.allowed)
            span.set_attribute("ragbot.input_safety.reason", input_safety.reason)
            self._set_json_attribute(span, "ragbot.input_safety.warnings", input_safety.warnings)

            with tracer.start_as_current_span("chat.retrieve") as retrieve_span:
                self._set_chat_kind(retrieve_span)
                retrieve_span.set_attribute("ragbot.node.name", "retrieve")
                self._set_text_attribute(retrieve_span, "ragbot.node.retrieve.input", masked_input.text)
                sources = self.store.search(masked_input.text, top_k=top_k)
                retrieve_span.set_attribute("ragbot.source_count", len(sources))
                retrieve_span.set_attribute("ragbot.query_length", len(masked_input.text))
                if sources:
                    retrieve_span.set_attribute("ragbot.top_score", float(sources[0].score))
                self._set_json_attribute(
                    retrieve_span,
                    "ragbot.node.retrieve.output.sources",
                    self._source_summaries(sources),
                )
                self._set_json_attribute(
                    retrieve_span,
                    "ragbot.node.retrieve.output.scores",
                    [float(hit.score) for hit in sources],
                )
                self._set_chat_trace_fields(
                    retrieve_span,
                    prefix="ragbot.node.retrieve",
                    input_value=masked_input.text,
                    output_value=self._source_summaries(sources),
                    output_role="tool",
                )
                
                # Evaluate routing (retriever quality)
                if self.evaluator_runner:
                    await self.evaluate_specific_step(
                        retrieve_span,
                        "retrieve",
                        evaluators=["routing"],
                        input_text=masked_input.text,
                        retrieved_docs=sources,
                        scores=[float(hit.score) for hit in sources],
                    )
            
            context = self._build_context(sources)
            span.set_attribute("ragbot.source_count", len(sources))
            self._set_json_attribute(span, "ragbot.sources", self._source_summaries(sources))

            with tracer.start_as_current_span("chat.generate") as gen_span:
                self._set_chat_kind(gen_span)
                gen_span.set_attribute("ragbot.node.name", "generate")
                self._set_text_attribute(gen_span, "ragbot.node.generate.input.query", masked_input.text)
                self._set_text_attribute(gen_span, "ragbot.node.generate.input.context", context)
                answer = self.answerer.generate(masked_input.text, context)
                gen_span.set_attribute("ragbot.context_length", len(context))
                gen_span.set_attribute("ragbot.answer_length", len(answer))
                self._set_text_attribute(
                    gen_span,
                    "ragbot.node.generate.output.answer",
                    self.pii_masker.mask(answer).text,
                )
                self._set_chat_trace_fields(
                    gen_span,
                    prefix="ragbot.node.generate",
                    input_value={
                        "query": masked_input.text,
                        "context": context,
                    },
                    output_value=answer,
                )
                
                # Evaluate generation quality (correctness, relevance, groundedness, format)
                if self.evaluator_runner:
                    await self.evaluate_specific_step(
                        gen_span,
                        "generate",
                        evaluators=["correctness", "relevance", "groundedness", "format"],
                        input_text=masked_input.text,
                        response_text=answer,
                        context=context,
                        reference=context,
                    )

            with tracer.start_as_current_span("chat.output_guardrails") as out_guardrail_span:
                self._set_chat_kind(out_guardrail_span)
                out_guardrail_span.set_attribute("ragbot.node.name", "output_guardrails")
                self._set_text_attribute(
                    out_guardrail_span,
                    "ragbot.node.output_guardrails.input",
                    self.pii_masker.mask(answer).text,
                )
                output_decision = self.guardrails.check_output(answer, has_sources=bool(sources))
                out_guardrail_span.set_attribute("ragbot.allowed", output_decision.allowed)
                out_guardrail_span.set_attribute("ragbot.reason", output_decision.reason)
                out_guardrail_span.set_attribute("ragbot.node.output_guardrails.output.allowed", output_decision.allowed)
                out_guardrail_span.set_attribute("ragbot.node.output_guardrails.output.reason", output_decision.reason)
                self._set_json_attribute(
                    out_guardrail_span,
                    "ragbot.node.output_guardrails.output.warnings",
                    output_decision.warnings,
                )
                self._set_chat_trace_fields(
                    out_guardrail_span,
                    prefix="ragbot.node.output_guardrails",
                    input_value=self.pii_masker.mask(answer).text,
                    output_value={
                        "allowed": output_decision.allowed,
                        "reason": output_decision.reason,
                        "warnings": output_decision.warnings,
                    },
                    output_role="system",
                )
                
                # Evaluate safety of output
                if self.evaluator_runner:
                    await self.evaluate_specific_step(
                        out_guardrail_span,
                        "output_guardrails",
                        evaluators=["safety"],
                        response_text=answer,
                    )

            with tracer.start_as_current_span("chat.mask_output") as out_mask_span:
                self._set_chat_kind(out_mask_span)
                out_mask_span.set_attribute("ragbot.node.name", "mask_output")
                self._set_text_attribute(out_mask_span, "ragbot.node.mask_output.input", answer)
                masked_output = self.pii_masker.mask(answer)
                out_mask_span.set_attribute("ragbot.was_masked", masked_output.was_masked)
                out_mask_span.set_attribute("ragbot.entities_count", len(masked_output.entities))
                self._set_text_attribute(out_mask_span, "ragbot.node.mask_output.output", masked_output.text)
                self._set_json_attribute(out_mask_span, "ragbot.node.mask_output.entities", masked_output.entities)
                self._set_chat_trace_fields(
                    out_mask_span,
                    prefix="ragbot.node.mask_output",
                    input_value=answer,
                    output_value=masked_output.text,
                )
            final_output = masked_output.text

            if not output_decision.allowed:
                final_output = output_decision.reason

            output_safety = SafetyDecision(
                allowed=output_decision.allowed,
                reason=output_decision.reason,
                warnings=output_decision.warnings + (["PII was masked in the answer."] if masked_output.was_masked else []),
                masked_text=final_output,
            )

            span.set_attribute("ragbot.allowed", output_decision.allowed)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            span.set_attribute("ragbot.duration_ms", int(elapsed_ms))
            self._set_chat_trace_fields(
                span,
                prefix="ragbot.trace",
                input_value=masked_input.text,
                output_value=final_output,
                output_role="assistant" if output_decision.allowed else "system",
            )
            self._set_text_attribute(span, "ragbot.output.answer", final_output)
            span.set_attribute("ragbot.output_safety.allowed", output_safety.allowed)
            span.set_attribute("ragbot.output_safety.reason", output_safety.reason)
            self._set_json_attribute(span, "ragbot.output_safety.warnings", output_safety.warnings)

            # Evaluate overall latency on main request span
            if self.evaluator_runner:
                await self.evaluate_specific_step(
                    span,
                    "request",
                    evaluators=["latency"],
                    input_text=masked_input.text,
                    response_text=final_output,
                    duration_ms=elapsed_ms,
                )

            return ChatResponse(
                answer=final_output,
                sources=sources,
                input_safety=input_safety,
                output_safety=output_safety,
                conversation_id=conversation_id,
            )

    def build_response(
        self,
        *,
        message: str,
        conversation_id: str | None,
        sources: list[RetrievalHit],
        input_decision: GuardrailDecision,
        output_decision: GuardrailDecision,
        masked_input_text: str,
        answer: str,
        masked_output_text: str,
    ) -> ChatResponse:
        input_safety = SafetyDecision(
            allowed=input_decision.allowed,
            reason=input_decision.reason,
            warnings=input_decision.warnings + (["PII was masked before retrieval."] if masked_input_text != message else []),
            masked_text=masked_input_text,
        )
        output_safety = SafetyDecision(
            allowed=output_decision.allowed,
            reason=output_decision.reason,
            warnings=output_decision.warnings + (["PII was masked in the answer."] if masked_output_text != answer else []),
            masked_text=masked_output_text if output_decision.allowed else output_decision.reason,
        )
        final_output = masked_output_text if output_decision.allowed else output_decision.reason
        return ChatResponse(
            answer=final_output,
            sources=sources,
            input_safety=input_safety,
            output_safety=output_safety,
            conversation_id=conversation_id,
        )

    def _build_context(self, sources: list[RetrievalHit]) -> str:
        if not sources:
            return ""
        parts = []
        for hit in sources:
            parts.append(f"Source: {hit.source_path} | Chunk: {hit.chunk_index}\n{hit.text}")
        return "\n\n".join(parts)

    def _to_safety(self, decision: GuardrailDecision) -> SafetyDecision:
        return SafetyDecision(allowed=decision.allowed, reason=decision.reason, warnings=decision.warnings)

    def _source_summaries(self, sources: list[RetrievalHit]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": hit.chunk_id,
                "source_path": hit.source_path,
                "chunk_index": hit.chunk_index,
                "score": float(hit.score),
                "text": self._truncate_text(hit.text),
            }
            for hit in sources
        ]

    def _truncate_text(self, text: str, max_length: int = MAX_SPAN_TEXT_LENGTH) -> str:
        if len(text) <= max_length:
            return text
        return f"{text[:max_length]}... [truncated {len(text) - max_length} chars]"

    def _set_chat_kind(self, span: Any) -> None:
        span.set_attribute("kind", CHAT_KIND)

    def _set_chat_trace_fields(
        self,
        span: Any,
        *,
        prefix: str,
        input_value: str | dict[str, Any],
        output_value: str | dict[str, Any] | None = None,
        input_role: str = "user",
        output_role: str = "assistant",
    ) -> None:
        serialized_input = self._serialize_trace_value(input_value)
        self._set_text_attribute(span, f"{prefix}.input", serialized_input)
        self._set_json_attribute(
            span,
            f"{prefix}.input_messages",
            [{"role": input_role, "content": input_value}],
        )
        if output_value is not None:
            serialized_output = self._serialize_trace_value(output_value)
            self._set_text_attribute(span, f"{prefix}.output", serialized_output)
            self._set_json_attribute(
                span,
                f"{prefix}.output_messages",
                [{"role": output_role, "content": output_value}],
            )
        else:
            serialized_output = None

        self._set_json_attribute(
            span,
            f"{prefix}.io",
            {
                "kind": CHAT_KIND,
                "input": {
                    "role": input_role,
                    "content": input_value,
                },
                "output": None if output_value is None else {
                    "role": output_role,
                    "content": output_value,
                },
            },
        )

    def _serialize_trace_value(self, value: str | dict[str, Any]) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str, ensure_ascii=True)

    def _set_text_attribute(self, span: Any, key: str, value: str | None) -> None:
        span.set_attribute(key, self._truncate_text(value or ""))

    def _set_json_attribute(self, span: Any, key: str, value: Any) -> None:
        try:
            serialized = json.dumps(value, default=str, ensure_ascii=True)
        except TypeError:
            serialized = json.dumps(str(value), ensure_ascii=True)
        self._set_text_attribute(span, key, serialized)

    async def evaluate_specific_step(
        self,
        span: Any,
        step_name: str,
        evaluators: list[str],
        input_text: str = "",
        response_text: str = "",
        **kwargs: Any
    ) -> None:
        """Run specific evaluators and log results to span attributes.
        
        Args:
            span: OpenTelemetry span to log attributes to
            step_name: Name of evaluation step
            evaluators: List of evaluator names to run
            input_text: User input or context
            response_text: Response or output being evaluated
            **kwargs: Additional context for evaluators
        """
        if not self.evaluator_runner:
            return

        async with self._evaluation_lock:
            evaluator_names = set(evaluators)
            all_evaluators = set(getattr(self.evaluator_runner, "evaluators", {}).keys())
            enabled_before = set(self.evaluator_runner.get_enabled_evaluators())
            enabled_for_step = enabled_before & evaluator_names
            if all_evaluators:
                enabled_for_step &= all_evaluators

            if not enabled_for_step:
                span.set_attribute(f"ragbot.eval.{step_name}.skipped", True)
                span.set_attribute(f"ragbot.eval.{step_name}.skip_reason", "No requested evaluators are enabled.")
                return

            self._set_enabled_evaluators(enabled_for_step)

            try:
                # Run evaluations
                result = await self.evaluator_runner.evaluate(
                    input_text=input_text,
                    response_text=response_text,
                    **kwargs
                )

                # Log evaluation results to span attributes
                span.set_attribute(f"ragbot.eval.{step_name}.trace_id", result.trace_id)
                span.set_attribute(f"ragbot.eval.{step_name}.pass_rate", result.pass_rate())
                span.set_attribute(f"ragbot.eval.{step_name}.evaluator_count", len(result.evaluations))
                self._set_json_attribute(
                    span,
                    f"ragbot.eval.{step_name}.evaluators",
                    sorted(result.evaluations.keys()),
                )
                for evaluator_name, eval_result in result.evaluations.items():
                    span.set_attribute(
                        f"ragbot.eval.{step_name}.{evaluator_name}.score",
                        eval_result.score
                    )
                    span.set_attribute(
                        f"ragbot.eval.{step_name}.{evaluator_name}.passed",
                        eval_result.passed
                    )
                    span.set_attribute(
                        f"ragbot.eval.{step_name}.{evaluator_name}.reason",
                        eval_result.reason,
                    )
                    self._set_json_attribute(
                        span,
                        f"ragbot.eval.{step_name}.{evaluator_name}.metadata",
                        eval_result.metadata,
                    )

                # Log aggregate score for this step
                span.set_attribute(
                    f"ragbot.eval.{step_name}.aggregate_score",
                    result.aggregate_score()
                )
            except Exception as e:
                span.set_attribute(f"ragbot.eval.{step_name}.error", str(e))
            finally:
                self._set_enabled_evaluators(enabled_before)

    def _set_enabled_evaluators(self, enabled: set[str]) -> None:
        all_evaluators = set(getattr(self.evaluator_runner, "evaluators", {}).keys())
        current_enabled = set(self.evaluator_runner.get_enabled_evaluators())

        for name in current_enabled - enabled:
            self.evaluator_runner.disable_evaluator(name)
        for name in enabled - current_enabled:
            if not all_evaluators or name in all_evaluators:
                self.evaluator_runner.enable_evaluator(name)

    async def evaluate_step(
        self,
        span: Any,
        step_name: str,
        input_text: str = "",
        response_text: str = "",
        **kwargs: Any
    ) -> None:
        """Run evaluators and log results to span attributes.
        
        Args:
            span: OpenTelemetry span to log attributes to
            step_name: Name of evaluation step
            input_text: User input or context
            response_text: Response or output being evaluated
            **kwargs: Additional context for evaluators
        """
        if not self.evaluator_runner:
            return

        try:
            result = await self.evaluator_runner.evaluate(
                input_text=input_text,
                response_text=response_text,
                **kwargs
            )

            # Log evaluation results to span attributes
            for evaluator_name, eval_result in result.evaluations.items():
                span.set_attribute(
                    f"ragbot.eval.{step_name}.{evaluator_name}.score",
                    eval_result.score
                )
                span.set_attribute(
                    f"ragbot.eval.{step_name}.{evaluator_name}.passed",
                    eval_result.passed
                )

            # Log aggregate score
            span.set_attribute(
                f"ragbot.eval.{step_name}.aggregate_score",
                result.aggregate_score()
            )
        except Exception as e:
            # Log evaluation errors but don't fail the main request
            span.set_attribute(f"ragbot.eval.{step_name}.error", str(e))


class LinearChatApp:
    def __init__(self, runtime: ChatRuntime) -> None:
        self.runtime = runtime

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        response = asyncio.run(
            self.runtime.run(
                payload["message"],
                top_k=payload.get("top_k", 4),
                conversation_id=payload.get("conversation_id"),
            )
        )
        return response.model_dump()


def build_chat_app(runtime: ChatRuntime, *, settings: Any | None = None):
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return LinearChatApp(runtime)

    def input_guardrail_node(state: dict[str, Any]) -> dict[str, Any]:
        decision = runtime.guardrails.check_input(state["message"])
        state["input_decision"] = decision
        state["blocked"] = not decision.allowed
        if state["blocked"]:
            state["response"] = runtime.build_response(
                message=state["message"],
                conversation_id=state.get("conversation_id"),
                sources=[],
                input_decision=decision,
                output_decision=GuardrailDecision(allowed=False, reason=decision.reason, warnings=[]),
                masked_input_text=runtime.pii_masker.mask(state["message"]).text,
                answer=decision.reason,
                masked_output_text=decision.reason,
            )
        return state

    def pii_mask_input_node(state: dict[str, Any]) -> dict[str, Any]:
        masked = runtime.pii_masker.mask(state["message"])
        state["message"] = masked.text
        state["input_masked"] = masked
        state["masked_input_text"] = masked.text
        return state

    def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
        sources = runtime.store.search(state["message"], top_k=state.get("top_k", 4))
        state["sources"] = sources
        state["scores"] = [float(s.score) for s in sources]
        state["context"] = runtime._build_context(sources)
        return state

    def generate_node(state: dict[str, Any]) -> dict[str, Any]:
        context = state.get("context", runtime._build_context(state.get("sources", [])))
        answer = runtime.answerer.generate(state["message"], context)
        state["answer"] = answer
        if runtime.evaluator_runner:
            asyncio.run(
                runtime.evaluate_specific_step(
                    span=_NullSpan(),
                    step_name="generate",
                    evaluators=["correctness", "relevance", "groundedness", "format"],
                    input_text=state["message"],
                    response_text=answer,
                    context=context,
                    reference=context,
                )
            )
        return state

    def output_guardrail_node(state: dict[str, Any]) -> dict[str, Any]:
        state["output_decision"] = runtime.guardrails.check_output(state.get("answer", ""), has_sources=bool(state.get("sources")))
        if runtime.evaluator_runner:
            asyncio.run(
                runtime.evaluate_specific_step(
                    span=_NullSpan(),
                    step_name="output_guardrails",
                    evaluators=["safety"],
                    response_text=state.get("answer", ""),
                )
            )
        return state

    def pii_mask_output_node(state: dict[str, Any]) -> dict[str, Any]:
        masked = runtime.pii_masker.mask(state.get("answer", ""))
        state["answer"] = masked.text
        state["output_masked"] = masked
        state["masked_output_text"] = masked.text
        return state

    def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("response") is not None:
            return state

        input_decision = state.get("input_decision")
        output_decision = state.get("output_decision")
        if input_decision is None:
            input_decision = GuardrailDecision(allowed=True, reason="Input passed guardrails.", warnings=[])
        if output_decision is None:
            output_decision = GuardrailDecision(allowed=True, reason="Output passed guardrails.", warnings=[])

        state["response"] = runtime.build_response(
            message=state["message"],
            conversation_id=state.get("conversation_id"),
            sources=state.get("sources", []),
            input_decision=input_decision,
            output_decision=output_decision,
            masked_input_text=state.get("masked_input_text", state["message"]),
            answer=state.get("answer", ""),
            masked_output_text=state.get("masked_output_text", state.get("answer", "")),
        )
        return state

    workflow = StateGraph(dict)
    workflow.add_node("input_guardrails", input_guardrail_node)
    workflow.add_node("mask_input", pii_mask_input_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("output_guardrails", output_guardrail_node)
    workflow.add_node("mask_output", pii_mask_output_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("input_guardrails")
    workflow.add_conditional_edges(
        "input_guardrails",
        lambda state: "finalize" if state.get("blocked") else "mask_input",
        {"mask_input": "mask_input", "finalize": "finalize", "__end__": END},
    )
    workflow.add_edge("mask_input", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "output_guardrails")
    workflow.add_edge("output_guardrails", "mask_output")
    workflow.add_edge("mask_output", "finalize")
    workflow.add_edge("finalize", END)
    return ChatGraphApp(runtime=runtime, graph=workflow.compile())
