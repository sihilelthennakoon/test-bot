from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import time
from typing import TYPE_CHECKING, Any

from ragbot.llm.gemini import GeminiAnswerer
from ragbot.safety.guardrails import GuardrailDecision, GuardrailEngine
from ragbot.safety.pii import PIIMasker
from ragbot.schemas import ChatResponse, RetrievalHit, SafetyDecision
from ragbot.vectorstore.faiss_store import FaissVectorStore

if TYPE_CHECKING:
    from ragbot.evaluations.runner import EvaluationRunner


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
    evaluator_runner: Any = None
    _evaluation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def run(self, message: str, *, top_k: int = 4, conversation_id: str | None = None) -> ChatResponse:
        started_at = time.perf_counter()
        input_decision = self.guardrails.check_input(message)
        masked_message = self.pii_masker.mask(message)

        if not input_decision.allowed:
            safety = self._to_safety(input_decision)
            return ChatResponse(
                answer=input_decision.reason,
                sources=[],
                input_safety=safety,
                output_safety=safety,
                conversation_id=conversation_id,
            )

        sources = self.store.search(masked_message.text, top_k=top_k)
        context = self._build_context(sources)
        answer = self.answerer.generate(masked_message.text, context)
        output_decision = self.guardrails.check_output(answer, has_sources=bool(sources))
        masked_output = self.pii_masker.mask(answer)

        final_output = masked_output.text if output_decision.allowed else output_decision.reason
        input_safety = SafetyDecision(
            allowed=True,
            reason=input_decision.reason,
            warnings=["PII was masked before retrieval."] if masked_message.was_masked else [],
            masked_text=masked_message.text,
        )
        output_safety = SafetyDecision(
            allowed=output_decision.allowed,
            reason=output_decision.reason,
            warnings=output_decision.warnings + (["PII was masked in the answer."] if masked_output.was_masked else []),
            masked_text=final_output,
        )

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if self.evaluator_runner:
            await self.evaluate_specific_step(
                step_name="request",
                evaluators=["latency"],
                input_text=masked_message.text,
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

    async def evaluate_specific_step(
        self,
        step_name: str,
        evaluators: list[str],
        input_text: str = "",
        response_text: str = "",
        **kwargs: Any,
    ) -> Any | None:
        if not self.evaluator_runner:
            return None

        async with self._evaluation_lock:
            evaluator_names = set(evaluators)
            all_evaluators = set(getattr(self.evaluator_runner, "evaluators", {}).keys())
            enabled_before = set(self.evaluator_runner.get_enabled_evaluators())
            enabled_for_step = enabled_before & evaluator_names
            if all_evaluators:
                enabled_for_step &= all_evaluators

            if not enabled_for_step:
                return None

            self._set_enabled_evaluators(enabled_for_step)

            try:
                return await self.evaluator_runner.evaluate(
                    input_text=input_text,
                    response_text=response_text,
                    **kwargs,
                )
            except Exception as exc:
                return exc
            finally:
                self._set_enabled_evaluators(enabled_before)

    async def evaluate_step(
        self,
        step_name: str,
        input_text: str = "",
        response_text: str = "",
        **kwargs: Any,
    ) -> Any | None:
        return await self.evaluate_specific_step(
            step_name=step_name,
            evaluators=list(getattr(self.evaluator_runner, "evaluators", {}).keys()),
            input_text=input_text,
            response_text=response_text,
            **kwargs,
        )

    def _set_enabled_evaluators(self, enabled: set[str]) -> None:
        all_evaluators = set(getattr(self.evaluator_runner, "evaluators", {}).keys())
        current_enabled = set(self.evaluator_runner.get_enabled_evaluators())

        for name in current_enabled - enabled:
            self.evaluator_runner.disable_evaluator(name)
        for name in enabled - current_enabled:
            if not all_evaluators or name in all_evaluators:
                self.evaluator_runner.enable_evaluator(name)


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
