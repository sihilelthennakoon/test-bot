from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragbot.llm.gemini import GeminiAnswerer
from ragbot.observability.phoenix import TraceRecorder
from ragbot.safety.guardrails import GuardrailDecision, GuardrailEngine
from ragbot.safety.pii import PIIMasker
from ragbot.schemas import ChatResponse, RetrievalHit, SafetyDecision
from ragbot.vectorstore.faiss_store import FaissVectorStore


@dataclass(slots=True)
class ChatRuntime:
    store: FaissVectorStore
    answerer: GeminiAnswerer
    pii_masker: PIIMasker
    guardrails: GuardrailEngine
    tracer: TraceRecorder

    def run(self, message: str, *, top_k: int = 4, conversation_id: str | None = None) -> ChatResponse:
        with self.tracer.span("chat.request", conversation_id=conversation_id or "anonymous"):
            input_decision = self.guardrails.check_input(message)
            if not input_decision.allowed:
                safety = self._to_safety(input_decision)
                return ChatResponse(
                    answer=input_decision.reason,
                    sources=[],
                    input_safety=safety,
                    output_safety=safety,
                    conversation_id=conversation_id,
                )

            masked_input = self.pii_masker.mask(message)
            input_safety = SafetyDecision(
                allowed=True,
                reason=input_decision.reason,
                warnings=["PII was masked before retrieval."] if masked_input.was_masked else [],
                masked_text=masked_input.text,
            )

            sources = self.store.search(masked_input.text, top_k=top_k)
            context = self._build_context(sources)
            answer = self.answerer.generate(masked_input.text, context)
            output_decision = self.guardrails.check_output(answer, has_sources=bool(sources))
            masked_output = self.pii_masker.mask(answer)
            final_output = masked_output.text

            if not output_decision.allowed:
                final_output = output_decision.reason

            output_safety = SafetyDecision(
                allowed=output_decision.allowed,
                reason=output_decision.reason,
                warnings=output_decision.warnings + (["PII was masked in the answer."] if masked_output.was_masked else []),
                masked_text=final_output,
            )
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


class LinearChatApp:
    def __init__(self, runtime: ChatRuntime) -> None:
        self.runtime = runtime

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.runtime.run(
            payload["message"],
            top_k=payload.get("top_k", 4),
            conversation_id=payload.get("conversation_id"),
        )
        return response.model_dump()


def build_chat_app(runtime: ChatRuntime):
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return LinearChatApp(runtime)

    def input_guardrail_node(state: dict[str, Any]) -> dict[str, Any]:
        decision = runtime.guardrails.check_input(state["message"])
        state["input_decision"] = decision
        state["blocked"] = not decision.allowed
        return state

    def pii_mask_input_node(state: dict[str, Any]) -> dict[str, Any]:
        masked = runtime.pii_masker.mask(state["message"])
        state["message"] = masked.text
        state["input_masked"] = masked
        return state

    def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
        state["sources"] = runtime.store.search(state["message"], top_k=state.get("top_k", 4))
        return state

    def generate_node(state: dict[str, Any]) -> dict[str, Any]:
        context = runtime._build_context(state.get("sources", []))
        state["answer"] = runtime.answerer.generate(state["message"], context)
        return state

    def output_guardrail_node(state: dict[str, Any]) -> dict[str, Any]:
        state["output_decision"] = runtime.guardrails.check_output(state.get("answer", ""), has_sources=bool(state.get("sources")))
        return state

    def pii_mask_output_node(state: dict[str, Any]) -> dict[str, Any]:
        masked = runtime.pii_masker.mask(state.get("answer", ""))
        state["answer"] = masked.text
        state["output_masked"] = masked
        return state

    workflow = StateGraph(dict)
    workflow.add_node("input_guardrails", input_guardrail_node)
    workflow.add_node("mask_input", pii_mask_input_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("output_guardrails", output_guardrail_node)
    workflow.add_node("mask_output", pii_mask_output_node)

    workflow.set_entry_point("input_guardrails")
    workflow.add_conditional_edges(
        "input_guardrails",
        lambda state: "__end__" if state.get("blocked") else "mask_input",
        {"mask_input": "mask_input", "__end__": END},
    )
    workflow.add_edge("mask_input", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "output_guardrails")
    workflow.add_edge("output_guardrails", "mask_output")
    workflow.add_edge("mask_output", END)
    return workflow.compile()
