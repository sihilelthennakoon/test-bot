from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GeminiAnswerer:
    model_name: str
    api_key: str | None = None

    def generate(self, question: str, context: str) -> str:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception:
            return self._fallback_answer(question, context)

        if not self.api_key:
            return self._fallback_answer(question, context)

        prompt = (
            "You are a careful enterprise RAG assistant.\n"
            "Answer only from the provided context.\n"
            "If the answer is not present, say you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        model = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key, temperature=0)
        response = model.invoke(prompt)
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        return self._fallback_answer(question, context)

    def _fallback_answer(self, question: str, context: str) -> str:
        summary = context.strip().splitlines()[0] if context.strip() else "no retrieval context was available"
        return (
            f"I can answer from the retrieved document context. Question: {question}. "
            f"Relevant context summary: {summary}"
        )
