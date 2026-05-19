from __future__ import annotations

from dataclasses import dataclass

from ragbot.schemas import EvaluationExample, EvaluationResult
from ragbot.service import ChatService


@dataclass(slots=True)
class Evaluator:
    service: ChatService

    def run(self, examples: list[EvaluationExample]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for example in examples:
            response = self.service.runtime.run(example.question)
            retrieved_ids = [hit.chunk_id for hit in response.sources]
            normalized_answer = response.answer.lower()
            contains_expected_terms = all(term.lower() in normalized_answer for term in example.expected_terms)
            result = EvaluationResult(
                question=example.question,
                answer=response.answer,
                retrieved_chunk_ids=retrieved_ids,
                contains_expected_terms=contains_expected_terms,
                trace_id=response.trace_id,
            )
            self.service.runtime.tracer.record_evaluation(result.model_dump())
            results.append(result)
        return results
