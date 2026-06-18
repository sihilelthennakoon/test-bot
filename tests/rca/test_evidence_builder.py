from __future__ import annotations

import pandas as pd

from ragbot.rca.evidence_builder import build_evidence_packages


def _base_row(**overrides):
    row = {
        "span_id": "span-1",
        "context_trace_id": "trace-1",
        "question": "What is AuroraLamp?",
        "answer": "AuroraLamp is a smart bedside lamp.",
        "correctness_score": 0.1,
        "relevance_score": 0.2,
        "faithfulness_score": 0.3,
        "safety_score": 1.0,
        "retrieved_doc_count": 1,
        "retrieval_scores": "[0.8]",
        "retrieved_context": "AuroraLamp is a smart bedside lamp.",
        "context_has_answer": True,
        "answer_supported_by_context": True,
        "pii_detected": False,
        "span_status": "OK",
        "latency_ms": 10.0,
        "langgraph_node": "generate",
    }
    row.update(overrides)
    return row


def test_evidence_builder_uses_finalize_scores_only():
    df = pd.DataFrame(
        [
            _base_row(
                span_id="span-generate",
                langgraph_node="generate",
                correctness_score=0.1,
                relevance_score=0.2,
                faithfulness_score=0.3,
                safety_score=0.4,
            ),
            _base_row(
                span_id="span-finalize",
                langgraph_node="finalize",
                correctness_score=1.0,
                relevance_score=0.9,
                faithfulness_score=0.8,
                safety_score=0.7,
            ),
        ]
    )

    [evidence] = build_evidence_packages(df)

    assert evidence["scores"] == {
        "correctness": 1.0,
        "relevance": 0.9,
        "faithfulness": 0.8,
        "safety": 0.7,
    }


def test_evidence_builder_normalizes_finalized_node_name():
    df = pd.DataFrame(
        [
            _base_row(
                span_id="span-finalized",
                langgraph_node="Finalized",
                correctness_score=0.95,
                relevance_score=0.85,
                faithfulness_score=0.75,
                safety_score=0.65,
            ),
            _base_row(
                span_id="span-other",
                langgraph_node="retrieve",
                correctness_score=0.1,
                relevance_score=0.1,
                faithfulness_score=0.1,
                safety_score=0.1,
            ),
        ]
    )

    [evidence] = build_evidence_packages(df)

    assert evidence["scores"] == {
        "correctness": 0.95,
        "relevance": 0.85,
        "faithfulness": 0.75,
        "safety": 0.65,
    }
    assert "finalize" in evidence["metadata"]["langgraph_nodes"]


def test_evidence_builder_falls_back_to_all_nodes_when_finalize_missing():
    df = pd.DataFrame(
        [
            _base_row(
                span_id="span-generate",
                langgraph_node="generate",
                correctness_score=0.4,
                relevance_score=0.9,
                faithfulness_score=0.8,
                safety_score=0.7,
            ),
            _base_row(
                span_id="span-retrieve",
                langgraph_node="retrieve",
                correctness_score=0.6,
                relevance_score=0.3,
                faithfulness_score=0.5,
                safety_score=0.2,
            ),
        ]
    )

    [evidence] = build_evidence_packages(df)

    assert evidence["scores"] == {
        "correctness": 0.4,
        "relevance": 0.3,
        "faithfulness": 0.5,
        "safety": 0.2,
    }
