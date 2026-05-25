"""Tests for evaluator factory functions (Phoenix SDK)."""

import pytest

from ragbot.evaluations import (
    correctness,
    format as format_eval,
    groundedness,
    latency,
    relevance,
    routing,
    safety,
)


class TestFactoryFunctions:
    """Test that all evaluator factory functions can be created."""

    def test_create_correctness_evaluator(self):
        """Test correctness evaluator factory."""
        evaluator = correctness.create_correctness_evaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'apply_async') or hasattr(evaluator, 'evaluate')

    def test_create_relevance_evaluator(self):
        """Test relevance evaluator factory."""
        evaluator = relevance.create_relevance_evaluator()
        assert evaluator is not None

    def test_create_groundedness_evaluator(self):
        """Test groundedness evaluator factory."""
        evaluator = groundedness.create_groundedness_evaluator()
        assert evaluator is not None

    def test_create_routing_evaluator(self):
        """Test routing evaluator factory."""
        evaluator = routing.create_routing_evaluator(
            min_doc_count=1,
            min_score_threshold=0.3,
        )
        assert evaluator is not None

    def test_create_latency_evaluator(self):
        """Test latency evaluator factory."""
        evaluator = latency.create_latency_evaluator(
            pass_threshold_ms=3000,
            warning_threshold_ms=5000,
        )
        assert evaluator is not None

    def test_create_format_evaluator(self):
        """Test format evaluator factory."""
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            response: str

        evaluator = format_eval.create_format_evaluator(schema=TestResponse)
        assert evaluator is not None

    def test_create_safety_evaluator(self):
        """Test safety evaluator factory."""
        evaluator = safety.create_safety_evaluator(
            check_pii=True,
            check_injection=True,
        )
        assert evaluator is not None
