"""Tests for evaluation runner with Phoenix SDK."""

import asyncio

import pytest
import pandas as pd

from ragbot.evaluations.results import EvaluationResult, TraceEvaluationResults
from ragbot.evaluations.runner import EvaluationRunner


class TestEvaluationRunnerInitialization:
    """Test EvaluationRunner initialization."""

    def test_runner_initialization_with_defaults(self):
        """Test EvaluationRunner initializes with default evaluators."""
        runner = EvaluationRunner()

        assert len(runner.evaluators) >= 7
        enabled = runner.get_enabled_evaluators()
        assert "correctness" in enabled or "routing" in enabled

    def test_runner_custom_evaluators(self):
        """Test EvaluationRunner with custom evaluators dict."""
        from ragbot.evaluations import routing

        custom_evaluators = {
            "routing": routing.create_routing_evaluator(),
        }
        runner = EvaluationRunner(evaluators=custom_evaluators)

        assert len(runner.evaluators) == 1
        assert "routing" in runner.evaluators


class TestEvaluationRunnerEvaluation:
    """Test evaluating requests with Phoenix SDK."""

    def test_runner_evaluate_single_request(self):
        """Test runner evaluates a single request."""
        runner = EvaluationRunner()

        result = asyncio.run(
            runner.evaluate(
                input_text="What is X?",
                response_text="X is a variable.",
                question="What is X?",
            )
        )

        assert isinstance(result, TraceEvaluationResults)
        assert result.trace_id is not None
        assert result.input_text == "What is X?"
        assert result.response_text == "X is a variable."
        assert isinstance(result.evaluations, dict)

    def test_runner_evaluate_with_trace_id(self):
        """Test runner evaluates with custom trace ID."""
        runner = EvaluationRunner()

        result = asyncio.run(
            runner.evaluate(
                trace_id="test_trace_123",
                input_text="What is X?",
                response_text="X is a variable.",
            )
        )

        assert result.trace_id == "test_trace_123"

    def test_runner_evaluate_batch(self):
        """Test runner evaluates a batch of requests."""
        runner = EvaluationRunner()

        requests = [
            {
                "input_text": "What is X?",
                "response_text": "X is a variable.",
            },
            {
                "input_text": "What is Y?",
                "response_text": "Y is another variable.",
            },
        ]

        results = asyncio.run(runner.evaluate_batch(requests))

        assert len(results) == 2
        assert all(isinstance(r, TraceEvaluationResults) for r in results)
        assert results[0].trace_id == "batch_0"
        assert results[1].trace_id == "batch_1"


class TestEvaluationRunnerControl:
    """Test evaluator enable/disable controls."""

    def test_runner_disable_evaluator(self):
        """Test disabling specific evaluator."""
        runner = EvaluationRunner()
        initial_enabled = len(runner.get_enabled_evaluators())

        if initial_enabled > 0:
            first_eval = runner.get_enabled_evaluators()[0]
            runner.disable_evaluator(first_eval)

            assert first_eval not in runner.get_enabled_evaluators()
            assert len(runner.get_enabled_evaluators()) == initial_enabled - 1

    def test_runner_enable_evaluator(self):
        """Test enabling specific evaluator."""
        runner = EvaluationRunner()
        
        if runner.get_enabled_evaluators():
            first_eval = runner.get_enabled_evaluators()[0]
            runner.disable_evaluator(first_eval)
            runner.enable_evaluator(first_eval)

            assert first_eval in runner.get_enabled_evaluators()

    def test_get_enabled_disabled_evaluators(self):
        """Test retrieving enabled/disabled evaluators."""
        runner = EvaluationRunner()

        enabled = runner.get_enabled_evaluators()
        disabled = runner.get_disabled_evaluators()

        assert isinstance(enabled, list)
        assert isinstance(disabled, list)
        assert len(enabled) + len(disabled) == len(runner.evaluators)


class TestTraceEvaluationResults:
    """Test TraceEvaluationResults schema and methods."""

    def test_trace_evaluation_results_aggregate_score(self):
        """Test TraceEvaluationResults computes aggregate score."""
        result1 = EvaluationResult(
            evaluator_name="test1",
            score=1.0,
            passed=True,
            reason="Perfect",
        )
        result2 = EvaluationResult(
            evaluator_name="test2",
            score=0.5,
            passed=False,
            reason="Okay",
        )

        trace = TraceEvaluationResults(
            trace_id="test",
            input_text="test",
            response_text="test",
            evaluations={"test1": result1, "test2": result2},
        )

        assert trace.aggregate_score() == 0.75

    def test_trace_evaluation_results_pass_rate(self):
        """Test TraceEvaluationResults computes pass rate."""
        result1 = EvaluationResult(
            evaluator_name="test1",
            score=1.0,
            passed=True,
            reason="Perfect",
        )
        result2 = EvaluationResult(
            evaluator_name="test2",
            score=0.0,
            passed=False,
            reason="Failed",
        )

        trace = TraceEvaluationResults(
            trace_id="test",
            input_text="test",
            response_text="test",
            evaluations={"test1": result1, "test2": result2},
        )

        assert trace.pass_rate() == 0.5

    def test_trace_to_dict(self):
        """Test converting trace to dictionary."""
        result = EvaluationResult(
            evaluator_name="test",
            score=0.8,
            passed=True,
            reason="Good",
        )

        trace = TraceEvaluationResults(
            trace_id="test",
            input_text="input",
            response_text="output",
            evaluations={"test": result},
        )

        trace_dict = trace.to_dict()
        assert trace_dict["trace_id"] == "test"
        assert "evaluations" in trace_dict
        assert "aggregate_score" in trace_dict
        assert "pass_rate" in trace_dict


class TestRunnerRepr:
    """Test runner string representation."""

    def test_runner_repr(self):
        """Test runner string representation."""
        runner = EvaluationRunner()
        repr_str = repr(runner)

        assert "EvaluationRunner" in repr_str
        assert "enabled" in repr_str
