"""End-to-end integration test for evaluation pipeline."""

import asyncio
import json
import tempfile
from pathlib import Path

from ragbot.evaluations.offline import OfflineEvaluationPipeline
from ragbot.evaluations.runner import EvaluationRunner


def test_full_evaluation_pipeline():
    """Test complete offline evaluation pipeline."""
    # Create evaluation runner with all evaluators
    runner = EvaluationRunner()

    # Prepare test data
    test_requests = [
        {
            "input_text": "What is the capital of France?",
            "response_text": "The capital of France is Paris.",
            "question": "What is the capital of France?",
            "duration_ms": 1500,
            "retrieved_docs": ["France's capital is Paris."],
            "context": "France's capital is Paris, located in the north-central part of the country.",
        },
        {
            "input_text": "How do I reset my password?",
            "response_text": "You can reset your password by clicking the 'Forgot Password' link on the login page.",
            "question": "How do I reset my password?",
            "duration_ms": 2000,
            "retrieved_docs": ["Password reset is available via the login page."],
        },
        {
            "input_text": "Contact support at example@company.com",
            "response_text": "PII information should not be repeated in responses.",
            "question": "How do I contact support?",
            "duration_ms": 800,
        },
    ]

    # Run evaluations
    results = asyncio.run(runner.evaluate_batch(test_requests))

    # Verify results
    assert len(results) == 3
    for result in results:
        assert result.trace_id is not None
        assert len(result.evaluations) > 0
        assert 0 <= result.aggregate_score() <= 1
        assert 0 <= result.pass_rate() <= 1


def test_offline_pipeline_complete_workflow():
    """Test complete offline evaluation workflow with file I/O."""
    pipeline = OfflineEvaluationPipeline()

    # Load actual test cases
    test_cases = pipeline.load_test_cases("data/eval/test_cases.json")
    assert len(test_cases) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run evaluation
        results = asyncio.run(
            pipeline.run_and_save(
                test_cases=test_cases[:5],  # Use first 5 for quick test
                output_dir=tmpdir,
                run_name="Integration Test",
            )
        )

        # Verify results
        assert results.total_requests == 5
        assert len(results.traces) == 5

        # Check output files
        output_path = Path(tmpdir)
        json_files = list(output_path.glob("*_results.json"))
        csv_files = list(output_path.glob("*.csv"))
        report_files = list(output_path.glob("evaluation_report.json"))

        assert len(json_files) >= 1, "Should have results JSON file"
        assert len(csv_files) >= 1, "Should have CSV export"
        assert len(report_files) >= 1, "Should have evaluation report"

        # Verify JSON structure
        with open(json_files[0], "r") as f:
            json_data = json.load(f)
            assert "total_requests" in json_data
            assert "summary" in json_data
            assert "traces" in json_data

        # Verify CSV has data
        csv_file = csv_files[0]
        with open(csv_file, "r") as f:
            lines = f.readlines()
            assert len(lines) > 1, "CSV should have header + data rows"

        # Verify summary report
        with open(report_files[0], "r") as f:
            summary = json.load(f)
            assert "total_requests" in summary
            assert "evaluators" in summary
            assert "mean_aggregate_score" in summary


def test_evaluation_runner_all_evaluators_enabled():
    """Test that all evaluators are available and can be configured."""
    runner = EvaluationRunner()

    # Check all evaluators are present
    expected_evaluators = {
        "correctness",
        "relevance",
        "groundedness",
        "routing",
        "safety",
        "latency",
        "format",
    }
    assert set(runner.evaluators.keys()) == expected_evaluators

    # Test disabling/enabling
    runner.disable_evaluator("correctness")
    assert "correctness" not in runner.get_enabled_evaluators()

    runner.enable_evaluator("correctness")
    assert "correctness" in runner.get_enabled_evaluators()


def test_evaluation_result_aggregation():
    """Test result aggregation and statistics."""
    runner = EvaluationRunner()

    requests = [
        {
            "input_text": f"Question {i}?",
            "response_text": f"Answer {i}.",
            "duration_ms": 1500 + i * 500,
        }
        for i in range(10)
    ]

    results = asyncio.run(runner.evaluate_batch(requests))

    # Verify all evals have results
    for result in results:
        assert len(result.evaluations) >= 5  # At least safety, latency, format

    # Verify scores are in valid range
    for result in results:
        for eval_result in result.evaluations.values():
            assert 0 <= eval_result.score <= 1
            assert isinstance(eval_result.passed, bool)


def test_eval_with_required_context():
    """Test evaluators with specific context requirements."""
    runner = EvaluationRunner()

    # Request with full context for all evaluators
    request = {
        "input_text": "What is X?",
        "response_text": "X is a variable in programming.",
        "question": "What is X?",
        "response": "X is a variable in programming.",
        "duration_ms": 2000,
        "retrieved_docs": ["Variables are named storage locations."],
        "context": "X commonly refers to a variable or unknown value.",
        "scores": [0.85, 0.70],
    }

    result = asyncio.run(
        runner.evaluate(
            input_text=request["input_text"],
            response_text=request["response_text"],
            **{k: v for k, v in request.items() if k not in ["input_text", "response_text"]},
        )
    )

    # All evaluators should have results
    assert len(result.evaluations) >= 7
    assert result.aggregate_score() > 0


if __name__ == "__main__":
    # Run offline evaluation from command line
    print("Running offline evaluation...")
    test_offline_pipeline_complete_workflow()
    print("✓ Offline evaluation complete")

    print("Running full pipeline test...")
    test_full_evaluation_pipeline()
    print("✓ Full pipeline test complete")

    print("All integration tests passed! ✓")
