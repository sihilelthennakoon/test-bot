"""Tests for offline evaluation pipeline."""

import asyncio
import json
import tempfile
from pathlib import Path

from ragbot.evaluations.offline import OfflineEvaluationPipeline


def test_offline_pipeline_evaluate_dataset():
    """Test offline pipeline evaluates a dataset."""
    pipeline = OfflineEvaluationPipeline()

    test_cases = [
        {
            "input_text": "What is X?",
            "response_text": "X is a variable.",
            "duration_ms": 1500,
        },
        {
            "input_text": "What is Y?",
            "response_text": "Y is another variable.",
            "duration_ms": 2000,
        },
    ]

    results = asyncio.run(
        pipeline.evaluate_dataset(
            test_cases=test_cases,
            run_name="Test Run",
        )
    )

    assert results.total_requests == 2
    assert len(results.traces) == 2
    assert results.run_name == "Test Run"


def test_offline_pipeline_save_json():
    """Test offline pipeline saves results to JSON."""
    pipeline = OfflineEvaluationPipeline()

    test_cases = [
        {
            "input_text": "What is X?",
            "response_text": "X is a variable.",
            "duration_ms": 1500,
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        results = asyncio.run(
            pipeline.run_and_save(
                test_cases=test_cases,
                output_dir=tmpdir,
                run_name="Test",
            )
        )

        # Check files were created
        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) >= 2  # results.json and report.json

        # Check JSON content
        json_file = [f for f in files if "_results.json" in f.name][0]
        with open(json_file, "r") as f:
            data = json.load(f)
            assert data["total_requests"] == 1


def test_offline_pipeline_save_csv():
    """Test offline pipeline saves results to CSV."""
    pipeline = OfflineEvaluationPipeline()

    test_cases = [
        {
            "input_text": "What is X?",
            "response_text": "X is a variable.",
            "duration_ms": 1500,
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        results = asyncio.run(
            pipeline.run_and_save(
                test_cases=test_cases,
                output_dir=tmpdir,
                run_name="Test",
            )
        )

        # Check CSV file exists
        csv_file = list(Path(tmpdir).glob("*.csv"))[0]
        assert csv_file.exists()

        # Check CSV content
        with open(csv_file, "r") as f:
            lines = f.readlines()
            assert len(lines) >= 2  # header + data


def test_offline_pipeline_load_test_cases():
    """Test loading test cases from JSON file."""
    test_data = [
        {
            "input_text": "What is X?",
            "response_text": "X is a variable.",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_cases.json"
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        loaded = OfflineEvaluationPipeline.load_test_cases(test_file)
        assert loaded == test_data


def test_offline_pipeline_save_test_cases():
    """Test saving test cases to JSON file."""
    test_data = [
        {
            "input_text": "What is X?",
            "response_text": "X is a variable.",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_cases.json"
        OfflineEvaluationPipeline.save_test_cases(test_data, test_file)

        assert test_file.exists()

        with open(test_file, "r") as f:
            loaded = json.load(f)
            assert loaded == test_data


def test_offline_pipeline_summary_statistics():
    """Test offline pipeline computes summary statistics."""
    pipeline = OfflineEvaluationPipeline()

    test_cases = [
        {
            "input_text": "What is X?",
            "response_text": "X is a variable.",
            "duration_ms": 1500,
        },
        {
            "input_text": "What is Y?",
            "response_text": "Y is another variable.",
            "duration_ms": 2000,
        },
    ]

    results = asyncio.run(
        pipeline.evaluate_dataset(
            test_cases=test_cases,
            run_name="Test",
        )
    )

    summary = results.to_summary()

    assert summary["total_requests"] == 2
    assert "evaluators" in summary
    assert "mean_aggregate_score" in summary


def test_offline_pipeline_run_and_save():
    """Test full run and save workflow."""
    pipeline = OfflineEvaluationPipeline()

    test_cases = [
        {
            "input_text": "What is X?",
            "response_text": "X is a variable.",
            "duration_ms": 1500,
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        results = asyncio.run(
            pipeline.run_and_save(
                test_cases=test_cases,
                output_dir=tmpdir,
                run_name="Test Run",
            )
        )

        # Verify output directory has files
        output_dir = Path(tmpdir)
        assert len(list(output_dir.glob("*.json"))) >= 2
        assert len(list(output_dir.glob("*.csv"))) >= 1
