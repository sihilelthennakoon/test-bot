"""Offline evaluation pipeline for batch testing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .results import EvaluationRunResults, TraceEvaluationResults
from .runner import EvaluationRunner


class OfflineEvaluationPipeline:
    """Runs evaluations on test datasets for regression testing and benchmarking."""

    def __init__(self, runner: EvaluationRunner | None = None):
        """Initialize offline evaluation pipeline.

        Args:
            runner: EvaluationRunner instance. If None, creates default.
        """
        self.runner = runner or EvaluationRunner()

    async def evaluate_dataset(
        self,
        test_cases: list[dict[str, Any]],
        run_name: str | None = None,
        run_id: str | None = None,
    ) -> EvaluationRunResults:
        """Evaluate a dataset of test cases.

        Args:
            test_cases: List of dicts with keys:
                - input_text (str): User query
                - response_text (str): Expected or actual response
                - Optional: context, retrieved_docs, sources, etc.
            run_name: Human-readable name for this evaluation run
            run_id: Unique run ID. Auto-generated if None.

        Returns:
            EvaluationRunResults with aggregated metrics.
        """
        run_id = run_id or f"run_{uuid4().hex[:12]}"
        run_name = run_name or f"Evaluation Run {run_id}"

        # Run evaluations
        trace_results = await self.runner.evaluate_batch(
            requests=test_cases,
            trace_id_prefix=run_id,
        )

        # Aggregate results
        run_results = EvaluationRunResults(
            run_id=run_id,
            run_name=run_name,
            total_requests=len(test_cases),
            traces=trace_results,
        )

        return run_results

    @staticmethod
    def load_test_cases(path: str | Path) -> list[dict[str, Any]]:
        """Load test cases from JSON file.

        Expected format:
        [
            {
                "input_text": "What is the color of the sky?",
                "response_text": "The sky appears blue due to Rayleigh scattering.",
                "context": "...",  # optional
            },
            ...
        ]

        Args:
            path: Path to JSON file

        Returns:
            List of test case dictionaries.
        """
        path = Path(path)
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_test_cases(
        test_cases: list[dict[str, Any]], path: str | Path
    ) -> None:
        """Save test cases to JSON file.

        Args:
            test_cases: List of test cases
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(test_cases, f, indent=2)

    async def run_and_save(
        self,
        test_cases: list[dict[str, Any]],
        output_dir: str | Path,
        run_name: str | None = None,
    ) -> EvaluationRunResults:
        """Run evaluation and save results to files.

        Args:
            test_cases: List of test cases
            output_dir: Directory to save results
            run_name: Name for this run

        Returns:
            EvaluationRunResults
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluation
        results = await self.evaluate_dataset(
            test_cases=test_cases,
            run_name=run_name,
        )

        # Save results in multiple formats
        json_path = output_dir / f"{results.run_id}_results.json"
        csv_path = output_dir / f"{results.run_id}_results.csv"
        summary_path = output_dir / "evaluation_report.json"

        results.save_json(json_path)
        results.save_csv(csv_path)

        # Also save summary at top level
        with open(summary_path, "w") as f:
            json.dump(results.to_summary(), f, indent=2)

        return results


async def run_offline_evaluation(
    test_dataset_path: str | Path,
    output_dir: str | Path = "data/eval/results",
    run_name: str | None = None,
) -> EvaluationRunResults:
    """Convenience function to run offline evaluation on a dataset file.

    Args:
        test_dataset_path: Path to JSON file with test cases
        output_dir: Where to save results
        run_name: Name for this evaluation run

    Returns:
        Aggregated evaluation results
    """
    pipeline = OfflineEvaluationPipeline()

    # Load test cases
    test_cases = pipeline.load_test_cases(test_dataset_path)

    # Run and save
    results = await pipeline.run_and_save(
        test_cases=test_cases,
        output_dir=output_dir,
        run_name=run_name,
    )

    return results
