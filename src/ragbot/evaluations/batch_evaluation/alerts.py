"""Batch evaluation degradation alert detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


logger = logging.getLogger(__name__)

DEFAULT_ALERT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "alert.yaml"


def _coerce_text(value: Any) -> str:
	if value is None:
		return ""
	return str(value).strip()


def load_alert_config(config_path: Path = DEFAULT_ALERT_CONFIG_PATH) -> dict[str, Any] | None:
	if not config_path.exists():
		return None

	try:
		loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
	except (OSError, yaml.YAMLError) as exc:
		logger.warning("Skipping degradation alert detection: invalid alert config at %s: %s", config_path, exc)
		return None

	if loaded is None:
		return None
	if not isinstance(loaded, dict):
		logger.warning("Skipping degradation alert detection: alert config at %s must be a mapping.", config_path)
		return None
	return loaded


def _validated_alert_evaluator_config(config: dict[str, Any]) -> dict[str, tuple[float, float]] | None:
	if not config.get("enabled", False):
		return {}

	evaluators = config.get("evaluators")
	if not isinstance(evaluators, dict):
		logger.warning("Skipping degradation alert detection: alert config must contain an evaluators mapping.")
		return None

	validated: dict[str, tuple[float, float]] = {}
	for evaluator_name, evaluator_config in evaluators.items():
		evaluator = _coerce_text(evaluator_name)
		if not evaluator:
			logger.warning("Skipping degradation alert detection: evaluator names must be non-empty.")
			return None
		if not isinstance(evaluator_config, dict):
			logger.warning("Skipping degradation alert detection: config for evaluator '%s' must be a mapping.", evaluator)
			return None

		try:
			score_threshold = float(evaluator_config["score_threshold"])
			percent_threshold = float(evaluator_config["percent_threshold"])
		except (KeyError, TypeError, ValueError):
			logger.warning("Skipping degradation alert detection: invalid thresholds for evaluator '%s'.", evaluator)
			return None

		if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= percent_threshold <= 100.0:
			logger.warning("Skipping degradation alert detection: thresholds for evaluator '%s' are out of range.", evaluator)
			return None

		validated[evaluator] = (score_threshold, percent_threshold)

	return validated


def detect_degradation_alerts(
	annotations_df: pd.DataFrame,
	config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
	if annotations_df.empty or config is None:
		return []

	evaluator_config = _validated_alert_evaluator_config(config)
	if not evaluator_config:
		return []

	required_columns = {"annotation_name", "score"}
	if not required_columns.issubset(annotations_df.columns):
		logger.warning("Skipping degradation alert detection: annotations are missing required alert columns.")
		return []

	alerts: list[dict[str, Any]] = []
	for evaluator, (score_threshold, percent_threshold) in evaluator_config.items():
		evaluator_rows = annotations_df.loc[annotations_df["annotation_name"] == evaluator]
		total_count = int(len(evaluator_rows))
		if total_count == 0:
			continue

		scores = pd.to_numeric(evaluator_rows["score"], errors="coerce")
		low_score_count = int((scores < score_threshold).sum())
		low_score_percent = (low_score_count / total_count) * 100
		if low_score_percent >= percent_threshold:
			alerts.append(
				{
					"evaluator": evaluator,
					"low_score_count": low_score_count,
					"total_count": total_count,
					"low_score_percent": low_score_percent,
					"score_threshold": score_threshold,
					"percent_threshold": percent_threshold,
				}
			)

	return alerts


def log_degradation_alerts(annotations_df: pd.DataFrame) -> None:
	for alert in detect_degradation_alerts(annotations_df, load_alert_config()):
		logger.warning(
			(
				"Performance degradation alert: evaluator=%s low_score_count=%s "
				"total_count=%s low_score_percent=%.2f score_threshold=%.3f "
				"percent_threshold=%.3f"
			),
			alert["evaluator"],
			alert["low_score_count"],
			alert["total_count"],
			alert["low_score_percent"],
			alert["score_threshold"],
			alert["percent_threshold"],
			extra={"performance_degradation_alert": alert},
		)
