from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _build_settings(**overrides):
	values = {
		"project_name": "ragbot",
		"phoenix_project_name": "phoenix-project",
		"phoenix_fetch_span_kind": "CHAIN",
		"batch_evaluation_cron_limit": 100,
		"phoenix_query_endpoint": "http://localhost:6006",
		"batch_evaluation_cron_sync_annotations": True,
		"batch_evaluation_cron_save_annotations": True,
		"batch_evaluation_use_checkpoint": True,
		"batch_evaluation_checkpoint_file": Path("data/batch_evaluation/checkpoint.json"),
		"batch_evaluation_cron_interval_seconds": 3600,
		"batch_evaluation_cron_enabled": False,
	}
	values.update(overrides)
	return SimpleNamespace(**values)


def _build_service():
	return SimpleNamespace(
		store=SimpleNamespace(_records=[], index_path="index.faiss"),
		ingestion=SimpleNamespace(ingest_file=lambda *args, **kwargs: None),
		chat_app=SimpleNamespace(invoke=lambda *args, **kwargs: {}),
		build_invocation_config=lambda *args, **kwargs: {},
	)


def _load_app_module(monkeypatch, settings):
	import ragbot.config as config_module
	import ragbot.service as service_module

	monkeypatch.setattr(config_module, "get_settings", lambda: settings)
	monkeypatch.setattr(service_module.ChatService, "create", classmethod(lambda cls, arg: _build_service()))
	sys.modules.pop("ragbot.api.app", None)
	return importlib.import_module("ragbot.api.app")


def test_create_app_passes_checkpoint_settings_to_scheduler(monkeypatch) -> None:
	captured = {}
	settings = _build_settings(batch_evaluation_cron_enabled=True)

	class FakeScheduler:
		def __init__(self, *, config, interval_seconds: int) -> None:
			captured["config"] = config
			captured["interval_seconds"] = interval_seconds

		def start(self) -> None:
			return None

		def stop(self) -> None:
			return None

	import ragbot.evaluations.batch_evaluation.scheduler as scheduler_module

	monkeypatch.setattr(scheduler_module, "BatchEvaluationScheduler", FakeScheduler)
	app_module = _load_app_module(monkeypatch, settings)

	app_module.create_app()

	assert captured["config"].use_checkpoint is True
	assert captured["config"].checkpoint_file == settings.batch_evaluation_checkpoint_file
	assert captured["interval_seconds"] == settings.batch_evaluation_cron_interval_seconds


def test_eval_endpoint_passes_checkpoint_config_to_batch_runner(monkeypatch) -> None:
	captured = {}
	settings = _build_settings()
	app_module = _load_app_module(monkeypatch, settings)

	def fake_run_span_batch(config):
		captured["batch_config"] = config
		return SimpleNamespace(
			span_count=2,
			evaluation_df=[{"row": 1}],
			annotation_count=4,
		)

	monkeypatch.setattr(app_module, "run_span_batch", fake_run_span_batch)

	client = TestClient(app_module.create_app())
	response = client.post(
		"/evaluations/batch/run",
		json={
			"from_time": "2026-06-12T09:00:00Z",
			"to_time": "2026-06-12T10:00:00Z",
		},
	)

	assert response.status_code == 200
	assert captured["batch_config"].from_time == datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)
	assert captured["batch_config"].to_time == datetime(2026, 6, 12, 10, 0, tzinfo=timezone.utc)
	assert captured["batch_config"].use_checkpoint is True
	assert captured["batch_config"].checkpoint_file == settings.batch_evaluation_checkpoint_file


def test_eval_endpoint_returns_error_for_invalid_checkpoint_file(monkeypatch, tmp_path: Path) -> None:
	checkpoint_file = tmp_path / "checkpoint.json"
	checkpoint_file.write_text("{not-json", encoding="utf-8")
	settings = _build_settings(batch_evaluation_checkpoint_file=checkpoint_file)

	import ragbot.evaluations.batch_evaluation.evaluate_batch as eval_module

	monkeypatch.setattr(eval_module, "_utc_now", lambda: datetime(2026, 6, 12, 10, 30, tzinfo=timezone.utc))
	app_module = _load_app_module(monkeypatch, settings)

	client = TestClient(app_module.create_app())
	response = client.post("/evaluations/batch/run", json={})

	assert response.status_code == 502
	assert "Invalid batch evaluation checkpoint JSON" in response.json()["detail"]


def test_scheduler_run_loop_delegates_to_batch_runner(monkeypatch) -> None:
	captured = {"runs": 0}

	import ragbot.evaluations.batch_evaluation.scheduler as scheduler_module

	class FakeStopEvent:
		def __init__(self) -> None:
			self.calls = 0

		def wait(self, timeout: int) -> bool:
			self.calls += 1
			return self.calls > 1

	scheduler = scheduler_module.BatchEvaluationScheduler(
		config=SimpleNamespace(project_name="phoenix-project", span_kind="CHAIN", limit=100),
		interval_seconds=1,
	)
	scheduler._stop_event = FakeStopEvent()

	monkeypatch.setattr(
		scheduler_module,
		"run_span_batch",
		lambda current_config: captured.update({"runs": captured["runs"] + 1}) or SimpleNamespace(
			span_count=1,
			evaluation_df=[{"row": 1}],
			annotation_count=4,
		),
	)

	scheduler._run_loop()

	assert captured["runs"] == 1
