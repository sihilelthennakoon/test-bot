from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


def test_settings_support_rca_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    monkeypatch.setenv("RAGBOT_PHOENIX_PULL_BASE_URL", "http://phoenix.internal:7007")
    monkeypatch.setenv("RAGBOT_RCA_PULL_SAVE_DIR", str(tmp_path / "rca-pulls"))

    import ragbot.config as config_module

    config_module = importlib.reload(config_module)
    settings = config_module.get_settings()

    assert settings.phoenix_pull_base_url == "http://phoenix.internal:7007"
    assert settings.rca_pull_save_dir == tmp_path / "rca-pulls"
    assert settings.rca_pull_save_dir.exists()


def test_default_pull_base_url_is_derived(monkeypatch) -> None:
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    monkeypatch.delenv("RAGBOT_PHOENIX_PULL_BASE_URL", raising=False)

    import ragbot.config as config_module

    config_module = importlib.reload(config_module)
    settings = config_module.Settings()

    assert settings.phoenix_pull_base_url == "http://localhost:6006"


def test_pull_traces_normalizes_inline_spans_and_persists(tmp_path: Path) -> None:
    from ragbot.rca.service import RCATraceService
    from ragbot.rca.storage import RCAPullStorage

    class FakeClient:
        def fetch_traces(self, filters):
            return {
                "data": [
                    {
                        "trace_id": "trace-1",
                        "start_time": "2026-05-27T10:00:00Z",
                        "end_time": "2026-05-27T10:00:02Z",
                        "spans": [
                            {
                                "span_id": "root-1",
                                "trace_id": "trace-1",
                                "name": "chat.request",
                                "start_time": "2026-05-27T10:00:00Z",
                                "end_time": "2026-05-27T10:00:02Z",
                                "attributes": {"session.id": "conv-1"},
                            },
                            {
                                "span_id": "child-1",
                                "trace_id": "trace-1",
                                "parent_span_id": "root-1",
                                "name": "retrieve",
                                "start_time": "2026-05-27T10:00:00.500000Z",
                                "end_time": "2026-05-27T10:00:01Z",
                            },
                        ],
                    }
                ],
                "next_cursor": "cursor-1",
            }

        def fetch_spans(self, filters):
            raise AssertionError("span fallback should not be used")

    service = RCATraceService(
        client=FakeClient(),
        storage=RCAPullStorage(tmp_path),
        default_project_name="ragbot",
    )

    result = service.pull_traces(conversation_id="conv-1")

    assert result.source == "phoenix.traces"
    assert result.trace_count == 1
    assert result.next_cursor == "cursor-1"
    assert result.snapshot_path is not None
    assert result.traces[0].conversation_id == "conv-1"
    assert result.traces[0].root_span_name == "chat.request"
    assert result.traces[0].span_count == 2

    snapshot = Path(result.snapshot_path)
    assert snapshot.exists()
    with snapshot.open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["trace_count"] == 1
    assert data["traces"][0]["spans"][0]["raw"]["attributes"]["session.id"] == "conv-1"


def test_pull_traces_falls_back_to_spans_for_unsupported_trace_endpoint(tmp_path: Path) -> None:
    from ragbot.rca.phoenix_client import PhoenixClientError
    from ragbot.rca.service import RCATraceService
    from ragbot.rca.storage import RCAPullStorage

    class FakeClient:
        def fetch_traces(self, filters):
            raise PhoenixClientError("not supported", status_code=404)

        def fetch_spans(self, filters):
            return {
                "spans": [
                    {
                        "span_id": "root-1",
                        "trace_id": "trace-1",
                        "name": "chat.request",
                        "start_time": "2026-05-27T10:00:00Z",
                        "end_time": "2026-05-27T10:00:01Z",
                        "attributes": {"session.id": "conv-22"},
                    },
                    {
                        "span_id": "child-1",
                        "trace_id": "trace-1",
                        "parent_span_id": "root-1",
                        "name": "generate",
                        "start_time": "2026-05-27T10:00:00.200000Z",
                        "end_time": "2026-05-27T10:00:00.900000Z",
                    },
                ],
                "meta": {"next_cursor": "cursor-2"},
            }

    service = RCATraceService(
        client=FakeClient(),
        storage=RCAPullStorage(tmp_path),
        default_project_name="ragbot",
    )

    result = service.pull_traces(conversation_id="conv-22")

    assert result.source == "phoenix.spans_fallback"
    assert result.trace_count == 1
    assert result.next_cursor == "cursor-2"
    assert result.traces[0].trace_id == "trace-1"
    assert result.traces[0].conversation_id == "conv-22"
    assert "trace_error" in result.raw_payload


def test_pull_storage_uses_json_snapshot_name(tmp_path: Path) -> None:
    from ragbot.rca.models import PhoenixPullFilters, PhoenixTracePullResult
    from ragbot.rca.storage import RCAPullStorage

    result = PhoenixTracePullResult(
        fetched_at=datetime(2026, 5, 27, 10, 0, 0, tzinfo=timezone.utc),
        source="phoenix.traces",
        project_name="ragbot",
        filters=PhoenixPullFilters(project_name="ragbot"),
        trace_count=0,
        traces=[],
    )
    storage = RCAPullStorage(tmp_path)

    path = storage.save(result)

    assert path.name == "phoenix-pull-ragbot-20260527T100000Z.json"
    assert path.exists()


def test_rca_pull_endpoint_returns_saved_result(monkeypatch, tmp_path: Path) -> None:
    from ragbot.rca.models import PhoenixPullFilters, PhoenixTracePullResult
    import ragbot.service as service_module

    @dataclass
    class FakeRCA:
        called_with: dict | None = None

        def pull_traces(self, **kwargs):
            self.called_with = kwargs
            return PhoenixTracePullResult(
                source="phoenix.traces",
                project_name="ragbot",
                filters=PhoenixPullFilters(project_name="ragbot", conversation_id="conv-1"),
                trace_count=0,
                traces=[],
                snapshot_path=str(tmp_path / "saved.json"),
            )

    fake_rca = FakeRCA()
    fake_service = SimpleNamespace(
        store=SimpleNamespace(_records=[]),
        ingestion=SimpleNamespace(ingest_file=lambda *args, **kwargs: None),
        rca=fake_rca,
        chat_app=SimpleNamespace(invoke=lambda *args, **kwargs: None),
        build_invocation_config=lambda conversation_id=None: {},
    )

    monkeypatch.setattr(service_module.ChatService, "create", lambda settings=None: fake_service)
    api_module = importlib.import_module("ragbot.api.app")
    api_module = importlib.reload(api_module)
    app = api_module.create_app()
    client = TestClient(app)

    response = client.get("/rca/traces/pull", params={"conversation_id": "conv-1", "limit": 12})

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot_path"] == str(tmp_path / "saved.json")
    assert fake_rca.called_with["conversation_id"] == "conv-1"
    assert fake_rca.called_with["limit"] == 12


def test_rca_pull_endpoint_maps_upstream_errors(monkeypatch) -> None:
    from ragbot.rca.phoenix_client import PhoenixClientError
    import ragbot.service as service_module

    class FakeRCA:
        def pull_traces(self, **kwargs):
            raise PhoenixClientError("Phoenix request failed: connection refused")

    fake_service = SimpleNamespace(
        store=SimpleNamespace(_records=[]),
        ingestion=SimpleNamespace(ingest_file=lambda *args, **kwargs: None),
        rca=FakeRCA(),
        chat_app=SimpleNamespace(invoke=lambda *args, **kwargs: None),
        build_invocation_config=lambda conversation_id=None: {},
    )

    monkeypatch.setattr(service_module.ChatService, "create", lambda settings=None: fake_service)
    api_module = importlib.import_module("ragbot.api.app")
    api_module = importlib.reload(api_module)
    app = api_module.create_app()
    client = TestClient(app)

    response = client.get("/rca/traces/pull")

    assert response.status_code == 502
    assert "Phoenix request failed" in response.json()["detail"]


def test_rca_pull_endpoint_validates_limit(monkeypatch) -> None:
    import ragbot.service as service_module

    fake_service = SimpleNamespace(
        store=SimpleNamespace(_records=[]),
        ingestion=SimpleNamespace(ingest_file=lambda *args, **kwargs: None),
        rca=SimpleNamespace(pull_traces=lambda **kwargs: None),
        chat_app=SimpleNamespace(invoke=lambda *args, **kwargs: None),
        build_invocation_config=lambda conversation_id=None: {},
    )

    monkeypatch.setattr(service_module.ChatService, "create", lambda settings=None: fake_service)
    api_module = importlib.import_module("ragbot.api.app")
    api_module = importlib.reload(api_module)
    app = api_module.create_app()
    client = TestClient(app)

    response = client.get("/rca/traces/pull", params={"limit": 0})

    assert response.status_code == 422
