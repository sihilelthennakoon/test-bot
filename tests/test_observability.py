from __future__ import annotations

from types import ModuleType
import sys

from ragbot.config import Settings
from ragbot.service import ChatService


def test_bootstrap_phoenix_registers_once(monkeypatch) -> None:
    from ragbot import observability

    calls: list[dict[str, object]] = []

    phoenix_module = ModuleType("phoenix")
    otel_module = ModuleType("phoenix.otel")

    def register(**kwargs):
        calls.append(kwargs)
        return object()

    otel_module.register = register  # type: ignore[attr-defined]
    phoenix_module.otel = otel_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "phoenix", phoenix_module)
    monkeypatch.setitem(sys.modules, "phoenix.otel", otel_module)
    monkeypatch.setattr(observability, "_REGISTERED", False)
    monkeypatch.setattr(observability, "_TRACER_PROVIDER", None)
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)

    tracer_provider = observability.bootstrap_phoenix(
        project_name="test-bot-with-eval",
        collector_endpoint="http://localhost:6006",
    )
    second_provider = observability.bootstrap_phoenix(
        project_name="ignored",
        collector_endpoint="http://example.com",
    )

    assert tracer_provider is second_provider
    assert len(calls) == 1
    assert calls[0] == {
        "project_name": "test-bot-with-eval",
        "endpoint": "http://localhost:6006/v1/traces",
        "protocol": "http/protobuf",
        "auto_instrument": True,
        "batch": True,
    }
    assert sys.modules["phoenix"] is phoenix_module
    assert sys.modules["phoenix.otel"] is otel_module


def test_chat_service_create_bootstraps_phoenix(monkeypatch) -> None:
    from ragbot import service as service_module

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        service_module,
        "bootstrap_phoenix",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(service_module, "build_embedding_provider", lambda settings: object())
    monkeypatch.setattr(
        service_module.FaissVectorStore,
        "load_or_create",
        lambda embedding_provider, index_dir: object(),
    )

    settings = Settings()
    service = ChatService.create(settings)

    assert service.settings == settings
    assert len(calls) == 1
    assert calls[0][1] == {
        "project_name": settings.phoenix_project_name,
        "collector_endpoint": settings.phoenix_collector_endpoint,
    }


def test_chat_service_builds_invocation_metadata(monkeypatch) -> None:
    from ragbot import service as service_module

    monkeypatch.setattr(service_module, "bootstrap_phoenix", lambda *args, **kwargs: object())
    monkeypatch.setattr(service_module, "build_embedding_provider", lambda settings: object())
    monkeypatch.setattr(
        service_module.FaissVectorStore,
        "load_or_create",
        lambda embedding_provider, index_dir: object(),
    )
    monkeypatch.setattr(service_module, "build_chat_app", lambda runtime, settings=None: object())

    service = ChatService.create(Settings())
    config = service.build_invocation_config("conv-123")

    assert config == {
        "metadata": {
            "conversation_id": "conv-123",
            "environment": service.settings.environment,
            "app_version": service.settings.app_version,
            "use_case": service.settings.use_case,
        },
        "tags": [],
    }