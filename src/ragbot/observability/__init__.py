from __future__ import annotations

import os

from ragbot.config import normalize_phoenix_collector_endpoint

_REGISTERED = False
_TRACER_PROVIDER = None


def bootstrap_phoenix(*, project_name: str, collector_endpoint: str = "http://localhost:6006"):
    global _REGISTERED, _TRACER_PROVIDER

    if _REGISTERED:
        return _TRACER_PROVIDER

    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", collector_endpoint)
    endpoint = normalize_phoenix_collector_endpoint(collector_endpoint)

    try:
        from phoenix.otel import register
    except Exception:
        _REGISTERED = True
        _TRACER_PROVIDER = None
        return None

    try:
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
            protocol="http/protobuf",
            auto_instrument=True,
            batch=True,
        )
    except Exception:
        _REGISTERED = True
        _TRACER_PROVIDER = None
        return None

    _REGISTERED = True
    _TRACER_PROVIDER = tracer_provider
    return tracer_provider
