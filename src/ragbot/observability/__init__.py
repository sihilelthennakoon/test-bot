from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

_REGISTERED = False
_TRACER_PROVIDER = None


def _normalize_collector_endpoint(collector_endpoint: str) -> str:
    parsed = urlparse(collector_endpoint)
    if not parsed.scheme or not parsed.netloc:
        return collector_endpoint
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1/traces"
    return urlunparse(parsed._replace(path=path))


def bootstrap_phoenix(*, project_name: str, collector_endpoint: str = "http://localhost:6006"):
    global _REGISTERED, _TRACER_PROVIDER

    if _REGISTERED:
        return _TRACER_PROVIDER

    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", collector_endpoint)
    endpoint = _normalize_collector_endpoint(collector_endpoint)

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

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:
        pass

    _REGISTERED = True
    _TRACER_PROVIDER = tracer_provider
    return tracer_provider