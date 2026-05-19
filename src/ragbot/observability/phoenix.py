from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Iterator


@dataclass(slots=True)
class TraceRecorder:
    project_name: str
    collector_endpoint: str
    enabled: bool = True
    evaluation_path: Path | None = None
    _evaluations: list[dict[str, Any]] = field(default_factory=list)

    def configure(self) -> None:
        if not self.enabled:
            return
        try:
            from phoenix.otel import register
        except Exception:
            return
        register(project_name=self.project_name, endpoint=self.collector_endpoint)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        try:
            from opentelemetry import trace as otel_trace
        except Exception:
            yield
            return

        tracer = otel_trace.get_tracer(self.project_name)
        with tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            yield

    def record_evaluation(self, payload: dict[str, Any]) -> None:
        self._evaluations.append(payload)
        if self.evaluation_path is None:
            return
        self.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        with self.evaluation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_trace_recorder(project_name: str, collector_endpoint: str, evaluation_path: Path | None = None) -> TraceRecorder:
    recorder = TraceRecorder(
        project_name=project_name,
        collector_endpoint=collector_endpoint,
        evaluation_path=evaluation_path,
    )
    recorder.configure()
    return recorder
