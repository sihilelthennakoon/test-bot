from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any
from urllib import error, parse, request

from .models import PhoenixPullFilters


class PhoenixClientError(RuntimeError):
    """Raised when Phoenix trace retrieval fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


@dataclass(slots=True)
class PhoenixClient:
    base_url: str
    timeout_seconds: float = 10.0

    def _build_url(self, path: str, params: dict[str, Any]) -> str:
        query = {
            key: value
            for key, value in params.items()
            if value is not None
        }
        encoded = parse.urlencode(query, doseq=True)
        return f"{self.base_url.rstrip('/')}{path}?{encoded}" if encoded else f"{self.base_url.rstrip('/')}{path}"

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self._build_url(path, params)
        req = request.Request(url, headers={"accept": "application/json"})
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PhoenixClientError(
                f"Phoenix request failed with status {exc.code}: {detail}",
                status_code=exc.code,
            ) from exc
        except error.URLError as exc:
            raise PhoenixClientError(f"Phoenix request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise PhoenixClientError("Phoenix request timed out") from exc

        try:
            return json.loads(payload) if payload else {}
        except json.JSONDecodeError as exc:
            raise PhoenixClientError("Phoenix response was not valid JSON") from exc

    def fetch_traces(self, filters: PhoenixPullFilters) -> dict[str, Any]:
        params = {
            "start_time": _isoformat(filters.start_time),
            "end_time": _isoformat(filters.end_time),
            "trace_id": filters.trace_id,
            "session_id": filters.conversation_id,
            "limit": filters.limit,
            "cursor": filters.cursor,
            "include_spans": "true",
            "sort": "start_time",
            "order": "desc",
        }
        return self._get_json(f"/v1/projects/{parse.quote(filters.project_name, safe='')}/traces", params)

    def fetch_spans(self, filters: PhoenixPullFilters) -> dict[str, Any]:
        params = {
            "project_name": filters.project_name,
            "start_time": _isoformat(filters.start_time),
            "end_time": _isoformat(filters.end_time),
            "trace_id": filters.trace_id,
            "session_id": filters.conversation_id,
            "limit": filters.limit,
            "cursor": filters.cursor,
            "sort": "start_time",
            "order": "desc",
        }
        return self._get_json("/v1/spans", params)
