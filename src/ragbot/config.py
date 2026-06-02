from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

load_dotenv()


def normalize_phoenix_collector_endpoint(collector_endpoint: str) -> str:
    parsed = urlparse(collector_endpoint)
    if not parsed.scheme or not parsed.netloc:
        return collector_endpoint
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1/traces"
    return urlunparse(parsed._replace(path=path))


def derive_phoenix_pull_base_url(collector_endpoint: str) -> str:
    parsed = urlparse(collector_endpoint)
    if not parsed.scheme or not parsed.netloc:
        return collector_endpoint.rstrip("/")
    return urlunparse(parsed._replace(path="", params="", query="", fragment="")).rstrip("/")


@dataclass(frozen=True, slots=True)
class Settings:
    project_name: str = "ragbot"
    environment: str = os.getenv("RAGBOT_ENVIRONMENT", "dev")
    app_version: str = os.getenv("RAGBOT_APP_VERSION", "v1")
    use_case: str = os.getenv("RAGBOT_USE_CASE", "rag_chatbot")
    data_dir: Path = Path(os.getenv("RAGBOT_DATA_DIR", "data"))
    raw_dir: Path = Path(os.getenv("RAGBOT_RAW_DIR", "data/raw"))
    index_dir: Path = Path(os.getenv("RAGBOT_INDEX_DIR", "data/index"))
    embedding_provider: str = os.getenv("RAGBOT_EMBEDDING_PROVIDER", "sentence-transformer")
    sentence_transformer_model: str = os.getenv(
        "RAGBOT_SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    gemini_model: str = os.getenv("RAGBOT_GEMINI_MODEL", "gemini-2.5-flash")
    gemini_embedding_model: str = os.getenv(
        "RAGBOT_GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"
    )
    max_input_chars: int = int(os.getenv("RAGBOT_MAX_INPUT_CHARS", "6000"))
    retrieval_top_k: int = int(os.getenv("RAGBOT_RETRIEVAL_TOP_K", "4"))
    phoenix_project_name: str = os.getenv("PHOENIX_PROJECT_NAME", "test-bot-with-eval")
    phoenix_collector_endpoint: str = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    phoenix_pull_base_url: str = os.getenv(
        "RAGBOT_PHOENIX_PULL_BASE_URL",
        derive_phoenix_pull_base_url(os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")),
    )
    rca_pull_save_dir: Path = Path(os.getenv("RAGBOT_RCA_PULL_SAVE_DIR", "src/ragbot/rca/pulls"))
    server_host: str = os.getenv("RAGBOT_HOST", "127.0.0.1")
    server_port: int = int(os.getenv("RAGBOT_PORT", "8000"))
    server_reload: bool = os.getenv("RAGBOT_RELOAD", "false").lower() in {"true", "1", "yes"}

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.rca_pull_save_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
