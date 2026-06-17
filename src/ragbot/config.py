from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


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
    rca_threshold: float = float(os.getenv("RCA_THRESHOLD", "0.5"))
    phoenix_project_name: str = os.getenv("PHOENIX_PROJECT_NAME", "test-bot-with-eval")
    phoenix_collector_endpoint: str = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    phoenix_query_endpoint: str = os.getenv("PHOENIX_QUERY_ENDPOINT", "http://localhost:6006")
    phoenix_fetch_output_dir: Path = Path(os.getenv("PHOENIX_FETCH_OUTPUT_DIR", "data/fetch/phoenix"))
    phoenix_fetch_checkpoint_file: Path = Path(
        os.getenv("PHOENIX_FETCH_CHECKPOINT_FILE", "data/fetch/phoenix/checkpoint.json")
    )
    phoenix_fetch_batch_size: int = int(os.getenv("PHOENIX_FETCH_BATCH_SIZE", "200"))
    phoenix_fetch_span_kind: str = os.getenv("PHOENIX_FETCH_SPAN_KIND", "CHAIN")
    batch_evaluation_cron_enabled: bool = os.getenv("BATCH_EVALUATION_CRON_ENABLED", "true").lower() in {
        "true",
        "1",
        "yes",
    }
    batch_evaluation_cron_interval_seconds: int = int(os.getenv("BATCH_EVALUATION_CRON_INTERVAL_SECONDS", "3600"))
    batch_evaluation_cron_limit: int = int(os.getenv("BATCH_EVALUATION_CRON_LIMIT", "1000"))
    batch_evaluation_cron_sync_annotations: bool = os.getenv(
        "BATCH_EVALUATION_CRON_SYNC_ANNOTATIONS", "true"
    ).lower() in {"true", "1", "yes"}
    batch_evaluation_cron_save_annotations: bool = os.getenv(
        "BATCH_EVALUATION_CRON_SAVE_ANNOTATIONS", "true"
    ).lower() in {"true", "1", "yes"}
    batch_evaluation_use_checkpoint: bool = os.getenv(
        "BATCH_EVALUATION_USE_CHECKPOINT", "false"
    ).lower() in {"true", "1", "yes"}
    batch_evaluation_checkpoint_file: Path = Path(
        os.getenv("BATCH_EVALUATION_CHECKPOINT_FILE", "data/batch_evaluation/checkpoint.json")
    )
    server_host: str = os.getenv("RAGBOT_HOST", "127.0.0.1")
    server_port: int = int(os.getenv("RAGBOT_PORT", "8000"))
    server_reload: bool = os.getenv("RAGBOT_RELOAD", "false").lower() in {"true", "1", "yes"}

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.phoenix_fetch_output_dir.mkdir(parents=True, exist_ok=True)
        self.phoenix_fetch_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.batch_evaluation_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
