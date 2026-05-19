from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True, slots=True)
class Settings:
    project_name: str = "ragbot"
    data_dir: Path = Path(os.getenv("RAGBOT_DATA_DIR", "data"))
    raw_dir: Path = Path(os.getenv("RAGBOT_RAW_DIR", "data/raw"))
    index_dir: Path = Path(os.getenv("RAGBOT_INDEX_DIR", "data/index"))
    phoenix_project_name: str = os.getenv("PHOENIX_PROJECT_NAME", "ragbot")
    phoenix_collector_endpoint: str = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    )
    gemini_model: str = os.getenv("RAGBOT_GEMINI_MODEL", "gemini-1.5-flash")
    gemini_embedding_model: str = os.getenv(
        "RAGBOT_GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"
    )
    max_input_chars: int = int(os.getenv("RAGBOT_MAX_INPUT_CHARS", "6000"))
    retrieval_top_k: int = int(os.getenv("RAGBOT_RETRIEVAL_TOP_K", "4"))

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
