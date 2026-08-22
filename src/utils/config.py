"""Process configuration, loaded once from the environment.

Nothing in this module reads the wall clock or contacts a network. The
`AS_OF` value lives here only as a raw string; `datastore.clock` is the
single place that parses and exposes it (ARCHITECTURE.md D6).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed view of the environment. Missing required keys fail at startup."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- model provider -------------------------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    model_cheap: str = ""
    model_strong: str = ""
    model_fallback: str = ""

    # --- embeddings (build time only) ------------------------------------
    embedding_model: str = ""
    embedding_dim: int | None = None

    gemini_api_key: str = ""

    # --- storage ---------------------------------------------------------
    chroma_persist_dir: Path = Field(default=Path("data/index"))
    chroma_api_key: str = ""
    sqlite_path: Path = Field(default=Path("data/parcelpilot.db"))

    # --- clock (raw; parsed by datastore.clock) --------------------------
    as_of: str = ""

    # --- runtime ---------------------------------------------------------
    log_level: str = "INFO"
    repair_budget: int = 2

    @property
    def chroma_dir(self) -> Path:
        """Absolute path to the committed Chroma store."""
        return self._absolute(self.chroma_persist_dir)

    @property
    def db_path(self) -> Path:
        """Absolute path to the committed SQLite build."""
        return self._absolute(self.sqlite_path)

    @staticmethod
    def _absolute(candidate: Path) -> Path:
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Import this rather than instantiating Settings."""
    return Settings()
