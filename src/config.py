"""Process configuration, loaded once from the environment.

Two properties matter here more than completeness.

**Slugs live in config, never in code.** Two model slugs died in a single
afternoon while the data pack was being read: an OpenRouter `:free` endpoint
was delisted and `gemini-2.5-flash` began returning 404 for new accounts. The
defaults below are the ones verified working on 2026-08-22, and every one is
overridable without touching a module.

**Misconfiguration fails at import, not mid-demo.** Selecting a provider
without its key, or Chroma Cloud without a token, raises here rather than
surfacing as a 401 five tool calls into a conversation.

Nothing in this module reads the wall clock or contacts a network. `AS_OF` is
held as a raw string; `src.clock` is the only place that parses it (D6).
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

ProviderName = Literal["gemini", "openrouter"]
VectorStoreName = Literal["chroma_cloud", "chroma_local"]

PROVIDERS: Final[tuple[str, ...]] = ("gemini", "openrouter")
VECTOR_STORES: Final[tuple[str, ...]] = ("chroma_cloud", "chroma_local")

#: Chroma allows alphanumerics, underscore and hyphen, and requires the first
#: and last character to be alphanumeric.
_ILLEGAL_IN_COLLECTION: Final = re.compile(r"[^A-Za-z0-9_-]+")


class ConfigError(RuntimeError):
    """Configuration is missing or incoherent. Never recoverable at runtime."""


@dataclass(frozen=True, slots=True)
class ChatConfig:
    """Everything a chat provider needs. Both providers speak the OpenAI wire format."""

    provider: str
    api_key: str
    base_url: str
    cheap_model: str
    strong_model: str
    #: Applied when a caller does not set max_tokens. OpenRouter reserves the
    #: requested budget against the account balance before running, so an
    #: uncapped request 402s on a low balance even when the reply is short.
    max_output_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Embedding provider plus the identity that namespaces the vector collection."""

    provider: str
    api_key: str
    base_url: str
    model: str
    dimensions: int

    @property
    def identity(self) -> str:
        """`provider/model/dim`. Changing any part invalidates stored vectors."""
        return f"{self.provider}/{self.model}/{self.dimensions}"

    @property
    def collection_name(self) -> str:
        """Chroma collection for this exact embedding identity (D20).

        Namespacing by identity is what stops a provider switch from silently
        comparing vectors that were never in the same space.
        """
        slug = _ILLEGAL_IN_COLLECTION.sub("_", f"{self.provider}_{self.model}")
        return f"pp_clauses_{slug}_{self.dimensions}".strip("_")


class Settings(BaseSettings):
    """Typed view of the environment."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- provider selection (D9a) ---------------------------------------
    llm_provider: str = "gemini"
    embedding_provider: str = "gemini"

    # --- Gemini: primary for dev, tests and the demo ---------------------
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model_cheap: str = "gemini-3.5-flash-lite"
    gemini_model_strong: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dim: int = 1536

    # --- OpenRouter: implemented, switchable, deliberately unfunded ------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model_cheap: str = "google/gemini-2.5-flash-lite"
    openrouter_model_strong: str = "google/gemini-2.5-flash"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    openrouter_embedding_dim: int = 1536

    # --- storage ---------------------------------------------------------
    vector_store: str = "chroma_cloud"
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = "parcelpilot"
    chroma_persist_dir: Path = Field(default=Path("data/index"))
    sqlite_path: Path = Field(default=Path("data/parcelpilot.db"))
    #: Sessions, the action log and run events. Separate from `sqlite_path`,
    #: which is committed, read-only at runtime and rebuilt rather than
    #: migrated - the next build would delete anything runtime wrote there.
    runtime_path: Path = Field(default=Path("data/runtime.db"))

    # --- clock (raw; parsed only by src.clock) ---------------------------
    as_of: str = ""

    # --- runtime ---------------------------------------------------------
    api_base_url: str = "http://127.0.0.1:8000"
    session_secret: str = ""
    log_level: str = "INFO"
    repair_budget: int = 2
    max_output_tokens: int = 4096
    request_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        self._check_choice("LLM_PROVIDER", self.llm_provider, PROVIDERS)
        self._check_choice("EMBEDDING_PROVIDER", self.embedding_provider, PROVIDERS)
        self._check_choice("VECTOR_STORE", self.vector_store, VECTOR_STORES)

        # Only a *selected* provider must be configured. OpenRouter stays
        # implemented and unfunded until someone chooses it.
        for provider in {self.llm_provider, self.embedding_provider}:
            if not getattr(self, f"{provider}_api_key"):
                raise ConfigError(
                    f"{provider.upper()}_API_KEY is required because {provider!r} is selected"
                )

        if self.vector_store == "chroma_cloud" and not self.chroma_api_key:
            raise ConfigError("CHROMA_API_KEY is required when VECTOR_STORE=chroma_cloud")

        if getattr(self, f"{self.embedding_provider}_embedding_dim") <= 0:
            raise ConfigError(
                f"{self.embedding_provider.upper()}_EMBEDDING_DIM must be a positive dimension"
            )

        if not self.session_secret:
            # A shipped default would be a forgeable session token. Random
            # per-process means sessions do not survive a restart, which is
            # the correct trade for a mocked login.
            object.__setattr__(self, "session_secret", secrets.token_urlsafe(32))

        return self

    @staticmethod
    def _check_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
        if value not in allowed:
            raise ConfigError(f"{name}={value!r} is not one of {list(allowed)}")

    def chat_config(self) -> ChatConfig:
        p = self.llm_provider
        return ChatConfig(
            provider=p,
            api_key=getattr(self, f"{p}_api_key"),
            base_url=getattr(self, f"{p}_base_url"),
            cheap_model=getattr(self, f"{p}_model_cheap"),
            strong_model=getattr(self, f"{p}_model_strong"),
            max_output_tokens=self.max_output_tokens,
        )

    def embedding_config(self) -> EmbeddingConfig:
        p = self.embedding_provider
        return EmbeddingConfig(
            provider=p,
            api_key=getattr(self, f"{p}_api_key"),
            base_url=getattr(self, f"{p}_base_url"),
            model=getattr(self, f"{p}_embedding_model"),
            dimensions=getattr(self, f"{p}_embedding_dim"),
        )

    @property
    def chroma_dir(self) -> Path:
        return self._absolute(self.chroma_persist_dir)

    @property
    def db_path(self) -> Path:
        return self._absolute(self.sqlite_path)

    @property
    def runtime_db_path(self) -> Path:
        return self._absolute(self.runtime_path)

    @staticmethod
    def _absolute(candidate: Path) -> Path:
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Import this rather than instantiating Settings."""
    try:
        return Settings()
    except ValidationError as exc:  # pragma: no cover - shape depends on pydantic
        raise ConfigError(f"invalid configuration: {exc}") from exc
