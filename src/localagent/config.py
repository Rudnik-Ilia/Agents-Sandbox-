"""Central configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings sourced from environment variables or the `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = "http://10.100.102.10:11434"
    llm_provider: str = "ollama"
    llm_model: str = "gemma4:e4b"
    embed_model: str = "mxbai-embed-large:latest"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_site_url: str = ""
    openrouter_app_name: str = "LocalAgent"

    llm_temperature: float = 0.2
    llm_request_timeout: int = 600
    llm_keep_alive: str = "30m"
    llm_num_ctx: int = 8192
    llm_think: bool = False

    # Reliability
    fallback_model: str = "llama3.1:8b"
    llm_max_retries: int = 2

    # Safety: ask for confirmation before running dangerous tools
    require_tool_approval: bool = True

    # Observability (LangSmith). Set langsmith_tracing=true and an API key to enable.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "localagent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    rag_top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120

    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fusion_candidates: int = 12

    log_level: str = "INFO"
    log_dir: str = "logs"

    @property
    def corpus_dir(self) -> Path:
        """Directory holding the sample RAG corpus."""
        return PROJECT_ROOT / "data" / "corpus"

    @property
    def skills_dir(self) -> Path:
        """Directory holding opt-in markdown skills."""
        return PROJECT_ROOT / "skills"

    @property
    def rules_dir(self) -> Path:
        """Directory holding always-on markdown rules."""
        return PROJECT_ROOT / "rules"

    @property
    def log_path(self) -> Path:
        """Absolute path of the logs directory."""
        return PROJECT_ROOT / self.log_dir

    @property
    def memory_db(self) -> Path:
        """SQLite file backing persistent chat memory."""
        return PROJECT_ROOT / "memory.sqlite3"


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance."""
    return Settings()
