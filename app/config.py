"""
Central settings, loaded from environment variables (and a local .env file
in development). Nothing here talks to a network or loads a model -- it's
pure configuration, safe to import anywhere without side effects.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- provider API keys (all optional -- a provider with no key is
    # simply skipped by the fallback chain, not a startup error) ---
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # --- router behaviour ---
    # Path to a fine-tuned difficulty-classifier checkpoint. Until this
    # exists (i.e. before the Colab training run has been done and the
    # checkpoint copied back here), DifficultyClassifier transparently
    # falls back to the heuristic scorer in router/heuristic.py -- the
    # whole API is usable end-to-end without it.
    classifier_checkpoint_dir: str | None = Field(default="training/checkpoints/difficulty-classifier")
    classifier_confidence_floor: float = Field(
        default=0.5,
        description="Below this confidence, the router escalates to the next tier up regardless of the predicted "
        "label. Tuned against the heuristic scorer's actual output range (0.55-0.65, capped below 0.7 by design -- "
        "see heuristic.py) rather than a trained classifier's, which typically spans much wider. A floor of 0.6 "
        "sat inside that range and caused near-universal escalation (confirmed via a real benchmark run: 0/30 "
        "requests reached the CHEAP tier). 0.5 keeps the escalation mechanism meaningful without silently "
        "defeating cost savings while running on the heuristic. Re-tune this once a trained classifier (with a "
        "wider, better-calibrated confidence range) is in use.",
    )

    # --- semantic cache ---
    semantic_cache_enabled: bool = True
    semantic_cache_similarity_threshold: float = Field(
        default=0.94,
        description="Cosine similarity above which a cached response is reused instead of calling a model. "
        "Tune this against the benchmark's cache-hit-rate / quality tradeoff, not by guessing.",
    )
    semantic_cache_max_entries: int = 50_000
    embedding_backend: Literal["openai", "sentence-transformers"] = "openai"
    embedding_model: str = "text-embedding-3-small"

    # --- rate limiting / cost tracking ---
    default_rate_limit_per_minute: int = 60
    cost_ledger_path: str = "data/cost_ledger.sqlite3"

    # --- tracing ---
    otel_enabled: bool = False
    otel_exporter_endpoint: str | None = None
    langsmith_enabled: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "llm-router"

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
