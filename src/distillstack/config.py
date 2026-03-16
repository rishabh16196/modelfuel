"""Pydantic-settings based configuration for DistillStack."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DISTILL_",
        env_file_encoding="utf-8",
    )

    teacher_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    vlm_model: str = "deepseek/deepseek-ocr-2"

    litellm_api_key: SecretStr = SecretStr("")

    vlm_confidence_threshold: float = 0.85
    max_chunk_tokens: int = 1024

    output_dir: Path = Path("output")
    log_level: str = "INFO"

    def configure_logging(self) -> None:
        """Apply the configured log level to the root logger."""
        logging.basicConfig(
            level=self.log_level.upper(),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
