"""Runtime configuration loaded from environment variables.

Centralizing config here keeps the rest of the codebase free of os.getenv()
calls and makes it trivial to swap models, data paths, or temperature without
touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    data_dir: Path
    cors_origins: list[str]

    @classmethod
    def from_env(cls) -> Settings:
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required. Copy .env.example to .env and set the value."
            )
        return cls(
            anthropic_api_key=key,
            model_name=os.getenv("MODEL_NAME", "claude-sonnet-4-20250514"),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
            data_dir=Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data")),
            cors_origins=[
                o.strip()
                for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000").split(",")
                if o.strip()
            ],
        )


# Lazy singleton — avoids requiring env vars at import time (e.g. during tests
# that mock the LLM).
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
