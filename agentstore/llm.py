"""Anthropic Claude factory.

We isolate model construction in one place so every agent gets the same
configuration (model, temperature, max_tokens) and tests can monkey-patch
this single function instead of every call site.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from .config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Return a configured Claude chat model. Cached per-process."""
    s = get_settings()
    return ChatAnthropic(
        model=s.model_name,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        api_key=s.anthropic_api_key,
        timeout=60,
        max_retries=2,
    )
