"""Provider factory."""

from __future__ import annotations

from ..config import ProviderConfig
from .anthropic import AnthropicProvider
from .base import LLMProvider
from .dummy import DummyProvider
from .openai import OpenAICompatibleProvider, OpenAIProvider


def build_provider(cfg: ProviderConfig) -> LLMProvider:
    kwargs = {
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
    }
    if cfg.kind == "dummy":
        return DummyProvider(model=cfg.model or "dummy-1", **kwargs)
    if cfg.kind == "anthropic":
        return AnthropicProvider(
            model=cfg.model,
            api_key_env=cfg.api_key_env or "ANTHROPIC_API_KEY",
            **kwargs,
        )
    if cfg.kind == "openai":
        return OpenAIProvider(
            model=cfg.model,
            api_key_env=cfg.api_key_env or "OPENAI_API_KEY",
            **kwargs,
        )
    if cfg.kind == "openai_compatible":
        if not cfg.base_url:
            raise ValueError("openai_compatible provider requires base_url")
        return OpenAICompatibleProvider(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            **kwargs,
        )
    raise ValueError(f"unknown provider kind: {cfg.kind}")
