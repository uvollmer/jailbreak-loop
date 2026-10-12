"""Anthropic provider."""

from __future__ import annotations

import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import Completion, LLMProvider

try:
    from anthropic import APIError, APIStatusError, AsyncAnthropic
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore[assignment]
    APIError = Exception  # type: ignore[assignment,misc]
    APIStatusError = Exception  # type: ignore[assignment,misc]


class AnthropicProvider(LLMProvider):
    kind = "anthropic"

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        if AsyncAnthropic is None:
            raise RuntimeError("anthropic package not installed")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"env var {api_key_env} is not set")
        self._client = AsyncAnthropic(api_key=api_key)

    @retry(
        reraise=True,
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type((APIError, APIStatusError)),
    )
    async def chat(
        self,
        system: str,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> Completion:
        anthropic_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        # Anthropic API rejects `system=None` explicitly — the arg must be omitted
        # entirely when there is no system prompt. Same for stop_sequences.
        kwargs: dict = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop
        resp = await self._client.messages.create(**kwargs)
        parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        text = "".join(parts)
        usage = resp.usage
        return Completion(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            raw={"stop_reason": resp.stop_reason},
        )
