"""OpenAI provider (also handles OpenAI-compatible base_url for Ollama/vLLM)."""

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
    from openai import APIError, APIStatusError, AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]
    APIError = Exception  # type: ignore[assignment,misc]
    APIStatusError = Exception  # type: ignore[assignment,misc]


class OpenAIProvider(LLMProvider):
    kind = "openai"

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        if AsyncOpenAI is None:
            raise RuntimeError("openai package not installed")
        api_key = os.environ.get(api_key_env, "not-required")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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
        oai_messages: list[dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            stop=stop or None,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        return Completion(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            raw={"finish_reason": choice.finish_reason},
        )


class OpenAICompatibleProvider(OpenAIProvider):
    """Same as OpenAI but requires a base_url (Ollama, vLLM, LM Studio, etc.)."""

    kind = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key_env: str | None = None, **kwargs) -> None:
        super().__init__(
            model,
            api_key_env=api_key_env or "OPENAI_COMPATIBLE_API_KEY",
            base_url=base_url,
            **kwargs,
        )
