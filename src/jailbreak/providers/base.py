"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)


class LLMProvider(ABC):
    kind: str = "base"

    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        self.max_tokens: int = kwargs.get("max_tokens", 1024)
        self.temperature: float = kwargs.get("temperature", 0.7)

    @abstractmethod
    async def chat(
        self,
        system: str,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> Completion:
        """Send a chat request and return a Completion.

        `messages` follows OpenAI shape: [{"role": "user"|"assistant", "content": str}, ...].
        Providers translate as needed.
        """
