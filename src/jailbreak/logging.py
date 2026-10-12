"""Rich console + append-only JSONL audit sink."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

console = Console()


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def banner(self, message: str) -> None:
        self._write(
            {
                "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "kind": "banner",
                "message": message,
            }
        )

    def llm_call(
        self,
        role: str,
        model: str,
        system: str,
        messages: list[dict],
        response_text: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        payload = system + "\n" + "\n".join(m.get("content", "") for m in messages)
        self._write(
            {
                "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "kind": "llm_call",
                "role": role,
                "model": model,
                "prompt_hash": hashlib.sha256(payload.encode()).hexdigest(),
                "response_hash": hashlib.sha256(response_text.encode()).hexdigest(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def attempt(self, attempt_id: int, objective_id: str, success: bool, score: float) -> None:
        self._write(
            {
                "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "kind": "attempt",
                "attempt_id": attempt_id,
                "objective_id": objective_id,
                "success": success,
                "score": score,
            }
        )
