"""LLM-as-judge: score whether the target complied with the objective across a transcript."""

from __future__ import annotations

import json
import re
from pathlib import Path

from jinja2 import Template

from .models import JudgeVerdict, Objective, Turn
from .providers.base import LLMProvider
from .providers.dummy import strip_code_fence

_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "judge.jinja"


class Judge:
    def __init__(self, provider: LLMProvider, template_path: Path | None = None) -> None:
        self.provider = provider
        path = template_path or _TEMPLATE_PATH
        self.template = Template(path.read_text())

    async def judge(
        self,
        objective: Objective,
        transcript: list[Turn],
    ) -> JudgeVerdict:
        rendered = self.template.render(objective=objective, transcript=transcript)
        comp = await self.provider.chat(
            system=rendered,
            messages=[{"role": "user", "content": "Return the verdict JSON now."}],
            temperature=0.0,
            max_tokens=500,
        )
        return _parse_verdict(comp.text)


def _parse_verdict(text: str) -> JudgeVerdict:
    text = strip_code_fence(text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return JudgeVerdict(success=False, score=0.0, rationale="unparseable judge output")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return JudgeVerdict(success=False, score=0.0, rationale="invalid judge JSON")
    success_turn = data.get("success_turn")
    if success_turn is not None:
        try:
            success_turn = int(success_turn)
        except (TypeError, ValueError):
            success_turn = None
    return JudgeVerdict(
        success=bool(data.get("success", False)),
        score=float(data.get("score", 0.0)),
        rationale=str(data.get("rationale", "")),
        refusal_signals=[str(x) for x in (data.get("refusal_signals") or [])],
        severity=int(data.get("severity", 0)),
        success_turn=success_turn,
    )
