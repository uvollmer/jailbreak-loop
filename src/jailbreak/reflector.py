"""Reflector: distill lessons from recent successes into a strategies note + propose objectives."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from .models import Attempt, Reflection
from .providers.base import LLMProvider
from .providers.dummy import strip_code_fence

_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "reflector.jinja"


@dataclass
class ProposedObjective:
    id: str
    description: str
    category: str
    rationale: str
    judge_rubric_override: str | None = None


@dataclass
class ReflectionOutput:
    reflection: Reflection
    proposed_objectives: list[ProposedObjective]


class Reflector:
    def __init__(
        self,
        provider: LLMProvider,
        strategies_path: str | Path,
        template_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.strategies_path = Path(strategies_path)
        path = template_path or _TEMPLATE_PATH
        self.template = Template(path.read_text())

    def read_strategies(self) -> str:
        if self.strategies_path.exists():
            return self.strategies_path.read_text()
        return ""

    def write_strategies(self, note: str) -> None:
        self.strategies_path.parent.mkdir(parents=True, exist_ok=True)
        self.strategies_path.write_text(note)

    async def reflect(
        self,
        iter_num: int,
        recent: list[Attempt],
        filled: int,
        total_grid: int,
        loaded_objectives: list[str],
        allow_categories: list[str],
    ) -> ReflectionOutput:
        style_counts = dict(Counter(a.features.attack_style for a in recent))
        previous = self.read_strategies()
        system = self.template.render(
            iter=iter_num,
            recent=recent,
            filled=filled,
            total_grid=total_grid,
            style_counts=style_counts,
            previous=previous,
            loaded_objectives=loaded_objectives,
            allow_categories=allow_categories,
        )
        comp = await self.provider.chat(
            system=system,
            messages=[{"role": "user", "content": "Return the JSON now."}],
            max_tokens=1500,
        )
        note, proposals = _parse_reflector_output(comp.text)
        if note:
            self.write_strategies(note)
        family_counts = Counter(t for a in recent for t in a.technique_ids)
        top_families = [f for f, _ in family_counts.most_common(5)]
        reflection = Reflection(iter=iter_num, summary=note, top_families=top_families)
        return ReflectionOutput(reflection=reflection, proposed_objectives=proposals)


def _parse_reflector_output(text: str) -> tuple[str, list[ProposedObjective]]:
    text = strip_code_fence(text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return (text.strip(), [])
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return (text.strip(), [])
    note = str(data.get("strategies_note") or "").strip()
    proposals: list[ProposedObjective] = []
    for item in data.get("proposed_objectives") or []:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id") or "").strip()
        desc = str(item.get("description") or "").strip()
        cat = str(item.get("category") or "").strip()
        rat = str(item.get("rationale") or "").strip()
        rubric = item.get("judge_rubric_override")
        rubric = str(rubric).strip() if rubric else None
        if not (pid and desc and cat and rat):
            continue
        proposals.append(
            ProposedObjective(
                id=pid,
                description=desc,
                category=cat,
                rationale=rat,
                judge_rubric_override=rubric,
            )
        )
    return (note, proposals)
