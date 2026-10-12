"""Proposer: generate candidate attacks and, in adaptive mode, next-turn messages."""

from __future__ import annotations

import json
import re
from pathlib import Path

from jinja2 import Template

from .models import Attempt, CandidateAttack, Objective, Op, Strategy, Technique, Turn
from .providers.base import LLMProvider
from .providers.dummy import strip_code_fence

_PROMPT_DIR = Path(__file__).parent / "prompts"
_PROPOSER_TEMPLATE = _PROMPT_DIR / "proposer.jinja"
_ADAPTIVE_TEMPLATE = _PROMPT_DIR / "adaptive_turn.jinja"


class Proposer:
    def __init__(
        self,
        provider: LLMProvider,
        adapter_provider: LLMProvider | None = None,
        template_path: Path | None = None,
        adaptive_template_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.adapter_provider = adapter_provider or provider
        self.template = Template((template_path or _PROPOSER_TEMPLATE).read_text())
        self.adaptive_template = Template(
            (adaptive_template_path or _ADAPTIVE_TEMPLATE).read_text()
        )

    async def propose(
        self,
        objective: Objective,
        op: Op,
        parents: list[Attempt],
        seed_techniques: list[Technique],
        strategies: str,
        n: int,
        max_turns: int,
    ) -> list[CandidateAttack]:
        prompt = self.template.render(
            objective=objective,
            op=op,
            parents=parents,
            seed_techniques=seed_techniques,
            strategies=strategies,
            n=n,
            max_turns=max_turns,
        )
        comp = await self.provider.chat(
            system=prompt,
            messages=[{"role": "user", "content": "Produce the JSON array now."}],
            max_tokens=3000,
        )
        return _parse_candidates(comp.text, op, parents, seed_techniques, max_turns)

    async def next_turn(
        self,
        objective: Objective,
        transcript: list[Turn],
        adaptive_strategy: str,
    ) -> str:
        system = self.adaptive_template.render(
            objective=objective,
            transcript=transcript,
            adaptive_strategy=adaptive_strategy,
        )
        comp = await self.adapter_provider.chat(
            system=system,
            messages=[{"role": "user", "content": "Draft the next user message now."}],
            max_tokens=500,
        )
        return comp.text.strip()


_STRATEGY_VALUES: set[Strategy] = {"single_turn", "scripted_multi_turn", "adaptive_multi_turn"}


def _parse_candidates(
    text: str,
    op: Op,
    parents: list[Attempt],
    seeds: list[Technique],
    max_turns: int,
) -> list[CandidateAttack]:
    text = strip_code_fence(text)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[CandidateAttack] = []
    parent_ids = [p.id for p in parents if p.id is not None]
    seed_ids = [s.id for s in seeds]
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Back-compat: proposer may still emit the old "prompt" field for single-turn.
        initial = str(item.get("initial_prompt") or item.get("prompt") or "").strip()
        if not initial:
            continue
        strategy = str(item.get("strategy") or "single_turn").strip()
        if strategy not in _STRATEGY_VALUES:
            strategy = "single_turn"
        followups_raw = item.get("scripted_followups") or []
        followups = [str(x).strip() for x in followups_raw if str(x).strip()]
        adaptive_strategy = str(item.get("adaptive_strategy") or "").strip()
        item_max_turns = int(item.get("max_turns") or 0) or max_turns
        item_max_turns = max(1, min(item_max_turns, max_turns))
        if strategy == "single_turn":
            followups = []
            adaptive_strategy = ""
            item_max_turns = 1
        out.append(
            CandidateAttack(
                strategy=strategy,  # type: ignore[arg-type]
                initial_prompt=initial,
                scripted_followups=followups,
                adaptive_strategy=adaptive_strategy,
                max_turns=item_max_turns,
                rationale=str(item.get("rationale", "")),
                technique_ids=list(item.get("technique_ids") or seed_ids),
                parent_attempt_ids=parent_ids,
                op=op,
            )
        )
    return out
