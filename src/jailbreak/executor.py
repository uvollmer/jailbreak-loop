"""Send a candidate attack to the target model. Supports single- and multi-turn."""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable

from .models import CandidateAttack, TargetResponse, TokenUsage, Turn
from .providers.base import LLMProvider

# Heuristic refusal detector — used to early-exit adaptive conversations when
# the target visibly refuses, so we don't burn K turns on a locked-in refusal.
_REFUSAL_RE = re.compile(
    r"\b(i can(?:not|'t)|i(?:'m| am) (?:not able|unable)|"
    r"against my (?:guidelines|policies)|i won't|i will not|"
    r"i(?:'m| am) sorry, but i cannot)\b",
    re.IGNORECASE,
)


class Executor:
    def __init__(self, target: LLMProvider) -> None:
        self.target = target

    async def execute(
        self,
        candidate: CandidateAttack,
        max_turns: int,
        next_turn_fn: Callable[[list[Turn]], Awaitable[str]] | None = None,
    ) -> TargetResponse:
        """Play the candidate attack against the target.

        `next_turn_fn` is required for adaptive_multi_turn — it receives the
        transcript-so-far (ending with the target's most recent reply) and
        returns the next user message. Pass None for single_turn / scripted.
        """
        started = time.perf_counter()
        transcript: list[Turn] = []
        model_name = ""
        total_input = 0
        total_output = 0

        # Effective per-attack turn cap.
        if candidate.strategy == "single_turn":
            cap = 1
        else:
            cap = max(1, min(max_turns, candidate.max_turns or max_turns))

        # Turn 1 — always the initial_prompt.
        transcript.append(Turn(role="user", content=candidate.initial_prompt))
        comp = await self.target.chat(
            system="",
            messages=[{"role": t.role, "content": t.content} for t in transcript],
        )
        model_name = comp.model
        total_input += comp.input_tokens
        total_output += comp.output_tokens
        transcript.append(Turn(role="assistant", content=comp.text))

        turn_idx = 1
        while turn_idx < cap:
            next_msg = await self._draft_next_user_turn(
                candidate, transcript, turn_idx, next_turn_fn
            )
            if not next_msg:
                break
            # If the adapter itself refused (drafted a refusal instead of an
            # attacker-style user turn), don't inject its refusal as the "user"
            # turn — end the attack here and let the judge score what we have.
            if candidate.strategy == "adaptive_multi_turn" and _looks_like_refusal(next_msg):
                break
            transcript.append(Turn(role="user", content=next_msg))
            comp = await self.target.chat(
                system="",
                messages=[{"role": t.role, "content": t.content} for t in transcript],
            )
            total_input += comp.input_tokens
            total_output += comp.output_tokens
            transcript.append(Turn(role="assistant", content=comp.text))
            turn_idx += 1
            # Early exit if target has cleanly refused twice in a row.
            if _looks_like_refusal(transcript[-1].content) and turn_idx >= 2:
                if _looks_like_refusal(transcript[-3].content):
                    break

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        assistant_texts = [t.content for t in transcript if t.role == "assistant"]
        return TargetResponse(
            text="\n\n---\n\n".join(assistant_texts),
            model=model_name,
            latency_ms=elapsed_ms,
            usage=TokenUsage(input=total_input, output=total_output),
            transcript=transcript,
            n_turns=turn_idx,
        )

    async def _draft_next_user_turn(
        self,
        candidate: CandidateAttack,
        transcript: list[Turn],
        turn_idx: int,
        next_turn_fn: Callable[[list[Turn]], Awaitable[str]] | None,
    ) -> str:
        if candidate.strategy == "scripted_multi_turn":
            # scripted_followups is 0-indexed for turns 2..K.
            i = turn_idx - 1
            if i < len(candidate.scripted_followups):
                return candidate.scripted_followups[i].strip()
            return ""
        if candidate.strategy == "adaptive_multi_turn":
            if next_turn_fn is None:
                return ""
            return (await next_turn_fn(transcript)).strip()
        return ""


def _looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))
