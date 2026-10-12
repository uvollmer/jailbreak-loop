"""Shared Pydantic types used across the loop."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Op = Literal["mutate", "crossover", "seed"]
Strategy = Literal["single_turn", "scripted_multi_turn", "adaptive_multi_turn"]
TurnDepth = Literal["single", "short", "long"]


def classify_turn_depth(n_turns: int) -> TurnDepth:
    if n_turns <= 1:
        return "single"
    if n_turns <= 3:
        return "short"
    return "long"


class Technique(BaseModel):
    id: str
    name: str
    family: str
    description: str
    exemplars: list[str] = Field(default_factory=list)
    source: str = ""


class Objective(BaseModel):
    id: str
    description: str
    category: str
    judge_rubric_override: str | None = None


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CandidateAttack(BaseModel):
    """A candidate attack the loop wants to run against the target.

    Depending on `strategy`:
    - single_turn:          only `initial_prompt` is used.
    - scripted_multi_turn:  `initial_prompt` is turn 1; `scripted_followups`
                            are turns 2..K.
    - adaptive_multi_turn:  `initial_prompt` is turn 1; the adapter (see
                            Proposer.next_turn) drafts each subsequent user
                            turn given the target's replies + `adaptive_strategy`.
    """

    strategy: Strategy = "single_turn"
    initial_prompt: str
    scripted_followups: list[str] = Field(default_factory=list)
    adaptive_strategy: str = ""
    max_turns: int = 1
    rationale: str = ""
    technique_ids: list[str] = Field(default_factory=list)
    parent_attempt_ids: list[int] = Field(default_factory=list)
    op: Op = "seed"

    def combined_representation(self) -> str:
        """Single string used for embedding / feature extraction.

        Includes every piece of the attack plan that differs meaningfully across
        candidates, so novelty dedup and feature classification see the whole
        strategy, not just turn 1.
        """
        parts: list[str] = [f"strategy={self.strategy}", self.initial_prompt]
        if self.scripted_followups:
            parts.append("followups: " + " || ".join(self.scripted_followups))
        if self.adaptive_strategy:
            parts.append("adaptive: " + self.adaptive_strategy)
        return " || ".join(parts)


# Backward-compat alias — callers importing CandidatePrompt still work.
CandidatePrompt = CandidateAttack


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0


class TargetResponse(BaseModel):
    """Result of the executor playing an attack against the target.

    For a single-turn attempt, `transcript` is length 2 (user, assistant) and
    `text` is that assistant reply. For multi-turn, `transcript` has all turns
    and `text` is the concatenated assistant text for convenience.
    """

    text: str
    model: str
    latency_ms: int
    usage: TokenUsage = Field(default_factory=TokenUsage)
    transcript: list[Turn] = Field(default_factory=list)
    n_turns: int = 1


class JudgeVerdict(BaseModel):
    success: bool
    score: float = Field(ge=0.0, le=10.0)
    rationale: str
    refusal_signals: list[str] = Field(default_factory=list)
    severity: int = Field(default=0, ge=0, le=3)
    success_turn: int | None = None


class AttemptFeatures(BaseModel):
    attack_style: str
    risk_category: str
    turn_depth: TurnDepth = "single"
    technique_family: str | None = None


class Attempt(BaseModel):
    id: int | None = None
    objective_id: str
    op: Op
    strategy: Strategy = "single_turn"
    parent_attempt_ids: list[int] = Field(default_factory=list)
    technique_ids: list[str] = Field(default_factory=list)
    prompt: str  # initial_prompt — kept for legacy readers/queries
    target_model: str
    response: str  # concatenated assistant text — kept for legacy readers
    transcript: list[Turn] = Field(default_factory=list)
    n_turns: int = 1
    verdict: JudgeVerdict
    features: AttemptFeatures
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ArchiveCell(BaseModel):
    cell_key: str
    attempt_id: int
    score: float
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Reflection(BaseModel):
    id: int | None = None
    iter: int
    summary: str
    top_families: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
