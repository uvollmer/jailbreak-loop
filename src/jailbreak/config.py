"""Config loading. TOML on disk → pydantic models."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    kind: str  # "anthropic" | "openai" | "openai_compatible" | "dummy"
    model: str = ""
    api_key_env: str | None = None
    base_url: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7


class ProvidersConfig(BaseModel):
    target: ProviderConfig
    proposer: ProviderConfig
    judge: ProviderConfig
    # Optional: used for drafting adaptive multi-turn follow-up messages.
    # Falls back to the proposer provider if omitted.
    adapter: ProviderConfig | None = None


class LoopConfig(BaseModel):
    iterations: int = 100
    parallelism: int = 4
    checkpoint_every: int = 25
    reflect_every: int = 50
    seed: int = 42
    max_turns: int = 5


class ArchiveGridConfig(BaseModel):
    dimensions: list[str] = Field(
        default_factory=lambda: ["attack_style", "risk_category", "turn_depth"]
    )
    attack_styles: list[str] = Field(
        default_factory=lambda: [
            "direct",
            "roleplay",
            "hypothetical",
            "encoded",
            "multi_turn",
            "authority",
            "obfuscated",
            "meta",
        ]
    )
    risk_categories: list[str] = Field(
        default_factory=lambda: [
            "policy_violation",
            "privacy",
            "misinformation",
            "cyber",
            "harassment",
            "self_harm",
            "cbrn",
            "other",
        ]
    )
    turn_depths: list[str] = Field(default_factory=lambda: ["single", "short", "long"])


class NoveltyConfig(BaseModel):
    threshold: float = 0.92
    embedding_model: str = "BAAI/bge-small-en-v1.5"


class StrategyMixConfig(BaseModel):
    single_turn: float = 0.4
    scripted_multi_turn: float = 0.2
    adaptive_multi_turn: float = 0.4


class ProposerBehaviorConfig(BaseModel):
    candidates_per_step: int = 4
    crossover_prob: float = 0.35
    seed_apply_prob: float = 0.25
    strategy_mix: StrategyMixConfig = Field(default_factory=StrategyMixConfig)


class PathsConfig(BaseModel):
    db: str = "./run.db"
    reflections: str = "./strategies.md"
    audit_log: str = "./audit.jsonl"


class EthicsConfig(BaseModel):
    allow_categories: list[str] = Field(
        default_factory=lambda: [
            "policy_violation",
            "privacy",
            "misinformation",
            "cyber",
            "harassment",
            "self_harm",
            "cbrn",
            "other",
            "sanity",
        ]
    )


class Config(BaseModel):
    providers: ProvidersConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    archive: ArchiveGridConfig = Field(default_factory=ArchiveGridConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    proposer: ProposerBehaviorConfig = Field(default_factory=ProposerBehaviorConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ethics: EthicsConfig = Field(default_factory=EthicsConfig)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    grid_raw = raw.get("archive", {}).get("grid", {}) if isinstance(raw.get("archive"), dict) else {}
    archive_cfg = ArchiveGridConfig(**grid_raw) if grid_raw else ArchiveGridConfig()
    return Config(
        providers=ProvidersConfig(**raw["providers"]),
        loop=LoopConfig(**raw.get("loop", {})),
        archive=archive_cfg,
        novelty=NoveltyConfig(**raw.get("novelty", {})),
        proposer=ProposerBehaviorConfig(**raw.get("proposer", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        ethics=EthicsConfig(**raw.get("ethics", {})),
    )
