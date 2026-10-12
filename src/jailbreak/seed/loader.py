"""Load vendored navigator snapshot + curated techniques.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..models import Technique

DATA_DIR = Path(__file__).parent / "data"


def load_navigator() -> dict:
    with (DATA_DIR / "navigator.snapshot.json").open() as f:
        return json.load(f)


def load_techniques() -> list[Technique]:
    """Load curated techniques.yaml PLUS convert navigator entries to Techniques."""
    with (DATA_DIR / "techniques.yaml").open() as f:
        raw = yaml.safe_load(f)
    curated = [
        Technique(
            id=t["id"],
            name=t["name"],
            family=t["family"],
            description=t["description"].strip(),
            exemplars=list(t.get("exemplars") or []),
            source=t.get("source", ""),
        )
        for t in raw
    ]
    # Also index navigator techniques (with a nav_ prefix on ids to avoid collisions).
    nav = load_navigator()
    nav_techs = [
        Technique(
            id=f"nav_{t['id']}",
            name=t["name"],
            family=t["tactic"],
            description=t["description"],
            exemplars=[],
            source="redteams.ai/navigator (snapshot)",
        )
        for t in nav.get("techniques", [])
    ]
    return curated + nav_techs
