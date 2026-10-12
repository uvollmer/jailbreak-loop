"""Load user objectives from a YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import EthicsConfig
from .models import Objective


def load_objectives(path: str | Path, ethics: EthicsConfig | None = None) -> list[Objective]:
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or []
    objectives = [Objective(**item) for item in raw]
    if ethics is not None:
        allowed = set(ethics.allow_categories)
        bad = [o for o in objectives if o.category not in allowed]
        if bad:
            raise ValueError(
                "Objective categories not in ethics.allow_categories: "
                + ", ".join(f"{o.id}({o.category})" for o in bad)
            )
    return objectives
