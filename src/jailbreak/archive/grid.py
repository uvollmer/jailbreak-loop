"""MAP-Elites archive over a configurable feature grid."""

from __future__ import annotations

from ..config import ArchiveGridConfig
from ..models import AttemptFeatures
from ..storage.repo import ArchiveRepo


def cell_key(features: AttemptFeatures, dims: list[str]) -> str:
    parts = []
    for d in dims:
        v = getattr(features, d, None)
        parts.append(str(v) if v is not None else "unknown")
    return "|".join(parts)


class MAPElitesArchive:
    def __init__(self, cfg: ArchiveGridConfig, repo: ArchiveRepo) -> None:
        self.cfg = cfg
        self.repo = repo

    def key(self, features: AttemptFeatures) -> str:
        return cell_key(features, self.cfg.dimensions)

    async def try_insert(
        self, features: AttemptFeatures, attempt_id: int, score: float
    ) -> bool:
        """Return True if accepted (cell empty or new score > incumbent)."""
        k = self.key(features)
        current = await self.repo.get(k)
        if current is None or score > current[1]:
            await self.repo.upsert(k, attempt_id, score)
            return True
        return False

    async def sample_parent_ids(self, k: int = 2) -> list[int]:
        rows = await self.repo.all()
        if not rows:
            return []
        import random

        picked = random.sample(rows, min(k, len(rows)))
        return [p[1] for p in picked]

    async def coverage(self) -> tuple[int, int]:
        """Return (filled_cells, total_grid_size)."""
        rows = await self.repo.all()
        total = 1
        for d in self.cfg.dimensions:
            vals = _dim_values(self.cfg, d)
            total *= max(1, len(vals))
        return (len(rows), total)


def _dim_values(cfg: ArchiveGridConfig, dim: str) -> list[str]:
    if dim == "attack_style":
        return cfg.attack_styles
    if dim == "risk_category":
        return cfg.risk_categories
    if dim == "turn_depth":
        return cfg.turn_depths
    return []
