"""End-to-end smoke test with DummyProvider only. No network."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from jailbreak.archive.grid import MAPElitesArchive
from jailbreak.archive.novelty import NoveltyFilter
from jailbreak.config import (
    ArchiveGridConfig,
    Config,
    EthicsConfig,
    LoopConfig,
    NoveltyConfig,
    PathsConfig,
    ProposerBehaviorConfig,
    ProviderConfig,
    ProvidersConfig,
)
from jailbreak.executor import Executor
from jailbreak.judge import Judge
from jailbreak.logging import AuditLog
from jailbreak.models import Objective
from jailbreak.orchestrator import Loop
from jailbreak.proposer import Proposer
from jailbreak.providers.dummy import DummyProvider
from jailbreak.reflector import Reflector
from jailbreak.seed.loader import load_techniques
from jailbreak.storage.db import open_db
from jailbreak.storage.repo import (
    ArchiveRepo,
    AttemptsRepo,
    ObjectivesRepo,
    ProposedObjectivesRepo,
    ReflectionsRepo,
    RunsRepo,
    TechniquesRepo,
)


@pytest.mark.asyncio
async def test_end_to_end_dummy():
    tmp = Path(tempfile.mkdtemp())
    cfg = Config(
        providers=ProvidersConfig(
            target=ProviderConfig(kind="dummy", model="dummy-target"),
            proposer=ProviderConfig(kind="dummy", model="dummy-proposer"),
            judge=ProviderConfig(kind="dummy", model="dummy-judge"),
        ),
        loop=LoopConfig(iterations=20, parallelism=2, checkpoint_every=5, reflect_every=10),
        archive=ArchiveGridConfig(),
        novelty=NoveltyConfig(threshold=0.999),
        proposer=ProposerBehaviorConfig(candidates_per_step=2),
        paths=PathsConfig(
            db=str(tmp / "run.db"),
            reflections=str(tmp / "strategies.md"),
            audit_log=str(tmp / "audit.jsonl"),
        ),
        ethics=EthicsConfig(),
    )
    target = DummyProvider(model="dummy-target")
    proposer_p = DummyProvider(model="dummy-proposer")
    judge_p = DummyProvider(model="dummy-judge")

    conn = await open_db(cfg.paths.db)
    try:
        attempts_repo = AttemptsRepo(conn)
        archive_repo = ArchiveRepo(conn)
        reflections_repo = ReflectionsRepo(conn)
        runs_repo = RunsRepo(conn)
        objectives_repo = ObjectivesRepo(conn)
        techniques_repo = TechniquesRepo(conn)
        proposed_objectives_repo = ProposedObjectivesRepo(conn)

        techniques = load_techniques()
        await techniques_repo.upsert_many(techniques)
        objective = Objective(id="banana", description="banana sanity", category="sanity")
        await objectives_repo.upsert(objective)

        novelty = NoveltyFilter(threshold=cfg.novelty.threshold)
        archive = MAPElitesArchive(cfg.archive, archive_repo)
        adapter_p = DummyProvider(model="dummy-adapter")
        loop = Loop(
            cfg=cfg,
            proposer=Proposer(proposer_p, adapter_provider=adapter_p),
            executor=Executor(target),
            judge=Judge(judge_p),
            reflector=Reflector(proposer_p, cfg.paths.reflections),
            classifier=proposer_p,
            novelty=novelty,
            archive=archive,
            attempts_repo=attempts_repo,
            archive_repo=archive_repo,
            reflections_repo=reflections_repo,
            runs_repo=runs_repo,
            objectives_repo=objectives_repo,
            proposed_objectives_repo=proposed_objectives_repo,
            audit=AuditLog(cfg.paths.audit_log),
            techniques=techniques,
            objectives=[objective],
        )
        await loop.run()

        # 1. Attempts were recorded.
        rows = await attempts_repo.query(limit=100)
        assert len(rows) > 0, "no attempts recorded"

        # 2. At least one success (dummy target replies with 'banana' when asked).
        successes = [a for a in rows if a.verdict.success]
        assert len(successes) > 0, "no successful attempts"

        # 3. Archive has ≥1 cell.
        cells = await archive_repo.all()
        assert len(cells) >= 1

        # 4. Reflector wrote a strategies note.
        assert Path(cfg.paths.reflections).exists()
        assert Path(cfg.paths.reflections).read_text().strip() != ""

        # 5. Audit log has content.
        assert Path(cfg.paths.audit_log).exists()
        assert Path(cfg.paths.audit_log).stat().st_size > 0

        # 6. Reflector produced a proposed_objectives row (dummy emits one).
        proposals = await proposed_objectives_repo.list_all()
        assert len(proposals) >= 1
        assert proposals[0]["id"] == "dummy-followup"
        # sanity category is in the default ethics allowlist → status=pending
        assert proposals[0]["status"] == "pending"

        # 7. Multi-turn code paths exercised: at least one attempt used a
        # non-single_turn strategy and had n_turns > 1.
        strategies_seen = {a.strategy for a in rows}
        assert strategies_seen != {"single_turn"}, "no multi-turn strategies exercised"
        multi_turn = [a for a in rows if a.n_turns > 1]
        assert multi_turn, "no attempts had n_turns > 1"
        # Transcript was persisted and reads back with real turns.
        assert all(len(a.transcript) == 2 * a.n_turns for a in rows)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_novelty_filter_dedups():
    nov = NoveltyFilter(threshold=0.99)
    vec1 = nov.encode("banana banana banana")
    vec2 = nov.encode("banana banana banana")
    assert nov.is_duplicate(vec1, [vec2])


def test_seed_loader():
    techs = load_techniques()
    assert len(techs) > 20
    families = {t.family for t in techs}
    assert "iterative_refinement" in families
    assert "persona" in families
