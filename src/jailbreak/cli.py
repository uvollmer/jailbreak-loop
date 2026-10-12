"""jb CLI."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import typer

from .archive.grid import MAPElitesArchive
from .archive.novelty import NoveltyFilter
from .config import load_config
from .executor import Executor
from .judge import Judge
from .logging import AuditLog, console
from .objectives import load_objectives
from .orchestrator import Loop
from .proposer import Proposer
from .providers.registry import build_provider
from .reflector import Reflector
from .seed.loader import load_techniques
from .storage.db import open_db
from .storage.repo import (
    ArchiveRepo,
    AttemptsRepo,
    ObjectivesRepo,
    ProposedObjectivesRepo,
    ReflectionsRepo,
    RunsRepo,
    TechniquesRepo,
)

app = typer.Typer(help="Autonomous jailbreak-research loop — authorized use only.")

_PACKAGE_ROOT = Path(__file__).parent
_REPO_EXAMPLES = _PACKAGE_ROOT.parent.parent / "examples"


@app.command()
def init(dir: str = typer.Option(".", help="Where to write scaffolding files.")) -> None:
    """Copy example config, objectives, and vendored seed data into a working directory."""
    dst = Path(dir).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.toml", "objectives.yaml"):
        src = _REPO_EXAMPLES / name
        if not src.exists():
            console.log(f"[warn] example {name} not found at {src}")
            continue
        shutil.copy2(src, dst / name)
        console.log(f"wrote {dst / name}")
    console.log("Edit config.toml to point at your provider(s), then run `jb run`.")


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c"),
    objectives: str = typer.Option(..., "--objectives", "-o"),
    iters: int | None = typer.Option(None, "--iters", help="Override loop.iterations."),
    parallelism: int | None = typer.Option(None, "--parallelism", help="Override loop.parallelism."),
    i_am_authorized: bool = typer.Option(
        False,
        "--i-am-authorized",
        help="Required. Confirms you have permission to red-team the target model(s).",
    ),
) -> None:
    """Run the autonomous jailbreak-research loop."""
    if not i_am_authorized:
        raise typer.BadParameter(
            "Refusing to run without --i-am-authorized. "
            "This tool is for authorized red-team research only."
        )
    cfg = load_config(config)
    if iters is not None:
        cfg.loop.iterations = iters
    if parallelism is not None:
        cfg.loop.parallelism = parallelism
    objectives_list = load_objectives(objectives, cfg.ethics)
    console.log(
        f"loaded {len(objectives_list)} objective(s); "
        f"target={cfg.providers.target.model}, "
        f"proposer={cfg.providers.proposer.model}, "
        f"judge={cfg.providers.judge.model}"
    )
    asyncio.run(_run_async(cfg, objectives_list))


async def _run_async(cfg, objectives_list) -> None:
    conn = await open_db(cfg.paths.db)
    try:
        target = build_provider(cfg.providers.target)
        proposer_llm = build_provider(cfg.providers.proposer)
        judge_llm = build_provider(cfg.providers.judge)
        classifier_llm = proposer_llm
        adapter_llm = (
            build_provider(cfg.providers.adapter)
            if cfg.providers.adapter is not None
            else proposer_llm
        )
        audit = AuditLog(cfg.paths.audit_log)
        audit.banner("--i-am-authorized acknowledged by operator")
        objectives_repo = ObjectivesRepo(conn)
        techniques_repo = TechniquesRepo(conn)
        attempts_repo = AttemptsRepo(conn)
        archive_repo = ArchiveRepo(conn)
        reflections_repo = ReflectionsRepo(conn)
        runs_repo = RunsRepo(conn)
        proposed_objectives_repo = ProposedObjectivesRepo(conn)
        techniques = load_techniques()
        await techniques_repo.upsert_many(techniques)
        for o in objectives_list:
            await objectives_repo.upsert(o)
        novelty = NoveltyFilter(cfg.novelty.embedding_model, cfg.novelty.threshold)
        archive = MAPElitesArchive(cfg.archive, archive_repo)
        loop = Loop(
            cfg=cfg,
            proposer=Proposer(proposer_llm, adapter_provider=adapter_llm),
            executor=Executor(target),
            judge=Judge(judge_llm),
            reflector=Reflector(classifier_llm, cfg.paths.reflections),
            classifier=classifier_llm,
            novelty=novelty,
            archive=archive,
            attempts_repo=attempts_repo,
            archive_repo=archive_repo,
            reflections_repo=reflections_repo,
            runs_repo=runs_repo,
            objectives_repo=objectives_repo,
            proposed_objectives_repo=proposed_objectives_repo,
            audit=audit,
            techniques=techniques,
            objectives=objectives_list,
        )
        await loop.run()
    finally:
        await conn.close()


@app.command()
def archive(
    config: str = typer.Option(..., "--config", "-c"),
    objective: str | None = typer.Option(None, "--objective"),
    format: str = typer.Option("table", "--format", help="table|json"),
) -> None:
    """Browse the current MAP-Elites archive."""
    cfg = load_config(config)
    asyncio.run(_archive_async(cfg, objective, format))


async def _archive_async(cfg, objective: str | None, format: str) -> None:
    conn = await open_db(cfg.paths.db)
    try:
        archive_repo = ArchiveRepo(conn)
        attempts_repo = AttemptsRepo(conn)
        cells = await archive_repo.all()
        rows = []
        for key, aid, score in cells:
            a = await attempts_repo.get(aid)
            if not a:
                continue
            if objective and a.objective_id != objective:
                continue
            rows.append((key, aid, score, a.objective_id, a.op, a.prompt[:120]))
        if format == "json":
            print(json.dumps(
                [
                    {"cell": k, "attempt_id": aid, "score": s, "objective": o, "op": op, "prompt": p}
                    for k, aid, s, o, op, p in rows
                ],
                indent=2,
            ))
            return
        for k, aid, s, o, op, p in rows:
            print(f"{k:40s}  score={s:5.2f}  obj={o:15s}  op={op:9s}  #{aid}  {p}")
    finally:
        await conn.close()


@app.command()
def attempts(
    config: str = typer.Option(..., "--config", "-c"),
    objective: str | None = typer.Option(None, "--objective"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """Query recent attempts."""
    cfg = load_config(config)
    asyncio.run(_attempts_async(cfg, objective, limit))


async def _attempts_async(cfg, objective: str | None, limit: int) -> None:
    conn = await open_db(cfg.paths.db)
    try:
        repo = AttemptsRepo(conn)
        rows = await repo.query(limit=limit, objective_id=objective)
        for a in rows:
            marker = "✓" if a.verdict.success else "·"
            print(
                f"{marker} #{a.id:5d}  obj={a.objective_id:15s}  op={a.op:9s}  "
                f"score={a.verdict.score:5.2f}  style={a.features.attack_style:11s}  "
                f"{a.prompt[:100]}"
            )
    finally:
        await conn.close()


@app.command()
def attempt(
    attempt_id: int = typer.Argument(..., help="Attempt id (integer)"),
    config: str = typer.Option(..., "--config", "-c"),
) -> None:
    """Print the full detail of a single attempt, including its transcript."""
    cfg = load_config(config)
    asyncio.run(_attempt_async(cfg, attempt_id))


async def _attempt_async(cfg, attempt_id: int) -> None:
    conn = await open_db(cfg.paths.db)
    try:
        a = await AttemptsRepo(conn).get(attempt_id)
        if not a:
            console.log(f"no attempt with id={attempt_id}")
            return
        marker = "✓" if a.verdict.success else "·"
        print(
            f"{marker} attempt #{a.id}  objective={a.objective_id}  op={a.op}  "
            f"strategy={a.strategy}  score={a.verdict.score:.2f}  "
            f"n_turns={a.n_turns}  latency_ms={a.latency_ms}"
        )
        print(f"target_model:   {a.target_model}")
        print(f"features:       {a.features.model_dump()}")
        print(f"technique_ids:  {a.technique_ids}")
        print(f"parent_ids:     {a.parent_attempt_ids}")
        print(f"verdict:        success={a.verdict.success}  score={a.verdict.score}  "
              f"severity={a.verdict.severity}  success_turn={a.verdict.success_turn}")
        print(f"rationale:      {a.verdict.rationale}")
        if a.verdict.refusal_signals:
            print(f"refusal_signals: {a.verdict.refusal_signals}")
        print(f"usage:          input={a.usage.input}  output={a.usage.output}")
        print(f"created_at:     {a.created_at.isoformat()}")
        print()
        for i, t in enumerate(a.transcript):
            turn_num = i // 2 + 1
            print(f"─── turn {turn_num} — {t.role.upper()} ───")
            print(t.content)
            print()
    finally:
        await conn.close()


@app.command()
def export(
    config: str = typer.Option(..., "--config", "-c"),
    out: str = typer.Option("successes.jsonl", "--out"),
    success_only: bool = typer.Option(True, "--success-only/--all"),
) -> None:
    """Export attempts to a local JSONL file."""
    cfg = load_config(config)
    asyncio.run(_export_async(cfg, out, success_only))


async def _export_async(cfg, out: str, success_only: bool) -> None:
    conn = await open_db(cfg.paths.db)
    try:
        repo = AttemptsRepo(conn)
        rows = await repo.query(limit=10_000)
        n = 0
        with Path(out).open("w") as f:
            for a in rows:
                if success_only and not a.verdict.success:
                    continue
                f.write(a.model_dump_json() + "\n")
                n += 1
        console.log(f"wrote {n} attempts to {out}")
    finally:
        await conn.close()


@app.command()
def queue(
    config: str = typer.Option(..., "--config", "-c"),
    status: str | None = typer.Option(None, "--status", help="pending|approved|rejected|blocked_by_ethics"),
) -> None:
    """List objectives the reflector has proposed. Run in a second terminal while `jb run` is active."""
    cfg = load_config(config)
    asyncio.run(_queue_async(cfg, status))


async def _queue_async(cfg, status: str | None) -> None:
    conn = await open_db(cfg.paths.db)
    try:
        repo = ProposedObjectivesRepo(conn)
        rows = await repo.list_all(status=status)
        if not rows:
            console.log("no proposals" + (f" with status={status}" if status else ""))
            return
        for r in rows:
            status_marker = {
                "pending": "○",
                "approved": "✓",
                "rejected": "✗",
                "blocked_by_ethics": "⚠",
            }.get(r["status"], "?")
            print(
                f"{status_marker} {r['id']:32s}  status={r['status']:18s}  "
                f"category={r['category']:20s}  iter={r['proposed_at_iter']}"
            )
            print(f"    description: {r['description']}")
            print(f"    rationale:   {r['rationale']}")
            if r.get("status_reason"):
                print(f"    reason:      {r['status_reason']}")
            print()
    finally:
        await conn.close()


@app.command()
def approve(
    prop_id: str = typer.Argument(..., help="Proposal id to approve"),
    config: str = typer.Option(..., "--config", "-c"),
) -> None:
    """Approve a pending proposal — copies it into the active objectives list."""
    cfg = load_config(config)
    asyncio.run(_approve_async(cfg, prop_id))


async def _approve_async(cfg, prop_id: str) -> None:
    conn = await open_db(cfg.paths.db)
    try:
        prop_repo = ProposedObjectivesRepo(conn)
        row = await prop_repo.get(prop_id)
        if not row:
            console.log(f"no proposal with id={prop_id!r}")
            return
        if row["status"] == "blocked_by_ethics":
            allowed = set(cfg.ethics.allow_categories)
            if row["category"] not in allowed:
                console.log(
                    f"cannot approve: category {row['category']!r} still not in "
                    f"ethics.allow_categories. Edit config.toml first if you truly want this."
                )
                return
        elif row["status"] != "pending":
            console.log(f"proposal {prop_id!r} is {row['status']}, not pending — refusing")
            return
        objectives_repo = ObjectivesRepo(conn)
        from .models import Objective

        obj = Objective(
            id=row["id"],
            description=row["description"],
            category=row["category"],
            judge_rubric_override=row["judge_rubric_override"],
        )
        await objectives_repo.upsert(obj)
        await prop_repo.mark(prop_id, "approved", status_reason="approved by operator")
        console.log(f"approved {prop_id!r} → objectives table")
    finally:
        await conn.close()


@app.command()
def reject(
    prop_id: str = typer.Argument(..., help="Proposal id to reject"),
    config: str = typer.Option(..., "--config", "-c"),
    reason: str = typer.Option("", "--reason", help="Optional reason recorded with the rejection"),
) -> None:
    """Reject a proposal so it does not clutter the pending queue."""
    cfg = load_config(config)
    asyncio.run(_reject_async(cfg, prop_id, reason))


async def _reject_async(cfg, prop_id: str, reason: str) -> None:
    conn = await open_db(cfg.paths.db)
    try:
        repo = ProposedObjectivesRepo(conn)
        row = await repo.get(prop_id)
        if not row:
            console.log(f"no proposal with id={prop_id!r}")
            return
        await repo.mark(prop_id, "rejected", status_reason=reason or "rejected by operator")
        console.log(f"rejected {prop_id!r}")
    finally:
        await conn.close()


if __name__ == "__main__":
    app()
