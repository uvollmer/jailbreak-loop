"""SQLite repositories for archive, attempts, objectives, techniques, reflections, runs."""

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite

from ..models import (
    Attempt,
    AttemptFeatures,
    JudgeVerdict,
    Objective,
    Reflection,
    Technique,
    TokenUsage,
    Turn,
)


class ObjectivesRepo:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def upsert(self, obj: Objective) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO objectives(id, description, category, judge_rubric_override) "
            "VALUES(?,?,?,?)",
            (obj.id, obj.description, obj.category, obj.judge_rubric_override),
        )
        await self.conn.commit()

    async def all(self) -> list[Objective]:
        async with self.conn.execute("SELECT * FROM objectives") as cur:
            rows = await cur.fetchall()
        return [
            Objective(
                id=r["id"],
                description=r["description"],
                category=r["category"],
                judge_rubric_override=r["judge_rubric_override"],
            )
            for r in rows
        ]


class TechniquesRepo:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def upsert_many(self, techniques: list[Technique]) -> None:
        await self.conn.executemany(
            "INSERT OR REPLACE INTO techniques(id, name, family, description, exemplars_json, source) "
            "VALUES(?,?,?,?,?,?)",
            [
                (t.id, t.name, t.family, t.description, json.dumps(t.exemplars), t.source)
                for t in techniques
            ],
        )
        await self.conn.commit()

    async def all(self) -> list[Technique]:
        async with self.conn.execute("SELECT * FROM techniques") as cur:
            rows = await cur.fetchall()
        return [
            Technique(
                id=r["id"],
                name=r["name"],
                family=r["family"],
                description=r["description"],
                exemplars=json.loads(r["exemplars_json"]),
                source=r["source"],
            )
            for r in rows
        ]


class AttemptsRepo:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def insert(self, attempt: Attempt, embedding: bytes | None = None) -> int:
        transcript_json = json.dumps([t.model_dump() for t in attempt.transcript])
        cur = await self.conn.execute(
            "INSERT INTO attempts("
            " objective_id, parent_ids_json, technique_ids_json, op, strategy,"
            " prompt, target_model, response, transcript_json, n_turns,"
            " judge_score, judge_success, judge_rationale, refusal_signals_json,"
            " success_turn, features_json, embedding, latency_ms, token_usage_json"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                attempt.objective_id,
                json.dumps(attempt.parent_attempt_ids),
                json.dumps(attempt.technique_ids),
                attempt.op,
                attempt.strategy,
                attempt.prompt,
                attempt.target_model,
                attempt.response,
                transcript_json,
                attempt.n_turns,
                attempt.verdict.score,
                int(attempt.verdict.success),
                attempt.verdict.rationale,
                json.dumps(attempt.verdict.refusal_signals),
                attempt.verdict.success_turn,
                attempt.features.model_dump_json(),
                embedding,
                attempt.latency_ms,
                attempt.usage.model_dump_json(),
            ),
        )
        await self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    async def get(self, attempt_id: int) -> Attempt | None:
        async with self.conn.execute(
            "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return _row_to_attempt(row)

    async def count_successful_by_objective(self, objective_id: str) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS c FROM attempts WHERE objective_id = ? AND judge_success = 1",
            (objective_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["c"] if row else 0)

    async def recent_successes(self, limit: int = 20) -> list[Attempt]:
        async with self.conn.execute(
            "SELECT * FROM attempts WHERE judge_success = 1 "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_attempt(r) for r in rows]

    async def query(self, limit: int = 50, objective_id: str | None = None) -> list[Attempt]:
        sql = "SELECT * FROM attempts"
        params: tuple = ()
        if objective_id:
            sql += " WHERE objective_id = ?"
            params = (objective_id,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params = (*params, limit)
        async with self.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [_row_to_attempt(r) for r in rows]

    async def embeddings_for_objective(self, objective_id: str) -> list[bytes]:
        async with self.conn.execute(
            "SELECT embedding FROM attempts WHERE objective_id = ? AND embedding IS NOT NULL",
            (objective_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [r["embedding"] for r in rows if r["embedding"]]


class ArchiveRepo:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def get(self, cell_key: str) -> tuple[int, float] | None:
        async with self.conn.execute(
            "SELECT attempt_id, score FROM archive_cells WHERE cell_key = ?", (cell_key,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return (int(row["attempt_id"]), float(row["score"]))

    async def upsert(self, cell_key: str, attempt_id: int, score: float) -> None:
        await self.conn.execute(
            "INSERT INTO archive_cells(cell_key, attempt_id, score, updated_at) "
            "VALUES(?,?,?, datetime('now')) "
            "ON CONFLICT(cell_key) DO UPDATE SET "
            "attempt_id=excluded.attempt_id, score=excluded.score, updated_at=excluded.updated_at",
            (cell_key, attempt_id, score),
        )
        await self.conn.commit()

    async def all(self) -> list[tuple[str, int, float]]:
        async with self.conn.execute(
            "SELECT cell_key, attempt_id, score FROM archive_cells"
        ) as cur:
            rows = await cur.fetchall()
        return [(r["cell_key"], int(r["attempt_id"]), float(r["score"])) for r in rows]


class ReflectionsRepo:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def insert(self, reflection: Reflection) -> None:
        await self.conn.execute(
            "INSERT INTO reflections(iter, summary, top_families_json) VALUES(?,?,?)",
            (reflection.iter, reflection.summary, json.dumps(reflection.top_families)),
        )
        await self.conn.commit()

    async def latest(self) -> Reflection | None:
        async with self.conn.execute(
            "SELECT * FROM reflections ORDER BY id DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return Reflection(
            id=row["id"],
            iter=row["iter"],
            summary=row["summary"],
            top_families=json.loads(row["top_families_json"]),
        )


class ProposedObjectivesRepo:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def insert(
        self,
        prop_id: str,
        description: str,
        category: str,
        rationale: str,
        proposed_at_iter: int,
        status: str,
        judge_rubric_override: str | None = None,
        status_reason: str | None = None,
    ) -> None:
        await self.conn.execute(
            "INSERT OR IGNORE INTO proposed_objectives("
            " id, description, category, judge_rubric_override, rationale,"
            " proposed_at_iter, status, status_reason"
            ") VALUES(?,?,?,?,?,?,?,?)",
            (
                prop_id,
                description,
                category,
                judge_rubric_override,
                rationale,
                proposed_at_iter,
                status,
                status_reason,
            ),
        )
        await self.conn.commit()

    async def list_all(self, status: str | None = None) -> list[dict]:
        if status:
            sql = "SELECT * FROM proposed_objectives WHERE status = ? ORDER BY proposed_at DESC"
            params: tuple = (status,)
        else:
            sql = "SELECT * FROM proposed_objectives ORDER BY proposed_at DESC"
            params = ()
        async with self.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get(self, prop_id: str) -> dict | None:
        async with self.conn.execute(
            "SELECT * FROM proposed_objectives WHERE id = ?", (prop_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def mark(self, prop_id: str, status: str, status_reason: str | None = None) -> None:
        await self.conn.execute(
            "UPDATE proposed_objectives SET status = ?, status_reason = ?, "
            "reviewed_at = datetime('now') WHERE id = ?",
            (status, status_reason, prop_id),
        )
        await self.conn.commit()


class RunsRepo:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn

    async def start(self, config_hash: str) -> int:
        cur = await self.conn.execute(
            "INSERT INTO runs(config_hash) VALUES(?)", (config_hash,)
        )
        await self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    async def finish(self, run_id: int, iters_completed: int) -> None:
        await self.conn.execute(
            "UPDATE runs SET ended_at = datetime('now'), iters_completed = ? WHERE id = ?",
            (iters_completed, run_id),
        )
        await self.conn.commit()


def _row_to_attempt(r: aiosqlite.Row) -> Attempt:
    features_data = json.loads(r["features_json"])
    # Row may lack the multi-turn columns if it was written by an older version.
    strategy = r["strategy"] if "strategy" in r.keys() else "single_turn"
    n_turns = int(r["n_turns"]) if "n_turns" in r.keys() and r["n_turns"] is not None else 1
    success_turn = r["success_turn"] if "success_turn" in r.keys() else None
    transcript_raw = r["transcript_json"] if "transcript_json" in r.keys() else None
    if transcript_raw:
        transcript = [Turn(**t) for t in json.loads(transcript_raw)]
    else:
        # Backfill from prompt/response for legacy rows.
        transcript = [
            Turn(role="user", content=r["prompt"]),
            Turn(role="assistant", content=r["response"]),
        ]
    verdict = JudgeVerdict(
        success=bool(r["judge_success"]),
        score=float(r["judge_score"]),
        rationale=r["judge_rationale"],
        refusal_signals=json.loads(r["refusal_signals_json"]),
        success_turn=int(success_turn) if success_turn is not None else None,
    )
    return Attempt(
        id=int(r["id"]),
        objective_id=r["objective_id"],
        op=r["op"],
        strategy=strategy,
        parent_attempt_ids=json.loads(r["parent_ids_json"]),
        technique_ids=json.loads(r["technique_ids_json"]),
        prompt=r["prompt"],
        target_model=r["target_model"],
        response=r["response"],
        transcript=transcript,
        n_turns=n_turns,
        verdict=verdict,
        features=AttemptFeatures(**features_data),
        created_at=datetime.fromisoformat(r["created_at"].replace(" ", "T")),
        latency_ms=int(r["latency_ms"]),
        usage=TokenUsage(**json.loads(r["token_usage_json"])),
    )
