"""Async SQLite connection + schema init."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def open_db(path: str | Path) -> aiosqlite.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    with SCHEMA_PATH.open() as f:
        schema = f.read()
    await conn.executescript(schema)
    await conn.commit()
    return conn
