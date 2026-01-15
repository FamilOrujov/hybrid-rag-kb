from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite


@asynccontextmanager
async def connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON;")
        await conn.execute("PRAGMA journal_mode = WAL;")
        yield conn


async def init_db(db_path: str, schema_path: str) -> None:
    schema_sql = Path(schema_path).read_text(encoding="utf-8")

    async with connect(db_path) as conn:
        await conn.executescript(schema_sql)
        await conn.commit()

