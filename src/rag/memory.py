from __future__ import annotations

import aiosqlite


async def add_message(
    db: aiosqlite.Connection,
    session_id: str,
    role: str,
    content: str,
) -> None:
    await db.execute(
        "INSERT INTO chat_messages(session_id, role, content) VALUES(?,?,?)",
        (session_id, role, content),
    )
    await db.commit()


async def get_recent_messages(
    db: aiosqlite.Connection,
    session_id: str,
    limit: int = 6,
) ->list[tuple[str, str]]:
    """
    Returns messages in chronological order.
    We map role "user" to ChatOllama role "human".
    """
    sql = """
    SELECT role, content
    FROM chat_messages
    WHERE session_id = ?
    ORDER BY id DESC
    LIMIT ?;
    """

    async with db.execute(sql, (session_id, limit)) as cursor:
        rows = await cursor.fetchall()

    rows = list(reversed(rows))

    out: list[tuple[str, str]] = []
    for r in rows:
        role = r["role"]
        if role == "user":
            out.append(("human", r["content"]))
        elif role == "assistant":
            out.append(("assistant", r["content"]))
        else:
            out.append(("system", r["content"]))

    return out

