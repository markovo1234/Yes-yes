"""SQLite state on the persistent volume: sessions, their Gemini chats, images, hourly counter."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    last_used_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    session_id  TEXT NOT NULL,
    cid         TEXT NOT NULL,
    model       TEXT,
    last_rid    TEXT,
    created_at  REAL NOT NULL,
    PRIMARY KEY (session_id, cid)
);
CREATE TABLE IF NOT EXISTS images (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    cid           TEXT NOT NULL,
    rid           TEXT NOT NULL,
    metadata      TEXT NOT NULL,
    turn_index    INTEGER NOT NULL,
    turn_count    INTEGER NOT NULL,
    ext           TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    user_request  TEXT,
    parent_id     TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS images_by_session ON images(session_id);
CREATE TABLE IF NOT EXISTS image_events (ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS image_events_ts ON image_events(ts);
"""


@dataclass(frozen=True)
class ImageRecord:
    id: str
    session_id: str
    cid: str
    rid: str
    metadata: list[Any]
    turn_index: int
    turn_count: int
    ext: str
    prompt: str
    user_request: str | None
    parent_id: str | None
    created_at: float


@dataclass(frozen=True)
class ChatRecord:
    session_id: str
    cid: str
    model: str | None
    last_rid: str | None


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)
        path.chmod(0o600)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # sessions ---------------------------------------------------------------
    def create_session(self, sid: str, now: float) -> None:
        self._exec("INSERT INTO sessions(id, created_at, last_used_at) VALUES (?,?,?)", (sid, now, now))

    def session_exists(self, sid: str) -> bool:
        return bool(self._all("SELECT 1 FROM sessions WHERE id=?", (sid,)))

    def touch_session(self, sid: str, now: float) -> None:
        self._exec("UPDATE sessions SET last_used_at=? WHERE id=?", (now, sid))

    def idle_sessions(self, before: float) -> list[str]:
        return [r["id"] for r in self._all("SELECT id FROM sessions WHERE last_used_at < ?", (before,))]

    def delete_session(self, sid: str) -> None:
        self._exec("DELETE FROM sessions WHERE id=?", (sid,))

    # chats --------------------------------------------------------------------
    def add_chat(self, sid: str, cid: str, model: str | None, now: float) -> None:
        self._exec(
            "INSERT OR IGNORE INTO chats(session_id, cid, model, created_at) VALUES (?,?,?,?)",
            (sid, cid, model, now),
        )

    def set_chat_last_rid(self, sid: str, cid: str, rid: str) -> None:
        self._exec("UPDATE chats SET last_rid=? WHERE session_id=? AND cid=?", (rid, sid, cid))

    def get_chat(self, sid: str, cid: str) -> ChatRecord | None:
        rows = self._all("SELECT * FROM chats WHERE session_id=? AND cid=?", (sid, cid))
        return _chat(rows[0]) if rows else None

    def list_chats(self, sid: str) -> list[ChatRecord]:
        return [_chat(r) for r in self._all("SELECT * FROM chats WHERE session_id=? ORDER BY created_at", (sid,))]

    def delete_chat(self, sid: str, cid: str) -> None:
        self._exec("DELETE FROM chats WHERE session_id=? AND cid=?", (sid, cid))

    # images -------------------------------------------------------------------
    def add_image(self, rec: ImageRecord) -> None:
        self._exec(
            "INSERT INTO images(id, session_id, cid, rid, metadata, turn_index, turn_count, ext, prompt,"
            " user_request, parent_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec.id, rec.session_id, rec.cid, rec.rid, json.dumps(rec.metadata), rec.turn_index,
                rec.turn_count, rec.ext, rec.prompt, rec.user_request, rec.parent_id, rec.created_at,
            ),
        )

    def get_image(self, sid: str, iid: str) -> ImageRecord | None:
        rows = self._all("SELECT * FROM images WHERE session_id=? AND id=?", (sid, iid))
        return _image(rows[0]) if rows else None

    def list_images(self, sid: str) -> list[ImageRecord]:
        return [_image(r) for r in self._all("SELECT * FROM images WHERE session_id=?", (sid,))]

    def delete_images(self, sid: str) -> int:
        return self._exec("DELETE FROM images WHERE session_id=?", (sid,)).rowcount

    # hourly image counter -----------------------------------------------------
    def record_images(self, count: int, now: float) -> None:
        if count > 0:
            with self._lock:
                self._db.executemany("INSERT INTO image_events(ts) VALUES (?)", [(now,)] * count)

    def images_since(self, since: float) -> int:
        return self._all("SELECT COUNT(*) AS n FROM image_events WHERE ts >= ?", (since,))[0]["n"]

    def oldest_image_since(self, since: float) -> float | None:
        return self._all("SELECT MIN(ts) AS t FROM image_events WHERE ts >= ?", (since,))[0]["t"]

    def prune_image_events(self, before: float) -> None:
        self._exec("DELETE FROM image_events WHERE ts < ?", (before,))


def _chat(r: sqlite3.Row) -> ChatRecord:
    return ChatRecord(session_id=r["session_id"], cid=r["cid"], model=r["model"], last_rid=r["last_rid"])


def _image(r: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        id=r["id"], session_id=r["session_id"], cid=r["cid"], rid=r["rid"], metadata=json.loads(r["metadata"]),
        turn_index=r["turn_index"], turn_count=r["turn_count"], ext=r["ext"], prompt=r["prompt"],
        user_request=r["user_request"], parent_id=r["parent_id"], created_at=r["created_at"],
    )
