"""Store — the source of truth.

Two tables:
  events   append-only, immutable log (the stream the SSM ingests; gives #1)
  entries  materialized active-state, mutable status (gives exact #2)

Retraction is exact because it's a database: a reverted/superseded row simply
stops being 'active', and active_entries() is all the rest of the system ever
reads. No model state to un-remember.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT    NOT NULL,
    kind     TEXT    NOT NULL,          -- publish | supersede | revert
    entry_id TEXT    NOT NULL,
    payload  TEXT    NOT NULL           -- JSON
);
CREATE TABLE IF NOT EXISTS entries (
    entry_id    TEXT PRIMARY KEY,
    author      TEXT    NOT NULL,
    type        TEXT    NOT NULL,       -- progress | decision | idea | doc | task ...
    body        TEXT    NOT NULL,
    refs        TEXT    NOT NULL,       -- JSON list (e.g. files touched)
    status      TEXT    NOT NULL,       -- active | superseded | reverted
    created_seq INTEGER NOT NULL,
    updated_seq INTEGER NOT NULL,
    supersedes  TEXT,                   -- entry_id this one replaced (nullable)
    project     TEXT                    -- structured grouping key (nullable)
);
CREATE INDEX IF NOT EXISTS idx_entries_status ON entries(status, created_seq);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: str = ":memory:"):
        # check_same_thread=False: the MCP server may handle requests on different
        # worker threads; short, committed transactions + SQLite's own locking keep
        # the two-dev write load safe. WAL lets a read not block the other's write.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        if db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- writes --------------------------------------------------------------

    def publish(
        self,
        author: str,
        etype: str,
        body: str,
        refs: Optional[list[str]] = None,
        entry_id: Optional[str] = None,
        ts: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        entry_id = entry_id or uuid.uuid4().hex
        refs = list(refs or [])
        seq = self._append_event(
            "publish", entry_id,
            {"author": author, "type": etype, "body": body, "refs": refs,
             "project": project}, ts,
        )
        self._insert_entry(entry_id, author, etype, body, refs, "active", seq,
                           None, project)
        self.conn.commit()
        return entry_id

    def supersede(
        self,
        entry_id: str,
        new_body: str,
        author: Optional[str] = None,
        etype: Optional[str] = None,
        refs: Optional[list[str]] = None,
        new_entry_id: Optional[str] = None,
        ts: Optional[str] = None,
    ) -> str:
        old = self._get_entry(entry_id)
        if old is None:
            raise KeyError(entry_id)
        if old["status"] != "active":
            raise ValueError(
                f"can only supersede an active entry; {entry_id} is {old['status']}"
            )
        new_entry_id = new_entry_id or uuid.uuid4().hex
        author = author or old["author"]
        etype = etype or old["type"]
        refs = list(old["refs"] if refs is None else refs)
        project = old["project"]  # a superseding revision inherits the grouping key
        seq = self._append_event(
            "supersede", new_entry_id,
            {"supersedes": entry_id, "author": author, "type": etype,
             "body": new_body, "refs": refs, "project": project}, ts,
        )
        self._insert_entry(new_entry_id, author, etype, new_body, refs,
                           "active", seq, supersedes=entry_id, project=project)
        self._set_status(entry_id, "superseded", seq)
        self.conn.commit()
        return new_entry_id

    def revert(self, entry_id: str, author: Optional[str] = None,
               ts: Optional[str] = None) -> None:
        e = self._get_entry(entry_id)
        if e is None:
            raise KeyError(entry_id)
        if e["status"] == "reverted":
            return  # idempotent: high-freq double-fire is a no-op
        if e["status"] == "superseded":
            raise ValueError(
                f"cannot revert a superseded entry ({entry_id}); revert its successor"
            )
        seq = self._append_event("revert", entry_id, {"author": author}, ts)
        self._set_status(entry_id, "reverted", seq)
        # revert is the inverse of supersede: restore what this entry replaced.
        if e["supersedes"]:
            prior = self._get_entry(e["supersedes"])
            if prior is not None and prior["status"] == "superseded":
                self._set_status(prior["entry_id"], "active", seq)
        self.conn.commit()

    # -- reads ---------------------------------------------------------------

    def get_entry(self, entry_id: str) -> dict[str, Any]:
        e = self._get_entry(entry_id)
        if e is None:
            raise KeyError(entry_id)
        return e

    def active_entries(self, author: Optional[str] = None,
                       etype: Optional[str] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM entries WHERE status = 'active'"
        params: list[Any] = []
        if author is not None:
            sql += " AND author = ?"
            params.append(author)
        if etype is not None:
            sql += " AND type = ?"
            params.append(etype)
        sql += " ORDER BY created_seq"
        return [self._row_to_entry(r) for r in self.conn.execute(sql, params)]

    def events_since(self, seq: int = 0) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE seq > ? ORDER BY seq", (seq,)
        )
        return [self._row_to_event(r) for r in rows]

    def all_events(self) -> list[dict[str, Any]]:
        return self.events_since(0)

    # -- internals -----------------------------------------------------------

    def _append_event(self, kind: str, entry_id: str,
                      payload: dict[str, Any], ts: Optional[str]) -> int:
        cur = self.conn.execute(
            "INSERT INTO events (ts, kind, entry_id, payload) VALUES (?, ?, ?, ?)",
            (ts or _now(), kind, entry_id, json.dumps(payload)),
        )
        return int(cur.lastrowid)

    def _insert_entry(self, entry_id: str, author: str, etype: str, body: str,
                     refs: list[str], status: str, seq: int,
                     supersedes: Optional[str], project: Optional[str] = None) -> None:
        self.conn.execute(
            "INSERT INTO entries (entry_id, author, type, body, refs, status, "
            "created_seq, updated_seq, supersedes, project) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entry_id, author, etype, body, json.dumps(refs), status,
             seq, seq, supersedes, project),
        )

    def _set_status(self, entry_id: str, status: str, seq: int) -> None:
        self.conn.execute(
            "UPDATE entries SET status = ?, updated_seq = ? WHERE entry_id = ?",
            (status, seq, entry_id),
        )

    def _get_entry(self, entry_id: str) -> Optional[dict[str, Any]]:
        r = self.conn.execute(
            "SELECT * FROM entries WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(r) if r is not None else None

    @staticmethod
    def _row_to_entry(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "entry_id": r["entry_id"],
            "author": r["author"],
            "type": r["type"],
            "body": r["body"],
            "refs": json.loads(r["refs"]),
            "status": r["status"],
            "created_seq": r["created_seq"],
            "updated_seq": r["updated_seq"],
            "supersedes": r["supersedes"],
            "project": r["project"],
        }

    @staticmethod
    def _row_to_event(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "seq": r["seq"],
            "ts": r["ts"],
            "kind": r["kind"],
            "entry_id": r["entry_id"],
            "payload": json.loads(r["payload"]),
        }
