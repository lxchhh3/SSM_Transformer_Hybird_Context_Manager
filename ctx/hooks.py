"""Read-side CC harness hooks — deterministic injection of the shared state.

The consumer loop splits the same way the product does: reads are deterministic
and free, so the harness INJECTS them instead of gambling on prompt compliance —
the verbatim board at SessionStart, and only-NEW teammate events at each
UserPromptSubmit via a per-session seq watermark (the stream). Writes
(publish / supersede / revert) stay judgment calls the CC makes through the
MCP tools.

Reads the DB directly (read-only intent, WAL allows it alongside the server)
rather than calling the MCP server: hooks run on every prompt, so they must be
ms-fast, immune to the system proxy, and alive even when the server is down.
Fail-soft everywhere — a hook must never break the user's prompt; errors land
in <state_dir>/last_error.txt instead of stderr.

Wire-up (each box, after `pip install` of the wheel):
    SessionStart      -> ctx-cc-hook session-start
    UserPromptSubmit  -> ctx-cc-hook prompt-submit
DB path from CTX_DB (default D:\\ctx\\team.db); watermarks in <db dir>/hook_state.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ctx.service import ContextService

STREAM_CAP = 20     # events per injection; the rest is surfaced, never silent
_BODY_CAP = 200


# -- per-session watermark -----------------------------------------------------

def _seq_file(state_dir: str, session_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
    return Path(state_dir) / f"{safe}.seq"


def _read_seq(state_dir: str, session_id: str) -> Optional[int]:
    try:
        return int(_seq_file(state_dir, session_id).read_text(encoding="ascii"))
    except (OSError, ValueError):
        return None


def _write_seq(state_dir: str, session_id: str, seq: int) -> None:
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    _seq_file(state_dir, session_id).write_text(str(seq), encoding="ascii")


# -- rendering -------------------------------------------------------------

def _fmt_event(e: dict[str, Any]) -> str:
    p = e["payload"]
    body = " ".join((p.get("body") or "").split())
    if len(body) > _BODY_CAP:
        body = body[:_BODY_CAP - 1] + "…"
    head = f"- #{e['seq']} {e['kind']} ({p.get('author', '?')}/{p.get('type', '?')}"
    if p.get("project"):
        head += f", {p['project']}"
    head += ")"
    if e["kind"] == "revert":
        return f"{head}: retracted entry {e['entry_id']}"
    line = f"{head}: {body}"
    if e["kind"] == "supersede" and p.get("supersedes"):
        line += f" (replaces {p['supersedes']})"
    if p.get("refs"):
        line += f" [{', '.join(p['refs'])}]"
    return line


# -- the two hooks ---------------------------------------------------------

def session_start(db_path: str, state_dir: str, session_id: str) -> str:
    """The current board, verbatim — and the stream watermark set to now,
    so the first prompt doesn't replay what the board already shows."""
    if not os.path.exists(db_path):
        return ""
    svc = ContextService(db_path)
    try:
        # watermark BEFORE the board read: a write landing in between shows up
        # twice (board + stream) rather than never
        tip = svc.store.max_seq()
        res = svc.status_board(now=datetime.now(timezone.utc), soften_caps=True)
    finally:
        svc.close()
    _write_seq(state_dir, session_id, tip)
    out = "[context-manager] team board (shared source of truth):"
    if not res["board"]:
        return out + "\n(board empty — no active entries yet)"
    out += "\n" + res["board"]
    if res["overflow"]:
        out += (f"\n(+{res['overflow']} older entries not shown — "
                "the status_board / recent MCP tools have the rest)")
    return out


def prompt_submit(db_path: str, state_dir: str, session_id: str) -> str:
    """New team events since this session last looked — the stream. Empty when
    there's nothing new (no context noise). First call only sets the watermark:
    history is the board's job, not the stream's."""
    if not os.path.exists(db_path):
        # first contact still marks time zero — if the DB appears later,
        # everything in it is NEW relative to this session, so it streams
        if _read_seq(state_dir, session_id) is None:
            _write_seq(state_dir, session_id, 0)
        return ""
    svc = ContextService(db_path)
    try:
        last = _read_seq(state_dir, session_id)
        if last is None:
            _write_seq(state_dir, session_id, svc.store.max_seq())
            return ""
        events = svc.recent(since_seq=last)
    finally:
        svc.close()
    if not events:
        return ""
    _write_seq(state_dir, session_id, events[-1]["seq"])
    shown = events[-STREAM_CAP:]
    out = ("[context-manager] team stream — new since your last prompt:\n"
           + "\n".join(_fmt_event(e) for e in shown))
    if len(events) > len(shown):
        out += (f"\n(+{len(events) - len(shown)} earlier new events not shown — "
                f"call recent(since_seq={last}) for all)")
    return out


# -- console entry point (`ctx-cc-hook`) ------------------------------------

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "session-start"
    db = os.environ.get("CTX_DB", r"D:\ctx\team.db")
    state_dir = os.environ.get("CTX_HOOK_STATE", str(Path(db).parent / "hook_state"))
    try:
        payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except Exception:
        payload = {}
    session_id = str(payload.get("session_id") or "default")
    try:
        fn = session_start if mode == "session-start" else prompt_submit
        text = fn(db, state_dir, session_id)
        if text:
            sys.stdout.buffer.write(text.encode("utf-8"))  # console may be cp1252
    except Exception as e:  # fail soft: never break the user's prompt
        try:
            Path(state_dir).mkdir(parents=True, exist_ok=True)
            (Path(state_dir) / "last_error.txt").write_text(
                f"{mode}: {type(e).__name__}: {e}", encoding="utf-8")
        except OSError:
            pass
    sys.exit(0)


if __name__ == "__main__":
    main()
