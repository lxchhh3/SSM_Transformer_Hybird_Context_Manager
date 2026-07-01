"""MCP HTTP server — the shared brain both CC sessions call over the LAN.

Thin adapter over ContextService. HTTP transport (not stdio) because the dev
machine must be reachable from both boxes. Each dev adds it as a remote MCP
server pointing at http://<dev-machine>:8765/mcp.

Run on the dev machine (call the env python directly — `conda run` buffers stdout):
    PYTHONPATH=. CTX_DB=D:/ctx/team.db <env>/python.exe -m ctx.mcp_server
"""

from __future__ import annotations

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from ctx.service import ContextService

DB_PATH = os.environ.get("CTX_DB", "ctx.db")     # source of truth on the dev machine
HOST = os.environ.get("CTX_HOST", "0.0.0.0")     # 0.0.0.0 = reachable over the LAN
PORT = int(os.environ.get("CTX_PORT", "8765"))

svc = ContextService(DB_PATH)
# stateless_http: each tool call is independent (our state lives in the store, not
# in a server-side MCP session) — simpler, and avoids the stateful-session 503s.
mcp = FastMCP("context-manager", host=HOST, port=PORT,
              stateless_http=True, json_response=True)


@mcp.tool()
def publish(author: str, etype: str, body: str,
            project: Optional[str] = None,
            refs: Optional[list[str]] = None,
            entry_id: Optional[str] = None) -> dict:
    """Log a progress note / decision / idea / doc / task to the shared team state.

    Call this whenever you finish a unit of work, make a decision, or start
    something notable. `author` is you (kevin or boss). `etype` is one of
    progress|decision|idea|doc|task. `project` groups the entry on the board.
    `refs` are the files touched (used for duplication detection).

    Returns the new entry_id plus `overlaps`: active entries by the OTHER dev
    whose files intersect yours — a duplication warning before it happens.
    """
    return svc.publish(author, etype, body, project=project, refs=refs,
                       entry_id=entry_id)


@mcp.tool()
def status_board() -> dict:
    """WHERE ARE WE — the current team state as a verbatim board, grouped by
    project with the driver named. Call this at the start of a session, or any
    time you need the clean current picture (superseded/reverted work already
    removed). Returns {board, shown, overflow}. This is the primary read."""
    return svc.status_board()


@mcp.tool()
def supersede(entry_id: str, new_body: str,
              author: Optional[str] = None,
              etype: Optional[str] = None,
              refs: Optional[list[str]] = None) -> dict:
    """Replace an entry with a new version (old becomes 'superseded')."""
    return svc.supersede(entry_id, new_body, author=author, etype=etype, refs=refs)


@mcp.tool()
def revert(entry_id: str) -> dict:
    """Retract an entry exactly: drop it from active state, and restore any prior
    entry it had superseded. The inverse of supersede."""
    svc.revert(entry_id)
    return {"ok": True, "entry_id": entry_id}


@mcp.tool()
def team_state() -> dict:
    """Where are we: active entries grouped by author -> type, plus the current
    cross-author file collisions."""
    return svc.team_state()


@mcp.tool()
def check_overlap(refs: list[str], author: Optional[str] = None) -> list:
    """Before you start: who else is currently touching these files?"""
    return svc.check_overlap(refs=refs, author=author)


@mcp.tool()
def recent(since_seq: int = 0) -> list:
    """Event stream since a sequence number — poll for the other dev's activity."""
    return svc.recent(since_seq=since_seq)


if __name__ == "__main__":
    print(f"[context-manager] serving on http://{HOST}:{PORT}/mcp", flush=True)
    print(f"[context-manager] source of truth: {os.path.abspath(DB_PATH)}", flush=True)
    print("[context-manager] add on each box: claude mcp add --transport http "
          f"context-manager http://<dev-machine-ip>:{PORT}/mcp", flush=True)
    mcp.run(transport="streamable-http")
