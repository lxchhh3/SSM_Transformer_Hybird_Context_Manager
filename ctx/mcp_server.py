"""MCP HTTP server — the shared brain both CC sessions call over the LAN.

Thin adapter over ContextService. HTTP transport (not stdio) because the dev
machine must be reachable from both boxes. Each dev adds it as a remote MCP
server pointing at http://<dev-machine>:8765/mcp.

Run on the dev machine (call the env python directly — `conda run` buffers stdout):
    PYTHONPATH=. CTX_DB=D:/ctx/team.db <env>/python.exe -m ctx.mcp_server
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
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

# Optional GPU-backed reads. Off by default so the server runs GPU-free; enable on
# the dev machine (which has the GPU) with CTX_GIST=1 / CTX_SSM=1. They COMPETE for
# VRAM on a 16 GB box (falcon-mamba-7b ~14 GB + Falcon-H1-3B ~6 GB) — enable one.
GIST_ENABLED = bool(os.environ.get("CTX_GIST"))
SSM_ENABLED = bool(os.environ.get("CTX_SSM"))
_compactor = None
_engine = None


def _get_compactor():
    global _compactor
    if _compactor is None:
        import torch
        from ctx.compaction import HybridCompactor
        _compactor = HybridCompactor(
            os.environ.get("CTX_GIST_MODEL", "tiiuae/Falcon-H1-3B-Instruct"),
            device="cuda", dtype=torch.float16)
    return _compactor


def _get_engine():
    global _engine
    if _engine is None:
        import torch
        from ctx.mamba_summarizer import MambaSummarizer
        from ctx.ssm_engine import ShardedSSMEngine
        summ = MambaSummarizer(
            os.environ.get("CTX_SSM_MODEL", "tiiuae/falcon-mamba-7b-instruct"),
            device="cuda", dtype=torch.float16)
        _engine = ShardedSSMEngine(svc.store, summ)
    return _engine


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
    removed). Each bullet carries a relative age (e.g. '3d ago') and long ALL-CAPS
    emphasis is softened for skimmability — facts (numbers/names/acronyms) are
    unchanged. Returns {board, shown, overflow}. This is the primary read."""
    return svc.status_board(now=datetime.now(timezone.utc), soften_caps=True)


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
    cross-author file collisions. Bodies are trimmed to a preview (with `body_len`
    and a per-author `totals` matrix) so the full active set can't overflow the
    read — drill into `status_board` for full board text."""
    return svc.team_state(preview=True)


@mcp.tool()
def check_overlap(refs: list[str], author: Optional[str] = None) -> list:
    """Before you start: who else is currently touching these files?"""
    return svc.check_overlap(refs=refs, author=author)


@mcp.tool()
def recent(since_seq: int = 0, limit: int = 50) -> dict:
    """Event stream since a sequence number — poll for the other dev's activity.
    Returns the newest `limit` events with bodies trimmed to a preview (the raw
    stream overflows the read); omitted older events are surfaced in `note`, and
    `latest_seq` is the watermark to pass next. Page by raising `limit`."""
    return svc.recent_summary(since_seq=since_seq, limit=limit)


@mcp.tool()
def overview() -> dict:
    """WHERE ARE WE as a short, readable GIST that ties related work together — a
    lossy human-glance summary over the capped board. Non-authoritative (it may
    compress or omit): drill into `status_board` for the exact set. Falls back to
    the verbatim board when the gist model is disabled (enable with CTX_GIST=1 on
    the dev machine). Returns {overview, shown, overflow, selector}."""
    board_opts = {"now": datetime.now(timezone.utc), "soften_caps": True}
    if not GIST_ENABLED:
        return svc.overview(**board_opts)  # deterministic verbatim board — always works
    try:
        return svc.overview(compactor=_get_compactor())
    except Exception as e:  # never take the server down for a model hiccup
        res = svc.overview(**board_opts)
        res["gist_error"] = f"{type(e).__name__}: {e}"
        return res


@mcp.tool()
def project_digests() -> dict:
    """Per-project streaming SSM digests — a constant-cost 'where are we' per
    project, each stream kept in its own faithful envelope. Requires CTX_SSM=1 on
    the dev machine (loads a Mamba on the GPU); otherwise returns a disabled note.
    Returns {digests: {project: text}, projects: [...]}"""
    if not SSM_ENABLED:
        return {"enabled": False,
                "note": "SSM disabled; set CTX_SSM=1 on the dev machine (GPU)."}
    return svc.project_digests(_get_engine())


def main() -> None:
    """Console entry point (`ctx-mcp-server`) — also `python -m ctx.mcp_server`."""
    print(f"[context-manager] serving on http://{HOST}:{PORT}/mcp", flush=True)
    print(f"[context-manager] source of truth: {os.path.abspath(DB_PATH)}", flush=True)
    print("[context-manager] add on each box: claude mcp add --transport http "
          f"context-manager http://<dev-machine-ip>:{PORT}/mcp", flush=True)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
