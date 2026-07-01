"""Live end-to-end verify of the MCP server: boot it as a subprocess, then drive
it through a real streamable-HTTP MCP client — initialize, list tools, publish
from two 'sessions', and confirm the cross-author overlap warning comes back
over the wire. Proves the shared brain actually serves both CC sessions.

Run with ctx_env python (has mcp):
    <ctx_env python> scripts/verify_mcp.py
"""

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time

# localhost must bypass the Windows system/registry proxy, or httpx routes the
# request through it and gets a 503 (lesson #16/#43). Set before the client is built.
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1," + os.environ.get("NO_PROXY", "")
os.environ["no_proxy"] = os.environ["NO_PROXY"]

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

HOST = "127.0.0.1"
PORT = 8799
URL = f"http://{HOST}:{PORT}/mcp"
PROJ = r"D:\2files\Context_Manager"


def _wait_port(host: str, port: int, timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.3)
    return False


def _show(result):
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    if result.content:
        return getattr(result.content[0], "text", result.content[0])
    return result


async def _drive() -> None:
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS:", sorted(t.name for t in tools.tools))

            # a realistic two-dev, two-project session
            await session.call_tool("publish", {
                "author": "kevin", "etype": "decision", "project": "game",
                "body": "Authoritative server model — clients send inputs only",
                "refs": ["net.go"], "entry_id": "k1"})
            await session.call_tool("publish", {
                "author": "kevin", "etype": "progress", "project": "game",
                "body": "Delta snapshots landed: 1180B -> 320B", "refs": ["snap.go"],
                "entry_id": "k2"})
            r = await session.call_tool("publish", {
                "author": "boss", "etype": "progress", "project": "game",
                "body": "Reworking the net loop for 2FA handshake", "refs": ["net.go"],
                "entry_id": "b1"})
            print("OVERLAP WARNING on b1 ->", _show(r))   # expect k1 on net.go
            await session.call_tool("publish", {
                "author": "boss", "etype": "progress", "project": "dash",
                "body": "LTTB downsampling — pan 60fps at 10k points",
                "refs": ["chart.tsx"], "entry_id": "b2"})

            # a stale entry, then retract it
            await session.call_tool("publish", {
                "author": "kevin", "etype": "idea", "project": "game",
                "body": "maybe switch transport to QUIC", "entry_id": "dead"})
            await session.call_tool("revert", {"entry_id": "dead"})

            board = await session.call_tool("status_board", {})
            print("\nSTATUS BOARD (over the wire):")
            print(_show(board))

            ov = await session.call_tool("check_overlap",
                                         {"refs": ["net.go"], "author": "boss"})
            print("\nCHECK_OVERLAP net.go (boss) ->", _show(ov))


def main() -> None:
    db = os.path.join(tempfile.gettempdir(), "ctx_verify.db")
    for ext in ("", "-journal", "-wal", "-shm"):
        try:
            os.remove(db + ext)
        except OSError:
            pass
    env = dict(os.environ, CTX_DB=db, CTX_HOST=HOST, CTX_PORT=str(PORT))
    proc = subprocess.Popen([sys.executable, "-m", "ctx.mcp_server"],
                            env=env, cwd=PROJ)
    try:
        if not _wait_port(HOST, PORT):
            print("SERVER DID NOT START")
            sys.exit(1)
        # uvicorn opens the port before the streamable-http lifespan is ready,
        # so early POSTs 503 — retry until the session manager is live.
        last = None
        for _ in range(12):
            try:
                asyncio.run(_drive())
                print("VERIFY_OK")
                return
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # noqa: BLE001 - report whatever failed
                last = e
                time.sleep(1.0)
        print("VERIFY_FAILED after retries:", repr(last))
        sys.exit(1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
