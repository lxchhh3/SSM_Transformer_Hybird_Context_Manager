"""HybridCompactor end-to-end demo — the real store path + the fixed prompt.

Populates a store with the Stage-3 stream (real cross-project link + noise), then
runs the production compactor (Falcon-H1-3B) over store.active_entries(). This is
the WHOLE path: DB -> BE render_board (verbatim) -> HybridCompactor (lossy gist).

The payoff to eyeball: Stage-3 (lesson #20) showed the model hallucinated false
dependencies when the prompt INVITED dependency-hunting. COMPACT_PROMPT forbids
inferring unstated deps, so we expect the REAL link (dashboard reads ETL's curated
Parquet zone) to survive while the FALSE ones (Game<->FW) do NOT reappear.

    HF_HUB_OFFLINE=1 <spike_sse python> scripts/compact_demo.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ctx.compaction import HybridCompactor
from ctx.service import ContextService

STREAM = [
    ("game", "kevin", "decision", "Authoritative server model: clients send inputs only, never positions."),
    ("fw",   "boss",  "progress", "Bumped the sensor poll loop to 5ms; watchdog resets if a frame is missed."),
    ("etl",  "kevin", "decision", "ETL writes the curated zone as Parquet, partitioned by region and date."),
    ("game", "kevin", "progress", "Wire payload capped at 1200 bytes to stay under a typical MTU."),
    ("dash", "boss",  "progress", "Switched dashboard charts to Tailwind + CSS vars for theming."),
    ("game", "boss",  "progress", "Added a server-side speed-hack detector that rejects impossible position deltas."),
    ("fw",   "kevin", "decision", "OTA updates signed with per-device keys; one rollback slot kept."),
    ("dash", "kevin", "progress", "Analytics view now reads the curated Parquet zone directly instead of raw tables."),
    ("etl",  "boss",  "progress", "Nightly job rescheduled to 02:00 UTC after the region merge."),
    ("game", "kevin", "progress", "Delta snapshots landed: avg packet 1180B -> 320B in the test arena."),
    ("fw",   "boss",  "progress", "Cut idle power draw 18% by gating the radio between beacons."),
    ("dash", "boss",  "decision", "Dashboard filters persist to URL query params so views are shareable."),
]


def main():
    svc = ContextService(":memory:")
    for proj, author, etype, body in STREAM:
        svc.publish(author, etype, body, project=proj)
    entries = svc.store.active_entries()

    compactor = HybridCompactor("tiiuae/Falcon-H1-3B-Instruct",
                                device="cuda", dtype=torch.float16, max_new_tokens=360)
    gist = compactor.compact(entries)

    print(f"\n{'#' * 72}\n# HybridCompactor gist ({len(entries)} active entries)\n"
          f"{'#' * 72}\n{gist}\n{'#' * 72}", flush=True)

    # a link is ASSERTED only when both ends appear in ONE line, not doc-wide
    lines = [ln.lower() for ln in gist.splitlines() if ln.strip()]
    real = any(("dashboard" in ln or "analytics" in ln) and
               ("curated" in ln or "parquet" in ln) for ln in lines)
    game_t = ("speed-hack", "speed hack", "netcode", "payload", "snapshot", "authoritative")
    fw_t = ("sensor", "radio", "firmware", "poll", "beacon", "watchdog")
    false_gamefw = any(any(a in ln for a in game_t) and any(b in ln for b in fw_t)
                       for ln in lines)
    print("[check] REAL link (dashboard <- curated Parquet):",
          "PRESENT" if real else "missing")
    print("[check] FALSE dep (game <-> fw) hallucinated:",
          "YES (bad)" if false_gamefw else "no (good)", flush=True)


if __name__ == "__main__":
    main()
