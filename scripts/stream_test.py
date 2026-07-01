"""Streaming-update stress test — the SSM's ACTUAL job (not comparison).

We load kevin's work A, then the boss streams UPDATES that evolve the state
(supersede a decision, add a finding, then revert a change). The SSM must keep a
faithful CURRENT digest through the stream — high-freq ingest (#1) + retraction
re-sync (#2) on real content. Comparison/dedup/conflict is deliberately ABSENT:
that goes to Claude (real attention), never the SSM (weak at associative recall).

Eval = does the final digest reflect the CURRENT active state? (Claude judges the
digest vs the ground-truth active set printed below; the keyword hints are hints.)
"""

import os
import statistics
import time

import torch

from ctx.mamba_summarizer import MambaSummarizer
from ctx.service import ContextService
from ctx.ssm_engine import SSMEngine

# kevin's work A (curated from D:\2files\fina\Rebuild\progress.md) — the start state
A_ENTRIES = [
    ("decision", "Base = EQUAL-WEIGHT; cap-weighting kills the illiquidity edge.",
     ["book.py"], "a_weight"),
    ("decision", "Factors = ILLIQ-ONLY; dropped LOWVOL and momentum.",
     ["factors.py"], "a_factors"),
    ("decision", "Concentration = smallest-mcap N=50.", ["book.py"], "a_conc"),
    ("decision", "Survival overlay = SMA120 + floor 0.5; maxDD -45% -> -25%.",
     ["overlay.py"], "a_overlay"),
    ("decision", "LLM shadow-only, LLM_LIVE=False; Sonnet per-name, Opus policy.",
     ["llm_advisory.py"], "a_llm"),
]

# boss's stream of UPDATES (not a parallel doc — the work evolving)
#   ("sup", old_id, new_id, new_body) | ("pub", id, etype, body, refs) | ("rev", id)
UPDATES = [
    ("sup", "a_overlay", "u_overlay",
     "Overlay tuned to SMA60 + floor 0.3 - faster reaction after the 2026 H1 drawdown."),
    ("pub", "u_backext", "progress",
     "Ran the all-A backward reconstruction: honest multi-cycle +9%/yr, maxDD -58%.",
     ["data.py"]),
    ("sup", "a_llm", "u_llm",
     "Promoted the LLM to LLM_LIVE=True after the shadow bar cleared (beat placebo +3% over 8 episodes)."),
    ("rev", "u_overlay"),  # boss reverts the SMA60 change -> SMA120 should come back
]


def main():
    mid = os.environ.get("M4_MODEL", "tiiuae/falcon-mamba-7b-instruct")
    print(f"[stream] loading {mid} on cuda...", flush=True)
    model = MambaSummarizer(mid, device="cuda", dtype=torch.float16, max_new_tokens=130)

    svc = ContextService(":memory:")
    eng = SSMEngine(svc.store, model, checkpoint_every=2)
    lat = []

    for etype, body, refs, eid in A_ENTRIES:
        svc.publish("kevin", etype, body, refs=refs, entry_id=eid)
    s = time.time(); eng.sync(); lat.append(time.time() - s)
    print("\n===== DIGEST after kevin's work A =====")
    print(eng.digest())

    for op in UPDATES:
        if op[0] == "sup":
            svc.supersede(op[1], op[3], author="boss", new_entry_id=op[2])
        elif op[0] == "pub":
            svc.publish("boss", op[2], op[3], refs=op[4], entry_id=op[1])
        elif op[0] == "rev":
            svc.revert(op[1])
        s = time.time(); eng.sync(); lat.append(time.time() - s)

    print("\n===== DIGEST after boss's update stream =====")
    digest = eng.digest()
    print(digest)

    print("\n===== GROUND-TRUTH active state (what the digest SHOULD reflect) =====")
    for e in svc.store.active_entries():
        print(f"  [{e['author']}] {e['body']}")

    print(f"\n===== throughput: {len(lat)} syncs, mean {statistics.mean(lat):.2f}s =====")

    d = digest.lower()
    print("\n===== faithfulness hints (Claude judges the digest above) =====")
    hints = {
        "equal-weight kept": "equal" in d,
        "ILLIQ-only kept": "illiq" in d,
        "N=50 kept": "50" in d,
        "overlay RESTORED to SMA120 (revert worked)": "120" in d,
        "SMA60 GONE (reverted out)": "60" not in d,
        "LLM_LIVE=True (promoted)": "live" in d or "true" in d,
        "backward-extension finding present": any(
            w in d for w in ("backward", "multi-cycle", "9%", "-58", "reconstruction")),
    }
    for k, v in hints.items():
        print(f"  [{'OK ' if v else 'MISS'}] {k}")

    print("\n[stream] done.", flush=True)


if __name__ == "__main__":
    main()
