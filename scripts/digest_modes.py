"""Controlled test: how to produce the team digest. Same stream (kevin's work A +
boss's updates incl. a mid-stream revert), four configs:

  mode 1 = INCREMENTAL FOLD  (prior_digest + event -> new digest, via the engine)
  mode 2 = FRESH SUMMARY of the current ACTIVE SET (store-driven, stateless)
  x {raw prompt, chat template}

Question: which gives a FAITHFUL + BOUNDED 'where are we' digest? If mode 2 wins,
the incremental SSM-fold is the wrong mechanism — the store already holds the exact
bounded current state, and the model just summarizes it on demand.
"""

import os

import torch

from ctx.mamba_summarizer import MambaSummarizer
from ctx.service import ContextService
from ctx.ssm_engine import SSMEngine

A_ENTRIES = [
    ("decision", "Base = EQUAL-WEIGHT; cap-weighting kills the illiquidity edge.", ["book.py"], "a_weight"),
    ("decision", "Factors = ILLIQ-ONLY; dropped LOWVOL and momentum.", ["factors.py"], "a_factors"),
    ("decision", "Concentration = smallest-mcap N=50.", ["book.py"], "a_conc"),
    ("decision", "Survival overlay = SMA120 + floor 0.5; maxDD -45% -> -25%.", ["overlay.py"], "a_overlay"),
    ("decision", "LLM shadow-only, LLM_LIVE=False; Sonnet per-name, Opus policy.", ["llm_advisory.py"], "a_llm"),
]
UPDATES = [
    ("sup", "a_overlay", "u_overlay", "Overlay tuned to SMA60 + floor 0.3 - faster reaction after the 2026 H1 drawdown."),
    ("pub", "u_backext", "progress", "Ran the all-A backward reconstruction: honest multi-cycle +9%/yr, maxDD -58%.", ["data.py"]),
    ("sup", "a_llm", "u_llm", "Promoted the LLM to LLM_LIVE=True after the shadow bar cleared (beat placebo +3% over 8 episodes)."),
    ("rev", "u_overlay"),
]


def apply_stream(svc):
    for etype, body, refs, eid in A_ENTRIES:
        svc.publish("kevin", etype, body, refs=refs, entry_id=eid)
    for op in UPDATES:
        if op[0] == "sup":
            svc.supersede(op[1], op[3], author="boss", new_entry_id=op[2])
        elif op[0] == "pub":
            svc.publish("boss", op[2], op[3], refs=op[4], entry_id=op[1])
        elif op[0] == "rev":
            svc.revert(op[1])


def fold_digest(model):
    """mode 1: replay the stream through the incremental-fold engine."""
    svc = ContextService(":memory:")
    eng = SSMEngine(svc.store, model, checkpoint_every=2)
    for etype, body, refs, eid in A_ENTRIES:
        svc.publish("kevin", etype, body, refs=refs, entry_id=eid)
    eng.sync()
    for op in UPDATES:
        if op[0] == "sup":
            svc.supersede(op[1], op[3], author="boss", new_entry_id=op[2])
        elif op[0] == "pub":
            svc.publish("boss", op[2], op[3], refs=op[4], entry_id=op[1])
        elif op[0] == "rev":
            svc.revert(op[1])
        eng.sync()
    return eng.digest()


def fresh_digest(model, active):
    """mode 2: one summary of the bounded current active set (store-driven)."""
    lines = "\n".join(f"- {e['author']} ({e['type']}): {e['body']}" for e in active)
    prompt = ("Below is a dev team's CURRENT state - each line is an active decision "
              "or update. Write a concise status board (<=120 words): what is decided "
              "and who owns what. SUMMARIZE in your own words; do NOT copy the lines "
              "verbatim.\n\n" + lines + "\n\nStatus board:")
    return model.generate(prompt, max_new_tokens=170)


def hints(digest):
    d = digest.lower()
    checks = {
        "equal-weight": "equal" in d,
        "ILLIQ-only": "illiq" in d,
        "N=50": "50" in d,
        "SMA120 restored": "120" in d,
        "SMA60 gone": "60" not in d,
        "LLM live": ("live" in d or "true" in d),
        "backward-ext": any(w in d for w in ("backward", "multi-cycle", "9%", "-58", "reconstruction")),
        f"BOUNDED ({len(digest)} chars, want<=700)": len(digest) <= 700,
    }
    return checks


def main():
    mid = os.environ.get("M4_MODEL", "tiiuae/falcon-mamba-7b-instruct")
    print(f"loading {mid}...", flush=True)
    model = MambaSummarizer(mid, device="cuda", dtype=torch.float16, max_new_tokens=170)

    svc = ContextService(":memory:")
    apply_stream(svc)
    active = svc.store.active_entries()

    for fmt in ("raw", "chat"):
        model.use_chat = (fmt == "chat")
        for mode, fn in (("1 INCREMENTAL-FOLD", lambda: fold_digest(model)),
                         ("2 FRESH-ACTIVE-SET", lambda: fresh_digest(model, active))):
            print("\n" + "#" * 72)
            print(f"### MODE {mode}  |  {fmt.upper()} prompt")
            print("#" * 72)
            dig = fn()
            print(dig)
            print("--- hints ---")
            for k, v in hints(dig).items():
                print(f"   [{'OK ' if v else 'MISS'}] {k}")

    print("\n[modes] done.", flush=True)


if __name__ == "__main__":
    main()
