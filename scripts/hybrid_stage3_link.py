"""Stage-3 linking probe — does the hybrid TIE related facts together?

The reframe (memory hybrid-compaction-gist) rests on one claim: the value isn't
verbatim recall (the DB's job) but LINKING — heavy input -> a clean gist that
connects related work. Stage-2 showed per-project synthesis; this tests the harder
thing: connecting entries that are RELATED but SCATTERED across the stream and
interleaved with noise.

Three planted link-chains, members deliberately far apart:
  1  cross-project dependency: ETL writes a curated Parquet zone  <->  the dashboard
     reads that curated zone
  2  shared theme: 1200-byte payload cap (MTU)  <->  delta snapshots 1180B->320B
     (i.e. the snapshots keep packets under the cap)
  3  decision + enforcement: authoritative server (inputs-only)  <->  server-side
     speed-hack detector

A link is CONNECTED if one bullet of the board mentions BOTH ends; SEPARATE if the
board only lists them apart. We print the board for eyeball + an automated co-occur
check so the verdict isn't just vibes.

    LINK_MODEL=tiiuae/Falcon-H1-3B-Instruct <spike_sse python> scripts/hybrid_stage3_link.py
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MID = os.environ.get("LINK_MODEL", "tiiuae/Falcon-H1-3B-Instruct")

# chronological stream (as ingested); chain members are separated by noise
STREAM = [
    ("game", "kevin", "decision", "Authoritative server model: clients send inputs only, never positions."),   # 3A
    ("fw",   "boss",  "progress", "Bumped the sensor poll loop to 5ms; watchdog resets if a frame is missed."),
    ("etl",  "kevin", "decision", "ETL writes the curated zone as Parquet, partitioned by region and date."),   # 1A
    ("game", "kevin", "progress", "Wire payload capped at 1200 bytes to stay under a typical MTU."),            # 2A
    ("dash", "boss",  "progress", "Switched dashboard charts to Tailwind + CSS vars for theming."),
    ("game", "boss",  "progress", "Added a server-side speed-hack detector that rejects impossible position deltas."),  # 3B
    ("fw",   "kevin", "decision", "OTA updates signed with per-device keys; one rollback slot kept."),
    ("dash", "kevin", "progress", "Analytics view now reads the curated Parquet zone directly instead of raw tables."),  # 1B
    ("etl",  "boss",  "progress", "Nightly job rescheduled to 02:00 UTC after the region merge."),
    ("game", "kevin", "progress", "Delta snapshots landed: avg packet 1180B -> 320B in the test arena."),        # 2B
    ("fw",   "boss",  "progress", "Cut idle power draw 18% by gating the radio between beacons."),
    ("dash", "boss",  "decision", "Dashboard filters persist to URL query params so views are shareable."),
]

# (name, terms_A, terms_B): a bullet is a link if it hits >=1 of each side
CHAINS = [
    ("ETL->dashboard curated zone", ["curated", "parquet"], ["dashboard", "analytics"]),
    ("MTU cap <-> delta snapshots", ["1200", "mtu"],       ["delta", "320", "1180"]),
    ("server-authority <-> anti-cheat", ["authoritative", "inputs only", "inputs-only"],
     ["speed-hack", "speed hack", "detector"]),
]

INSTR = ("Summarize the CURRENT state for a two-person team (kevin, boss) across these "
         "projects. Give a tight 'where are we' — and wherever two pieces of work connect "
         "or depend on each other, say so explicitly. Use ONLY the facts below.")


def stream_text():
    return "\n".join(f"- ({p}/{a}/{t}) {b}" for p, a, t, b in STREAM)


def main():
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to("cuda").eval()

    prompt = INSTR + "\n\n" + stream_text() + "\n\nStatus board:"
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
    n_new = int(os.environ.get("LINK_MAXNEW", "420"))
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n_new, do_sample=False)
    board = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"\n{'#' * 72}\n# {MID}\n{'#' * 72}\n{board}\n{'#' * 72}", flush=True)

    # a "bullet" = any non-empty line; a chain is CONNECTED if one line hits both sides
    lines = [ln.lower() for ln in board.splitlines() if ln.strip()]
    blob = board.lower()
    print("\n[link check]  (CONNECTED = both ends named in ONE bullet)")
    connected = 0
    for name, a_terms, b_terms in CHAINS:
        one_bullet = any(any(a in ln for a in a_terms) and any(b in ln for b in b_terms)
                         for ln in lines)
        both_present = (any(a in blob for a in a_terms)
                        and any(b in blob for b in b_terms))
        verdict = "CONNECTED" if one_bullet else ("both-present" if both_present
                                                  else "SEPARATE/missing")
        connected += one_bullet
        print(f"  {verdict:16s} {name}")
    print(f"\n[link check] {connected}/{len(CHAINS)} chains connected in a single bullet.",
          flush=True)


if __name__ == "__main__":
    main()
