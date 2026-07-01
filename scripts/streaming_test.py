"""Streaming-path test — the real usage pattern.

Instead of one prefill of the whole working set, feed it INCREMENTALLY (token by
token, carrying cache_params), and:
  1. read out the board at growing checkpoints (can we answer 'where are we?' at
     ANY point mid-stream, faithfully to what's streamed so far?);
  2. compare the FINAL streamed-state readout to a one-shot prefill of the same
     content (does incremental carry DRIFT from batch, or match it?).

Readout is non-destructive: we deep-copy the live cache, feed the '\n\nStatus
board:' suffix into the copy, decode from the copy, and keep streaming on the
original. Every step is seq_len==1 (the only safe path once the cache is populated).
"""

import copy
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MID = os.environ["M4_MODEL"]
CORPUS = os.environ.get(
    "CORPUS",
    "E:/Temp/claude/D--2files-Context-Manager/bdf87e91-d089-43f3-864a-65bb5f077c5b/scratchpad/corpus.json")
PROJ = {"game": "Game server (Go)", "dash": "Analytics dashboard (React/TS)",
        "etl": "ETL pipeline (Spark)", "fw": "IoT firmware (Rust)"}
PREAMBLE = ("You maintain a live shared status board for a two-person dev team "
            "(kevin, boss). Team facts stream in below as they happen. Keep the "
            "CURRENT state. For each project give 2-4 tight bullets: the current "
            "technical choices and who is driving. Use only the facts.")
ENVELOPE_TOK = 1000  # the faithful envelope from the knee sweep

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to("cuda").eval()


def active_for(corpus, keys):
    entries = []
    for k in keys:
        for e in corpus[k]:
            e = dict(e); e["proj"] = k
            entries.append(e)
    revised = {e["revises"] for e in entries if e.get("revises")}
    return [e for e in entries if e["id"] not in revised]


def render(active):
    keys = list(dict.fromkeys(e["proj"] for e in active))
    lines = []
    for k in keys:
        lines.append(f"\n# {PROJ[k]}")
        lines += [f"- ({e['author']}/{e['type']}) {e['body']}"
                  for e in active if e["proj"] == k]
    return "\n".join(lines)


def prefill(ids):
    with torch.no_grad():
        out = model(ids, use_cache=True)
    return out.cache_params, out.logits[0, -1]


def feed(cache, tok_id):
    inp = torch.tensor([[tok_id]], device="cuda")
    with torch.no_grad():
        out = model(inp, cache_params=cache, use_cache=True)
    return out.cache_params, out.logits[0, -1]


def greedy(cache, logits, n):
    ids, eos = [], tok.eos_token_id
    for _ in range(n):
        nxt = int(logits.argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        cache, logits = feed(cache, nxt)
    return tok.decode(ids, skip_special_tokens=True).strip()


def readout(cache, suffix_ids, n=260):
    """Non-destructive: decode a board from a COPY of the live streaming cache."""
    ro = copy.deepcopy(cache)
    logits = None
    for s in suffix_ids:
        ro, logits = feed(ro, s)
    return greedy(ro, logits, n)


def main():
    with open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    active_all = active_for(corpus, ["game", "dash"])

    # trim to the ~1000-token faithful envelope
    prefix = []
    for e in active_all:
        prefix.append(e)
        if tok(render(prefix), return_tensors="pt").input_ids.shape[1] >= ENVELOPE_TOK:
            break
    facts = render(prefix)

    body = PREAMBLE + "\n\n" + facts
    body_ids = tok(body, return_tensors="pt").input_ids[0].tolist()
    suffix_ids = tok("\n\nStatus board:", add_special_tokens=False,
                     return_tensors="pt").input_ids[0].tolist()
    N = len(body_ids)
    checkpoints = {int(N * 0.35), int(N * 0.7), N}
    print(f"streaming {len(prefix)} entries | {N} body tokens | "
          f"checkpoints @ {sorted(checkpoints)}", flush=True)

    # ---- STREAM token by token, carrying the cache ----
    cache, logits = None, None
    for i, tid in enumerate(body_ids):
        if cache is None:
            cache, logits = prefill(torch.tensor([[tid]], device="cuda"))
        else:
            cache, logits = feed(cache, tid)
        if (i + 1) in checkpoints:
            board = readout(cache, suffix_ids)
            print(f"\n{'=' * 72}\nCHECKPOINT @ {i + 1}/{N} body tokens "
                  f"| [x]-count={board.count('[x]')}\n{'=' * 72}\n{board}", flush=True)

    streamed_final = readout(cache, suffix_ids)

    # ---- ONE-SHOT baseline: prefill the whole thing in one pass ----
    ids = tok(body + "\n\nStatus board:", return_tensors="pt").input_ids.to("cuda")
    c1, l1 = prefill(ids)
    oneshot = greedy(c1, l1, 260)

    print(f"\n{'=' * 72}\nDRIFT CHECK: streamed-final vs one-shot prefill "
          f"(identical={streamed_final == oneshot})\n{'=' * 72}")
    print("--- streamed final ---\n" + streamed_final, flush=True)
    print("\n--- one-shot ---\n" + oneshot, flush=True)
    print("\n[streaming_test] done.", flush=True)


if __name__ == "__main__":
    main()
