"""Streaming-path test, CORRECTED — uses falcon's chat template (matching the
faithful knee-sweep/state-decode runs), not raw tokenization. The v1 script dropped
the template and collapsed early; that was a prompt-format artifact (the one-shot
baseline collapsed identically), NOT streaming.

Validates two things properly:
  1. Faithful mid-stream readout: read the board at growing checkpoints, decoding
     with the SAME templated assistant-cue the faithful runs used.
  2. Drift check IN THE FAITHFUL REGIME: streamed-state readout vs one-shot prefill
     of the same content, where matching rich (non-collapsed) text is meaningful.

Template wrapper (user-open head / assistant-cue tail) is recovered by diffing two
templated strings, so we can inject the readout suffix mid-stream without re-encoding.
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
PREAMBLE = ("Summarize the CURRENT state of this team's projects for a two-person "
            "team (kevin, boss). For each project give 2-4 tight bullets: the current "
            "technical choices and who is driving. Use ONLY the facts below.")
ENVELOPE_TOK = 1000

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


def encode_chat(user_text):
    enc = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return ids[0].tolist()


def split_template():
    """Recover (head, tail) template wrapper by diffing two templated strings:
    head = shared opening tokens, tail = shared closing (assistant-cue) tokens."""
    a = encode_chat("AAAA")
    b = encode_chat("AAAA BBBB")
    h = 0
    while h < min(len(a), len(b)) and a[h] == b[h]:
        h += 1
    t = 0
    while t < min(len(a), len(b)) - h and a[-1 - t] == b[-1 - t]:
        t += 1
    return a[:h], (a[len(a) - t:] if t else [])


def prefill(ids):
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda"), use_cache=True)
    return out.cache_params, out.logits[0, -1]


def feed(cache, tok_id):
    with torch.no_grad():
        out = model(torch.tensor([[tok_id]], device="cuda"),
                    cache_params=cache, use_cache=True)
    return out.cache_params, out.logits[0, -1]


def greedy(cache, logits, n=260):
    ids, eos = [], tok.eos_token_id
    for _ in range(n):
        nxt = int(logits.argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        cache, logits = feed(cache, nxt)
    return tok.decode(ids, skip_special_tokens=True).strip()


def main():
    with open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    active_all = active_for(corpus, ["game", "dash"])
    prefix = []
    for e in active_all:
        prefix.append(e)
        if tok(render(prefix), return_tensors="pt").input_ids.shape[1] >= ENVELOPE_TOK:
            break
    facts = render(prefix)

    head, tail = split_template()
    pre_ids = tok(PREAMBLE + "\n\n", add_special_tokens=False).input_ids
    fact_ids = tok(facts, add_special_tokens=False).input_ids
    suffix_ids = tok("\n\nStatus board:", add_special_tokens=False).input_ids
    stream_ids = head + pre_ids + fact_ids  # fed one token at a time
    base = len(head) + len(pre_ids)         # index where facts start

    def readout(cache):
        ro, logits = copy.deepcopy(cache), None
        for s in suffix_ids + tail:
            ro, logits = feed(ro, s)
        return greedy(ro, logits)

    cps = {base + round(f * len(fact_ids)) for f in (0.35, 0.7, 1.0)}
    print(f"streaming {len(prefix)} entries | {len(fact_ids)} fact tokens "
          f"| head={len(head)} tail={len(tail)} | checkpoints(fact-frac) "
          f"@ {sorted(round((c-base)/len(fact_ids), 2) for c in cps)}", flush=True)

    cache, logits = None, None
    for i, tid in enumerate(stream_ids):
        if cache is None:
            cache, logits = prefill([tid])
        else:
            cache, logits = feed(cache, tid)
        if (i + 1) in cps:
            board = readout(cache)
            frac = (i + 1 - base) / len(fact_ids)
            print(f"\n{'=' * 72}\nCHECKPOINT @ {frac:.0%} of facts streamed "
                  f"| [x]-count={board.count('[x]')}\n{'=' * 72}\n{board}", flush=True)

    streamed_final = readout(cache)
    oneshot_ids = head + pre_ids + fact_ids + suffix_ids + tail
    c1, l1 = prefill(oneshot_ids)
    oneshot = greedy(c1, l1)
    print(f"\n{'=' * 72}\nDRIFT CHECK (faithful regime): streamed==one-shot? "
          f"{streamed_final == oneshot}\n{'=' * 72}")
    print("--- streamed final ---\n" + streamed_final, flush=True)
    print("\n--- one-shot ---\n" + oneshot, flush=True)
    print("\n[streaming_test2] done.", flush=True)


if __name__ == "__main__":
    main()
