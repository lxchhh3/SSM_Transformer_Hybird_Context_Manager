"""Knee sweep — pin the faithful-envelope size of falcon-mamba's fixed state.

Same 'decode from the carried state alone' method as scripts/state_decode.py, but
across finely-graded prefix lengths that bracket the 775-faithful -> 1507-conflating
band. For each target token count we build a prefix of the active set, prefill
[instr + facts + 'Status board:'], DROP the input tokens, and greedy-decode the
board one token at a time from the cache alone. We print the '[x]' checklist count
(the collapse signature) so the knee is crisp, not just eyeballed.
"""

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
TARGETS = [850, 1000, 1150, 1300, 1450, 1600]

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
    return "\n".join(lines), keys


def encode_chat(user_text):
    enc = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return ids.to("cuda")


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


INSTR = ("Summarize the CURRENT state of these {n} project(s) for a two-person team "
         "(kevin, boss). For each project give 2-4 tight bullets: the current "
         "technical choices and who is driving. Use ONLY the facts below.")


def main():
    with open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    active_all = active_for(corpus, ["game", "dash", "etl", "fw"])

    # cumulative token count for each prefix length
    cum = []
    for k in range(1, len(active_all) + 1):
        text, keys = render(active_all[:k])
        ntok = tok(text, return_tensors="pt").input_ids.shape[1]
        cum.append((k, ntok, text, keys))

    chosen = []
    for target in TARGETS:
        k, ntok, text, keys = min(cum, key=lambda r: abs(r[1] - target))
        if k not in [c[0] for c in chosen]:
            chosen.append((k, ntok, text, keys, target))

    for k, ntok, text, keys, target in chosen:
        instr = INSTR.format(n=len(keys))
        ids = encode_chat(instr + "\n\n" + text + "\n\nStatus board:")
        cache, logits = prefill(ids)
        del ids
        torch.cuda.empty_cache()
        board = greedy(cache, logits, 300)
        nx = board.count("[x]")
        print(f"\n{'=' * 72}\n~{target} target | {ntok} fact tokens | {k} entries | "
              f"{len(keys)} proj | [x]-count={nx}\n{'=' * 72}\n{board}", flush=True)

    print("\n[knee_sweep] done.", flush=True)


if __name__ == "__main__":
    main()
