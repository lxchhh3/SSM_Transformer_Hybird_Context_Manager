"""Stage-2 hybrid benchmark — envelope depth + footprint, hybrid vs pure Mamba.

Extends the knee-sweep method (scripts/knee_sweep.py, global #75) to answer the
sharp question from lesson #17/#18: within an escalating fact stream, does a small
HYBRID (Falcon-H1-3B) reconstruct the team state faithfully DEEPER than pure Mamba
(falcon-mamba-7b), and at what carried-state cost?

Per global #75 a one-pass greedy generate() on Mamba IS a pure state-decode (no KV
to lean on), so it's the fair single code path for both families:
  - pure Mamba  -> carries ONLY a fixed recurrent state; collapses past its envelope
  - hybrid      -> carries the Mamba state + a GROWING attention KV; holds deeper,
                   but the cache grows (the #17 caveat, measured here)

Three signals per depth:
  recall   fraction of active entries whose lowest-doc-frequency 'signature' word
           survives into the decoded board (how many distinct facts got read out)
  x_count  runaway '[x]' checklist = the collapse signature from the knee sweep
  cache_MB carried cache bytes (generic tensor walk) -> constant for Mamba, rising
           for the hybrid; THIS is the tradeoff the swap would buy

Run once per model (can't co-load 14GB + 6GB in 16GB):
    STAGE2_MODEL=tiiuae/falcon-mamba-7b-instruct <spike_sse python> scripts/hybrid_stage2.py
    STAGE2_MODEL=tiiuae/Falcon-H1-3B-Instruct    <spike_sse python> scripts/hybrid_stage2.py
Writes stage2_<model>.json to the scratchpad; compare after both runs.
"""

import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MID = os.environ["STAGE2_MODEL"]
CORPUS = os.environ.get(
    "CORPUS",
    "E:/Temp/claude/D--2files-Context-Manager/bdf87e91-d089-43f3-864a-65bb5f077c5b/scratchpad/corpus.json")
OUTDIR = os.environ.get(
    "STAGE2_OUT",
    "E:/Temp/claude/D--2files-Context-Manager/7f57f5bd-c84d-422e-acb8-7614ddde3b42/scratchpad")
PROJ = {"game": "Game server (Go)", "dash": "Analytics dashboard (React/TS)",
        "etl": "ETL pipeline (Spark)", "fw": "IoT firmware (Rust)"}
TARGETS = [750, 1000, 1500, 2000, 2600]

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to("cuda").eval()

INSTR = ("Summarize the CURRENT state of these {n} project(s) for a two-person team "
         "(kevin, boss). For each project give 2-4 tight bullets: the current "
         "technical choices and who is driving. Use ONLY the facts below.")

_WORD = re.compile(r"[a-z0-9]{4,}")


def active_for(corpus, keys):
    entries = []
    for k in keys:
        for e in corpus[k]:
            e = dict(e); e["proj"] = k
            entries.append(e)
    revised = {e["revises"] for e in entries if e.get("revises")}
    return [e for e in entries if e["id"] not in revised]


def render(active, keys):
    lines = []
    for k in keys:
        lines.append(f"\n# {PROJ[k]}")
        lines += [f"- ({e['author']}/{e['type']}) {e['body']}"
                  for e in active if e["proj"] == k]
    return "\n".join(lines)


def content_words(body):
    return [w for w in _WORD.findall(body.lower()) if any(c.isalpha() for c in w)]


def build_signatures(active):
    """Each entry's most distinctive word = lowest document-frequency content word
    across the active set (ties -> longest). Presence of it in a board == that
    entry's fact survived the compression."""
    df = {}
    for e in active:
        for w in set(content_words(e["body"])):
            df[w] = df.get(w, 0) + 1
    sig = {}
    for e in active:
        ws = content_words(e["body"])
        if ws:
            sig[e["id"]] = min(ws, key=lambda w: (df[w], -len(w)))
    return sig


def encode_chat(user_text):
    enc = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return ids.to("cuda")


def cache_bytes(cache):
    """Sum bytes of every tensor reachable in the cache object — works for the
    Mamba cache (conv+recurrent states) AND a hybrid cache (those + KV)."""
    seen, total, stack = set(), 0, [(cache, 0)]
    while stack:
        o, d = stack.pop()
        if o is None or d > 6 or id(o) in seen:
            continue
        seen.add(id(o))
        if torch.is_tensor(o):
            total += o.numel() * o.element_size()
        elif isinstance(o, (list, tuple)):
            stack += [(x, d + 1) for x in o]
        elif isinstance(o, dict):
            stack += [(x, d + 1) for x in o.values()]
        else:
            dd = getattr(o, "__dict__", None)
            if dd:
                stack += [(x, d + 1) for x in dd.values()]
    return total


def measure_cache(ids):
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = getattr(out, "cache_params", None)
    if cache is None:
        cache = getattr(out, "past_key_values", None)
    return cache_bytes(cache)


def board_of(ids, n_new=320):
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n_new, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    with open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    active_all = active_for(corpus, ["game", "dash", "etl", "fw"])
    sig = build_signatures(active_all)

    cum = []
    for k in range(1, len(active_all) + 1):
        keys = list(dict.fromkeys(e["proj"] for e in active_all[:k]))
        text = render(active_all[:k], keys)
        ntok = tok(text, return_tensors="pt").input_ids.shape[1]
        cum.append((k, ntok, keys, text))

    chosen, picked_k = [], set()
    for target in TARGETS:
        k, ntok, keys, text = min(cum, key=lambda r: abs(r[1] - target))
        if k not in picked_k:
            picked_k.add(k)
            chosen.append((k, ntok, keys, text, target))

    rows = []
    print(f"\n{'#' * 72}\n# {MID}\n{'#' * 72}", flush=True)
    for k, ntok, keys, text, target in chosen:
        instr = INSTR.format(n=len(keys))
        ids = encode_chat(instr + "\n\n" + text + "\n\nStatus board:")
        mb = measure_cache(ids) / 1e6
        board = board_of(ids)
        del ids
        torch.cuda.empty_cache()

        entries = active_all[:k]
        recalled = sum(1 for e in entries if sig.get(e["id"], "\0") in board.lower())
        x_count = board.count("[x]")
        rows.append({"target": target, "entries": k, "fact_tokens": ntok,
                     "cache_MB": round(mb, 2), "recall": recalled,
                     "n_entries": len(entries),
                     "recall_frac": round(recalled / max(len(entries), 1), 3),
                     "x_count": x_count, "board_len": len(board)})
        print(f"\n{'=' * 72}\n{k} entries | {ntok} fact tok | {len(keys)} proj | "
              f"cache {mb:.2f} MB | recall {recalled}/{len(entries)} "
              f"({recalled / max(len(entries),1):.0%}) | [x]={x_count}\n{'=' * 72}\n"
              f"{board[:900]}", flush=True)

    safe = MID.split("/")[-1].replace(".", "_")
    outp = os.path.join(OUTDIR, f"stage2_{safe}.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump({"model": MID, "rows": rows}, fh, indent=2)
    print(f"\n[stage2] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
