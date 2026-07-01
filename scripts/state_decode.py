"""Decode from the CARRIED SSM state alone — the clean 'is the compression that
lives in the tensor readable?' test.

Mamba has no KV cache: the ONLY thing carried from prompt to generation is the
fixed-size recurrent state (conv_states + recurrent_states per layer). So:

  A (facts in state): prefill [instruction + FACTS + "Status board:"] -> grab the
      cache -> DROP the input tokens -> greedy-decode the board from the cache
      alone. Whatever facts appear can ONLY have come from the fixed-size tensor.
  B (control, no facts): prefill [instruction + "Status board:"] with the facts
      REMOVED. Same instruction, same-size state, but the facts never entered it.

If A reads out real, current facts and B cannot, the compression provably lives
in the tensor. We also print the cache byte-size: it is IDENTICAL for A and B and
for 775 vs 2464 input tokens — that constant size IS the compression.

Decode is strictly one-token-at-a-time: once the cache is populated the FalconMamba
decode path collapses to seq_len=1 (modeling_falcon_mamba.py L354-356), which is
exactly what broke the old chunked-carry probe.
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


def render(active, keys):
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
    return ids.to("cuda")


def prefill(ids):
    """Full-length prefill on a fresh cache (handles arbitrary seq_len)."""
    with torch.no_grad():
        out = model(ids, use_cache=True)
    return out.cache_params, out.logits[0, -1]


def feed(cache, tok_id):
    """Single-token decode step against a populated cache (seq_len==1)."""
    inp = torch.tensor([[tok_id]], device="cuda")
    with torch.no_grad():
        out = model(inp, cache_params=cache, use_cache=True)
    return out.cache_params, out.logits[0, -1]


def greedy(cache, logits, n):
    ids = []
    eos = tok.eos_token_id
    for _ in range(n):
        nxt = int(logits.argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        cache, logits = feed(cache, nxt)
    return tok.decode(ids, skip_special_tokens=True).strip()


def cache_bytes(cache):
    total = 0
    for layer in cache.layers:
        for name in ("conv_states", "recurrent_states"):
            t = getattr(layer, name, None)
            if t is not None:
                total += t.numel() * t.element_size()
    return total


INSTR = ("Summarize the CURRENT state of {scope} for a two-person team (kevin, "
         "boss). For each project give 2-4 tight bullets: the current technical "
         "choices and who is driving. Use ONLY the facts below.")


def run(keys, corpus, n_new=300):
    active = active_for(corpus, keys)
    facts = render(active, keys)
    scope = "this project" if len(keys) == 1 else f"these {len(keys)} projects"
    instr = INSTR.format(scope=scope)
    ntok = tok(facts, return_tensors="pt").input_ids.shape[1]

    ids_A = encode_chat(instr + "\n\n" + facts + "\n\nStatus board:")
    ids_B = encode_chat(instr + "\n\n(no facts provided)\n\nStatus board:")

    print(f"\n{'=' * 72}\n{len(keys)} project(s) | {len(active)} active | "
          f"{ntok} fact tokens\n{'=' * 72}", flush=True)

    cA, lA = prefill(ids_A)
    nbytes = cache_bytes(cA)
    print(f"[cache] carried state = {nbytes/1e6:.2f} MB  "
          f"(fixed — independent of the {ntok} fact tokens)", flush=True)
    del ids_A  # the fact tokens are GONE; only the fixed-size cache cA remains
    torch.cuda.empty_cache()
    print("\n--- A: decoded from the state that SAW the facts ---", flush=True)
    print(greedy(cA, lA, n_new), flush=True)

    cB, lB = prefill(ids_B)
    print("\n--- B: control, same instruction, state NEVER saw the facts ---",
          flush=True)
    print(greedy(cB, lB, n_new), flush=True)


def main():
    with open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    run(["game"], corpus)                       # 775 tok — faithful regime
    run(["game", "dash", "etl"], corpus)        # 2464 tok — degrading regime
    print("\n[state_decode] done.", flush=True)


if __name__ == "__main__":
    main()
