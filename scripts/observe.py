"""Observe the SSM doing its actual job: stream the team's CURRENT state (the
active set across 4 heterogeneous projects, ~100 entries) through falcon-mamba in
ONE forward, and read out a status board. No accuracy scoring yet (the index comes
later) — just see whether a coherent, compressed 'where are we' falls out of a
long, messy, multi-domain stream. This is the SSM used correctly: the recurrent
state compresses the whole stream; the board decodes from it.
"""

import json
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows GBK stdout (#5/#10)

from ctx.mamba_summarizer import MambaSummarizer

CORPUS = os.environ.get(
    "CORPUS",
    "E:/Temp/claude/D--2files-Context-Manager/bdf87e91-d089-43f3-864a-65bb5f077c5b/scratchpad/corpus.json")
PROJ = {"game": "Game server (Go)", "dash": "Analytics dashboard (React/TS)",
        "etl": "ETL pipeline (Spark)", "fw": "IoT firmware (Rust)"}


def main():
    with open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    entries = []
    for key, arr in corpus.items():
        for e in arr:
            e["proj"] = key
            entries.append(e)
    revised = {e["revises"] for e in entries if e.get("revises")}
    active = [e for e in entries if e["id"] not in revised]

    lines = []
    for key in corpus:
        lines.append(f"\n# {PROJ[key]}")
        lines += [f"- ({e['author']}/{e['type']}) {e['body']}"
                  for e in active if e["proj"] == key]
    state_text = "\n".join(lines)

    model = MambaSummarizer(os.environ["M4_MODEL"], device="cuda",
                            dtype=torch.float16, max_new_tokens=480)
    ntok = model.tok(state_text, return_tensors="pt").input_ids.shape[1]
    print(f"total entries={len(entries)}  active(current)={len(active)}  "
          f"state tokens={ntok}", flush=True)

    prompt = ("Below is the CURRENT state of a two-person team (kevin and boss) across "
              "four projects. Each line is an active decision, progress note, idea, or "
              "bug (superseded/reverted items already removed). Write a concise STATUS "
              "BOARD: for EACH project give 2-4 tight bullets on where it stands now, "
              "the key current technical choices, and who is driving. Be specific.\n\n"
              + state_text + "\n\nSTATUS BOARD:")
    print("\n===== falcon-mamba status board (decoded from the streamed state) =====\n",
          flush=True)
    print(model.generate(prompt, max_new_tokens=480,
                         repetition_penalty=1.3, no_repeat_ngram_size=3))
    print("\n[observe] done.", flush=True)


if __name__ == "__main__":
    main()
