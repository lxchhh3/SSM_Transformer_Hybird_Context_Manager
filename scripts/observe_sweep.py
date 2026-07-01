"""Length sweep with ONE sane decoding setting (repetition_penalty only, no n-gram
block) to isolate the length effect from decoding artifacts. Summarize the current
state at 1 -> 2 -> 4 projects and watch WHERE coherence breaks. This tells us if the
SSM's fixed state blurs as the streamed input grows (the real question), separate
from the greedy-loop / n-gram-salad artifacts.
"""

import json
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ctx.mamba_summarizer import MambaSummarizer

CORPUS = os.environ.get(
    "CORPUS",
    "E:/Temp/claude/D--2files-Context-Manager/bdf87e91-d089-43f3-864a-65bb5f077c5b/scratchpad/corpus.json")
PROJ = {"game": "Game server (Go)", "dash": "Analytics dashboard (React/TS)",
        "etl": "ETL pipeline (Spark)", "fw": "IoT firmware (Rust)"}


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


def main():
    with open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    model = MambaSummarizer(os.environ["M4_MODEL"], device="cuda",
                            dtype=torch.float16, max_new_tokens=340)

    for keys in (["game"], ["game", "dash"], ["game", "dash", "etl"],
                 ["game", "dash", "etl", "fw"]):
        active = active_for(corpus, keys)
        text = render(active, keys)
        ntok = model.tok(text, return_tensors="pt").input_ids.shape[1]
        scope = "this project" if len(keys) == 1 else f"these {len(keys)} projects"
        prompt = (f"Summarize the CURRENT state of {scope} for a two-person team "
                  "(kevin, boss). For each project give 2-4 tight bullets: the current "
                  "technical choices and who is driving. Use ONLY the facts below.\n\n"
                  + text + "\n\nStatus board:")
        print(f"\n{'=' * 72}\n{len(keys)} project(s) | {len(active)} active | "
              f"{ntok} state tokens\n{'=' * 72}", flush=True)
        print(model.generate(prompt, max_new_tokens=340), flush=True)  # plain greedy

    print("\n[sweep] done.", flush=True)


if __name__ == "__main__":
    main()
