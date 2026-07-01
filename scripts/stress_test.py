"""Realistic stress test on REAL dev content (D:\\2files\\fina\\Rebuild\\progress.md).

kevin's entries are curated from the real progress log; boss's are a mocked
parallel branch that forks at the actual decision points. We then exercise the
context manager the way two parallel CC devs would actually collide:

  Part 1  deterministic file-collisions (cheap, no model)   -> same-file overlaps
  Part 2  semantic DUPLICATE work across DIFFERENT files     -> only the model
  Part 3  CONFLICT vs compatible on same-component decisions -> the model
  Part 4  MERGE two conflicting decisions w/o losing either  -> the model
  Part 5  RETRACTION: kevin drops LOWVOL; is boss's LOWVOL work flagged stale?

Run on GPU falcon:
  PYTHONPATH=. HF_HUB_OFFLINE=1 NO_PROXY="*" TORCHDYNAMO_DISABLE=1 M4_DEVICE=cuda \
  M4_MODEL=<snapshot> <spike_sse python> scripts/stress_test.py
"""

import os

import torch

from ctx.mamba_summarizer import MambaSummarizer
from ctx.service import ContextService

# (author, etype, body, refs, entry_id)
KEVIN = [
    ("kevin", "decision", "Base = EQUAL-WEIGHT. Cap-weighting kills the illiquidity "
     "edge by underweighting the small illiquid names (Probe 2a).",
     ["model.py", "book.py"], "k_weight"),
    ("kevin", "idea", "Add a LOWVOL leg on top of ILLIQ to widen the factor set.",
     ["factors.py"], "k_lowvol"),
    ("kevin", "decision", "Factors = ILLIQ-ONLY. LOWVOL incremental IC ~0 post-2024 "
     "and adding it HURTS the net book (IR 3.30->1.68); momentum does not transfer "
     "down-cap.", ["factors.py", "book.py"], "k_factors"),
    ("kevin", "decision", "Concentration N=50, re-anchored from N=100 for "
     "lot-feasibility at the 240K sleeve.", ["book.py", "paper.py"], "k_conc"),
    ("kevin", "decision", "Survival overlay = SMA120 + floor 0.5 (PARTIAL de-gross). "
     "Slow signal avoids whipsaw; floor keeps return; maxDD -45%->-25%.",
     ["overlay.py"], "k_overlay"),
    ("kevin", "progress", "Built the paper-trading engine paper.py: current_selection "
     "/ current_exposure / target_weights / generate_orders (100-share lot floor) / "
     "Account ledger.", ["paper.py"], "k_paper"),
    ("kevin", "decision", "LLM = Sonnet per-name / Opus policy, FORCED TOOL USE so the "
     "verdict shape is schema-guaranteed by the API.",
     ["llm_client.py", "llm_advisory.py"], "k_llm"),
]

BOSS = [
    ("boss", "decision", "Base = CAP-WEIGHT with sqrt-mktcap damping - more "
     "investable and matches a standard risk model.", ["model.py", "book.py"], "b_weight"),
    ("boss", "decision", "Factors = ILLIQ + LOWVOL + a quality tilt; ILLIQ-only "
     "leaves alpha on the table.", ["factors.py", "book.py"], "b_factors"),
    ("boss", "progress", "Integrated the LOWVOL factor and added its tests.",
     ["factors.py"], "b_lowvol"),
    ("boss", "progress", "Concentration N=100 for diversification across the "
     "smallest-cap book.", ["book.py"], "b_conc"),
    ("boss", "decision", "Overlay = SMA60 + FULL de-gross to cash (floor 0) - react "
     "faster and fully protect in a crash.", ["overlay.py"], "b_overlay"),
    ("boss", "progress", "Built the order/broker layer trader.py: selection -> target "
     "weights -> round-lot orders -> account reconcile.", ["trader.py"], "b_trader"),
    ("boss", "decision", "LLM = Opus per-name / Sonnet policy with native JSON output "
     "(parse the response) - Opus handles the nuanced filing judgment better.",
     ["llm_client.py"], "b_llm"),
]


def _yn(out: str, *words: str) -> bool:
    low = out.lower()[:200]
    return any(w in low for w in words)


MODE = os.environ.get("STRESS_MODE", "gen")


def verdict(model, prompt, labels, gen_words):
    """Positive label is labels[0]. classify mode = logprob-scored verdict (no
    prose); gen mode = generate + keyword scrape (the old, hallucination-prone path)."""
    if MODE == "classify":
        r = model.classify(prompt, labels)
        probs = ", ".join(f"{k}:{v:.2f}" for k, v in r["probs"].items())
        return (r["verdict"] == labels[0]), f"[classify] {r['verdict']} (conf {r['confidence']:.2f}; {probs})"
    out = model.generate(prompt + " Then one sentence why.")
    return _yn(out, *gen_words), f"[gen] {out[:240]}"


def main() -> None:
    model_id = os.environ.get("M4_MODEL", "tiiuae/falcon-mamba-7b-instruct")
    device = os.environ.get("M4_DEVICE", "cpu")
    print(f"[stress] loading {model_id} on {device}...", flush=True)
    if device == "cuda":
        model = MambaSummarizer(model_id, device="cuda", dtype=torch.float16,
                                max_new_tokens=140)
    else:
        model = MambaSummarizer(model_id, device="cpu", dtype=torch.bfloat16,
                                max_new_tokens=140)

    print(f"[stress] use_chat={model.use_chat}", flush=True)
    svc = ContextService(":memory:")
    by_id = {}
    for author, etype, body, refs, eid in KEVIN + BOSS:
        svc.publish(author, etype, body, refs=refs, entry_id=eid)
        by_id[eid] = body

    print("\n========== PART 1: deterministic file-collisions (no model) ==========")
    for c in svc.overlaps():
        print(f"  [{c['a']} x {c['b']}] share {c['shared_refs']}")
    print("  (note: paper.py vs trader.py is NOT here - different files)")

    print("\n========== PART 2: semantic DUPLICATE across different files ==========")
    p = (f"Two developers each logged work.\n"
         f"  A (files paper.py): {by_id['k_paper']}\n"
         f"  B (files trader.py): {by_id['b_trader']}\n"
         "Are they building the SAME component (duplicated effort)?")
    flagged, detail = verdict(model, p, ["Yes", "No"], ("yes", "same", "duplicate"))
    print(f"  flagged_duplicate={flagged}\n  {detail}")

    print("\n========== PART 3: CONFLICT vs compatible (same component) ==========")
    pairs = [("k_weight", "b_weight", "base weighting"),
             ("k_overlay", "b_overlay", "survival overlay"),
             ("k_llm", "b_llm", "LLM model split")]
    for ka, kb, label in pairs:
        p = (f"Two developers made a decision about the {label} of the SAME system.\n"
             f"  A: {by_id[ka]}\n  B: {by_id[kb]}\n"
             "Are these CONFLICTING (mutually exclusive) or COMPATIBLE?")
        conf, detail = verdict(model, p, ["Conflict", "Compatible"],
                               ("conflict", "mutual", "exclusive", "cannot"))
        print(f"  [{label}] conflict={conf}\n     {detail}")

    print("\n========== PART 4: MERGE a conflict without losing either side ==========")
    p = (f"Merge these two developers' overlay decisions into ONE status that keeps "
         f"BOTH positions and explicitly flags the disagreement (do not silently pick one):\n"
         f"  kevin: {by_id['k_overlay']}\n  boss: {by_id['b_overlay']}\nMerged status:")
    out = model.generate(p, max_new_tokens=200)
    keeps_kevin = _yn(out.lower(), "sma120", "floor 0.5", "120", "partial")
    keeps_boss = _yn(out.lower(), "sma60", "floor 0", "60", "full")
    print(f"  keeps_kevin={keeps_kevin} keeps_boss={keeps_boss}\n  merged: {out[:320]}")

    print("\n========== PART 5: RETRACTION propagation ==========")
    # kevin drops the LOWVOL idea; boss's LOWVOL work now rests on a retracted premise
    svc.revert("k_lowvol")
    kevin_lowvol_status = svc.store.get_entry("k_lowvol")["status"]
    boss_lowvol_active = [e["entry_id"] for e in svc.store.active_entries(author="boss")
                          if "lowvol" in e["body"].lower()]
    print(f"  kevin k_lowvol status -> {kevin_lowvol_status}")
    print(f"  boss entries still building on LOWVOL (now a dropped premise): {boss_lowvol_active}")
    p = (f"kevin RETRACTED this idea: '{by_id['k_lowvol']}' (decided ILLIQ-only instead). "
         f"boss is still doing: '{by_id['b_lowvol']}' and '{by_id['b_factors']}'. "
         "Is boss building on a premise kevin dropped?")
    flagged, detail = verdict(model, p, ["Yes", "No"],
                              ("yes", "dropped", "retract", "stale", "premise"))
    print(f"  stale_dependency_flagged={flagged}\n  {detail}")

    print("\n[stress] done.", flush=True)


if __name__ == "__main__":
    main()
