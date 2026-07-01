"""Mechanistic diagnosis of the two stress-test anomalies. We have the model
loaded locally, so instrument it directly instead of concluding from black-box
verdicts (lesson #29/#31/#63: check the measurement before the headline).

A1: cold classify() gave 1/3 conflicts vs 3/3 generation -> measurement artifact
    (label tokenization / leading-space BPE / label prior) or real model error?
A2: duplicate verdict flipped Yes(raw) -> No(chat) -> what actually drives it?

Run: PYTHONPATH=. HF_HUB_OFFLINE=1 NO_PROXY=* TORCHDYNAMO_DISABLE=1 M4_DEVICE=cuda \
     M4_MODEL=<snap> <spike_sse python> scripts/diagnose.py
"""

import os

import torch

from ctx.mamba_summarizer import MambaSummarizer

K_PAPER = ("Built the paper-trading engine paper.py: current_selection / "
           "current_exposure / target_weights / generate_orders (100-share lot "
           "floor) / Account ledger.")
B_TRADER = ("Built the order/broker layer trader.py: selection -> target weights "
            "-> round-lot orders -> account reconcile.")
K_WEIGHT = ("Base = EQUAL-WEIGHT. Cap-weighting kills the illiquidity edge by "
            "underweighting the small illiquid names (Probe 2a).")
B_WEIGHT = ("Base = CAP-WEIGHT with sqrt-mktcap damping - more investable and "
            "matches a standard risk model.")
K_OVERLAY = ("Survival overlay = SMA120 + floor 0.5 (PARTIAL de-gross). Slow "
             "signal avoids whipsaw; floor keeps return; maxDD -45%->-25%.")
B_OVERLAY = ("Overlay = SMA60 + FULL de-gross to cash (floor 0) - react faster "
             "and fully protect in a crash.")


def conflict_prompt(a, b, label):
    return (f"Two developers made a decision about the {label} of the SAME "
            f"system.\n  A: {a}\n  B: {b}\n"
            "Are these CONFLICTING (mutually exclusive) or COMPATIBLE?")


DUP_PROMPT = (f"Two developers each logged work.\n  A (files paper.py): {K_PAPER}\n"
              f"  B (files trader.py): {B_TRADER}\n"
              "Are they building the SAME component (duplicated effort)?")


def topk_first(model, prompt, k=18):
    ids = model._encode(prompt)
    with torch.no_grad():
        logits = model.model(ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k)
    return [(repr(model.tok.decode([int(i)])), round(float(v), 4))
            for v, i in zip(vals, idx)]


def label_logp(model, prompt, labels):
    base = model._encode(prompt)
    blen = base.shape[1]
    res = {}
    for lab in labels:
        lid = model.tok(lab, add_special_tokens=False,
                        return_tensors="pt").input_ids.to(model.device)
        full = torch.cat([base, lid], dim=1)
        with torch.no_grad():
            lp = torch.log_softmax(model.model(full).logits, dim=-1).float()
        n = lid.shape[1]
        toks = [model.tok.decode([int(t)]) for t in lid[0]]
        mean = sum(lp[0, blen + i - 1, full[0, blen + i]].item() for i in range(n)) / n
        res[lab] = {"n_tok": n, "toks": toks, "mean_logp": round(mean, 3)}
    return res


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    mid = os.environ.get("M4_MODEL", "tiiuae/falcon-mamba-7b-instruct")
    model = MambaSummarizer(mid, device="cuda", dtype=torch.float16, max_new_tokens=24)
    print(f"use_chat={model.use_chat}")

    section("0. What the chat template ACTUALLY feeds the model (literal)")
    lit = model.tok.apply_chat_template(
        [{"role": "user", "content": "ARE THESE CONFLICTING OR COMPATIBLE?"}],
        add_generation_prompt=True, tokenize=False)
    print(repr(lit))

    cases = [
        ("A1a CONFLICT base-weighting (classify said COMPATIBLE = wrong)",
         conflict_prompt(K_WEIGHT, B_WEIGHT, "base weighting")),
        ("A1b CONFLICT overlay (classify said CONFLICT = right) - contrast",
         conflict_prompt(K_OVERLAY, B_OVERLAY, "survival overlay")),
    ]
    variants = ["Conflict", " Conflict", "conflict", "CONFLICT",
                "Compatible", " Compatible", "compatible", "COMPATIBLE"]
    for title, prompt in cases:
        section(title)
        print("-- top-18 tokens the model WANTS to emit first:")
        for tok, p in topk_first(model, prompt):
            print(f"     {p:.4f}  {tok}")
        print("-- label scores (mean logprob; note leading-space + casing):")
        for lab, d in label_logp(model, prompt, variants).items():
            print(f"     {d['mean_logp']:8.3f}  {lab!r:14} -> {d['n_tok']} tok {d['toks']}")
        print(f"-- greedy first words: {model.generate(prompt, max_new_tokens=14)!r}")

    section("A2 DUPLICATE flip: raw prompt vs chat template (same content)")
    print("-- top-18 first tokens (chat):")
    for tok, p in topk_first(model, DUP_PROMPT):
        print(f"     {p:.4f}  {tok}")
    print(f"-- CHAT greedy:  {model.generate(DUP_PROMPT, max_new_tokens=20)!r}")
    model.use_chat = False
    print(f"-- RAW  greedy:  {model.generate(DUP_PROMPT, max_new_tokens=20)!r}")
    model.use_chat = True
    print("-- label scores yes/no (chat):")
    for lab, d in label_logp(model, DUP_PROMPT,
                             ["Yes", " Yes", "No", " No"]).items():
        print(f"     {d['mean_logp']:8.3f}  {lab!r:8} -> {d['toks']}")

    print("\n[diagnose] done.")


if __name__ == "__main__":
    main()
