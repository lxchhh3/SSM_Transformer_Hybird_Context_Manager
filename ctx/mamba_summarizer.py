"""Mamba-backed Summarizer — the streaming team-state digest (branch A wiring).

Plugs into SSMEngine's fold/render contract. The state IS the running digest
text: fold() asks Mamba to integrate one new event into it; render() returns it.
Store stays the source of truth, so the engine never relies on the model to
forget — a revert just replays the affected tail.

Cost note: each fold is ONE bounded generation (the digest stays short), which is
the #1 win over re-stuffing an ever-growing log into a transformer on every
update. A mid-stream revert replays at most `checkpoint_every` folds, so that
knob trades revert latency against checkpoint memory. Real cost + merge quality
get measured at M4.

Requires ctx_env (torch cu128 + transformers). Not imported by the GPU-free tests.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ctx.prompts import FOLD_PROMPT as _PROMPT


class MambaSummarizer:
    def __init__(self, model: str = "state-spaces/mamba-2.8b-hf",
                 device: str | None = None, dtype=None, max_new_tokens: int = 200,
                 device_map=None, max_memory=None, use_chat: bool | None = None):
        self.tok = AutoTokenizer.from_pretrained(model)
        # Use the model's instruct chat template when present: cleaner
        # instruction-following, no raw-prompt scaffolding leaking into output.
        self.use_chat = (getattr(self.tok, "chat_template", None) is not None
                         if use_chat is None else use_chat)
        if device_map is not None:
            # accelerate spreads layers across GPU/CPU; max_memory caps the GPU so a
            # model slightly too big for VRAM doesn't OOM (overflow goes to CPU).
            # Inputs go to cuda:0 and accelerate dispatches to the offloaded layers.
            if dtype is None:
                dtype = torch.float16
            kwargs = {"dtype": dtype, "device_map": device_map}
            if max_memory is not None:
                kwargs["max_memory"] = max_memory
            self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            self.device = "cuda"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            if dtype is None:
                dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _encode(self, prompt: str):
        if self.use_chat:
            enc = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt")
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            return ids.to(self.device)
        return self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int | None = None,
                 repetition_penalty: float | None = None,
                 no_repeat_ngram_size: int | None = None) -> str:
        ids = self._encode(prompt)
        kw = dict(max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False)
        if repetition_penalty:
            kw["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size:
            kw["no_repeat_ngram_size"] = no_repeat_ngram_size
        with torch.no_grad():
            out = self.model.generate(ids, **kw)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def classify(self, prompt: str, labels: list[str]) -> dict:
        """Verdict-not-prose: score each candidate label by length-normalized
        logprob given the prompt, return the argmax + confidence. The model emits
        NO prose, so there is no free-text rationale to hallucinate."""
        import math
        base = self._encode(prompt)
        base_len = base.shape[1]
        scores = []
        for label in labels:
            lab = self.tok(label, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(self.device)
            full = torch.cat([base, lab], dim=1)
            with torch.no_grad():
                logp = torch.log_softmax(self.model(full).logits, dim=-1)
            n = lab.shape[1]
            s = sum(logp[0, base_len + i - 1, full[0, base_len + i]].item()
                    for i in range(n))
            scores.append(s / max(n, 1))
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        z = sum(exps)
        probs = {lab: e / z for lab, e in zip(labels, exps)}
        best = max(range(len(labels)), key=lambda i: scores[i])
        return {"verdict": labels[best], "confidence": probs[labels[best]],
                "probs": probs}

    def initial(self) -> str:
        return "(no activity yet)"

    def fold(self, state: str, entry: dict) -> str:
        prompt = _PROMPT.format(state=state, author=entry["author"],
                                type=entry["type"], body=entry["body"])
        return self.generate(prompt) or state

    def render(self, state: str) -> str:
        return state
