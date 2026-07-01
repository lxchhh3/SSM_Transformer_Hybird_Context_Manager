"""Hybrid feasibility probe (Stage 1) — does a small SSM/attention HYBRID load
and generate coherently on THIS box, kernel-free?

Context (research/deep_research_report.md): pure Mamba is weak on retrieval +
format adherence (findings 3, 11); the field's fix is a sparse ~8% attention
hybrid (findings 9-11). Report OQ1 flags that the strong hybrid evidence is all
7-8B and it's UNKNOWN whether it holds at 2.8-3B on one consumer GPU. This is
that test at our scale. Default candidate is the report's small hybrid,
Zamba2-2.7B (Mamba2 + shared attention).

The gate this answers: on Blackwell/Windows we have NO mamba-ssm / causal-conv1d
CUDA kernels (see memory context-manager-env), so the model must run on the pure
-PyTorch fallback. This probe reports whether it does, plus VRAM, tokens/sec, and
whether the output is coherent (not kernel-fallback garbage).

Run (model must be fetched first via scripts/fetch_model.py):
    HYBRID_MODEL=Zyphra/Zamba2-2.7B-instruct <spike_sse python> scripts/hybrid_probe.py

NOTE: needs the GPU. Do NOT set CUDA_VISIBLE_DEVICES=-1 here (that flag is for the
CPU-only DOWNLOAD step). Check the GPU is free first if training may be running.
"""

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("HYBRID_MODEL", "Zyphra/Zamba2-2.7B-instruct")

# A status-board fold, the real workload: fold one event into a short board.
PROMPT = (
    "You maintain a concise shared status board for a small dev team.\n"
    "Current status:\n- [progress] kevin wired the MCP server over LAN\n\n"
    "New event from boss (decision): use SQLite WAL for the store\n\n"
    "Rewrite the status to fold in this event in <= 60 words. Favor the most "
    "recent and active work; let older detail fade. Note who owns what.\n"
    "Updated status:\n"
)


def _kernel_flags() -> dict:
    """Best-effort: report whether the fast SSM kernels are visible to
    transformers. Absent -> the model must use the sequential fallback."""
    import transformers.utils.import_utils as iu
    flags = {}
    for fn in ("is_mamba_ssm_available", "is_mamba_2_ssm_available",
               "is_causal_conv1d_available"):
        f = getattr(iu, fn, None)
        try:
            flags[fn] = bool(f()) if f else "no-such-check"
        except Exception as e:  # pragma: no cover
            flags[fn] = f"err:{type(e).__name__}"
    return flags


def main() -> None:
    print(f"[probe] transformers on GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[probe] kernel availability: {_kernel_flags()}", flush=True)
    print(f"[probe] loading {MODEL} (fp16, cuda)...", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    model.eval()
    load_s = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"[probe] LOADED in {load_s:.1f}s | weights+load peak VRAM {vram:.2f} GB", flush=True)

    use_chat = getattr(tok, "chat_template", None) is not None
    if use_chat:
        ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    else:
        ids = tok(PROMPT, return_tensors="pt").input_ids
    ids = ids.to("cuda")

    # warmup (kernel/graph init) then timed run
    with torch.no_grad():
        model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=120, do_sample=False)
    gen_s = time.time() - t0
    n_new = out.shape[1] - ids.shape[1]
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n[probe] generated {n_new} tok in {gen_s:.1f}s "
          f"= {n_new / gen_s:.1f} tok/s | gen peak VRAM {peak:.2f} GB", flush=True)
    print("[probe] ---- output ----", flush=True)
    print(text, flush=True)
    print("[probe] ----------------", flush=True)
    print("[probe] PASS if the above is a coherent, format-following board.", flush=True)


if __name__ == "__main__":
    main()
