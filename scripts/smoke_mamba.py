"""M0 smoke test: prove a clean-float Mamba loads + runs in pure-PyTorch
transformers on the RTX 5070 Ti (Blackwell), and get a first throughput number.

Uses the smallest HF-native Mamba (state-spaces/mamba-790m-hf) for a fast signal.
This checkpoint runs WITHOUT the mamba-ssm CUDA kernels (the thing that's painful
on Windows/Blackwell) — transformers' slow path is pure torch. We scale up to
2.8b / falcon-mamba-7b-instruct for the real M4 benchmark once the path is proven.
"""

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("SMOKE_MODEL", "state-spaces/mamba-130m-hf")


def main() -> None:
    dev = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| device:", dev, gpu)
    dtype = torch.float16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(dev)
    model.eval()

    # --- criterion #1 first signal: ingest throughput (prefill a long log) ---
    log = ("Kevin: finished auth module, touching auth.py. "
           "Boss: starting 2FA on auth.py and totp.py. ") * 64
    ids = tok(log, return_tensors="pt").input_ids.to(dev)
    n_in = ids.shape[1]
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        model(ids)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"INGEST: prefilled {n_in} tokens in {dt:.3f}s = {n_in / dt:,.0f} tok/s")

    # --- generation sanity ---
    prompt = "Team status: Kevin finished auth. Boss is currently working on"
    pid = tok(prompt, return_tensors="pt").input_ids.to(dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(pid, max_new_tokens=40, do_sample=False)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    ng = out.shape[1] - pid.shape[1]
    print("GEN:", tok.decode(out[0], skip_special_tokens=True))
    print(f"GEN: {ng} tokens in {dt:.2f}s = {ng / dt:.1f} tok/s")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
