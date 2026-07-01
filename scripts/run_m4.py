"""Run the M4 benchmark on a real model, on CPU (per 'don't put it in my gpu').

falcon-mamba-7b is loaded in bf16 to fit ~14 GB RAM (fp32 would need ~28 GB).
Streaming task #1 is slow on CPU for a 7B — keep M4_NSTREAM small; on GPU it
would be far faster. The quality tasks (#3/#4) are just 2 generations.

Usage:
    HF_ENDPOINT=https://hf-mirror.com NO_PROXY=hf-mirror.com CUDA_VISIBLE_DEVICES=-1 \
    M4_MODEL=tiiuae/falcon-mamba-7b-instruct M4_NSTREAM=4 \
    <spike_sse python> scripts/run_m4.py
"""

import os

import torch

from ctx.benchmark import format_report, run_benchmark
from ctx.mamba_summarizer import MambaSummarizer

MODEL = os.environ.get("M4_MODEL", "tiiuae/falcon-mamba-7b-instruct")
N = int(os.environ.get("M4_NSTREAM", "4"))
DEVICE = os.environ.get("M4_DEVICE", "cpu")


def main() -> None:
    if DEVICE == "cuda":
        print(f"[M4] loading {MODEL} fully on GPU (fp16)...", flush=True)
        model = MambaSummarizer(MODEL, device="cuda",
                                dtype=torch.float16, max_new_tokens=160)
    else:
        print(f"[M4] loading {MODEL} on CPU (bf16)...", flush=True)
        model = MambaSummarizer(MODEL, device="cpu", dtype=torch.bfloat16,
                                max_new_tokens=160)
    print(f"[M4] running benchmark (n_stream={N})...", flush=True)
    report = run_benchmark(model, n_stream=N)
    print(format_report(report), flush=True)


if __name__ == "__main__":
    main()
