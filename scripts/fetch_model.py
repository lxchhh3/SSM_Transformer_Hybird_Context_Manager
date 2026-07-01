"""Fetch a model snapshot to the HF cache via the configured endpoint (mirror).

PURE DOWNLOAD — does not import torch and never touches the GPU. Grabs
safetensors only (skips redundant .bin/.pth) to save data, and resumes a
partial download if re-run (huggingface_hub caches + resumes by default).

Usage (direct mirror route, #43):
    HF_ENDPOINT=https://hf-mirror.com NO_PROXY=hf-mirror.com \
    FETCH_MODEL=tiiuae/falcon-mamba-7b-instruct <spike_sse python> scripts/fetch_model.py
"""

import os
import time

from huggingface_hub import snapshot_download

REPO = os.environ.get("FETCH_MODEL", "tiiuae/falcon-mamba-7b-instruct")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def main() -> None:
    print(f"[fetch] {REPO} via {ENDPOINT} (safetensors only, resumable)", flush=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=REPO,
        ignore_patterns=["*.bin", "*.pth", "*.pt", "*.h5", "*.msgpack",
                         "*.gguf", "*.onnx", "original/*"],
    )
    dt = time.time() - t0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    print(f"[fetch] DONE in {dt:.1f}s -> {path}", flush=True)
    print(f"[fetch] cache size: {total / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
