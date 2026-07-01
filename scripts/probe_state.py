"""Feasibility probe: can we carry the Mamba recurrent state (cache_params) across
chunked inputs and match one-shot processing? If yes, the streaming-state design
(stream events -> carry state -> read out digest from state) is viable, and we're
finally using the SSM as an SSM rather than looping an LLM over text.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = os.environ["M4_MODEL"]
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()

text = ("Team change log. "
        + " ".join(f"event {i}: file_{i % 9}.py edited by dev{i % 2}; note {i}."
                   for i in range(60)))
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
T = ids.shape[1]
print("sequence length T =", T)

with torch.no_grad():
    full = model(ids, use_cache=True)
last_full = full.logits[0, -1].float()

half = T // 2
try:
    with torch.no_grad():
        o1 = model(ids[:, :half], use_cache=True)
        o2 = model(ids[:, half:], cache_params=o1.cache_params, use_cache=True,
                   cache_position=torch.arange(half, T, device="cuda"))
    last_chunk = o2.logits[0, -1].float()
    diff = (last_full - last_chunk).abs().max().item()
    print("max|delta logit|  one-shot vs carried-cache:", round(diff, 4))
    print("next-token argmax match:",
          int(last_full.argmax()) == int(last_chunk.argmax()))
    print("one-shot next:", repr(tok.decode([int(last_full.argmax())])),
          "| carried next:", repr(tok.decode([int(last_chunk.argmax())])))
    print("VERDICT:", "STATE CARRIES (streaming-state viable)" if diff < 0.5
          else "MISMATCH (need different cache handling)")
except Exception as e:  # noqa: BLE001 - probe surfaces the API shape
    print("cache-carry call raised:", type(e).__name__, str(e)[:300])
    print("=> need to inspect the FalconMamba cache API for this transformers build")

print("[probe] done.")
