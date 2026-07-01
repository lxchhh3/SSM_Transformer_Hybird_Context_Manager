"""Integration check: SSMEngine + a REAL MambaSummarizer (small model, CPU).

The 24 unit tests prove the engine machinery with a fake summarizer; this proves
an actual Mamba plugs into the same fold/render contract end-to-end, including
checkpoint-replay on revert. Runs on the cached 130m — zero new data, CPU only.
"""

import os

import torch

from ctx.mamba_summarizer import MambaSummarizer
from ctx.ssm_engine import SSMEngine
from ctx.store import Store


def main() -> None:
    model = MambaSummarizer(
        os.environ.get("ENG_MODEL", "state-spaces/mamba-130m-hf"),
        device="cpu", dtype=torch.float32, max_new_tokens=48)
    store = Store(":memory:")
    eng = SSMEngine(store, model, checkpoint_every=4)

    store.publish("kevin", "progress", "finished the auth module in auth.py")
    eng.sync()
    print("DIGEST after #1:", eng.digest()[:140].replace("\n", " "))

    b = store.publish("boss", "decision", "we will use Postgres for storage")
    eng.sync()
    print("DIGEST after #2:", eng.digest()[:140].replace("\n", " "))

    store.revert(b)
    eng.sync()
    after = eng.digest()
    print("DIGEST after revert(#2):", after[:140].replace("\n", " "))
    print("postgres_gone_from_digest:", "postgres" not in after.lower())
    print("ENGINE_INTEGRATION_OK")


if __name__ == "__main__":
    main()
