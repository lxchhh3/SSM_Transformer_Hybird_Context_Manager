"""M4 benchmark — runs the 4 acceptance criteria through a model and reports.

Model-agnostic: works with any object exposing initial/fold/render (the
Summarizer contract) plus generate(prompt)->str. A FakeModel drives the harness
tests; MambaSummarizer (falcon / mamba) drives the real eval. The harness only
ORCHESTRATES + measures — you read the outputs and judge "powerful enough".

Note: criterion #2 (inverted change) is correct for ANY model, because the
retraction is the store's job — the engine rebuilds the digest from the active
set, so the superseded fact is gone by construction, not by the model choosing
to forget. The benchmark verifies that end-to-end.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

from ctx.ssm_engine import SSMEngine
from ctx.store import Store

# A realistic two-dev stream — the kind of parallel work that collides:
# both drift toward caching, and the two cache *ideas* are duplicates.
_STREAM = [
    ("kevin", "progress", "starting the price-fetch module in fetch.py"),
    ("boss", "progress", "setting up the DB schema in models.py"),
    ("kevin", "progress", "fetch.py pulls quotes ok, adding retry logic"),
    ("boss", "decision", "going with SQLite for the local cache"),
    ("kevin", "progress", "added a small cache inside fetch.py"),
    ("boss", "progress", "adding a cache layer in cache.py"),
    ("kevin", "idea", "we should memoize the repeated quote calls"),
    ("boss", "idea", "add an LRU cache for repeated quote lookups"),
]


def run_benchmark(model: Any, n_stream: int = 8) -> dict:
    return {
        "high_freq": _high_freq(model, n_stream),
        "inverted": _inverted(model),
        "duplicate": _duplicate(model),
        "merge": _merge(model),
    }


def _high_freq(model: Any, n_stream: int) -> dict:
    store = Store(":memory:")
    eng = SSMEngine(store, model, checkpoint_every=8)
    stream = _STREAM[:n_stream]
    lat = []
    t0 = time.perf_counter()  # perf_counter: time.time() is ~15ms-coarse on Windows
    for author, etype, body in stream:
        store.publish(author, etype, body)
        s = time.perf_counter()
        eng.sync()
        lat.append(time.perf_counter() - s)
    dt = time.perf_counter() - t0
    return {
        "n_events": len(stream),
        "total_s": round(dt, 3),
        "events_per_s": round(len(stream) / dt, 3) if dt else None,
        "mean_sync_s": round(statistics.mean(lat), 3) if lat else None,
        "final_digest": eng.digest(),
    }


def _inverted(model: Any) -> dict:
    store = Store(":memory:")
    eng = SSMEngine(store, model, checkpoint_every=8)
    d = store.publish("boss", "decision", "we will use Postgres for storage")
    eng.sync()
    before = eng.digest()
    store.supersede(d, "we will use SQLite for storage")  # the inversion
    eng.sync()
    after = eng.digest()
    return {
        "digest_before": before,
        "digest_after": after,
        "reverted_fact_absent": "postgres" not in after.lower(),
        "new_fact_present": "sqlite" in after.lower(),
    }


def _duplicate(model: Any) -> dict:
    store = Store(":memory:")
    store.publish("kevin", "idea", "add an LRU cache in front of the price API",
                  refs=["cache.py"])
    store.publish("boss", "idea", "memoize the quote fetcher to cut API calls",
                  refs=["quotes.py"])  # different files -> file-overlap misses it
    ideas = [e for e in store.active_entries() if e["type"] == "idea"]
    prompt = (
        "Two developers each logged an idea:\n"
        f"  A) {ideas[0]['body']}\n  B) {ideas[1]['body']}\n"
        "Do these describe the same underlying work? "
        "Answer 'yes' or 'no', then one sentence why."
    )
    out = model.generate(prompt)
    low = out.lower()
    flagged = any(w in low for w in ("yes", "same", "duplicate", "similar", "overlap"))
    return {"prompt": prompt, "model_output": out, "flagged_duplicate": flagged}


def _merge(model: Any) -> dict:
    docs = [
        "Note A: the price service polls upstream. KEYFACT: request timeout is 8s.",
        "Note B: results are cached locally. KEYFACT: cache TTL is 60 seconds.",
        "Note C: failed calls are retried. KEYFACT: backoff is exponential.",
    ]
    facts = ["8s", "60 seconds", "exponential"]
    prompt = ("Merge these three notes into one coherent note, preserving every "
              "fact:\n\n" + "\n\n".join(docs) + "\n\nMerged note:")
    out = model.generate(prompt)
    low = out.lower()
    kept = [f for f in facts if f.lower() in low]
    return {
        "prompt": prompt, "model_output": out,
        "facts": facts, "facts_kept": kept,
        "recall": round(len(kept) / len(facts), 3),
    }


def format_report(report: dict) -> str:
    h, inv = report["high_freq"], report["inverted"]
    dup, mrg = report["duplicate"], report["merge"]
    return "\n".join([
        "=== M4 BENCHMARK ===",
        "",
        f"[#1 high-freq]  {h['n_events']} events in {h['total_s']}s "
        f"= {h['events_per_s']} ev/s (mean sync {h['mean_sync_s']}s)",
        f"    digest: {h['final_digest'][:240]}",
        "",
        f"[#2 inverted]   reverted-fact-absent={inv['reverted_fact_absent']} "
        f"new-fact-present={inv['new_fact_present']}",
        f"    after: {inv['digest_after'][:240]}",
        "",
        f"[#3 duplicate]  flagged={dup['flagged_duplicate']}",
        f"    model: {dup['model_output'][:240]}",
        "",
        f"[#4 merge]      key-fact recall={mrg['recall']} kept={mrg['facts_kept']}",
        f"    merged: {mrg['model_output'][:240]}",
    ])
