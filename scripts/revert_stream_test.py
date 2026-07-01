"""Revert-in-stream — the store<->SSM boundary (criterion #2) on the TRUE
cache-carry path. The store gives exact retraction; the SSM keeps a faithful
streaming view. On a change we re-derive the active set from the store and re-fold
the stream (or replay from a checkpoint) — the SSM never has to 'un-remember'.

Two scenarios through the real ctx.store.Store:
  S1 plain revert: a distinctive fact (Kafka bus) is streamed, then reverted.
     Replay the tail from a pre-fact checkpoint -> the fact must VANISH, the rest
     survive. Bonus: checkpoint-replay == full re-stream of active_after (the
     ssm_engine replay optimization is exact on the real cache path).
  S2 supersede->revert->restore: Redis superseded by Postgres, then reverted; the
     store restores Redis and the board must flip Postgres->Redis.
"""

import copy
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ctx.store import Store

MID = os.environ["M4_MODEL"]
PREAMBLE = ("Summarize the CURRENT state of the game-server project for a two-person "
            "team (kevin, boss). Give tight bullets: current technical choices and "
            "who is driving. Use ONLY the facts below.")

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).to("cuda").eval()


def encode_chat(user_text):
    enc = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return ids[0].tolist()


def split_template():
    a, b = encode_chat("AAAA"), encode_chat("AAAA BBBB")
    h = 0
    while h < min(len(a), len(b)) and a[h] == b[h]:
        h += 1
    t = 0
    while t < min(len(a), len(b)) - h and a[-1 - t] == b[-1 - t]:
        t += 1
    return a[:h], (a[len(a) - t:] if t else [])


HEAD, TAIL = split_template()
SUFFIX = tok("\n\nStatus board:", add_special_tokens=False).input_ids
PRE = tok(PREAMBLE + "\n\n# Game server (Go)", add_special_tokens=False).input_ids


def toks(text):
    return tok(text, add_special_tokens=False).input_ids


def prefill(ids):
    with torch.no_grad():
        out = model(torch.tensor([ids], device="cuda"), use_cache=True)
    return out.cache_params, out.logits[0, -1]


def feed(cache, tid):
    with torch.no_grad():
        out = model(torch.tensor([[tid]], device="cuda"), cache_params=cache, use_cache=True)
    return out.cache_params, out.logits[0, -1]


def stream(seed, seg_ids_list):
    """Fold token segments onto a starting cache (fresh if seed is None)."""
    cache, logits = seed, None
    for seg in seg_ids_list:
        for tid in seg:
            if cache is None:
                cache, logits = prefill([tid])
            else:
                cache, logits = feed(cache, tid)
    return cache, logits


def greedy(cache, logits, n=480):
    ids, eos = [], tok.eos_token_id
    for _ in range(n):
        nxt = int(logits.argmax())
        if nxt == eos:
            break
        ids.append(nxt)
        cache, logits = feed(cache, nxt)
    return tok.decode(ids, skip_special_tokens=True).strip()


def readout(cache, n=480):
    ro, logits = copy.deepcopy(cache), None
    for s in SUFFIX + TAIL:
        ro, logits = feed(ro, s)
    return greedy(ro, logits, n)


def line(e):
    return f"\n- ({e['author']}/{e['type']}) {e['body']}"


def has(board, kw):
    return kw.lower() in board.lower()


def check(label, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}", flush=True)
    return cond


# ---------------------------------------------------------------- Scenario 1
def scenario1():
    print(f"\n{'#' * 72}\n# S1: plain revert — a streamed fact must vanish, rest survive\n{'#' * 72}")
    s = Store()
    E = [
        ("kevin", "progress", "Wire format settled: 2-byte length prefix, 1-byte type, varint seq, payload capped at 1200 bytes."),
        ("boss", "decision", "World sim runs at a fixed 60Hz internal step, snapshots to clients at the network tick."),
        ("kevin", "progress", "Fixed g7: RTT*1.5 backoff with a 250ms cap; resend rate dropped from 340/s to ~40/s."),
        ("boss", "decision", "Authoritative server model — clients send inputs only, never positions."),
        ("kevin", "progress", "Delta snapshots landed: avg packet 1180B -> 320B; ring buffer holds the last 32 acks."),
        ("boss", "decision", "Adopted Kafka for the match-event bus — every game node publishes deltas to a shared topic."),  # X
        ("kevin", "progress", "Reverting to 20Hz tick — 30Hz doubled CPU; client interpolation hides it fine."),
        ("boss", "progress", "Speed-hack detector live; threshold tuned to 1.10x after false positives."),
        ("kevin", "decision", "Session state goes in Redis with a 30s TTL."),
        ("boss", "progress", "Wired Prometheus metrics on the tick loop; p99 tick time is 6.2ms."),
        ("kevin", "progress", "Lag compensation: server rewinds up to 200ms to validate hits."),
        ("boss", "progress", "Match nodes autoscale on CCU; the 200 players/box ceiling holds."),
    ]
    ids = [s.publish(a, t, b) for a, t, b in E]
    x_id = ids[5]  # the Kafka entry
    before = s.active_entries()
    xi = [e["entry_id"] for e in before].index(x_id)
    prefix, X, tail = before[:xi], before[xi], before[xi + 1:]

    # stream prefix -> CHECKPOINT -> + X + tail  (full state, before revert)
    ck, _ = stream(None, [HEAD, PRE] + [toks(line(e)) for e in prefix])
    full_cache, full_logits = stream(copy.deepcopy(ck), [toks(line(X))] + [toks(line(e)) for e in tail])
    board_full = readout(full_cache)

    # revert in the store; re-derive active set
    s.revert(x_id)
    after = s.active_entries()
    assert [e["entry_id"] for e in after] == [e["entry_id"] for e in (prefix + tail)]

    # replay from the checkpoint, streaming only the tail (X skipped)
    rev_cache, _ = stream(copy.deepcopy(ck), [toks(line(e)) for e in tail])
    board_reverted = readout(rev_cache)
    # and a full re-stream of active_after from scratch (should match the replay)
    fresh_cache, _ = stream(None, [HEAD, PRE] + [toks(line(e)) for e in after])
    board_fresh = readout(fresh_cache)

    print(f"\nstore: {len(before)} active -> revert Kafka -> {len(after)} active "
          f"(Kafka status={s.get_entry(x_id)['status']})")
    print("\n--- board BEFORE revert ---\n" + board_full, flush=True)
    print("\n--- board AFTER revert (checkpoint-replay) ---\n" + board_reverted, flush=True)
    print("\nchecks:")
    check("Kafka present before revert", has(board_full, "kafka"))
    check("Kafka GONE after revert", not has(board_reverted, "kafka"))
    check("tail survived (Prometheus present after revert)", has(board_reverted, "prometheus"))
    check("prefix survived (60Hz present after revert)", has(board_reverted, "60hz") or has(board_reverted, "60 hz"))
    check("checkpoint-replay == full re-stream of active_after", board_reverted == board_fresh)


# ---------------------------------------------------------------- Scenario 2
def scenario2():
    print(f"\n{'#' * 72}\n# S2: supersede -> revert -> restore — board must flip Postgres->Redis\n{'#' * 72}")
    s = Store()
    base = [
        s.publish("kevin", "progress", "Wire format settled: 2-byte length prefix, payload capped at 1200 bytes."),
        s.publish("boss", "decision", "World sim runs at a fixed 60Hz internal step, snapshots at the network tick."),
        s.publish("boss", "decision", "Authoritative server model — clients send inputs only, never positions."),
    ]
    y = s.publish("kevin", "decision", "Session store: Redis, with a 30s TTL on session keys.")
    yp = s.supersede(y, "Session store: Postgres, with a small in-process cache in front.")

    now = s.active_entries()
    board_super = readout(stream(None, [HEAD, PRE] + [toks(line(e)) for e in now])[0])

    s.revert(yp)  # restores y (Redis), yp reverted
    restored = s.active_entries()
    board_restored = readout(stream(None, [HEAD, PRE] + [toks(line(e)) for e in restored])[0])

    print(f"\nafter supersede: Y={s.get_entry(y)['status']}, Y'={s.get_entry(yp)['status']}")
    print("--- board after supersede (expect Postgres) ---\n" + board_super, flush=True)
    s_y, s_yp = s.get_entry(y)['status'], s.get_entry(yp)['status']
    print(f"\nafter revert:    Y={s_y}, Y'={s_yp}")
    print("--- board after revert/restore (expect Redis) ---\n" + board_restored, flush=True)
    print("\nchecks:")
    check("Postgres present after supersede", has(board_super, "postgres"))
    check("Redis absent after supersede", not has(board_super, "redis"))
    check("store restored Y to active on revert", s_y == "active" and s_yp == "reverted")
    check("Redis present after revert/restore", has(board_restored, "redis"))
    check("Postgres GONE after revert/restore", not has(board_restored, "postgres"))


if __name__ == "__main__":
    scenario1()
    scenario2()
    print("\n[revert_stream_test] done.", flush=True)
