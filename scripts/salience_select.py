"""Does 'SSM selects' actually earn its keep over recency?

Build an OVERSIZED active set (48 entries, one project): 3 OLD load-bearing
decisions published FIRST, then buried under 45 recent minor tweaks. A recency cap
(keep newest 25) necessarily drops the 3 old decisions. Question: can the SSM's
salience pick surface them back?

The SSM only INFLUENCES selection — it lists the load-bearing facts per envelope-
sized batch; match_back maps its lines to real store entries (hallucinations map to
nothing and are dropped). The board is verbatim store text either way. Honest test:
we print which of the 3 decisions each strategy retained. No assuming it works.
"""

import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ctx.index import match_back
from ctx.mamba_summarizer import MambaSummarizer
from ctx.service import ContextService

MID = os.environ["M4_MODEL"]

DECISIONS = [
    ("boss", "Authoritative server model — clients send inputs only, never positions."),
    ("boss", "Wire protocol frozen: 2-byte length prefix, 1-byte type, varint seq."),
    ("kevin", "Session state lives in Redis with a 30s TTL — everything assumes this."),
]
PROBE = ["authoritative", "wire protocol", "redis"]  # keyword per decision

TWEAKS = [
    "Bumped log verbosity on the tick loop", "Renamed FooHandler to BarHandler",
    "Fixed a typo in the metrics readme", "Tweaked the retry jitter by 5ms",
    "Reordered imports in net.go", "Added a debug flag for the packet dumper",
    "Split the config into two files", "Nudged the GC target percentage",
    "Cleaned up an unused struct field", "Adjusted a log line's wording",
    "Formatted the codebase with gofmt", "Renamed a test helper",
    "Bumped a dependency patch version", "Added a comment to the ring buffer",
    "Shortened a variable name in the parser", "Removed a stale TODO",
    "Tuned the batch flush interval slightly", "Fixed flaky test ordering",
    "Added a nil check in the handler", "Renamed the metrics namespace",
    "Dropped an unused import", "Reworded an error message",
    "Adjusted the histogram buckets", "Added a trailing newline to a file",
    "Bumped the CI timeout by 30s", "Cleaned whitespace in the yaml",
    "Renamed a channel variable", "Added a log on shutdown",
    "Tweaked a retry backoff constant", "Fixed a comment typo in parser.go",
    "Reordered two struct fields", "Added a metric for queue depth",
    "Adjusted a default port number", "Renamed an internal package",
    "Bumped the linter version", "Removed dead code in util.go",
    "Added a helper for hex dumps", "Tweaked the log timestamp format",
    "Fixed an off-by-one in a debug print", "Renamed a mutex field",
    "Added a build tag for integration tests", "Cleaned up a switch statement",
    "Bumped the test coverage threshold", "Added a comment above the scan loop",
    "Adjusted a sleep in a flaky test",
]


def build_service():
    svc = ContextService(":memory:")
    for i, (author, body) in enumerate(DECISIONS):
        svc.publish(author, "decision", body, project="game", entry_id=f"dec{i}")
    for j, body in enumerate(TWEAKS):
        svc.publish("kevin" if j % 2 else "boss", "progress", body, project="game")
    return svc


def mamba_pick(model, batch):
    facts = "\n".join(f"- ({e['author']}/{e['type']}) {e['body']}" for e in batch)
    prompt = ("Below are facts from a dev team's log. List ONLY the ones that still "
              "DEFINE the current state — the load-bearing decisions and current "
              "technical choices other work depends on. Skip one-off progress notes "
              "and minor tweaks. Copy each chosen fact VERBATIM, one per line.\n\n"
              + facts + "\n\nLoad-bearing facts:")
    out = model.generate(prompt, max_new_tokens=300)
    lines = [ln.strip("-•* \t") for ln in out.splitlines() if ln.strip()]
    matched = match_back(lines, batch)
    # diagnostic: did this batch even CONTAIN decisions, did the model EMIT them,
    # did match_back CATCH them? Distinguishes judgment-failure from match bug.
    has_dec = [e["entry_id"] for e in batch if e["type"] == "decision"]
    emitted_dec = [p for p in PROBE if p in out.lower()]
    matched_dec = [e["entry_id"] for e in matched if e["type"] == "decision"]
    print(f"\n[batch of {len(batch)}] decisions_in_batch={has_dec} "
          f"| model_emitted_decision_kw={emitted_dec} | matched_decisions={matched_dec}")
    print(f"  raw model output (first 400 chars): {out[:400]!r}", flush=True)
    return matched


def retained(board):
    return [p for p in PROBE if p in board.lower()]


def main():
    svc = build_service()
    n = len(svc.store.active_entries())
    model = MambaSummarizer(MID, device="cuda", dtype=torch.float16, max_new_tokens=300)

    recency = svc.status_board(cap_entries=25)
    ssm = svc.status_board(cap_entries=25, pick=lambda b: mamba_pick(model, b))

    print(f"\n{'=' * 72}\n{n} active entries | 3 old load-bearing decisions buried "
          f"under {len(TWEAKS)} tweaks\n{'=' * 72}")
    print(f"\nRECENCY  [{recency['selector']}]: shown={recency['shown']} "
          f"overflow={recency['overflow']} | decisions retained: {retained(recency['board'])}")
    print(f"SSM-PICK [{ssm['selector']}]: shown={ssm['shown']} "
          f"overflow={ssm['overflow']} | decisions retained: {retained(ssm['board'])}")

    rec_keeps, ssm_keeps = retained(recency["board"]), retained(ssm["board"])
    print(f"\nVERDICT: recency kept {len(rec_keeps)}/3, SSM kept {len(ssm_keeps)}/3 "
          f"of the buried load-bearing decisions.")
    print("  -> SSM selection earns its keep" if len(ssm_keeps) > len(rec_keeps)
          else "  -> no gain over recency here")

    print(f"\n{'-' * 72}\nSSM-SELECTED BOARD (verbatim store text):\n{'-' * 72}")
    print(ssm["board"])
    print("\n[salience_select] done.", flush=True)


if __name__ == "__main__":
    main()
