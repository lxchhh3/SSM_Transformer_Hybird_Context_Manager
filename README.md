# Context_Manager

A shared context layer for a small team (kevin + boss) who each drive Claude Code,
so parallel work stops duplicating and "where are we?" is always answerable — from
a *clean* source of truth, not from a model's polluted context window.

## The problem this solves

A transformer's context window is a bad place to keep the team's evolving truth,
for two reasons that are really the same disease — *truth living inside a model's
head, where it can't be cleanly retracted and can't be shared*:

1. **Within one session — poisoning.** Over a long CC session the window fills with
   stale decisions, reverted approaches, dead ends, contradictions. The transformer
   *can't drop* any of it. Ask "where are we" and it reasons over a soup of current
   + abandoned + contradictory facts.
2. **Across two sessions — divergence.** Your CC and boss's CC are separate windows.
   Neither reflects the other's work → you rebuild what the other already did.

**The fix:** move the team's truth *out* of the transformer into an external place
that (a) retracts exactly, (b) is shared, (c) always yields a clean current state.
Each CC pulls clean truth instead of trusting its own polluted/partial context.

## The stack (DB / BE / cache / FE)

```
  Your CC ──┐            ┌── Boss's CC        (consumers: the CC sessions…
            │  MCP / LAN  │                     …and you two, directly)
            └──────┬──────┘
             DEV MACHINE
  ┌───────────────────────────────────────────────────────────┐
  │ FE     display: the exact board  |  or the SSM gist         │ what a consumer reads
  │ CACHE  SSM — constant-size, always-fresh "where are we"      │ O(1) in AND out; recency-fresh
  │ BE     index / service — query, shape, render (deterministic)│ structured, GPU-free, exact
  │ DB     store — the source of truth; clean retraction         │ append-only log + active state
  └───────────────────────────────────────────────────────────┘
  Claude — the judgment calls (dedup / conflict / merge), elsewhere, on demand
```

- **DB — `ctx/store.py`.** SQLite: an append-only event log + a materialized active
  state. Exact, retractable: supersede/revert flip status tags, so a dropped fact is
  *gone from the active set* (recoverable from the log). This is the source of truth.
- **BE — `ctx/index.py` + `ctx/service.py`.** Deterministic, GPU-free logic over the
  DB: group by project, name the driver, cap to the envelope, and render a **verbatim**
  board (store text only → a model can't hallucinate into it). Serves scoped queries
  cheaply via structured keys.
- **CACHE — the SSM (`ctx/mamba_summarizer.py` + engine).** A compact, lossy, always-
  fresh *materialized view* of the update stream. Derived, never the truth — you fall
  back to the DB for exact. Earns its seat at scale (see "the SSM's job" below).
- **FE.** Whatever displays a view: the exact board for drill-down, or the SSM's gist
  for a human glance. The consumer is often another CC (likes structure) — sometimes
  a human (wants the readable overview).

## Invariants (the load-bearing rules — don't blur these)

- **Truth lives in the DB, never in a model's context.**
- **Every input is an update.** The SSM folds *all* updates into its state; it never
  decides which are real, true, or important.
- **Exact retraction = the DB** (explicit supersede/revert). **Graceful fade of the
  merely-old = the SSM** (recency decay). Complementary.
- **Judgment = Claude or a deterministic structured rule** (dedup, conflict, merge,
  salience/priority). **Never the SSM** — comparison is its architectural weakness,
  confirmed repeatedly. See `memory/ssm-never-judges.md`.
- **The cache is derived and lossy; the DB is exact.** Dropped never means lost —
  overview → active set → full log, each one drill-down recovers the layer below.

## The SSM's job — and its honest limit

**Pro (why it's here, and what a transformer cannot do):** an SSM ingests an
unbounded, high-frequency, ever-changing stream at **constant memory and constant
cost**, and the fixed state it keeps *is already* the compressed, always-current
"where are we." The store appends O(1) too, but its *readout* is O(N active entries);
the SSM is **O(1) in and out**, and **recency-fresh for free** (old fades as new folds
in — anti-poisoning by decay, no judgment). That constant-size readout of an endless
stream is the differentiated value.

**Limit (measured):** the fixed state is *faithful* only within a **~25-entry /
~1k-token envelope**; past that it saturates and the gist garbles. So "how big a fresh
view can I hold" is exactly the axis a **bigger-state model** buys — self-host a larger
SSM to keep it local + constant-cost, or narrate with Claude if cloud is acceptable.

**Rule of thumb:** small/medium project → DB + BE alone are cheap and exact; the SSM is
optional. Large live surface area (feeding the active set verbatim gets expensive) →
the constant-size SSM cache earns its seat, and wants a bigger state to stay faithful.

## Build status

- [x] **DB store** — exact truth; supersede / revert / restore validated end-to-end
- [x] **BE index/service** — deterministic verbatim board + structured salience/driver
      (69 tests, TDD, GPU-free)
- [x] **MCP server** — live-verified over streamable-HTTP (both CCs call it over LAN)
- [x] **SSM streaming cache** — validated: constant-cost fold, **zero drift vs batch**,
      live query at any mid-stream point, recency-fresh (`scripts/streaming_test2.py`)
- [~] **SSM faithful envelope** ~25 entries / ~1k tok — bigger view ⇒ bigger-state model
- [x] **Per-project state sharding** — one SSM state per project (`ShardedSSMEngine`,
      keyed by `index.project_of`) complements the envelope above: it's per-STATE, so N
      projects ≈ N× fresh headroom with no bigger model, and churn replays only its shard;
      exposed as the `project_digests` MCP tool (per-project streaming digests)
- [x] **Hybrid compaction gist** — Falcon-H1-3B (SSM/attention, kernel-free on Blackwell)
      folds the BE-capped board into a lossy *linked* gist (links only stated relations),
      wired as the `overview` MCP tool (`ctx/compaction.py`); degrades gracefully where pure
      Mamba collapses past its envelope — lossy + non-authoritative, the DB stays exact
- [x] **SSM selection/judgment** — tested and **rejected**: judging salience is not the
      SSM's job (`scripts/salience_select.py`, 0/3 — it copies input order, ignores intent)
- [x] **Fold prompt corrected** — the summarizer no longer asks the SSM to flag overlap
      or keep every fact (judgment + fights its recency bias); it leans into recency
      (`ctx/prompts.py`, torch-free + test-guarded)
- [ ] **Claude judgment integration** (dedup / conflict) — elsewhere, on demand

## Run it (real deployment)

**1. On the dev machine — start the shared brain** (any env with `mcp`; call the env
python directly, `conda run` buffers):

```bash
PYTHONPATH=. CTX_DB=D:/ctx/team.db CTX_PORT=8765 <env>/python.exe -m ctx.mcp_server
# prints the URL + the absolute DB path it's serving
```

`CTX_DB` is the file that IS your team's memory — back it up, don't delete it.
Find the machine's LAN IP (`ipconfig` / `ip addr`); both boxes point at it.

The optional GPU-backed reads are off by default (the server runs GPU-free): set
`CTX_GIST=1` for the `overview` gist or `CTX_SSM=1` for `project_digests` — on a 16 GB
box they compete for VRAM (falcon-mamba-7b ~14 GB + Falcon-H1-3B ~6 GB), so enable one.

**2. On each box — add the server to Claude Code:**

```bash
claude mcp add --transport http context-manager http://<dev-machine-ip>:8765/mcp
```

**3. On each box — set identity + habit** in that box's `CLAUDE.md` (so the CC passes
the right author and actually uses it):

```
You are `kevin` (boss's box: `boss`). Use the context-manager MCP server:
- At the start of a coding session, call `status_board` to see where the team is.
- Before starting work on files, call `check_overlap` with those files.
- When you finish a unit of work or make a decision, `publish` it (with project + refs).
- If you change your mind, `supersede` the old entry; if you abandon it, `revert`.
```

That's the whole loop: read the board → check for collisions → do work → publish it.
Both CCs now share one clean, retractable "where are we."

## Dev

```bash
PYTHONPATH=. python -m pytest tests/ -q     # deterministic core (stdlib only, no GPU)
```

Model/GPU work uses conda env with torch cu128 (RTX 5070 Ti / Blackwell); scripts in
`scripts/` load falcon-mamba-7b and need `PYTHONPATH=.`.
```
