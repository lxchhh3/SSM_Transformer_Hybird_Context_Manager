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

## The stack (DB / BE / FE)

```
  Your CC ──┐            ┌── Boss's CC        (consumers: the CC sessions…
            │  MCP / LAN  │                     …and you two, directly)
            └──────┬──────┘
             DEV MACHINE
  ┌───────────────────────────────────────────────────────────┐
  │ FE     display: the exact verbatim board                    │ what a consumer reads
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
- **FE.** Whatever displays the board — usually another CC (likes structure), sometimes
  a human. Always exact store text.
- **Claude — judgment, on demand.** Dedup, conflict, merge — the reasoning calls, made by
  the CC when it needs them. Not hosted here.

> **A model-backed cache (SSM / hybrid gist) was explored and archived** — it is *not*
> part of this stack. It worked, but wasn't worth the resident VRAM at this scale: the
> store is exact and free, and judgment is Claude's. Record + measured results:
> [`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md).

## Invariants (the load-bearing rules — don't blur these)

- **Truth lives in the DB, never in a model's context.**
- **Every input is an update to the log.** The store records each publish / supersede /
  revert; nothing decides *on ingest* which are real, true, or important.
- **Exact retraction = the DB** (explicit supersede/revert) — the active set is always
  clean, so there is no stale soup to reason over.
- **Judgment = Claude or a deterministic structured rule** (dedup, conflict, merge,
  salience/priority). **Never a model-in-the-loop cache** — reasoning is not an SSM's
  job: *it compresses, it does not compare*, confirmed across four probes
  (`research/SSM_POSTMORTEM.md`, `memory/ssm-never-judges.md`).
- **The board is derived from the log; the DB is exact.** A dropped fact is gone from the
  active set, recoverable from the full log — active set → full log, drill down.

## What we explored and archived — the model-backed cache

We spent a research cycle testing whether a state-space model could be a **constant-size,
always-fresh cache** of "where are we." Four framings — store, salience picker, lossy
linking gist, drift detector — and they all hit the same wall: **an SSM compresses, it
does not compare.** Three failed on that; the one that worked (a Falcon-H1-3B lossy
linking gist) wasn't worth its resident VRAM, because the store already renders the set
*exactly and for free*, and the judgment cases the cache can't cover belong to Claude —
already in the loop, no model to host. So the model-backed cache is **archived**: the code
stays in the tree as the experiment, off by default and unwired from the live path. Full
record + measured results: [`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md).

## Build status

**The live product** — deterministic, GPU-free:
- [x] **DB store** — exact truth; supersede / revert / restore validated end-to-end
- [x] **BE index/service** — deterministic verbatim board + structured driver / overlap
      (69 tests, TDD, GPU-free)
- [x] **MCP server** — live-verified over streamable-HTTP (both CCs call it over LAN)
- [ ] **Claude judgment integration** (dedup / conflict) — elsewhere, on demand

**Explored and archived** — the model-backed cache
([`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md)):
- SSM streaming cache + per-project sharding (`ShardedSSMEngine`) — validated mechanics
  (constant-cost fold, zero drift), archived: a lossy view of a set the BE renders exactly
- Hybrid compaction gist (Falcon-H1-3B, `ctx/compaction.py`) — works, but a bad
  resident-VRAM trade at this scale
- SSM as store / salience / drift — **rejected**: *it compresses, it does not compare*
  (`scripts/salience_select.py` 0/3; `scripts/drift_probe.py` — drift is a reasoning task)

## Run it (real deployment)

**1. On the dev machine — start the shared brain** (any env with `mcp`; call the env
python directly, `conda run` buffers):

```bash
PYTHONPATH=. CTX_DB=D:/ctx/team.db CTX_PORT=8765 <env>/python.exe -m ctx.mcp_server
# prints the URL + the absolute DB path it's serving
```

`CTX_DB` is the file that IS your team's memory — back it up, don't delete it.
Find the machine's LAN IP (`ipconfig` / `ip addr`); both boxes point at it.

The `overview` / `project_digests` MCP tools are **archived experiments**, off by default —
the server runs fully deterministic and GPU-free. Read `research/SSM_POSTMORTEM.md` before
enabling them (`CTX_GIST=1` / `CTX_SSM=1`); at this scale they aren't worth the VRAM.

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

The model/GPU scripts in `scripts/` (falcon-mamba-7b, Falcon-H1-3B) are the **archived SSM
experiments** — see `research/SSM_POSTMORTEM.md`. The live product needs none of them.
```
