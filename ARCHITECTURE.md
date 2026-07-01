# Architecture

How Context_Manager is built and *why* it's shaped this way. The README is the
one-screen north star; this is the design rationale, including the empirical results
that pushed each decision. The model-backed cache layer was explored and archived; that
arc is recorded in [`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md) (with the
underlying survey in [`research/deep_research_report.md`](research/deep_research_report.md)).

---

## 1. The problem

A transformer's context window is a poor home for a team's *evolving* truth, and it
fails two ways that share one root — **truth living inside a model's head, where it
can neither be cleanly retracted nor shared:**

- **Within one session — poisoning.** A long Claude Code session accumulates stale
  decisions, reverted approaches, dead ends, contradictions. The transformer holds
  every token and can't drop any; "where are we?" is answered over a soup of current
  + abandoned facts.
- **Across two sessions — divergence.** Two devs each run their own CC with its own
  window. Neither reflects the other's work, so they rebuild each other's output.

**Thesis:** move the team's truth *out* of the model into an external store that
(1) retracts exactly, (2) is shared, (3) always yields a clean current state — and
have each CC read from it instead of trusting its own polluted/partial context.

---

## 2. Principles (invariants)

These are load-bearing; violating one is how the design drifts.

1. **Truth lives in the DB, never in a model's context.**
2. **Every input is an update.** The streaming model folds *all* updates; it never
   decides which are real, true, or important.
3. **Exact retraction = the DB** (explicit supersede/revert). There is no separate
   "fade" tier: because retraction prunes the active set, the current state is always
   clean — there is no stale soup left over to decay past.
4. **Judgment = Claude or a deterministic rule** (dedup, conflict, merge, salience).
   **Never the model** — *the SSM compresses, it does not compare.* This is the
   load-bearing survivor of the archived cache exploration: every framing that asked a
   fixed lossy state to *make a call* failed the same way (§6 and
   [`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md)).
5. **The cache is derived and lossy; the DB is exact.** "Dropped" never means "lost":
   overview → active set → full log, each layer drills down to the one below.

---

## 3. Component stack (DB / BE / FE + Claude)

```
  Your CC ──┐            ┌── Boss's CC        (consumers: the CC sessions + humans)
            │  MCP / LAN  │
            └──────┬──────┘
             DEV MACHINE
  ┌───────────────────────────────────────────────────────────┐
  │ FE     display: the exact board                              │  what a consumer reads
  │ BE     index / service — query, shape, render (deterministic)│  structured, GPU-free
  │ DB     store — source of truth; clean retraction             │  append log + active state
  └───────────────────────────────────────────────────────────┘
  Claude — judgment (dedup / conflict / merge), elsewhere, on demand
```

| Layer | Module | Job | Key property |
|-------|--------|-----|--------------|
| DB    | `ctx/store.py` | source of truth | exact, append-only + retractable |
| BE    | `ctx/index.py`, `ctx/service.py` | query/shape/render | deterministic, GPU-free |
| FE    | the board string / a CC / a human | display | exact, verbatim store text |
| judge | Claude (in the CC sessions) | dedup / conflict / merge | on demand, not hosted here |

The model-backed cache (`ctx/compaction.py`, `ctx/ssm_engine.py`,
`ctx/mamba_summarizer.py`) was explored and archived — see §6 and
[`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md).

---

## 4. Data model (the DB)

Two tables (`ctx/store.py`):

- **`events`** — append-only, immutable log. Every `publish` / `supersede` / `revert`
  is one row (`seq`, `ts`, `kind`, `entry_id`, JSON `payload`). This is the stream the
  SSM ingests and the complete history; nothing is ever deleted.
- **`entries`** — the *materialized active state*. One row per entry with a mutable
  `status ∈ {active, superseded, reverted}`, plus `author`, `type`, `body`, `refs`
  (files), `project`, `created_seq`, `updated_seq`, `supersedes`.

`active_entries()` returns only `status='active'`, ordered by `created_seq`. That set —
pruned by supersede/revert — is all the rest of the system ever reads. Its size grows
with the **live surface area**, not with history: 10k events can be ~200 active entries.

---

## 5. Retraction semantics (why the DB beats a window)

- **publish** → new `active` entry + a `publish` event.
- **supersede(old, new_body)** → `new` becomes `active` (with `supersedes=old`), `old`
  becomes `superseded`. A revision, with the lineage kept.
- **revert(id)** → `id` becomes `reverted`; **and if it superseded a prior entry, that
  prior is restored to `active`.** Revert is the *true inverse* of supersede. Idempotent
  (a double-fire from a high-frequency client is a no-op).

This is why an external store beats keeping state in the transformer: a reverted fact
is *gone from the active set* (though still in the log), exactly and instantly — the
"clean retraction" a stateful context window structurally cannot do. Validated
end-to-end on the real streaming path in `scripts/revert_stream_test.py` (a streamed
fact vanishes on revert; a supersede→revert flips the board back to the original).

---

## 6. Explored and archived: the model-backed cache

For one cycle we explored a **model-backed cache** on top of the exact core — an SSM
(or a small SSM/attention hybrid) as a constant-size, always-fresh readout of "where are
we." Four framings were tried: the SSM as **store**, as **salience picker**, as a lossy
linking **gist**, and as a **drift detector**. Every one hit the same wall, the one law
of this exploration:

> **The SSM compresses; it does not compare.**

Store, salience, and drift each secretly asked a fixed lossy state to hold itself up
against a criterion and *make a call* — reasoning, which is Claude's job (Principle 4). A
fixed lossy state has nowhere to run a comparison; the one thing it genuinely does — fold
a stream into a lossy state and read it back — is real but narrow.

The single framing that *worked* was the hybrid gist (Falcon-H1-3B): it compacts the
capped board into a readable, linked overview and never collapses. We archived it anyway,
because at this scale it is a bad trade — it spends ~6 GB resident VRAM to produce a
*lossy* view of a set the deterministic BE already renders *exactly and for free*, and the
subtle-judgment cases it cannot cover fall back to Claude, who is already in the loop.
Working was never the bar; worth-it was. The code stays in-tree as the record
(`ctx/compaction.py`, `ctx/ssm_engine.py`, `ctx/mamba_summarizer.py`, `scripts/*`), off by
default and unwired from the live path.

The full arc — every framing, the measured envelopes and VRAM footprints, the sharding
lever, and the drift probe — is in
[`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md).

---

## 7. Coordination & rendering (the BE)

All deterministic and GPU-free (`ctx/service.py`, `ctx/index.py`):

- **Duplication warning.** `publish` returns `overlaps`: active entries by the *other*
  author whose `refs` (files) intersect yours — a collision caught before the work is
  redone. `check_overlap(refs)` answers it *before* you start.
- **The board.** `status_board()` renders the active set **verbatim** — grouped by
  `project`, the `driver` named (author with the most entries, ties → most recent),
  bullets author-tagged only when the author isn't the driver. Because it's verbatim
  store text, no model can hallucinate into it.
- **Envelope + overflow.** `working_set()` recency-orders, dedups exact-duplicate
  bodies, and caps to the ~25-entry envelope, **surfacing** the overflow (`overflow`,
  `overflow_ids`) rather than silently dropping it — those entries stay in the DB for
  exact recall. If prioritization is ever needed over a huge set, it is a **structured**
  rule (type/recency), never a model judgment (see the rejected `select_salient` +
  `match_back` framework, retained for a future *Claude* picker only).

---

## 8. Interface & topology (the MCP server)

`ctx/mcp_server.py` is a thin FastMCP HTTP adapter over `ContextService`, stateless
(`stateless_http=True`) because all state lives in the DB. The live server is fully
**deterministic and GPU-free** — seven tools: `publish`, `status_board`, `check_overlap`,
`supersede`, `revert`, `team_state`, `recent`. Two GPU-backed reads remain wired but are
**archived experiments, off by default** — `overview` (the hybrid gist, `CTX_GIST=1`) and
`project_digests` (per-project streaming digests, `CTX_SSM=1`); they are the experimental
record (§6, [`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md)), not part of the
live surface.

```
  kevin's box (CC)  ──http──┐
                            ├──►  dev machine :8765/mcp  ──►  team.db (source of truth)
  boss's box  (CC)  ──http──┘
```

The working loop, per box (set in each box's `CLAUDE.md`): **read the board →
`check_overlap` before touching files → do the work → `publish` it.** Identity
(`kevin` / `boss`) is set per box. Deployment steps are in the README.

---

## 9. What is validated

- **DB** — exact retraction incl. supersede→revert→restore (`tests/test_store.py`,
  `scripts/revert_stream_test.py`).
- **BE** — verbatim board, driver/attribution, envelope + surfaced overflow, file-ref
  overlap (`tests/test_index.py`, `tests/test_service.py`). **69 tests green, GPU-free**
  (incl. prompt-invariant, sharding, compaction, and wiring guards).
- **MCP** — the seven deterministic tools live over streamable-HTTP, overlap warning +
  board over the wire (`scripts/verify_mcp.py`).
- **Explored and archived (not a live capability)** — the model-backed cache: fixed-state
  streaming, per-project sharding, and the hybrid compaction gist all *ran* on Blackwell,
  but each hit "the SSM compresses, it does not compare," and the one working piece was a
  bad VRAM trade. The full arc and measured numbers are in
  [`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md).

---

## 10. Open questions (candidates, not commitments)

The model-backed probes are closed — sharding, the bigger-state lever, the same-size
hybrid test, and the mechanism question are all retired with the cache exploration (see
[`research/SSM_POSTMORTEM.md`](research/SSM_POSTMORTEM.md)). What remains is on the live
deterministic path:

- **Wire the Claude judgment tier** (dedup / conflict escalation) — the last core gap.
```
