# Architecture

How Context_Manager is built and *why* it's shaped this way. The README is the
one-screen north star; this is the design rationale, including the empirical results
that pushed each decision. For the SSM strength/limit menu see
[`research/advantage_surface.md`](research/advantage_surface.md).

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
3. **Exact retraction = the DB** (explicit supersede/revert). **Graceful fade of the
   merely-old = the SSM** (recency decay). Complementary, not redundant.
4. **Judgment = Claude or a deterministic rule** (dedup, conflict, merge, salience).
   **Never the SSM** — comparison is its architectural weakness (see §6).
5. **The cache is derived and lossy; the DB is exact.** "Dropped" never means "lost":
   overview → active set → full log, each layer drills down to the one below.

---

## 3. Component stack (DB / BE / cache / FE)

```
  Your CC ──┐            ┌── Boss's CC        (consumers: the CC sessions + humans)
            │  MCP / LAN  │
            └──────┬──────┘
             DEV MACHINE
  ┌───────────────────────────────────────────────────────────┐
  │ FE     display: exact board  |  or SSM gist                 │  what a consumer reads
  │ CACHE  SSM — constant-size, always-fresh "where are we"      │  O(1) in AND out
  │ BE     index / service — query, shape, render (deterministic)│  structured, GPU-free
  │ DB     store — source of truth; clean retraction             │  append log + active state
  └───────────────────────────────────────────────────────────┘
  Claude — judgment (dedup / conflict / merge), elsewhere, on demand
```

| Layer | Module | Job | Key property |
|-------|--------|-----|--------------|
| DB    | `ctx/store.py` | source of truth | exact, append-only + retractable |
| BE    | `ctx/index.py`, `ctx/service.py` | query/shape/render | deterministic, GPU-free |
| cache | `ctx/mamba_summarizer.py`, `ctx/ssm_engine.py` | streaming compressed view | O(1) fold, constant-size readout |
| FE    | the board string / a CC / a human | display | exact *or* glanceable |
| judge | Claude (in the CC sessions) | dedup / conflict / merge | on demand, not hosted here |

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

## 6. The SSM cache — pro, mechanism, and honest limits

**Why an SSM at all.** It ingests an unbounded, high-frequency stream at **constant
memory and constant cost**, and the fixed state it keeps *is already* the compressed,
current "where are we." The DB appends O(1) too, but its readout is O(N active); the
SSM is **O(1) in and out**, and **recency-fresh for free** (old fades as new folds in —
anti-poisoning by decay, no judgment). That constant-size readout of an endless stream
is the differentiated value. Full strength/limit menu: `research/advantage_surface.md`.

**Mechanism.** Each event is folded once into the recurrent state (`cache_params` =
conv + recurrent states). Exact revert re-syncs by replaying the affected tail from the
nearest checkpoint (`ctx/ssm_engine.py`), never by asking the model to un-remember.

**What we measured** (falcon-mamba-7b-instruct, fp16, on an RTX 5070 Ti):

| Finding | Result | Evidence |
|---------|--------|----------|
| State is fixed-size | **~21 MB**, identical for 775 vs 2464 input tokens | `scripts/state_decode.py` |
| Faithful envelope | clean ≤ ~1000 tok / ~25 entries; conflates ~1150; garbles ~1300+ | `scripts/knee_sweep.py` |
| Streaming == batch | byte-identical readout, **zero drift**; live query at any mid-stream point | `scripts/streaming_test2.py` |
| Recency bias | old facts fade first as the state saturates | knee sweep |
| **Selection/judgment** | **rejected — 0/3**: asked to pick load-bearing entries it just copies input order, ignoring intent | `scripts/salience_select.py` |

**The limits define the boundaries with the other layers** (see also
`advantage_surface.md` §Limits): no random access → exact recall is the DB's job;
capacity-bounded to ~25 entries → the DB is the archive, the SSM the recent view;
**cannot judge** → salience/dedup/conflict is Claude's or a deterministic rule's job.

**Current status of this layer.** The DB + BE deliver the live product today (the board
is fully deterministic — no model in the path). The SSM cache is **validated in
principle** (the table above) but **not yet wired into the live readout**; it's the
*scale-insurance* layer, to switch on when the active set outgrows cheap verbatim
feeding — and at that scale it wants a **bigger state** (a larger-`d_state` SSM), the
one honest lever for a bigger faithful envelope.

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
(`stateless_http=True`) because all state lives in the DB. Tools: `publish`,
`status_board`, `check_overlap`, `supersede`, `revert`, `team_state`, `recent`.

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
  overlap (`tests/test_index.py`, `tests/test_service.py`). 48 tests green, GPU-free.
- **MCP** — all 7 tools live over streamable-HTTP, overlap warning + board over the
  wire (`scripts/verify_mcp.py`).
- **SSM cache** — fixed state, faithful envelope, zero-drift streaming, revert re-sync
  (scripts in §6); **selection explicitly disproven**.

---

## 10. Open questions (candidates, not commitments)

From `research/advantage_surface.md` — pick *from* these, don't do all of them:

- **Shard SSM state by project** so each stream stays under the ~25-entry envelope.
- **Bigger state, not bigger model** (Mamba-2 `d_state`, RWKV-7) as the envelope lever.
- **Δ carries real elapsed time** so recency-fade tracks wall-clock, not event count.
- **Input-shaping before folding** to fit more entries per 1k-token budget.
- **Wire the Claude judgment tier** (dedup / conflict escalation) — the last core gap.
```
