# Architecture

How Context_Manager is built and *why* it's shaped this way. The README is the
one-screen north star; this is the design rationale, including the empirical results
that pushed each decision. For the cited SSM/hybrid research record see
[`research/deep_research_report.md`](research/deep_research_report.md).

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
3. **Exact retraction = the DB** (explicit supersede/revert). **Lossy fade of the
   merely-old = the cache** (recency decay in the streaming carry, compaction in the
   gist). Complementary, not redundant.
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
  │ FE     display: exact board  |  or the gist                 │  what a consumer reads
  │ CACHE  lossy linking gist + O(1) streaming carry (dual)      │  derived, lossy, optional
  │ BE     index / service — query, shape, render (deterministic)│  structured, GPU-free
  │ DB     store — source of truth; clean retraction             │  append log + active state
  └───────────────────────────────────────────────────────────┘
  Claude — judgment (dedup / conflict / merge), elsewhere, on demand
```

| Layer | Module | Job | Key property |
|-------|--------|-----|--------------|
| DB    | `ctx/store.py` | source of truth | exact, append-only + retractable |
| BE    | `ctx/index.py`, `ctx/service.py` | query/shape/render | deterministic, GPU-free |
| cache | `ctx/compaction.py`, `ctx/ssm_engine.py`, `ctx/mamba_summarizer.py` | lossy linking gist (hybrid) + O(1) streaming carry (Mamba) | derived, lossy, optional |
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

## 6. The model-backed cache — two roles over the exact core

The DB + BE already deliver the live product *deterministically and GPU-free*: the
verbatim board needs no model. Everything here is an **optional readout layer on top**
of that exact core — derived, lossy, never the truth (Principle 5), with the DB always
the fallback for exact. And it is **not one model doing one thing**: a hybrid probe this
cycle (`scripts/hybrid_stage2.py`) split it into two roles with different mechanics,
footprints, and jobs — the readable *gist* most consumers actually reach for, and the
constant-cost *streaming carry* for an unbounded firehose. Cited record:
`research/deep_research_report.md`.

### 6a. The lossy compaction gist — the readable "where are we" (`Falcon-H1-3B` → `overview`)

What a consumer (a human, or another CC) usually wants: not a verbatim relist but a
short, **linked** overview that ties related work together. `HybridCompactor`
(`ctx/compaction.py`) feeds the BE's *capped* verbatim board into a small SSM/attention
**hybrid** and compacts it into that gist.

- **Lossy is the point, not a defect.** It paraphrases, groups, and drops detail *by
  design* — exact recall is the DB's job (Principle 5). Its job is to *link* facts into a
  glanceable picture; judging it on verbatim recall grades it against the wrong job.
- **It links only what the board states.** A linking probe (`scripts/hybrid_stage3_link.py`)
  showed that *inviting* it to hunt dependencies makes it hallucinate false ones, so the
  prompt forbids speculation — it connects an explicit reference (one entry reads
  another's output), never an inferred one. LINK ≠ JUDGE (Principle 4): it synthesizes
  what it is *given*; truth, salience, and conflict stay the DB's and Claude's.
- **It degrades gracefully.** Where a pure Mamba *collapses* past its envelope into
  garbage, the hybrid keeps emitting a coherent, format-clean board — its attention holds
  a near-verbatim view of the (bounded) input instead of compressing into a saturating
  fixed state. Cost: a **growing** KV cache (94→156 MB in the sweep), bounded only
  *because the input is the BE-capped set* — that cap is exactly what makes a hybrid safe
  here.

**Measured** (Falcon-H1-3B-Instruct, fp16, RTX 5070 Ti — the deployed gist model):

| Finding | Result | Evidence |
|---------|--------|----------|
| Loads kernel-free on Blackwell | 3.2 s; **~6.3 GB** VRAM weights — half the 7B's ~14 GB | `scripts/hybrid_probe.py` |
| **Never collapses under load** | coherent, format-clean board at **2600 tok**, where pure Mamba degrades to `[x]`×78 garbage | `scripts/hybrid_stage2.py` |
| Footprint is the trade-off | attention KV grows **94 → 156 MB** across the depth sweep (vs Mamba's flat ~21 MB), bounded by the capped input | `scripts/hybrid_stage2.py` |
| Throughput | **~17.5 tok/s** on the kernel-free naive fallback | `scripts/hybrid_probe.py` |
| Links, doesn't fabricate | ties an explicitly-stated cross-project dependency; would hallucinate if asked to *hunt* deps → the prompt forbids it (verified) | `scripts/hybrid_stage3_link.py`, `scripts/compact_demo.py` |

### 6b. The constant-size streaming carry — the O(1) firehose primitive (pure Mamba → `project_digests`)

The original SSM value, and still the only thing that does it: fold an **unbounded,
high-frequency** stream at **constant memory and constant cost**, where the fixed state
it keeps *is already* the current "where are we." The DB appends O(1) too, but its
readout is O(N active); a pure Mamba is **O(1) in and out**, recency-fresh for free (old
fades as new folds in — anti-poisoning by decay, no judgment).

- **Mechanism.** Each event folds once into the recurrent state (`cache_params` = conv +
  recurrent states); an exact revert re-syncs by replaying the affected tail from the
  nearest checkpoint (`ctx/ssm_engine.py`), never by asking the model to un-remember.
- **Sharding is the envelope lever.** `ShardedSSMEngine` keeps one fold state *per
  project* (report finding 5: recall scales with state-per-stream), so each stream stays
  under the faithful envelope and a hot-project revert replays only *that* shard — the
  first lever to reach for, before a bigger-state model.
- **Measured** (falcon-mamba-7b-instruct — the *benchmark baseline* for this role, not a
  deployment target; fp16, RTX 5070 Ti):

| Finding | Result | Evidence |
|---------|--------|----------|
| State is fixed-size | **~21 MB**, identical for 775 vs 2464 input tokens | `scripts/state_decode.py` |
| Faithful envelope | clean ≤ ~1000 tok / ~25 entries; conflates ~1150; garbles ~1300+ | `scripts/knee_sweep.py` |
| Streaming == batch | byte-identical readout, **zero drift**; live query at any mid-stream point | `scripts/streaming_test2.py` |
| Recency bias | old facts fade first as the state saturates | knee sweep |
| **Selection/judgment** | **rejected — 0/3**: asked to pick load-bearing entries it just copies input order, ignoring intent | `scripts/salience_select.py` |

- **Honest limit.** Past ~25 entries / ~1k tokens the fixed state saturates and
  **collapses** (2600 tok → runaway `[x]` garbage in the sweep) — exactly why capping +
  sharding matter here, and why the gist (6a), which never collapses, is the better fit
  for a *readable* overview. The one lever for a bigger faithful envelope is a **bigger
  state** (larger-`d_state` SSM); 6a sidesteps the wall by not compressing into a fixed
  state at all.

### Shared boundaries (both roles)

Both inherit the same limits (deep-research report, confirmed): no random access → exact
recall is the DB's; capacity-bounded → the DB is the archive; **cannot judge** →
salience/dedup/conflict is Claude's or a deterministic rule's (Principle 4; the rejected
`salience_select.py`, 0/3). Neither role is ever the source of truth — the gist and the
streaming state are both derived views the DB can always reconstruct.

**Status.** The DB + BE are the deterministic product today. Both model roles are **wired
but optional**: MCP `overview` (the gist, `CTX_GIST=1`) and `project_digests` (sharded
streaming, `CTX_SSM=1`), each lazy-loaded on the dev-machine GPU with a deterministic
board fallback — they compete for VRAM on a 16 GB box, so enable one.

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
(`stateless_http=True`) because all state lives in the DB. Deterministic tools:
`publish`, `status_board`, `check_overlap`, `supersede`, `revert`, `team_state`,
`recent`. Optional GPU-backed reads (off by default; enable on the dev machine):
`overview` (the lossy linking gist, `CTX_GIST=1`) and `project_digests` (per-project
streaming digests, `CTX_SSM=1`) — each lazy-loads its model and falls back to the
verbatim board, and the two compete for VRAM on a 16 GB box.

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
- **MCP** — the deterministic tools live over streamable-HTTP, overlap warning + board
  over the wire (`scripts/verify_mcp.py`); the optional `overview`/`project_digests`
  reads are wired with a deterministic fallback.
- **SSM cache** — fixed state, faithful envelope, zero-drift streaming, revert re-sync
  (scripts in §6); **per-project sharding** (`ShardedSSMEngine`); **selection explicitly
  disproven**.
- **Hybrid compaction gist** — `Falcon-H1-3B` runs kernel-free on Blackwell, compacts +
  links the capped board without collapsing, links only stated relations (no
  speculation); verified end-to-end (`scripts/hybrid_stage2.py`,
  `scripts/hybrid_stage3_link.py`, `scripts/compact_demo.py`).

---

## 10. Open questions (candidates, not commitments)

From `research/deep_research_report.md` — pick *from* these, don't do all of them:

- ~~**Shard SSM state by project**~~ — **done** (`ShardedSSMEngine`, §6).
- **Bigger state, not bigger model** (Mamba-2 `d_state`, RWKV-7) as the envelope lever —
  still the honest lever for role (i); the hybrid gist is the answer for role (ii).
- **Same-size hybrid vs pure test** (e.g. mamba-2.8b vs a 2.8B hybrid) to cleanly settle
  the report's mechanism claim — the shipped 3B-hybrid-vs-7B-Mamba comparison confounds
  size (it answered the *swap* decision, not the mechanism).
- **Δ carries real elapsed time** so recency-fade tracks wall-clock, not event count.
- **Wire the Claude judgment tier** (dedup / conflict escalation) — the last core gap.
```
