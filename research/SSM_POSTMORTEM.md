# SSM post-mortem — what we explored, and why we archived it

*2026-07-01 → 07-02. This records an exploration that did not land, so the next person
(probably us) inherits the fence instead of re-deriving it. The code is kept in the tree
as the experimental record; it is **not** part of the live product.*

## The hypothesis

Use a state-space model (Mamba, or a small SSM/attention hybrid) as a **constant-size,
always-fresh cache** of the team's "where are we." An SSM folds an unbounded event stream
into a fixed state at O(1) cost per event, and that state is *already* a compressed current
view — so the pitch went, it could be the cheap always-on readout the store can't be (the
store's readout is O(N active)).

## What we tried, and what each probe showed

| Framing | We asked the SSM to… | Result | Evidence |
|---|---|---|---|
| **SSM as store** | hold the team's truth | ✗ the DB is exact, retractable, free; the SSM is lossy — the store wins on its own turf | invariants; `ctx/store.py` |
| **SSM as salience picker** | pick which entries matter | ✗ **0/3** — it copies input order, ignores intent | `scripts/salience_select.py` |
| **SSM as lossy gist** (hybrid, Falcon-H1-3B) | compact the capped board into a readable, *linked* overview | ✅ *works* — never collapses, links stated relations; BUT needs ~6 GB resident VRAM, and "narrate with Claude on demand" beats it at our scale | `scripts/hybrid_stage2.py`, `scripts/compact_demo.py` |
| **SSM as drift detector** | flag when an agent wanders off the goal | ~ partial — state-distance catches only *coarse topical* drift, needs the faithful 7B state (130m fails), and is blind to *subtle invariant-violation* drift (the valuable case) | `scripts/drift_probe.py` |

## The law it kept hitting

Every framing failed the *same* way, and it fits in one sentence:

> **The SSM compresses; it does not compare.**

Store, salience, drift — each secretly asked the model to hold its state up against a
criterion and *make a call*. That is reasoning / judgment, and it is architecturally not
the SSM's: a fixed lossy state has nowhere to do a comparison. The one thing it genuinely
does — fold a stream into a lossy fixed state and read it back — is real but narrow.
Spotting drift, in particular, is a reasoning task, and an SSM cannot solve it.

## Why we archived it (even the part that works)

The hybrid gist *does* produce a clean lossy overview. We archived it anyway, because at
this scale it is a bad trade: it costs resident VRAM to produce a *lossy* view of a set the
deterministic BE already renders *exactly and for free*, and the subtle-judgment cases it
cannot cover collapse back to "ask Claude" — who is already in the loop and needs no
resident model. Working was never the bar; worth-it was.

## What survived (the actual product)

A working, GPU-free, **exact coordination substrate**: the append-only log + active-state
store, the deterministic verbatim board, the MCP server, and the two-CC loop
(read board → `check_overlap` → `publish`). It solves the problem we started with — two
agents not duplicating, always a clean "where are we" — and it never needed a model.

## What's reusable

- The **invariants** (truth in the DB; judgment is Claude's; every input is an update) are
  portable to any transformer-based agent setup — the problem is architectural, not
  vendor-specific.
- The **negative results** are the fence: don't ask an SSM to store, judge, or reason.
- The **code** stays as the record (`ctx/compaction.py`, `ctx/ssm_engine.py`,
  `ctx/mamba_summarizer.py`, `scripts/*`), off by default and unwired from the live path.
