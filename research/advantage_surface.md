# The SSM + Spike Advantage Surface

A living menu of what the *stateful/streaming* model family (SSMs and spiking nets)
is natively good at — so we design **from its strengths**, not from transformer
habits. This is not a task list and not a plan. It's a surface we prune from: each
entry is a capability we could spend, tagged by whether we already spend it.

**Why this file exists.** Kevin is new to SSMs and still carries transformer
intuitions; so does CC — we both keep re-deriving the same facts and mis-framing the
design. This is the shared brain that stops the re-discovery. Read it before touching
SSM/spike work.

## The framing discipline (read this or the rest misleads you)

1. **The SSM/spike family is the default here, the transformer is a *branch*.** The
   transformer (Claude) is a subroutine the streaming layer calls at its *known*
   weaknesses (exact recall → the DB; judgment/merge → Claude). It is not the master
   the SSM assists.
2. **Never make the SSM justify itself against a transformer baseline.** "Why not just
   let Claude read the entries" is the transformer reflex — it treats the transformer
   as free and the SSM as on trial. Start from the SSM's native strengths and ask what
   they unlock.
3. **Roles, per the README stack:** DB = truth (exact, retractable). SSM = *cache*
   (constant-size, always-fresh readout) — the trunk of the *readout path*, never the
   trunk of *truth*. Claude = judgment, on demand.
4. **Status tags:** `USED` (we spend it today) · `LATENT` (we own it, unspent — these
   are the seeds) · `HYPOTHESIS` (plausible, needs a probe before we trust it).
5. **Anchors (ground truth, don't re-derive):** README invariants; lessons #72 (RWKV-7
   runs on this box), #73 (envs: torch cu128 + `spike_sse`), #75 (falcon-mamba state
   faithful to ~25 entries / ~1k tok, recency evicts oldest); rejected salience test
   (`scripts/salience_select.py`, 0/3).

## The shared spine

What unites SSMs and spiking nets — and is the exact opposite of a transformer:

> **stateful · streaming · event/time-driven · recency-native · cheap-per-event · bounded footprint**

A transformer is stateless, batch, random-access, order-tagged, all-at-once, and
grows with context. Every advantage below is one facet of that spine. When in doubt,
reason from the spine, not from the specific model.

---

## SSM advantages

**O(1)/token inference, constant state** — `USED`
- The state is fixed-size no matter how long the stream; readout is O(1).
- *Transformer prior it corrects:* "readout cost grows with history" (KV cache). Here
  it doesn't.
- *Lever for us:* the differentiated value — constant-size readout of an unbounded
  stream, where the DB's readout is O(N active).

**Train parallel (scan/conv) / run recurrent** — `LATENT`
- Same model has a parallel training form (O(N)) and a recurrent inference form (O(1)).
- *Prior it corrects:* "recurrent = slow to train." Not for SSMs.
- *Lever:* if we ever fit a state-shaper/compressor to our event schema, training is
  affordable offline and deploys O(1). Unspent because we use an off-the-shelf model.

**Unbounded context, never re-process history** — `USED`
- Fold each new event once; the past is already in the state. No window, no re-read.
- *Prior it corrects:* "long context = re-stuff the whole log each turn."
- *Lever:* high-frequency ingest is nearly free; the firehose is the happy path.

**Continuous-time core (the Δ step)** — `LATENT` *(underrated)*
- SSMs come from a continuous ODE; the discretization step Δ can vary per event, so
  *irregularly-timed* input is native. A burst then an hour of silence is
  representable as-is.
- *Prior it corrects:* "position is an integer index" — a transformer has no clock.
- *Lever:* our event stream is irregular by nature. Δ could carry real elapsed time
  so the gist's recency-fade tracks wall-clock, not event-count.

**HiPPO memory — state = principled projection of the whole past** — `LATENT`
- The state isn't an arbitrary blob; classic SSMs (S4) keep an optimal polynomial
  approximation of the entire history on a fixed basis.
- *Prior it corrects:* "the summary is whatever the model happens to emit."
- *Lever:* "what survives compression" is *designable*, not just emergent — a handle
  on the gist's content, if we ever shape the basis/decay.

**Selectivity (Mamba/S6 input-dependent gates)** — `USED (partial)`
- falcon-mamba has S6: the model conditions remember/ignore on content.
- *Prior it corrects:* "recurrence forgets uniformly."
- *Lever:* content-based weighting of update *types* is already in the model — usable
  without asking it to *judge* (judgment stays a branch).

**Tunable decay (state eigenvalues / Δ)** — `LATENT`
- The recency-fade *rate* is a parameter, not a fixed behavior.
- *Prior it corrects:* "forgetting is an accident of the architecture."
- *Lever:* a knob — fast fade for a fast-moving board, slow for a stable one; per-
  project fade rates.

**Deterministic, checkpointable, diffable state** — `USED`
- The state is a plain tensor: snapshot it, diff two of them, replay from a checkpoint.
- *Prior it corrects:* "you can't rewind a model's belief."
- *Lever:* already how the engine does exact revert (replay from nearest checkpoint).

**Linear-core algebra (state composition)** — `HYPOTHESIS`
- The sequence recurrence is associative (that's *why* the parallel scan works), so
  states compose along a stream. Whether two *independent* streams' states can be
  merged into one meaningful state is unproven.
- *Prior it corrects:* "combining contexts means concatenating tokens."
- *Lever if real:* cheap merge of per-shard states → a global gist without re-folding.
  Selectivity breaks clean linearity, so **treat as a thing to test, not assume.**

---

## Spiking advantages

**Event-driven compute** — `LATENT` *(potentially big)*
- Neurons compute only when they fire; no fire, no work. Cost scales with *change*,
  not with wall-clock or sequence length.
- *Prior it corrects:* "every step costs a full forward pass."
- *Lever:* maps 1:1 to "every input is an update." Silence is free; bursts cost
  proportional to the burst. A near-perfect fit for a sparse team stream.

**Sparse activations** — `LATENT`
- Most units are silent at any moment.
- *Prior it corrects:* "the whole network lights up per token."
- *Lever:* cheap on the local box; extends the small-footprint constraint.

**Temporal coding (info in spike *timing*)** — `LATENT`
- Information lives in *when* spikes occur, not only how many.
- *Prior it corrects:* "order is a position embedding bolted on."
- *Lever:* update timing/order carried in the representation itself.

**Neuromorphic energy floor** — `LATENT` *(future-only)*
- Orders-of-magnitude lower power on neuromorphic hardware.
- *Prior it corrects:* "inference means a GPU."
- *Lever:* only matters if this ever leaves the dev box; noted for completeness.

**Local/online plasticity (STDP)** — `LATENT`
- Local, spike-timing learning rules adapt online without a global backprop pass.
- *Prior it corrects:* "adaptation requires a training run."
- *Lever:* online adaptation to a team's update patterns, no retrain loop.

---

## The intersection: spiking-SSM

A spiking state-space layer = the SSM's principled long memory **+** the SNN's
fire-only-on-change compute. On paper it's the sharpest fit for a sparse, unbounded,
irregular event stream — exactly our shape — and the `spike_sse` env is already
standing on this ground.

Status: `HYPOTHESIS`. This is an active research area, not a settled tool; specifics
(which spiking-SSM variant, does it hold the ~25-entry envelope, does event-driven
compute survive real folding) need a dedicated probe before we bank on it. Kevin may
know the local state of `spike_sse` better than this file does — reconcile before
building.

---

## The limits (anti-advantages — what NOT to ask of it)

These are the walls. Each one is where the SSM *branches out* to the DB or Claude.

- **No random access / no lookback.** The past is compressed; you cannot query "event
  #40 verbatim." → exact recall is the **DB's** job.
- **Weak associative recall, capacity-bounded.** Faithful only within ~25 entries /
  ~1k tokens (#75); past that the gist garbles, oldest-first (recency eviction). →
  the DB is the archive; the SSM is the *recent* view.
- **Comparison/judgment is its architectural weakness.** It cannot decide which
  updates are salient, duplicate, or conflicting — `salience_select.py` scored 0/3,
  it just copies input order. → judgment is **Claude** or a deterministic rule, per
  the README invariant. **The SSM never judges.**
- **It doesn't know what it forgot.** The state won't flag its own gaps. → completeness
  is the DB's guarantee, never the model's.
- **Order is destiny.** Left-to-right recurrence means late dominates, early fades —
  ingest order changes the state. → don't assume order-invariance; if order matters,
  control it upstream.

---

## Transformer priors that keep biting (the anti-re-discovery list)

The short version of the corrections above, as a checklist. When a design decision
feels obvious, check it isn't one of these:

1. "Context is a buffer I can re-read." → No buffer. Fixed-size state; carry forward
   only what survived.
2. "More context is strictly better." → Past capacity, new input *evicts* old.
3. "Order is just a tag." → Order is a decay schedule.
4. "Recall is exact and content-addressable." → SSM recall is weak and bounded; that's
   the DB's job.
5. "Bigger model = better memory." → State *size* is the lever, not param count.
6. "It knows its whole context." → It knows only its lossy state, and not what it lost.
7. *(meta)* "Why not just let the transformer do it." → That makes the transformer the
   default and the SSM a defendant. Invert it.

---

## Open probes (seeds — candidates to prune, NOT a committed plan)

Latent advantages worth a small experiment, if/when we choose one. Listed so we pick
*from* them, not so we do all of them:

- **Shard state by project.** One SSM state per project (the BE already groups by
  project) → each stream stays under the ~25-entry envelope. Turns one global wall
  into many rarely-hit ones. (Exploits: constant state + tunable per-shard decay.)
- **Δ carries real elapsed time.** Make recency-fade track wall-clock, not event
  count. (Exploits: continuous-time core.)
- **Input-shaping before folding.** Compress each entry to essential tokens first →
  more entries per 1k-token budget. (Stretches the envelope with no model change.)
- **Bigger *state*, not bigger model.** Mamba-2's larger `d_state`, or RWKV-7 (#72) —
  state size as the faithful-envelope lever.
- **Spiking-SSM feasibility on `spike_sse`.** Does fire-on-change compute survive real
  event folding, and does it hold the envelope? (Tests the intersection above.)
- **Per-shard state merge.** If the linear-algebra hypothesis holds, merge shard states
  into a global gist without re-folding. (Depends on the HYPOTHESIS above.)
