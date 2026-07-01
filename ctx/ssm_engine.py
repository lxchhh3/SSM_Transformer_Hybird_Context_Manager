"""SSM engine — the streaming compressed view over the store.

Model-agnostic on purpose: it depends only on a `Summarizer` (fold/render). A
deterministic fake drives the tests; a Mamba-backed summarizer drops in at M4
with zero changes here. The store stays the source of truth, so the engine never
has to "un-remember" — on a retraction it just replays the affected tail.

Cost model:
  - pure append (publish)         -> fold only the new tail        (#1 cheap)
  - mid-stream change (revert/sup) -> restore nearest checkpoint    (#2 correct,
    at-or-before the change, replay forward                          bounded cost)

Invariant: digest() == render(fold over store.active_entries()), always.
"""

from __future__ import annotations

from typing import Any, Protocol


class Summarizer(Protocol):
    def initial(self) -> Any: ...
    def fold(self, state: Any, entry: dict) -> Any: ...
    def render(self, state: Any) -> str: ...


class SSMEngine:
    def __init__(self, store, summarizer: Summarizer, checkpoint_every: int = 8):
        self.store = store
        self.summ = summarizer
        self.every = max(1, checkpoint_every)
        self._order: list[str] = []                 # cached active ids, created_seq order
        self._state: Any = summarizer.initial()
        self._ckpts: list[tuple[int, Any]] = [(0, summarizer.initial())]

    def sync(self, active: list[dict] | None = None) -> None:
        # `active` lets a caller (e.g. ShardedSSMEngine) push a pre-partitioned
        # slice instead of each engine re-querying the store; must be in
        # created_seq order, same as store.active_entries().
        if active is None:
            active = self.store.active_entries()     # ordered by created_seq
        new_ids = [e["entry_id"] for e in active]
        old = self._order

        # first index where the active sequence diverges from our cached view
        d = 0
        while d < len(old) and d < len(new_ids) and old[d] == new_ids[d]:
            d += 1
        if d == len(old) and d == len(new_ids):
            return  # nothing changed

        if d == len(old):
            # pure append: keep folding from where we are — no re-processing (#1)
            start, state = len(old), self._state
        else:
            # membership changed mid-stream: rewind to nearest checkpoint <= d (#2)
            start, state = self._checkpoint_at_or_before(d)
            self._ckpts = [c for c in self._ckpts if c[0] <= start]

        for i in range(start, len(new_ids)):
            state = self.summ.fold(state, active[i])
            if (i + 1) % self.every == 0:
                self._ckpts = [c for c in self._ckpts if c[0] != i + 1]
                self._ckpts.append((i + 1, state))

        self._state = state
        self._order = new_ids

    def digest(self) -> str:
        return self.summ.render(self._state)

    def size(self) -> int:
        """Active entries currently folded into the state (0 == empty shard)."""
        return len(self._order)

    def _checkpoint_at_or_before(self, idx: int) -> tuple[int, Any]:
        best = self._ckpts[0]
        for k, state in self._ckpts:
            if k <= idx and k >= best[0]:
                best = (k, state)
        return best


class ShardedSSMEngine:
    """One SSMEngine per project — each stream keeps its own fixed-size state.

    Why: the faithful envelope (~25 entries / ~1k tok) is a property of a single
    state, not of the whole team (deep_research_report.md finding 5 — recall
    capacity scales with state-per-stream). Folding every project into one shared
    state saturates it fast; a state PER project keeps each stream small. The BE
    already groups by project (`index.project_of`), so we reuse that exact key.

    Structural win: shards are independent, so churn in a hot project replays only
    THAT shard — a revert/supersede never re-folds a cold project's state.
    """

    def __init__(self, store, summarizer: Summarizer, checkpoint_every: int = 8,
                 project_of=None):
        from ctx.index import project_of as _default_project_of
        self.store = store
        self.summ = summarizer
        self.every = checkpoint_every
        self._project_of = project_of or _default_project_of
        self._engines: dict[str, SSMEngine] = {}
        self._last_seq: dict[str, int] = {}          # newest active seq per shard

    def sync(self) -> None:
        # Partition the active set ONCE (store returns created_seq order, so each
        # per-project slice stays ordered). Seed with known shards so a project
        # that just emptied out drains its state to initial instead of lingering.
        groups: dict[str, list[dict]] = {name: [] for name in self._engines}
        for e in self.store.active_entries():
            groups.setdefault(self._project_of(e), []).append(e)
        for name, slice_ in groups.items():
            eng = self._engines.get(name)
            if eng is None:
                eng = SSMEngine(None, self.summ, self.every)
                self._engines[name] = eng
            eng.sync(slice_)
            if slice_:
                self._last_seq[name] = max(e["created_seq"] for e in slice_)

    def digest(self, project: str | None = None) -> str:
        if project is not None:
            eng = self._engines.get(project)
            return (eng.digest() if eng is not None
                    else self.summ.render(self.summ.initial()))
        return "\n\n".join(f"## {name}\n{self._engines[name].digest()}"
                           for name in self._live())

    def projects(self) -> list[str]:
        """Non-empty shards, most-recently-active first (ties by name)."""
        return self._live()

    def _live(self) -> list[str]:
        names = [n for n, e in self._engines.items() if e.size() > 0]
        names.sort(key=lambda n: (-self._last_seq.get(n, 0), n))
        return names
