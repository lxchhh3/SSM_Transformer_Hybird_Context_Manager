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

    def sync(self) -> None:
        active = self.store.active_entries()         # ordered by created_seq
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

    def _checkpoint_at_or_before(self, idx: int) -> tuple[int, Any]:
        best = self._ckpts[0]
        for k, state in self._ckpts:
            if k <= idx and k >= best[0]:
                best = (k, state)
        return best
