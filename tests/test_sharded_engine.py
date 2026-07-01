"""Sharded SSM engine — one fixed-size state per project.

The report's finding 5 (memory<->recall Pareto, recall capacity scales with
state-per-stream) says the ~25-entry envelope is per-STATE, not global. Sharding
by project gives each stream its own state, so a busy team with several projects
keeps each stream small instead of saturating one shared state. Structural win
we also assert here: churn in a hot project (revert/supersede) replays only that
shard — cold shards are never re-folded.

Model-agnostic, like test_ssm_engine: a deterministic FAKE summarizer stands in
for Mamba, so none of this needs a GPU.
"""

import pytest

from ctx.index import project_of
from ctx.service import ContextService
from ctx.ssm_engine import ShardedSSMEngine


class FakeSummarizer:
    """State = tuple of folded entry ids. Records fold order so tests can prove
    which entries (and which shards) were re-folded on a re-sync."""

    def __init__(self):
        self.fold_calls = 0
        self.folded: list[str] = []

    def initial(self):
        return ()

    def fold(self, state, entry):
        self.fold_calls += 1
        self.folded.append(entry["entry_id"])
        return state + (entry["entry_id"],)

    def render(self, state):
        return "|".join(state)


def ground_truth(svc, project):
    ids = [e["entry_id"] for e in svc.store.active_entries()
           if project_of(e) == project]
    return "|".join(ids)


@pytest.fixture
def setup():
    svc = ContextService(":memory:")
    summ = FakeSummarizer()
    eng = ShardedSSMEngine(svc.store, summ, checkpoint_every=2)
    return svc, summ, eng


# --- isolation --------------------------------------------------------------

def test_shards_isolate_by_project(setup):
    svc, summ, eng = setup
    svc.publish("kevin", "progress", "a", entry_id="a1", project="A")
    svc.publish("kevin", "progress", "b", entry_id="b1", project="B")
    svc.publish("kevin", "progress", "a2", entry_id="a2", project="A")
    eng.sync()
    assert eng.digest("A") == "a1|a2"
    assert eng.digest("B") == "b1"


def test_unknown_project_is_empty(setup):
    svc, summ, eng = setup
    svc.publish("kevin", "progress", "a", entry_id="a1", project="A")
    eng.sync()
    assert eng.digest("ghost") == ""  # render(initial()) — no such shard


# --- cheap appends, per-shard ----------------------------------------------

def test_cross_project_append_is_incremental(setup):
    svc, summ, eng = setup
    svc.publish("kevin", "progress", "a", entry_id="a1", project="A")
    svc.publish("kevin", "progress", "b", entry_id="b1", project="B")
    eng.sync()
    mark = summ.fold_calls
    svc.publish("kevin", "progress", "a2", entry_id="a2", project="A")
    eng.sync()
    assert summ.fold_calls - mark == 1  # only the new A entry folded
    assert eng.digest("A") == "a1|a2"


def test_revert_only_refolds_its_shard(setup):
    """The structural payoff: reverting in A must not re-fold any of B."""
    svc, summ, eng = setup
    for i in (1, 2, 3):
        svc.publish("kevin", "progress", str(i), entry_id=f"a{i}", project="A")
    for i in (1, 2, 3):
        svc.publish("kevin", "progress", str(i), entry_id=f"b{i}", project="B")
    eng.sync()
    mark = len(summ.folded)
    svc.revert("a2")
    eng.sync()
    replayed = summ.folded[mark:]
    assert replayed, "revert should have replayed A's tail"
    assert all(x.startswith("a") for x in replayed)  # B untouched
    assert eng.digest("A") == "a1|a3"
    assert eng.digest("B") == "b1|b2|b3"  # unchanged


# --- combined view ----------------------------------------------------------

def test_combined_digest_recency_ordered_skips_empty(setup):
    svc, summ, eng = setup
    svc.publish("kevin", "progress", "a", entry_id="a1", project="A")
    svc.publish("kevin", "progress", "b", entry_id="b1", project="B")
    eng.sync()
    combined = eng.digest()  # no project arg -> all shards
    assert "a1" in combined and "b1" in combined
    assert combined.index("## B") < combined.index("## A")  # B is newer -> first
    assert eng.projects() == ["B", "A"]

    svc.revert("b1")  # drain B
    eng.sync()
    combined = eng.digest()
    assert "## B" not in combined
    assert eng.projects() == ["A"]


# --- default grouping matches the BE ---------------------------------------

def test_default_grouping_matches_index(setup):
    svc, summ, eng = setup
    svc.publish("kevin", "progress", "x", entry_id="x1", refs=["foo/bar.py"])
    svc.publish("kevin", "progress", "y", entry_id="y1")  # no project, no ref
    eng.sync()
    assert eng.digest("foo") == "x1"       # top path segment of the ref
    assert eng.digest("unfiled") == "y1"   # fallback bucket


# --- the invariant, per shard, under a mixed stream ------------------------

def test_shard_digest_never_drifts_per_project(setup):
    svc, summ, eng = setup
    ops = [
        ("pub", "A", "a1"), ("pub", "B", "b1"), ("pub", "A", "a2"),
        ("rev", "a1", None),
        ("pub", "B", "b2"),
        ("sup", "a2", "a2b"),
        ("rev", "b1", None),
    ]
    for op, arg, extra in ops:
        if op == "pub":
            svc.publish("kevin", "progress", extra, entry_id=extra, project=arg)
        elif op == "rev":
            svc.revert(arg)
        elif op == "sup":
            svc.supersede(arg, "v2", new_entry_id=extra)
        eng.sync()
        for p in ("A", "B"):
            assert eng.digest(p) == ground_truth(svc, p)
