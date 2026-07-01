"""M3 SSM-engine tests — the streaming compressed view, model-agnostic.

The engine maintains a digest incrementally (cheap appends = #1) and re-syncs by
bounded replay from a checkpoint when the active set changes under it (revert/
supersede = #2). We test that machinery with a deterministic FAKE summarizer so
none of it needs a GPU; Mamba is just a real Summarizer plugged into the same
contract at M4.

Core invariant: engine.digest() == render(fold over store.active_entries()) — the
incremental view never drifts from ground truth, no matter the update pattern.
"""

import pytest

from ctx.service import ContextService
from ctx.ssm_engine import SSMEngine


class FakeSummarizer:
    """Deterministic stand-in for Mamba. State = tuple of folded entry ids.

    Counts folds so tests can assert the engine is incremental (cheap appends)
    and that revert replay is bounded by the checkpoint window.
    """

    def __init__(self):
        self.fold_calls = 0

    def initial(self):
        return ()

    def fold(self, state, entry):
        self.fold_calls += 1
        return state + (entry["entry_id"],)

    def render(self, state):
        return "|".join(state)


def ground_truth(svc, summ):
    state = summ.initial()
    for e in svc.store.active_entries():
        state = state + (e["entry_id"],)
    return summ.render(state)


@pytest.fixture
def setup():
    svc = ContextService(":memory:")
    summ = FakeSummarizer()
    eng = SSMEngine(svc.store, summ, checkpoint_every=2)
    return svc, summ, eng


# --- #1: streaming ingest ---------------------------------------------------

def test_ingest_builds_digest_from_active(setup):
    svc, summ, eng = setup
    svc.publish("kevin", "progress", "a", entry_id="k1")
    svc.publish("kevin", "progress", "b", entry_id="k2")
    eng.sync()
    assert eng.digest() == "k1|k2"


def test_sync_is_incremental_for_appends(setup):
    svc, summ, eng = setup
    svc.publish("kevin", "progress", "a", entry_id="k1")
    eng.sync()
    folds_after_first = summ.fold_calls
    svc.publish("boss", "progress", "b", entry_id="b1")
    eng.sync()
    # only the new entry was folded, not the whole history re-processed (#1)
    assert summ.fold_calls - folds_after_first == 1
    assert eng.digest() == "k1|b1"


# --- #2: re-sync on retraction ----------------------------------------------

def test_revert_resyncs_digest(setup):
    svc, summ, eng = setup
    for i in (1, 2, 3):
        svc.publish("kevin", "progress", str(i), entry_id=f"k{i}")
    eng.sync()
    svc.revert("k2")
    eng.sync()
    assert eng.digest() == "k1|k3"  # reverted entry gone, no residue


def test_supersede_resyncs_digest(setup):
    svc, summ, eng = setup
    svc.publish("kevin", "decision", "v1", entry_id="d1")
    eng.sync()
    svc.supersede("d1", new_body="v2", new_entry_id="d2")
    eng.sync()
    assert eng.digest() == "d2"


def test_late_revert_replay_is_bounded_by_checkpoint(setup):
    """A revert near the end must NOT re-fold the whole history — checkpoints
    bound the replay cost so #1 (cheap) and #2 (correct) coexist."""
    svc, summ, eng = setup  # checkpoint_every=2
    for i in range(1, 7):  # k1..k6
        svc.publish("kevin", "progress", str(i), entry_id=f"k{i}")
    eng.sync()
    base = summ.fold_calls
    svc.revert("k6")  # last entry
    eng.sync()
    replayed = summ.fold_calls - base
    assert eng.digest() == "k1|k2|k3|k4|k5"
    assert replayed <= 2 + 1  # at most one checkpoint window, not all 5


# --- the invariant ----------------------------------------------------------

def test_digest_never_drifts_from_ground_truth(setup):
    """High-frequency mixed stream: after every sync the incremental digest
    must equal a from-scratch summary of the active set."""
    svc, summ, eng = setup
    ops = [
        ("pub", "k1"), ("pub", "k2"), ("pub", "k3"),
        ("rev", "k2"),
        ("pub", "k4"), ("pub", "k5"),
        ("sup", "k1"),            # -> k1b
        ("rev", "k4"),
        ("pub", "k6"),
    ]
    for op, arg in ops:
        if op == "pub":
            svc.publish("kevin", "progress", arg, entry_id=arg)
        elif op == "rev":
            svc.revert(arg)
        elif op == "sup":
            svc.supersede(arg, new_body="v2", new_entry_id=arg + "b")
        eng.sync()
        assert eng.digest() == ground_truth(svc, summ)
