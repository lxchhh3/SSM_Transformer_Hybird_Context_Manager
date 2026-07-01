"""M1 store tests — the spine.

The store is the SOURCE OF TRUTH. Its whole job is exact, structured state with
clean retraction (criterion #2). The SSM never has to "un-remember" because the
store can drop a fact exactly, transformer-style: only ever feed what's active.
"""

import pytest

from ctx.store import Store


@pytest.fixture
def store():
    return Store(":memory:")


# --- basics -----------------------------------------------------------------

def test_publish_creates_active_entry(store):
    eid = store.publish(
        author="kevin", etype="progress", body="started auth module",
        refs=["auth.py"], entry_id="e1", ts="t0",
    )
    assert eid == "e1"
    e = store.get_entry("e1")
    assert e["status"] == "active"
    assert e["author"] == "kevin"
    assert e["type"] == "progress"
    assert e["body"] == "started auth module"
    assert e["refs"] == ["auth.py"]
    assert e["supersedes"] is None
    assert [x["entry_id"] for x in store.active_entries()] == ["e1"]


def test_event_log_is_monotonic_and_unique(store):
    store.publish("kevin", "progress", "a", entry_id="e1")
    store.publish("boss", "progress", "b", entry_id="e2")
    evs = store.all_events()
    seqs = [ev["seq"] for ev in evs]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert [ev["kind"] for ev in evs] == ["publish", "publish"]


# --- criterion #2: exact retraction -----------------------------------------

def test_revert_removes_from_active_exactly(store):
    """Clean retraction, no residue — the heart of #2."""
    store.publish("kevin", "idea", "use JWT", entry_id="A")
    store.publish("boss", "idea", "use sessions", entry_id="B")
    store.revert("A")
    assert [e["entry_id"] for e in store.active_entries()] == ["B"]  # A gone, B intact
    assert store.get_entry("A")["status"] == "reverted"             # history preserved


def test_supersede_replaces_in_active_keeps_history(store):
    store.publish("kevin", "decision", "db = postgres", entry_id="d1")
    new_id = store.supersede("d1", new_body="db = sqlite", new_entry_id="d2")
    assert new_id == "d2"
    assert [e["entry_id"] for e in store.active_entries()] == ["d2"]
    assert store.get_entry("d1")["status"] == "superseded"
    assert store.get_entry("d2")["body"] == "db = sqlite"
    assert store.get_entry("d2")["supersedes"] == "d1"


def test_revert_of_supersede_restores_prior(store):
    """revert is the exact inverse of supersede."""
    store.publish("kevin", "decision", "db = postgres", entry_id="d1")
    store.supersede("d1", new_body="db = sqlite", new_entry_id="d2")
    store.revert("d2")
    assert [e["entry_id"] for e in store.active_entries()] == ["d1"]  # prior restored
    assert store.get_entry("d1")["status"] == "active"
    assert store.get_entry("d2")["status"] == "reverted"


def test_revert_unknown_raises(store):
    with pytest.raises(KeyError):
        store.revert("nope")


def test_revert_is_idempotent(store):
    """High-freq sessions may double-fire; revert must not duplicate events."""
    store.publish("kevin", "progress", "x", entry_id="e1")
    store.revert("e1")
    n_before = len(store.all_events())
    store.revert("e1")
    assert len(store.all_events()) == n_before
    assert store.get_entry("e1")["status"] == "reverted"


def test_cannot_revert_superseded_entry(store):
    store.publish("kevin", "decision", "v1", entry_id="d1")
    store.supersede("d1", new_body="v2", new_entry_id="d2")
    with pytest.raises(ValueError):
        store.revert("d1")  # revert the successor d2 instead


# --- criterion #1: streaming feed for the SSM -------------------------------

def test_events_since_streams_new_only(store):
    store.publish("kevin", "progress", "a", entry_id="e1")
    last = store.all_events()[-1]["seq"]
    store.publish("boss", "progress", "b", entry_id="e2")
    new = store.events_since(last)
    assert [ev["entry_id"] for ev in new] == ["e2"]


def test_active_entries_filter(store):
    store.publish("kevin", "idea", "a", entry_id="e1")
    store.publish("boss", "progress", "b", entry_id="e2")
    assert [e["entry_id"] for e in store.active_entries(author="kevin")] == ["e1"]
    assert [e["entry_id"] for e in store.active_entries(etype="progress")] == ["e2"]


# --- durability -------------------------------------------------------------

def test_persistence_across_reopen(tmp_path):
    db = str(tmp_path / "ctx.db")
    s1 = Store(db)
    s1.publish("kevin", "decision", "keep", entry_id="d1")
    s1.revert("d1")
    s1.publish("boss", "idea", "live", entry_id="i1")
    s1.close()

    s2 = Store(db)
    assert [e["entry_id"] for e in s2.active_entries()] == ["i1"]
    assert s2.get_entry("d1")["status"] == "reverted"
