"""Service wiring for the SSM/hybrid layer — GPU-free (injected fakes).

The MCP exposes two new reads that need a model in production, but the SERVICE
methods take the model injected, so the coordination logic (cap to the envelope,
fall back to the verbatim board, sync the shards) is testable without a GPU.
"""

import pytest

from ctx.service import ContextService
from ctx.ssm_engine import ShardedSSMEngine


@pytest.fixture
def svc():
    return ContextService(":memory:")


class FakeCompactor:
    """Records the entries it was handed; returns a canned gist."""

    def __init__(self):
        self.got = None

    def compact(self, entries):
        self.got = entries
        return "GIST"


class FakeSummarizer:
    """Deterministic stand-in for Mamba (state = tuple of folded ids)."""

    def initial(self):
        return ()

    def fold(self, state, entry):
        return state + (entry["entry_id"],)

    def render(self, state):
        return "|".join(state)


# --- overview: lossy gist over the capped set, board fallback --------------

def test_overview_without_compactor_falls_back_to_board(svc):
    svc.publish("kevin", "progress", "raw UDP netcode", project="game", entry_id="k1")
    res = svc.overview()
    assert res["selector"] == "board"
    assert "raw UDP netcode" in res["overview"]  # verbatim store text


def test_overview_with_compactor_returns_gist(svc):
    svc.publish("kevin", "progress", "a", project="game", entry_id="k1")
    fake = FakeCompactor()
    res = svc.overview(compactor=fake)
    assert res["selector"] == "gist"
    assert res["overview"] == "GIST"
    assert [e["entry_id"] for e in fake.got] == ["k1"]  # fed the active set


def test_overview_caps_to_envelope_before_compacting(svc):
    for i in range(4):
        svc.publish("kevin", "progress", f"body{i}", project="game", entry_id=f"k{i}")
    fake = FakeCompactor()
    res = svc.overview(compactor=fake, cap_entries=2)
    assert res["shown"] == 2
    assert res["overflow"] == 2
    assert len(res["overflow_ids"]) == 2
    assert len(fake.got) == 2  # the compactor only ever sees the capped set (bounded KV)


# --- project_digests: per-project SSM shards --------------------------------

def test_project_digests_are_per_project(svc):
    svc.publish("kevin", "progress", "a", project="game", entry_id="g1")
    svc.publish("boss", "progress", "b", project="dash", entry_id="d1")
    svc.publish("kevin", "progress", "c", project="game", entry_id="g2")
    engine = ShardedSSMEngine(svc.store, FakeSummarizer(), checkpoint_every=2)
    res = svc.project_digests(engine)
    assert res["digests"]["game"] == "g1|g2"
    assert res["digests"]["dash"] == "d1"
    assert set(res["projects"]) == {"game", "dash"}
