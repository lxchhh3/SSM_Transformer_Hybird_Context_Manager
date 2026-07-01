"""HybridCompactor plumbing — GPU-free.

The compactor turns the BE's verbatim board into a lossy linked gist via a local
LM (Falcon-H1-3B in production). By injecting the generator we test the wiring —
does it feed the VERBATIM board through the COMPACTION prompt (not the fold one),
and short-circuit on an empty set — without loading a model.
"""

from ctx.compaction import HybridCompactor


class FakeGen:
    """Stands in for the local LM: records the prompt, returns a canned gist."""

    def __init__(self):
        self.last_prompt = None
        self.calls = 0

    def generate(self, prompt, **kw):
        self.calls += 1
        self.last_prompt = prompt
        return "GIST"


def _entry(eid, author, etype, body, proj=None, seq=0):
    return {"entry_id": eid, "author": author, "type": etype, "body": body,
            "refs": [], "status": "active", "created_seq": seq, "updated_seq": seq,
            "supersedes": None, "project": proj}


def test_compact_empty_skips_the_model():
    g = FakeGen()
    c = HybridCompactor(generator=g)
    assert c.compact([]) == "(nothing active)"
    assert g.calls == 0  # never bothered the model


def test_compact_feeds_verbatim_board_into_the_prompt():
    g = FakeGen()
    c = HybridCompactor(generator=g)
    entries = [
        _entry("k1", "kevin", "progress", "raw UDP netcode at 20Hz", proj="game", seq=1),
        _entry("k2", "boss", "decision", "authoritative server, inputs only", proj="game", seq=2),
    ]
    out = c.compact(entries)
    assert out == "GIST"
    # verbatim store text reaches the model, wrapped by the compaction template
    assert "raw UDP netcode at 20Hz" in g.last_prompt
    assert "authoritative server, inputs only" in g.last_prompt
    assert "Overview:" in g.last_prompt


def test_compact_uses_compaction_prompt_not_fold():
    g = FakeGen()
    c = HybridCompactor(generator=g)
    c.compact([_entry("k1", "kevin", "progress", "x", proj="game", seq=1)])
    # the fold prompt's per-event scaffolding must NOT appear
    assert "New event" not in g.last_prompt
