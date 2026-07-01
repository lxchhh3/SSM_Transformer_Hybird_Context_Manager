"""M2 service-layer tests — the model-free coordination brain.

This is where the project earns its keep: stop two devs duplicating. All of it
is deterministic on the structured store (file-ref collisions, team digest).
The SSM (M3) later upgrades this with semantic overlap + NL digests, but the
exact, cheap path lives here and never needs a GPU.
"""

import pytest

from ctx.service import ContextService


@pytest.fixture
def svc():
    return ContextService(":memory:")


# --- overlap detection (#3, deterministic file-ref path) --------------------

def test_publish_warns_on_file_overlap_by_other_author(svc):
    svc.publish("kevin", "progress", "refactoring login", refs=["auth.py"],
                entry_id="k1")
    res = svc.publish("boss", "progress", "adding 2FA", refs=["auth.py", "totp.py"],
                      entry_id="b1")
    assert res["entry_id"] == "b1"
    assert len(res["overlaps"]) == 1
    ov = res["overlaps"][0]
    assert ov["entry_id"] == "k1"
    assert ov["author"] == "kevin"
    assert ov["shared_refs"] == ["auth.py"]


def test_publish_does_not_warn_on_own_files(svc):
    svc.publish("kevin", "progress", "part 1", refs=["auth.py"], entry_id="k1")
    res = svc.publish("kevin", "progress", "part 2", refs=["auth.py"], entry_id="k2")
    assert res["overlaps"] == []  # your own work is not duplication


def test_reverted_entries_dont_cause_false_overlap(svc):
    """#2 feeding clean coordination: a retracted file-claim must not collide."""
    svc.publish("kevin", "progress", "abandoned attempt", refs=["auth.py"],
                entry_id="k1")
    svc.revert("k1")
    res = svc.publish("boss", "progress", "fresh start", refs=["auth.py"],
                      entry_id="b1")
    assert res["overlaps"] == []


def test_check_overlap_query_before_publishing(svc):
    svc.publish("kevin", "idea", "cache layer", refs=["cache.py"], entry_id="k1")
    hits = svc.check_overlap(refs=["cache.py", "db.py"], author="boss")
    assert [h["entry_id"] for h in hits] == ["k1"]
    assert hits[0]["shared_refs"] == ["cache.py"]


def test_overlaps_lists_current_cross_author_collisions(svc):
    svc.publish("kevin", "progress", "x", refs=["auth.py"], entry_id="k1")
    svc.publish("boss", "progress", "y", refs=["auth.py"], entry_id="b1")
    svc.publish("kevin", "progress", "z", refs=["solo.py"], entry_id="k2")
    collisions = svc.overlaps()
    assert len(collisions) == 1
    pair = collisions[0]
    assert {pair["a"], pair["b"]} == {"k1", "b1"}
    assert pair["shared_refs"] == ["auth.py"]


# --- team state digest ------------------------------------------------------

def test_team_state_groups_active_by_author_and_type(svc):
    svc.publish("kevin", "progress", "on auth", refs=["auth.py"], entry_id="k1")
    svc.publish("kevin", "idea", "use JWT", entry_id="k2")
    svc.publish("boss", "decision", "db = sqlite", entry_id="b1")
    svc.publish("boss", "progress", "scrapped", entry_id="b2")
    svc.revert("b2")  # reverted -> must not appear

    state = svc.team_state()
    assert set(state["by_author"]) == {"kevin", "boss"}
    assert [e["entry_id"] for e in state["by_author"]["kevin"]["progress"]] == ["k1"]
    assert [e["entry_id"] for e in state["by_author"]["kevin"]["idea"]] == ["k2"]
    assert [e["entry_id"] for e in state["by_author"]["boss"]["decision"]] == ["b1"]
    assert "progress" not in state["by_author"]["boss"]  # b2 reverted, group empty


def test_status_board_renders_verbatim_grouped_and_excludes_reverted(svc):
    svc.publish("boss", "decision", "authoritative server model", project="game")
    svc.publish("boss", "progress", "delta snapshots 1180B -> 320B", project="game")
    svc.publish("kevin", "progress", "LineChart 22fps on pan", project="dash")
    svc.publish("kevin", "idea", "scrapped approach", project="dash", entry_id="dead")
    svc.revert("dead")  # boss drives game (2 entries); reverted entry must vanish

    r = svc.status_board()
    board = r["board"]
    assert "## game — " in board
    assert "- [decision] authoritative server model" in board       # verbatim
    assert "- [progress] delta snapshots 1180B -> 320B" in board
    assert "scrapped approach" not in board                         # reverted gone
    assert r["shown"] == 3 and r["overflow"] == 0


def test_status_board_surfaces_overflow_not_silently(svc):
    for i in range(30):
        svc.publish("kevin", "progress", f"entry number {i}", project="game")
    r = svc.status_board(cap_entries=25)
    assert r["shown"] == 25
    assert r["overflow"] == 5 and len(r["overflow_ids"]) == 5


def test_status_board_uses_picker_to_keep_salient_over_recency(svc):
    svc.publish("boss", "decision", "authoritative server model — the spine", project="game",
                entry_id="spine")  # OLD load-bearing decision, published first
    for i in range(30):
        svc.publish("kevin", "progress", f"minor tweak {i}", project="game")

    recency = svc.status_board(cap_entries=25)
    assert recency["selector"] == "recency"
    assert "spine" in recency["overflow_ids"]           # recency drops the old spine

    pick = lambda batch: [e for e in batch if e["type"] == "decision"]
    ssm = svc.status_board(cap_entries=25, pick=pick)
    assert ssm["selector"] == "ssm"
    assert "the spine" in ssm["board"]                  # salience keeps it on the board


def test_recent_returns_events_since(svc):
    svc.publish("kevin", "progress", "a", entry_id="k1")
    cut = svc.recent()[-1]["seq"]
    svc.publish("boss", "progress", "b", entry_id="b1")
    new = svc.recent(since_seq=cut)
    assert [ev["entry_id"] for ev in new] == ["b1"]
