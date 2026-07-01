"""Index tests — the deterministic 'SSM selects, store renders' layer.

The board is rendered VERBATIM from store entries (no model prose -> no drift);
this module only groups, picks the driver, orders by recency, and caps the working
set to the SSM's faithful envelope. All structured-keys, GPU-free, TDD.
"""

import pytest

from ctx.index import (driver_of, match_back, project_of, render_board,
                       select_salient, working_set)


def E(entry_id, author, etype, body, seq, project=None, refs=None, status="active"):
    return {"entry_id": entry_id, "author": author, "type": etype, "body": body,
            "created_seq": seq, "project": project, "refs": refs or [],
            "status": status}


# --- project_of: structured grouping key ------------------------------------

def test_project_of_prefers_explicit_tag():
    assert project_of(E("a", "kevin", "progress", "x", 1, project="dash")) == "dash"


def test_project_of_derives_from_ref_path():
    e = E("a", "kevin", "progress", "x", 1, refs=["game/server/net.go"])
    assert project_of(e) == "game"


def test_project_of_falls_back_to_unfiled():
    assert project_of(E("a", "kevin", "progress", "x", 1, refs=["auth.py"])) == "unfiled"
    assert project_of(E("a", "kevin", "progress", "x", 1)) == "unfiled"


# --- driver_of: who is driving (structured, by entry count) -----------------

def test_driver_is_author_with_most_entries():
    es = [E("a", "kevin", "progress", "x", 1), E("b", "kevin", "progress", "y", 2),
          E("c", "boss", "decision", "z", 3)]
    assert driver_of(es) == "kevin"


def test_driver_tie_breaks_to_most_recent():
    es = [E("a", "kevin", "progress", "x", 1), E("b", "boss", "decision", "z", 2)]
    assert driver_of(es) == "boss"  # 1-1 tie -> the more recent entry's author


# --- working_set: recency order + cap to the envelope + overflow ------------

def test_working_set_orders_by_recency_desc():
    es = [E("a", "kevin", "progress", "old", 1), E("b", "boss", "progress", "new", 3),
          E("c", "kevin", "idea", "mid", 2)]
    kept = working_set(es)["kept"]
    assert [e["entry_id"] for e in kept] == ["b", "c", "a"]


def test_working_set_caps_entries_and_reports_overflow():
    es = [E(str(i), "kevin", "progress", f"e{i}", i) for i in range(30)]
    ws = working_set(es, cap_entries=25)
    assert len(ws["kept"]) == 25
    assert len(ws["dropped"]) == 5
    # the 5 OLDEST fall to overflow (recall via the store/index, not the SSM)
    assert {e["entry_id"] for e in ws["dropped"]} == {"0", "1", "2", "3", "4"}


def test_working_set_caps_by_measured_tokens():
    # distinct bodies (else dedup collapses them), 5 words each
    es = [E(str(i), "kevin", "progress", f"e{i} w w w w", i) for i in range(10)]
    ws = working_set(es, cap_entries=99, cap_tokens=12, measure=lambda s: len(s.split()))
    assert len(ws["kept"]) == 2  # 5 + 5 = 10 <= 12; a third would hit 15 > 12


def test_working_set_dedups_exact_duplicate_body_keeping_newest():
    es = [E("a", "kevin", "progress", "same fact", 1),
          E("b", "boss", "progress", "same fact", 5),
          E("c", "kevin", "idea", "other", 3)]
    kept = working_set(es)["kept"]
    ids = [e["entry_id"] for e in kept]
    assert "a" not in ids and "b" in ids and "c" in ids  # older exact-dup dropped


# --- render_board: verbatim, grouped, driver header, exact attribution ------

def test_render_groups_by_project_with_driver_header():
    es = [E("a", "boss", "decision", "authoritative server", 2, project="game"),
          E("b", "boss", "progress", "delta snapshots", 3, project="game"),
          E("c", "kevin", "progress", "linechart", 1, project="dash")]
    board = render_board(es)
    assert "## game — boss driving" in board
    assert "## dash — kevin driving" in board


def test_render_bullets_are_verbatim_store_text():
    es = [E("a", "boss", "decision", "1180B -> 320B, ring buffer 32 acks", 1, project="game")]
    board = render_board(es)
    assert "- [decision] 1180B -> 320B, ring buffer 32 acks" in board


def test_render_shows_author_only_when_not_the_driver():
    es = [E("a", "boss", "decision", "d1", 3, project="game"),
          E("b", "boss", "progress", "d2", 2, project="game"),
          E("c", "kevin", "idea", "guest idea", 1, project="game")]  # boss drives
    board = render_board(es)
    assert "- [decision] d1" in board                 # driver: no author tag
    assert "- [idea] (kevin) guest idea" in board     # non-driver: tagged


# --- match_back: SSM lines -> exact store entries (never fake content) -------

def test_match_back_maps_paraphrase_to_the_real_entry():
    cands = [E("a", "boss", "decision", "Authoritative server model, clients send inputs only", 1),
             E("b", "kevin", "progress", "Delta snapshots cut packets 1180B to 320B", 2)]
    got = match_back(["authoritative server model — clients only send inputs"], cands)
    assert [e["entry_id"] for e in got] == ["a"]


def test_match_back_drops_hallucinated_lines():
    cands = [E("a", "boss", "decision", "Authoritative server model", 1)]
    got = match_back(["Migrated the whole stack to Kubernetes with a service mesh"], cands)
    assert got == []  # no real entry -> nothing injected


def test_match_back_claims_each_entry_at_most_once():
    cands = [E("a", "boss", "decision", "Authoritative server model clients inputs only", 1)]
    got = match_back(["authoritative server model", "clients send inputs only, authoritative"], cands)
    assert [e["entry_id"] for e in got] == ["a"]  # both lines hit 'a' -> one selection


# --- select_salient: keep old-but-load-bearing that recency would drop -------

def test_select_salient_retains_old_decision_recency_drops():
    # 30 recent progress notes + one OLD decision at the very bottom (seq 0)
    es = [E("dec", "boss", "decision", "Authoritative server model", 0)]
    es += [E(str(i), "kevin", "progress", f"tweak number {i}", i) for i in range(1, 31)]

    recency_kept = {e["entry_id"] for e in working_set(es, cap_entries=25)["kept"]}
    assert "dec" not in recency_kept  # recency alone loses the old decision

    pick_decisions = lambda batch: [e for e in batch if e["type"] == "decision"]
    salient = select_salient(es, pick_decisions, cap_entries=25)
    assert "dec" in {e["entry_id"] for e in salient}  # salience keeps it


def test_select_salient_noop_when_within_envelope():
    es = [E(str(i), "kevin", "progress", f"e{i}", i) for i in range(10)]
    assert select_salient(es, lambda b: [], cap_entries=25) == es


def test_render_orders_projects_and_entries_by_recency():
    es = [E("old", "kevin", "progress", "old proj", 1, project="alpha"),
          E("new1", "boss", "progress", "new a", 5, project="beta"),
          E("new2", "boss", "decision", "new b", 4, project="beta")]
    board = render_board(es)
    # beta (newest entry seq 5) appears before alpha; within beta, seq5 before seq4
    assert board.index("## beta") < board.index("## alpha")
    assert board.index("new a") < board.index("new b")
