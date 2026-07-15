"""Tool-surface polish — the four read-side defects from the 2026-07-15 eval.

All four keep the load-bearing invariants: the STORE stays verbatim, defaults
stay byte-compatible for library callers (only the MCP edge opts in), and
nothing is silently dropped — every omission is surfaced in a note or count.

  #1 recent()'s pagination note must name the registered TOOL (`recent`), not
     the internal service method (`recent_summary`)
  #2 team_state preview must be bounded for ANY active-set size: newest-first
     entry `limit` (omitted surfaced) + per-entry refs cap (76KB at 117 entries)
  #3 status_board/overview can omit the overflow_ids blob (count stays) — 92
     raw UUIDs were ~1/3 of the payload with no tool able to consume them
  #4 get_entry: verbatim single-entry drill-down by full id or unique prefix,
     the reader that finally makes surfaced ids actionable
"""

import pytest

from ctx.index import CAP_ENTRIES
from ctx.service import ContextService


@pytest.fixture
def svc():
    return ContextService(":memory:")


# --- #1 recent note names the tool ------------------------------------------

def test_recent_note_names_the_tool(svc):
    for i in range(5):
        svc.publish("kevin", "progress", f"event {i}")
    res = svc.recent_summary(since_seq=0, limit=2)
    assert res["omitted"] == 3
    assert "recent(" in res["note"]
    assert "recent_summary" not in res["note"]


# --- #2 bounded team_state ---------------------------------------------------

def _flatten(res):
    return [e for types in res["by_author"].values()
            for lst in types.values() for e in lst]


def test_team_state_limit_keeps_newest_and_surfaces_omitted(svc):
    ids = [svc.publish("kevin", "progress", f"body {i}")["entry_id"]
           for i in range(8)]
    res = svc.team_state(preview=True, limit=3)
    listed = _flatten(res)
    assert len(listed) == 3
    assert {e["entry_id"] for e in listed} == set(ids[-3:])  # newest kept
    assert res["omitted"] == 5                               # never silent
    assert "5" in res["note"] and "limit" in res["note"]
    assert res["totals"]["kevin"]["progress"] == 8           # totals count ALL


def test_team_state_caps_refs_in_preview(svc):
    refs = [f"D:/proj/file{i}.py" for i in range(9)]
    svc.publish("kevin", "progress", "refy", refs=refs)
    item = _flatten(svc.team_state(preview=True))[0]
    assert item["refs"] == refs[:4]
    assert item["refs_omitted"] == 5


def test_team_state_no_limit_or_raw_mode_unchanged(svc):
    refs = [f"f{i}" for i in range(9)]
    svc.publish("kevin", "progress", "x", refs=refs)
    raw = svc.team_state()                       # library default: full entries
    assert _flatten(raw)[0]["refs"] == refs
    assert "omitted" not in raw and "note" not in raw
    unlimited = svc.team_state(preview=True)     # preview without limit: all shown
    assert len(_flatten(unlimited)) == 1 and "omitted" not in unlimited


# --- #3 overflow_ids opt-out --------------------------------------------------

def test_status_board_can_omit_overflow_ids(svc):
    for i in range(CAP_ENTRIES + 5):
        svc.publish("kevin", "progress", f"e{i}")
    default = svc.status_board()                 # library default unchanged
    assert default["overflow"] == 5 and len(default["overflow_ids"]) == 5
    slim = svc.status_board(include_overflow_ids=False)
    assert "overflow_ids" not in slim
    assert slim["overflow"] == 5                 # the count survives


def test_overview_can_omit_overflow_ids(svc):
    for i in range(CAP_ENTRIES + 2):
        svc.publish("kevin", "progress", f"e{i}")
    default = svc.overview()
    assert len(default["overflow_ids"]) == 2
    slim = svc.overview(include_overflow_ids=False)
    assert "overflow_ids" not in slim and slim["overflow"] == 2


# --- #4 get_entry drill-down ---------------------------------------------------

def test_get_entry_exact_and_unique_prefix(svc):
    eid = svc.publish("kevin", "decision", "the locked call",
                      refs=["a.py"])["entry_id"]
    full = svc.get_entry(eid)
    assert full["body"] == "the locked call" and full["status"] == "active"
    assert full["created_ts"]                    # ts joined for age context
    assert svc.get_entry(eid[:8])["entry_id"] == eid


def test_get_entry_ambiguous_prefix_lists_matches(svc):
    a = svc.publish("kevin", "progress", "one",
                    entry_id="aaaa1122" + "0" * 24)["entry_id"]
    b = svc.publish("kevin", "progress", "two",
                    entry_id="aaaa1133" + "0" * 24)["entry_id"]
    res = svc.get_entry("aaaa11")                # shared prefix -> ambiguous
    assert "error" in res and set(res["matches"]) == {a, b}
    assert svc.get_entry("aaaa1122")["entry_id"] == a   # longer prefix resolves


def test_get_entry_misses_are_friendly_not_raising(svc):
    assert "error" in svc.get_entry("ffffffffffff")   # no match
    svc.publish("kevin", "progress", "x", entry_id="abcde" + "0" * 27)
    assert "error" in svc.get_entry("abc")            # <6 chars: exact only
