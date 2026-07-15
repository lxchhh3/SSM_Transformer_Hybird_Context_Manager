"""Readability fixes — display-only board upgrades (kevin's 4-point review).

All four respect the load-bearing invariants: the STORE stays verbatim (lesson
#4), overflow is surfaced never silent (#11). These only change PRESENTATION at
the read edges (MCP tools / hook), so the pure `render_board` default and every
existing test are untouched.

  #1 relative age on board entries      (injected `now` -> deterministic)
  #2 soften shouty ALL-CAPS emphasis    (acronym-safe, display-only)
  #3 bounded team_state / recent        (preview + pagination, was 70KB+)
  #4 honest overview board-fallback     (says it's authoritative, not a stub)
"""

from datetime import datetime, timedelta, timezone

import pytest

from ctx.index import render_board
from ctx.service import ContextService
from ctx.store import Store

NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)


def E(entry_id, author, etype, body, seq, project=None, created_ts=None):
    return {"entry_id": entry_id, "author": author, "type": etype, "body": body,
            "created_seq": seq, "project": project, "refs": [], "status": "active",
            "created_ts": created_ts}


@pytest.fixture
def svc():
    return ContextService(":memory:")


# --- #1 relative age (render_board, injected now) ---------------------------

def test_render_shows_relative_age_when_now_and_ts_given():
    e = E("a", "kevin", "progress", "did a thing", 1,
          created_ts=(NOW - timedelta(days=3)).isoformat())
    board = render_board([e], now=NOW)
    assert "(3d ago)" in board and "did a thing" in board


def test_render_age_buckets_minutes_hours_days():
    mk = lambda delta: render_board(
        [E("a", "k", "progress", "x", 1, created_ts=(NOW - delta).isoformat())], now=NOW)
    assert "(just now)" in mk(timedelta(seconds=5))
    assert "(2m ago)" in mk(timedelta(minutes=2))
    assert "(5h ago)" in mk(timedelta(hours=5))
    assert "(4d ago)" in mk(timedelta(days=4))


def test_render_no_age_by_default_or_when_ts_missing():
    e = E("a", "kevin", "progress", "body text", 1)  # created_ts=None
    assert "ago" not in render_board([e])              # pure default: no now
    assert "ago" not in render_board([e], now=NOW)     # now but no ts -> no crash
    assert "just now" not in render_board([e], now=NOW)


def test_render_age_survives_unparseable_ts():
    e = E("a", "kevin", "progress", "x", 1, created_ts="t0")  # store test-style ts
    assert "ago" not in render_board([e], now=NOW)  # guarded, no crash


# --- #2 soften shouty caps (acronym-safe, display-only) ---------------------

def test_soften_lowers_long_shouty_words_keeps_short_acronyms():
    e = E("a", "k", "progress",
          "M5 BC GATE PASSED: SSM beats MLP, memory is LOAD-BEARING", 1)
    out = render_board([e], soften_caps=True)
    assert "passed" in out and "load-bearing" in out    # >=5 letters -> softened
    for keep in ("SSM", "MLP", "BC", "M5", "GATE"):     # <=4 / has-digit -> kept
        assert keep in out
    assert "PASSED" not in out


def test_soften_is_off_by_default():
    e = E("a", "k", "progress", "GATE PASSED", 1)
    assert "PASSED" in render_board([e])  # verbatim unless opted in


def test_soften_recapitalizes_a_softened_leading_word():
    e = E("a", "k", "progress", "LANDED the executor", 1)
    assert "Landed the executor" in render_board([e], soften_caps=True)


def test_soften_preserves_mixed_case_and_digit_tokens():
    e = E("a", "k", "progress", "DAgger over AoW at 500food, feudal 8:45", 1)
    out = render_board([e], soften_caps=True)
    for keep in ("DAgger", "AoW", "500food", "8:45"):
        assert keep in out


# --- #1+#2 wired through status_board (opt-in), default stays verbatim -------

def test_status_board_softens_and_ages_when_opted_in(svc):
    svc.publish("kevin", "progress", "GATE PASSED and LANDED", project="game",
                entry_id="k1")
    r = svc.status_board(now=datetime.now(timezone.utc), soften_caps=True)
    assert "passed" in r["board"] and "landed" in r["board"]
    assert "PASSED" not in r["board"]
    assert "(just now)" in r["board"]  # freshly published


def test_status_board_default_is_verbatim_and_ageless(svc):
    svc.publish("kevin", "progress", "GATE PASSED", project="game", entry_id="k1")
    r = svc.status_board()
    assert "- [progress] GATE PASSED" in r["board"]
    assert "ago" not in r["board"] and "just now" not in r["board"]


# --- store helper: ts_by_seq (creating-event ts) ----------------------------

def test_ts_by_seq_maps_creating_event_ts():
    s = Store(":memory:")
    s.publish("kevin", "progress", "a", entry_id="e1", ts="2026-01-01T00:00:00+00:00")
    seq = s.get_entry("e1")["created_seq"]
    assert s.ts_by_seq([seq]) == {seq: "2026-01-01T00:00:00+00:00"}
    assert s.ts_by_seq([999999]) == {}
    assert s.ts_by_seq([]) == {}


# --- #3 bounded team_state --------------------------------------------------

def test_team_state_preview_truncates_bodies_and_adds_totals(svc):
    svc.publish("kevin", "progress", "x" * 500, project="game", entry_id="k1")
    svc.publish("boss", "decision", "short", project="game", entry_id="b1")
    r = svc.team_state(preview=True, body_chars=100)
    ent = r["by_author"]["kevin"]["progress"][0]
    assert len(ent["body"]) <= 101 and ent["body_len"] == 500  # trimmed, true len kept
    assert ent["entry_id"] == "k1"                             # id survives trim
    assert r["totals"]["kevin"]["progress"] == 1
    assert r["totals"]["boss"]["decision"] == 1
    assert r["preview"] is True


def test_team_state_raw_default_is_unchanged(svc):
    svc.publish("kevin", "progress", "full body here", entry_id="k1")
    r = svc.team_state()
    assert r["by_author"]["kevin"]["progress"][0]["body"] == "full body here"
    assert "totals" not in r  # default shape preserved for existing callers


# --- #3 bounded recent (pagination) -----------------------------------------

def test_recent_summary_keeps_newest_window_and_surfaces_omitted(svc):
    for i in range(10):
        svc.publish("kevin", "progress", f"event {i} " + "y" * 300, entry_id=f"e{i}")
    r = svc.recent_summary(since_seq=0, limit=3, body_chars=50)
    assert r["returned"] == 3 and r["total"] == 10 and r["omitted"] == 7
    assert [ev["entry_id"] for ev in r["events"]] == ["e7", "e8", "e9"]  # newest
    assert all(len(ev["payload"]["body"]) <= 51 for ev in r["events"])   # trimmed
    assert r["latest_seq"] == 10 and "not shown" in r["note"]


def test_recent_summary_no_note_when_all_fit(svc):
    svc.publish("kevin", "progress", "a", entry_id="e1")
    r = svc.recent_summary(since_seq=0, limit=50)
    assert r["returned"] == 1 and r["omitted"] == 0 and "note" not in r


# --- #4 honest overview board-fallback --------------------------------------

class _FakeCompactor:
    def compact(self, entries):
        return "GIST"


def test_overview_board_fallback_declares_itself_authoritative(svc):
    svc.publish("kevin", "progress", "x", project="game", entry_id="k1")
    r = svc.overview()
    assert r["selector"] == "board"
    note = r["note"].lower()
    assert "authoritative" in note or "verbatim" in note


def test_overview_gist_path_has_no_board_note(svc):
    svc.publish("kevin", "progress", "a", project="game", entry_id="k1")
    r = svc.overview(compactor=_FakeCompactor())
    assert r["selector"] == "gist" and "note" not in r
