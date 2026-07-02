"""CC hook layer — deterministic read-side injection.

SessionStart injects the verbatim board; UserPromptSubmit injects only NEW
team events via a per-session seq watermark (the "stream"). Reads the DB
directly (fast, proxy-immune, works with the server down); writes stay
judgment calls the CC makes through the MCP tools. Fail-soft everywhere:
a hook must never break the user's prompt.
"""

import pytest

from ctx.hooks import prompt_submit, session_start
from ctx.service import ContextService
from ctx.store import Store


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "team.db")


@pytest.fixture
def state(tmp_path):
    return str(tmp_path / "hook_state")


def _seed(db, *pubs):
    svc = ContextService(db)
    for kw in pubs:
        svc.publish(**kw)
    svc.close()


# --- store: the watermark primitive -----------------------------------------

def test_store_max_seq(db):
    s = Store(db)
    assert s.max_seq() == 0
    s.publish("kevin", "progress", "a", entry_id="k1")
    s.publish("boss", "progress", "b", entry_id="b1")
    assert s.max_seq() == 2
    s.close()


# --- session-start: the board, verbatim --------------------------------------

def test_session_start_renders_verbatim_board(db, state):
    _seed(db, dict(author="kevin", etype="decision", project="game",
                   body="Authoritative server model", entry_id="k1"))
    out = session_start(db, state, "s1")
    assert "Authoritative server model" in out
    assert "game" in out


def test_session_start_empty_board_says_so(db, state):
    ContextService(db).close()  # create an empty store
    assert "empty" in session_start(db, state, "s1").lower()


def test_session_start_missing_db_is_silent_and_creates_nothing(tmp_path, state):
    missing = tmp_path / "nope" / "team.db"
    assert session_start(str(missing), state, "s1") == ""
    assert not missing.exists()


def test_session_start_resets_watermark(db, state):
    # the board already shows current state — the first prompt must not replay it
    _seed(db, dict(author="kevin", etype="progress", body="seed", entry_id="k1"))
    session_start(db, state, "s1")
    assert prompt_submit(db, state, "s1") == ""


# --- prompt-submit: the stream ------------------------------------------------

def test_first_prompt_initializes_silently(db, state):
    _seed(db, dict(author="kevin", etype="progress", body="history", entry_id="k1"))
    assert prompt_submit(db, state, "s1") == ""  # history is not a stream
    assert prompt_submit(db, state, "s1") == ""  # watermark was set


def test_prompt_streams_new_events_exactly_once(db, state):
    _seed(db, dict(author="kevin", etype="progress", body="old", entry_id="k1"))
    assert prompt_submit(db, state, "s1") == ""
    _seed(db, dict(author="boss", etype="progress", project="dash",
                   body="LTTB downsampling", refs=["chart.tsx"], entry_id="b1"))
    out = prompt_submit(db, state, "s1")
    assert "boss" in out and "LTTB downsampling" in out and "chart.tsx" in out
    assert prompt_submit(db, state, "s1") == ""  # consumed


def test_watermarks_are_per_session(db, state):
    assert prompt_submit(db, state, "s1") == ""
    assert prompt_submit(db, state, "s2") == ""
    _seed(db, dict(author="boss", etype="progress", body="parallel work", entry_id="b1"))
    assert "parallel work" in prompt_submit(db, state, "s1")
    assert "parallel work" in prompt_submit(db, state, "s2")  # s1 didn't consume s2's


def test_supersede_and_revert_render(db, state):
    _seed(db, dict(author="kevin", etype="idea", body="v1", entry_id="k1"))
    assert prompt_submit(db, state, "s1") == ""
    svc = ContextService(db)
    svc.supersede("k1", "v2 replaces it")
    svc.close()
    out = prompt_submit(db, state, "s1")
    assert "supersede" in out and "v2 replaces it" in out and "k1" in out
    svc = ContextService(db)
    svc.publish("boss", "idea", "dead end", entry_id="dead")
    svc.revert("dead")
    svc.close()
    out = prompt_submit(db, state, "s1")
    assert "revert" in out and "dead" in out


def test_stream_caps_and_surfaces_overflow(db, state):
    assert prompt_submit(db, state, "s1") == ""
    _seed(db, *[dict(author="boss", etype="progress", body=f"unit {i}",
                     entry_id=f"b{i}") for i in range(25)])
    out = prompt_submit(db, state, "s1")
    assert "unit 24" in out                    # newest shown
    assert "unit 0" not in out                 # oldest capped...
    assert "not shown" in out and "5" in out   # ...but surfaced, never silent


def test_prompt_submit_missing_db_is_silent(tmp_path, state):
    missing = tmp_path / "nope" / "team.db"
    assert prompt_submit(str(missing), state, "s1") == ""
    assert not missing.exists()
