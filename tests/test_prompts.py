"""Guard the fold prompt against the two report-confirmed weaknesses.

These are regression tests for the invariants in `ctx/prompts.py`, derived
strictly from `research/deep_research_report.md`:
  - findings 3/6 + the README "SSM never judges" rule -> the fold ask must not
    request comparison / dedup / overlap-flagging / salience ranking.
  - findings 5/8 (recency bias + finite capacity) -> the fold ask must lean
    into recency, not order total preservation of every past fact.

Torch-free: they assert the shape of the *prompt string*, so the stdlib-only
suite runs them without a GPU.
"""

from ctx.prompts import COMPACT_PROMPT, FOLD_PROMPT


def _body() -> str:
    # the instruction text only, not the {state}/{body} slots the caller fills
    return FOLD_PROMPT.lower()


# --- invariant: the SSM never judges (findings 3/6) -------------------------

def test_fold_prompt_asks_for_no_judgment():
    text = _body()
    banned = ["overlap", "flag", "duplicate", "dedup", "conflict",
              "salien", "prioriti", "rank", "decide", "compare",
              "most important", "which matters"]
    hits = [w for w in banned if w in text]
    assert not hits, f"fold prompt asks the SSM to judge: {hits}"


# --- invariant: lean into recency, don't order total preservation (5/8) -----

def test_fold_prompt_does_not_order_total_preservation():
    text = _body()
    banned = ["keep every", "drop nothing", "every fact", "everything",
              "lose nothing", "never drop", "all facts"]
    hits = [w for w in banned if w in text]
    assert not hits, f"fold prompt fights recency/capacity: {hits}"


def test_fold_prompt_favors_recency():
    text = _body()
    assert "recent" in text or "fade" in text, \
        "fold prompt should lean into recency (favor recent / let old fade)"


# --- keep the template usable -----------------------------------------------

def test_fold_prompt_keeps_its_slots():
    for slot in ("{state}", "{author}", "{type}", "{body}"):
        assert slot in FOLD_PROMPT, f"fold prompt lost its {slot} slot"


# --- compaction prompt: lossy linking gist over the capped board ------------
# Stage-3 (lesson #20) showed that INVITING dependency-hunting makes the hybrid
# hallucinate false links. So the compaction prompt must (a) link only what the
# board explicitly states, (b) stay lossy + recency-leaning, (c) keep its slot.

def test_compact_prompt_keeps_its_slot():
    assert "{board}" in COMPACT_PROMPT


def test_compact_prompt_favors_recency():
    t = COMPACT_PROMPT.lower()
    assert "recent" in t or "fade" in t, \
        "compaction prompt should lean into recency (lossy gist, not archive)"


def test_compact_prompt_links_only_explicit():
    t = COMPACT_PROMPT.lower()
    # must CONSTRAIN linking to what the board states, not invite speculation
    assert "only" in t and ("state" in t or "written" in t or "reference" in t), \
        "compaction prompt must connect only explicitly-stated links (#20)"
