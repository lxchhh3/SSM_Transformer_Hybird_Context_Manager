"""Prompt templates for the SSM summarizer — pure strings, no deps.

Kept torch-free on purpose so the load-bearing invariants they must obey
(from the deep-research report + README) are testable by the stdlib-only
suite, without a GPU. The Mamba wiring in `mamba_summarizer.py` imports from
here; the tests assert the *shape* of the ask here.

Invariants these prompts must respect (see `research/deep_research_report.md`):
  - The SSM NEVER judges (finding 6; salience_select 0/3). So the fold prompt
    must not ask it to compare, dedup, flag overlap, or rank salience — that is
    Claude's / the BE's job.
  - The SSM has exponential recency bias + finite capacity (findings 5, 8). So
    the fold prompt must lean INTO recency (let old fade), not order it to
    preserve every fact past the envelope — the thing it provably can't do.
"""

from __future__ import annotations

FOLD_PROMPT = (
    "You maintain a concise shared status board for a small dev team.\n"
    "Current status:\n{state}\n\n"
    "New event from {author} ({type}): {body}\n\n"
    "Rewrite the status to fold in this event in <= 120 words. Favor the most "
    "recent and active work; let older detail fade. Note who owns what.\n"
    "Updated status:\n"
)

# Compaction / gist prompt — the LOSSY linking overview over the (BE-capped)
# verbatim board. This is the hybrid's job per memory hybrid-compaction-gist:
# heavy input -> a clean readable "where are we" that ties related work together.
# The linking rule is load-bearing: Stage-3 (lesson #20) showed that inviting the
# model to HUNT dependencies makes it hallucinate false ones, so we constrain it
# to connect ONLY what the board explicitly states, and never to judge/rank/resolve
# (that stays Claude + the DB — invariant #1). The gist is non-authoritative.
COMPACT_PROMPT = (
    "You write a short, readable 'where are we' overview for a small dev team, "
    "from the status board below.\n\n"
    "{board}\n\n"
    "Compact it into a tight overview (a lossy summary, not a full list): group "
    "related work, favor the most recent and active items, and let old detail "
    "fade. Connect two items ONLY when the board explicitly states they relate — "
    "e.g. one entry references another's output. Never guess or infer a dependency "
    "that is not written. Note who owns what. Do not rank which items matter most "
    "or resolve conflicts; only summarize what is on the board.\n"
    "Overview:\n"
)
