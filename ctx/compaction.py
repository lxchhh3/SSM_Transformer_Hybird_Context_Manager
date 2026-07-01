"""HybridCompactor — the lossy linking gist over the capped board.

This is the hybrid's role per memory hybrid-compaction-gist: the DB/BE renders the
active set VERBATIM (exact, grouped), and this compacts that board into a short,
readable "where are we" that ties explicitly-related work together. Falcon-H1-3B is
the production model (synthesizes + links + never collapses; global lesson #18/#19).

Roles it must NOT overstep (invariant #1, lesson #20):
  - it summarizes and links what the board STATES; it does not judge truth, rank
    salience, resolve conflicts, or infer unstated dependencies;
  - the gist is the lossy TOP layer — non-authoritative, with DB drill-down beneath.

The generator is injected (any object with `.generate(prompt, **kw)`), so this
module imports torch-free and the plumbing is unit-testable without a GPU; the real
model (reusing MambaSummarizer's model-agnostic load+generate) loads lazily only
when no generator is supplied.
"""

from __future__ import annotations

from ctx.index import render_board
from ctx.prompts import COMPACT_PROMPT

_EMPTY = "(nothing active)"


class HybridCompactor:
    def __init__(self, model: str = "tiiuae/Falcon-H1-3B-Instruct",
                 generator=None, max_new_tokens: int = 320, **model_kwargs):
        if generator is None:
            # MambaSummarizer's load+generate are model-agnostic (only its
            # fold/render are Mamba-streaming); reuse them as the local-LM backend.
            from ctx.mamba_summarizer import MambaSummarizer
            generator = MambaSummarizer(model, **model_kwargs)
        self._gen = generator
        self.max_new_tokens = max_new_tokens

    def compact(self, entries: list[dict]) -> str:
        """Verbatim board -> lossy linked gist. Empty set short-circuits (no model
        call). The board fed in is store-exact text, so the model can only compress
        what is real — it cannot inject a fact that was never in the store."""
        if not entries:
            return _EMPTY
        board = render_board(entries)
        return self._gen.generate(COMPACT_PROMPT.format(board=board),
                                  max_new_tokens=self.max_new_tokens)
