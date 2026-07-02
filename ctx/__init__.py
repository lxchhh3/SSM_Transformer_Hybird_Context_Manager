"""Context_Manager — shared context layer for parallel CC-driven dev.

The live product (see README) is deterministic and GPU-free:
  store      = DB: source of truth (append-only log, exact supersede/revert)
  index      = BE: deterministic grouping/driver/cap over the store
  service    = BE: the query surface the tools call
  mcp_server = both CC sessions call these tools over the LAN

Judgment (dedup / conflict / merge) = Claude, on demand, elsewhere.

compaction / ssm_engine / mamba_summarizer / prompts / benchmark are the
ARCHIVED model-backed cache experiments — kept as the record, off by default
(research/SSM_POSTMORTEM.md). Nothing in the live path imports them.
"""

__version__ = "0.1.0"
