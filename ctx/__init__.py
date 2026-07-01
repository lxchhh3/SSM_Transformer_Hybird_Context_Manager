"""Context_Manager — shared context layer for parallel CC-driven dev.

Architecture (see README):
  store      = source of truth (structured, mutable, exact reverts)   [M1]
  mcp_server = tools both CC sessions call over LAN                   [M2]
  ssm_engine = Mamba streaming compressed view (digest/overlap/merge) [M3]

The transformer (Claude, in the CC sessions) stays "elsewhere" and is
called on demand for quality-critical reasoning. We build the SSM + plumbing.
"""

__version__ = "0.0.1"
