"""Index — the deterministic 'SSM selects, store renders' layer.

Every past failure was the MODEL's prose drifting (wrong numbers, wrong author,
hallucinated bullets) while the store's facts stayed correct. So the board is
rendered VERBATIM from store entries: this module never asks a model to write a
sentence. It only does structured, GPU-free work:

  - project_of / driver_of : structured grouping keys (the user's chosen retrieval)
  - working_set            : recency-order, dedup, cap to the SSM's ~1000-tok /
                             ~25-entry faithful envelope; overflow -> `dropped`
                             (recall via the store, not the saturated SSM state)
  - render_board           : verbatim bullets, driver in the header, exact author
                             shown only when it differs from the driver

The SSM's job (added on top) is SELECTION when the active set overflows the
envelope: pick which entries are salient. Whatever it picks, the text shown is
always exact store text, so the model's drift can never reach the board.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

Entry = dict[str, Any]
Picker = Callable[[list["Entry"]], list["Entry"]]

# tuned to the measured faithful envelope (see knee sweep: ~1000 tok / ~25 entries)
CAP_ENTRIES = 25
CAP_TOKENS = 1000


def project_of(entry: Entry) -> str:
    """Structured grouping key: explicit tag, else the top path segment of the
    first file ref, else 'unfiled'. No model, no guessing."""
    tag = entry.get("project")
    if tag:
        return tag
    for ref in entry.get("refs") or []:
        if "/" in ref:
            return ref.split("/", 1)[0]
    return "unfiled"


def driver_of(entries: list[Entry]) -> Optional[str]:
    """Who is driving: the author with the most entries; ties break to the author
    of the single most-recent entry."""
    if not entries:
        return None
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["author"]] = counts.get(e["author"], 0) + 1
    top = max(counts.values())
    leaders = {a for a, c in counts.items() if c == top}
    if len(leaders) == 1:
        return next(iter(leaders))
    newest = max(entries, key=lambda e: e["created_seq"])
    return newest["author"]


def _recency(entries: list[Entry]) -> list[Entry]:
    return sorted(entries, key=lambda e: e["created_seq"], reverse=True)


def working_set(
    active: list[Entry],
    cap_entries: int = CAP_ENTRIES,
    cap_tokens: Optional[int] = None,
    measure: Optional[Callable[[str], int]] = None,
) -> dict[str, list[Entry]]:
    """Recency-order, dedup exact-duplicate bodies (keep newest), and cap to the
    envelope. Returns {'kept': [...], 'dropped': [...]}; `dropped` is the overflow
    the store/index answers by exact recall (never crammed into the SSM state)."""
    ordered = _recency(active)
    seen_bodies: set[str] = set()
    deduped: list[Entry] = []
    for e in ordered:  # newest first -> the newest of a duplicate body wins
        if e["body"] in seen_bodies:
            continue
        seen_bodies.add(e["body"])
        deduped.append(e)

    kept: list[Entry] = []
    used = 0
    for e in deduped:
        if len(kept) >= cap_entries:
            break
        if cap_tokens is not None and measure is not None:
            cost = measure(e["body"])
            if used + cost > cap_tokens:
                break
            used += cost
        kept.append(e)
    dropped = [e for e in deduped if e not in kept]
    return {"kept": kept, "dropped": dropped}


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def match_back(lines: list[str], candidates: list[Entry],
               thresh: float = 0.5) -> list[Entry]:
    """Map each model-emitted line back to the store entry it best overlaps
    (containment of shared word-tokens). Below threshold -> the line matched no
    real entry (a hallucination) and is dropped; each entry is claimed at most
    once. This is why SSM selection can never inject fake content: the output is
    always a subset of REAL entries, never the model's text."""
    scored = [(e, _tokens(e["body"])) for e in candidates]
    picked: list[Entry] = []
    used: set[str] = set()
    for line in lines:
        lt = _tokens(line)
        if not lt:
            continue
        best, best_score = None, 0.0
        for e, et in scored:
            if e["entry_id"] in used or not et:
                continue
            score = len(lt & et) / min(len(lt), len(et))
            if score > best_score:
                best, best_score = e, score
        if best is not None and best_score >= thresh:
            picked.append(best)
            used.add(best["entry_id"])
    return picked


def _chunk(items: list[Entry], size: int) -> list[list[Entry]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def select_salient(active: list[Entry], pick: Picker,
                   cap_entries: int = CAP_ENTRIES,
                   max_rounds: int = 5) -> list[Entry]:
    """Reduce an OVERSIZED active set to the salient ~cap via map-reduce, each
    batch kept within the SSM's faithful envelope. `pick(batch)` returns the
    salient SUBSET of a batch (for the SSM: match_back(mamba_pick(batch), batch)).
    Rounds shrink the pool until it fits; recency only decides display order later,
    so this can retain an OLD load-bearing decision that a recency cap would drop."""
    if len(active) <= cap_entries:
        return active
    pool = _recency(active)
    for _ in range(max_rounds):
        if len(pool) <= cap_entries:
            break
        picked: list[Entry] = []
        seen: set[str] = set()
        for batch in _chunk(pool, cap_entries):
            for e in pick(batch):
                if e["entry_id"] not in seen:
                    seen.add(e["entry_id"])
                    picked.append(e)
        if not picked or len(picked) >= len(pool):
            break  # no reduction -> stop rather than loop forever
        pool = picked
    return pool[:cap_entries]


def render_board(
    entries: list[Entry],
    project_of: Callable[[Entry], str] = project_of,
) -> str:
    """Verbatim, structured status board. Bullets are exact store bodies; the
    driver is named in the header; a bullet is author-tagged only when its author
    is not the project driver (exact attribution, straight from the store)."""
    groups: dict[str, list[Entry]] = {}
    for e in entries:
        groups.setdefault(project_of(e), []).append(e)

    # projects ordered by their most recent activity (tie -> name)
    def proj_key(name: str) -> tuple:
        newest = max(e["created_seq"] for e in groups[name])
        return (-newest, name)

    lines: list[str] = []
    for name in sorted(groups, key=proj_key):
        proj_entries = _recency(groups[name])
        driver = driver_of(proj_entries)
        lines.append(f"## {name} — {driver} driving")
        for e in proj_entries:
            tag = "" if e["author"] == driver else f" ({e['author']})"
            lines.append(f"- [{e['type']}]{tag} {e['body']}")
        lines.append("")
    return "\n".join(lines).rstrip()
