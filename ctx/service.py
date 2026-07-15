"""Service layer — the model-free coordination brain over the store.

Everything here is deterministic and GPU-free: file-ref collision detection and
a structured team digest. It directly attacks the duplication problem. The SSM
(M3) layers semantic overlap + natural-language digests on top, but this exact,
cheap path always works and needs no model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from ctx.index import (CAP_ENTRIES, CAP_TOKENS, render_board, select_salient,
                       working_set)
from ctx.store import Store


def _preview(body: str, n: int) -> str:
    """Collapse whitespace and cap a body to n chars (ellipsis if cut) — the
    bounded-read primitive so team_state/recent can't blow the token budget."""
    s = " ".join(body.split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _split_pinned(active: list[dict[str, Any]],
                  pin_types: tuple[str, ...]) -> tuple[list, list]:
    """Split the active set into (pinned, rest) by entry type. Pinned entries are
    exempt from the recency cap — a standing [decision] must never fall off the
    board just because progress notes outpaced it."""
    if not pin_types:
        return [], list(active)
    pinned = [e for e in active if e["type"] in pin_types]
    rest = [e for e in active if e["type"] not in pin_types]
    return pinned, rest


def _merge_recency(pinned: list[dict[str, Any]],
                   kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Weave pinned entries back into the capped set, newest-first — the same
    display order working_set produces, so the board reads uniformly."""
    return sorted(pinned + kept, key=lambda e: e["created_seq"], reverse=True)


class ContextService:
    def __init__(self, db_path: str = ":memory:", store: Optional[Store] = None):
        self.store = store if store is not None else Store(db_path)

    def close(self) -> None:
        self.store.close()

    # -- writes --------------------------------------------------------------

    def publish(self, author: str, etype: str, body: str,
                refs: Optional[list[str]] = None,
                entry_id: Optional[str] = None,
                project: Optional[str] = None) -> dict[str, Any]:
        refs = list(refs or [])
        overlaps = self.check_overlap(refs=refs, author=author)
        eid = self.store.publish(author, etype, body, refs=refs, entry_id=entry_id,
                                 project=project)
        return {"entry_id": eid, "overlaps": overlaps}

    def supersede(self, entry_id: str, new_body: str, **kwargs: Any) -> dict[str, Any]:
        new_id = self.store.supersede(entry_id, new_body, **kwargs)
        e = self.store.get_entry(new_id)
        overlaps = self.check_overlap(refs=e["refs"], author=e["author"])
        return {"entry_id": new_id, "overlaps": overlaps}

    def revert(self, entry_id: str) -> None:
        self.store.revert(entry_id)

    # -- coordination reads --------------------------------------------------

    def check_overlap(self, refs: Optional[list[str]] = None,
                      author: Optional[str] = None) -> list[dict[str, Any]]:
        """Active entries by OTHER authors whose files intersect `refs`."""
        refs = list(refs or [])
        if not refs:
            return []
        hits = []
        for e in self.store.active_entries():
            if author is not None and e["author"] == author:
                continue
            other = set(e["refs"])
            shared = [r for r in refs if r in other]  # preserve query order
            if shared:
                hits.append({
                    "entry_id": e["entry_id"],
                    "author": e["author"],
                    "type": e["type"],
                    "shared_refs": shared,
                })
        return hits

    def overlaps(self) -> list[dict[str, Any]]:
        """All current cross-author file collisions in the active set."""
        active = self.store.active_entries()  # ordered by created_seq
        out = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                if a["author"] == b["author"]:
                    continue
                bset = set(b["refs"])
                shared = [r for r in a["refs"] if r in bset]
                if shared:
                    out.append({"a": a["entry_id"], "b": b["entry_id"],
                                "shared_refs": shared})
        return out

    def team_state(self, preview: bool = False,
                   body_chars: int = 220,
                   limit: Optional[int] = None,
                   max_refs: int = 4) -> dict[str, Any]:
        """Structured 'where are we' digest: active entries by author -> type.

        `preview=True` (what the MCP tool uses) trims each body to `body_chars`
        (true length kept in `body_len`), caps refs at `max_refs` (rest counted in
        `refs_omitted`), and adds a `totals` count matrix. `limit` bounds the
        LISTED entries (newest by created_seq kept; older ones counted in
        `omitted` + a note, never silent) so the read stays bounded for ANY
        active-set size — 220-char previews alone still hit 76KB at 117 entries.
        `totals` always counts the WHOLE active set. Default is raw/unchanged for
        existing library callers."""
        active = self.store.active_entries()  # created_seq ascending
        totals: dict[str, dict[str, int]] = {}
        if preview:
            for e in active:
                t = totals.setdefault(e["author"], {})
                t[e["type"]] = t.get(e["type"], 0) + 1
        omitted = 0
        if preview and limit and len(active) > limit:
            omitted = len(active) - limit
            active = active[-limit:]  # newest kept
        by_author: dict[str, dict[str, list]] = {}
        for e in active:
            if preview:
                refs = e["refs"]
                item: dict[str, Any] = {
                    "entry_id": e["entry_id"], "author": e["author"],
                    "type": e["type"], "project": e["project"],
                    "created_seq": e["created_seq"], "refs": refs[:max_refs],
                    "body_len": len(e["body"]),
                    "body": _preview(e["body"], body_chars),
                }
                if len(refs) > max_refs:
                    item["refs_omitted"] = len(refs) - max_refs
            else:
                item = e
            by_author.setdefault(e["author"], {}).setdefault(e["type"], []).append(item)
        res: dict[str, Any] = {"by_author": by_author, "overlaps": self.overlaps()}
        if preview:
            res["totals"] = totals
            res["preview"] = True
            if omitted:
                res["omitted"] = omitted
                res["note"] = (f"{omitted} older active entries not listed "
                               f"(limit={limit}, newest kept; totals count "
                               f"everything). Raise limit to page, or "
                               f"get_entry(id) for one entry verbatim.")
        return res

    def get_entry(self, entry_id: str) -> dict[str, Any]:
        """Read ONE entry VERBATIM by full id or unique prefix (>= 6 chars) —
        the drill-down that makes ids surfaced by team_state/check_overlap
        actionable. Never raises on a miss: returns {"error": ...} (plus
        `matches` when a prefix is ambiguous) so the MCP edge stays friendly."""
        try:
            e = self.store.get_entry(entry_id)
        except KeyError:
            if len(entry_id) < 6:
                return {"error": f"no entry with id {entry_id!r} "
                                 "(prefix lookup needs >= 6 chars)"}
            matches = self.store.entries_by_prefix(entry_id)
            if not matches:
                return {"error": f"no entry matches {entry_id!r}"}
            if len(matches) > 1:
                return {"error": f"ambiguous prefix {entry_id!r}",
                        "matches": [m["entry_id"] for m in matches]}
            e = matches[0]
        self._attach_ts([e])
        return e

    def _attach_ts(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stamp each entry with its creating event's `created_ts` (from the event
        log) so the board can show a relative age. Entry dicts are fresh per read,
        so this in-place stamp is safe."""
        ts_map = self.store.ts_by_seq([e.get("created_seq") for e in entries])
        for e in entries:
            e["created_ts"] = ts_map.get(e.get("created_seq"))
        return entries

    def status_board(self, cap_entries: int = CAP_ENTRIES,
                     cap_tokens: Optional[int] = None,
                     measure: Optional[Any] = None,
                     pick: Optional[Any] = None,
                     now: Optional[datetime] = None,
                     soften_caps: bool = False,
                     include_overflow_ids: bool = True,
                     pin_types: tuple[str, ...] = ()) -> dict[str, Any]:
        """The 'SSM selects, store renders' board — always VERBATIM store text.

        When the active set overflows the envelope AND a `pick` (SSM salience
        picker) is given, the SSM chooses the salient subset (map-reduce, then
        match-back to real entries). Otherwise the deterministic recency cap runs.
        `overflow` is surfaced, never silently dropped — those entries stay in the
        store for exact recall.

        `now`/`soften_caps` are presentation-only (the MCP tool and the hook pass
        them): add a relative-age marker and down-case shouty emphasis. Default OFF
        keeps the library call byte-verbatim. `include_overflow_ids=False` (the MCP
        edge) drops the raw id blob — the count stays, and ids are reachable via
        team_state + get_entry. `pin_types` (the MCP edge and hook pass
        ("decision",), kevin's ruling) exempts those entry types from the cap
        entirely: every active one always renders, the cap applies to the rest."""
        active = self.store.active_entries()
        pinned, rest = _split_pinned(active, pin_types)
        if pick is not None and len(rest) > cap_entries:
            selected = select_salient(rest, pick, cap_entries=cap_entries)
            sel_ids = {e["entry_id"] for e in selected}
            dropped = [e for e in rest if e["entry_id"] not in sel_ids]
            kept = _merge_recency(pinned, selected)
            res: dict[str, Any] = {
                "board": render_board(self._attach_ts(kept), now=now,
                                      soften_caps=soften_caps),
                "shown": len(kept),
                "overflow": len(dropped),
                "selector": "ssm",
            }
            if pin_types:
                res["pinned"] = len(pinned)
            if include_overflow_ids:
                res["overflow_ids"] = [e["entry_id"] for e in dropped]
            return res
        ws = working_set(rest, cap_entries=cap_entries,
                         cap_tokens=cap_tokens, measure=measure)
        kept = _merge_recency(pinned, ws["kept"])
        res = {
            "board": render_board(self._attach_ts(kept), now=now,
                                  soften_caps=soften_caps),
            "shown": len(kept),
            "overflow": len(ws["dropped"]),
            "selector": "recency",
        }
        if pin_types:
            res["pinned"] = len(pinned)
        if include_overflow_ids:
            res["overflow_ids"] = [e["entry_id"] for e in ws["dropped"]]
        return res

    def overview(self, compactor: Optional[Any] = None,
                 cap_entries: int = CAP_ENTRIES,
                 cap_tokens: Optional[int] = None,
                 measure: Optional[Any] = None,
                 now: Optional[datetime] = None,
                 soften_caps: bool = False,
                 include_overflow_ids: bool = True,
                 pin_types: tuple[str, ...] = ()) -> dict[str, Any]:
        """Lossy linked 'where are we' gist over the CAPPED working set.

        `compactor` is a HybridCompactor (or any object with `.compact(entries)`);
        without one this falls back to the deterministic verbatim board, so the
        read always works GPU-free. Capping first keeps the compactor's input
        bounded (so a hybrid's KV stays bounded — memory hybrid-compaction-gist),
        and the overflow is surfaced, never hidden: those entries stay in the store
        for exact recall.

        On the board fallback a `note` states plainly that this IS the authoritative
        verbatim board (identical to status_board), not a lossy summary awaiting a
        model — the gist layer is archived (research/SSM_POSTMORTEM.md)."""
        active = self.store.active_entries()
        pinned, rest = _split_pinned(active, pin_types)
        ws = working_set(rest, cap_entries=cap_entries, cap_tokens=cap_tokens,
                         measure=measure)
        kept = _merge_recency(pinned, ws["kept"])
        res: dict[str, Any] = {"shown": len(kept), "overflow": len(ws["dropped"])}
        if pin_types:
            res["pinned"] = len(pinned)
        if include_overflow_ids:
            res["overflow_ids"] = [e["entry_id"] for e in ws["dropped"]]
        if compactor is None:
            res["overview"] = render_board(self._attach_ts(kept), now=now,
                                           soften_caps=soften_caps)
            res["selector"] = "board"
            res["note"] = ("gist layer archived (research/SSM_POSTMORTEM.md) — this "
                           "IS the authoritative verbatim board, identical to "
                           "status_board, not a lossy summary awaiting a model.")
        else:
            res["overview"] = compactor.compact(kept)
            res["selector"] = "gist"
        return res

    def recent_summary(self, since_seq: int = 0, limit: int = 50,
                       body_chars: int = 220) -> dict[str, Any]:
        """Bounded `recent`: the newest `limit` events since `since_seq`, with
        bodies trimmed — the raw stream is 70KB+ and overflows the read. Omitted
        (older) events are surfaced with a `note`, never silently dropped (#11);
        page them by re-querying with a larger limit. `latest_seq` is the watermark
        to advance to."""
        events = self.store.events_since(since_seq)
        total = len(events)
        shown = events[-limit:] if (limit and total > limit) else events
        trimmed = []
        for e in shown:
            p = dict(e["payload"])
            if p.get("body"):
                p["body"] = _preview(p["body"], body_chars)
            trimmed.append({**e, "payload": p})
        omitted = total - len(shown)
        res: dict[str, Any] = {
            "events": trimmed, "returned": len(shown), "total": total,
            "omitted": omitted, "since_seq": since_seq,
            "latest_seq": shown[-1]["seq"] if shown else since_seq,
        }
        if omitted:
            # name the registered TOOL (recent), not this service method — the
            # note is rendered to MCP callers who can only call tool names
            res["note"] = (f"{omitted} older new events not shown; "
                           f"call recent(since_seq={since_seq}, "
                           f"limit={total}) for all.")
        return res

    def project_digests(self, engine: Any) -> dict[str, Any]:
        """Per-project streaming SSM digests. `engine` is a ShardedSSMEngine; this
        syncs it against the store, then reads each non-empty shard — each stream
        kept in its own faithful envelope (report finding 5)."""
        engine.sync()
        projects = engine.projects()
        return {"digests": {p: engine.digest(p) for p in projects},
                "projects": projects}

    def recent(self, since_seq: int = 0) -> list[dict[str, Any]]:
        return self.store.events_since(since_seq)
