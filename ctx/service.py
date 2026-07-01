"""Service layer — the model-free coordination brain over the store.

Everything here is deterministic and GPU-free: file-ref collision detection and
a structured team digest. It directly attacks the duplication problem. The SSM
(M3) layers semantic overlap + natural-language digests on top, but this exact,
cheap path always works and needs no model.
"""

from __future__ import annotations

from typing import Any, Optional

from ctx.index import (CAP_ENTRIES, CAP_TOKENS, render_board, select_salient,
                       working_set)
from ctx.store import Store


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

    def team_state(self) -> dict[str, Any]:
        """Structured 'where are we' digest: active entries by author -> type."""
        by_author: dict[str, dict[str, list]] = {}
        for e in self.store.active_entries():
            by_author.setdefault(e["author"], {}).setdefault(e["type"], []).append(e)
        return {"by_author": by_author, "overlaps": self.overlaps()}

    def status_board(self, cap_entries: int = CAP_ENTRIES,
                     cap_tokens: Optional[int] = None,
                     measure: Optional[Any] = None,
                     pick: Optional[Any] = None) -> dict[str, Any]:
        """The 'SSM selects, store renders' board — always VERBATIM store text.

        When the active set overflows the envelope AND a `pick` (SSM salience
        picker) is given, the SSM chooses the salient subset (map-reduce, then
        match-back to real entries). Otherwise the deterministic recency cap runs.
        `overflow` is surfaced, never silently dropped — those entries stay in the
        store for exact recall."""
        active = self.store.active_entries()
        if pick is not None and len(active) > cap_entries:
            selected = select_salient(active, pick, cap_entries=cap_entries)
            sel_ids = {e["entry_id"] for e in selected}
            dropped = [e for e in active if e["entry_id"] not in sel_ids]
            return {
                "board": render_board(selected), "shown": len(selected),
                "overflow": len(dropped),
                "overflow_ids": [e["entry_id"] for e in dropped],
                "selector": "ssm",
            }
        ws = working_set(active, cap_entries=cap_entries,
                         cap_tokens=cap_tokens, measure=measure)
        return {
            "board": render_board(ws["kept"]),
            "shown": len(ws["kept"]),
            "overflow": len(ws["dropped"]),
            "overflow_ids": [e["entry_id"] for e in ws["dropped"]],
            "selector": "recency",
        }

    def overview(self, compactor: Optional[Any] = None,
                 cap_entries: int = CAP_ENTRIES,
                 cap_tokens: Optional[int] = None,
                 measure: Optional[Any] = None) -> dict[str, Any]:
        """Lossy linked 'where are we' gist over the CAPPED working set.

        `compactor` is a HybridCompactor (or any object with `.compact(entries)`);
        without one this falls back to the deterministic verbatim board, so the
        read always works GPU-free. Capping first keeps the compactor's input
        bounded (so a hybrid's KV stays bounded — memory hybrid-compaction-gist),
        and the overflow is surfaced, never hidden: those entries stay in the store
        for exact recall."""
        active = self.store.active_entries()
        ws = working_set(active, cap_entries=cap_entries, cap_tokens=cap_tokens,
                         measure=measure)
        kept = ws["kept"]
        if compactor is None:
            text, selector = render_board(kept), "board"
        else:
            text, selector = compactor.compact(kept), "gist"
        return {"overview": text, "shown": len(kept),
                "overflow": len(ws["dropped"]),
                "overflow_ids": [e["entry_id"] for e in ws["dropped"]],
                "selector": selector}

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
