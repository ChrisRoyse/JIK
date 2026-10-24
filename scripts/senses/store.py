"""Transactional, content-bound UCR index. Canonical bytes remain authoritative.

One transaction per ingestion; immutable records and slots; generation chain binds
the input scope and fixed panel. Old generations remain readable. Context joins
are indexed and bounded by an explicit limit, with overflow a typed refusal.
"""
from __future__ import annotations

import json
import sqlite3
import zlib
from pathlib import Path

from .contract import SenseError, canonical, digest, is_sha256, require, time_value, validate_record


def records_contract():
    """Stable observation format; deliberately independent of lens code/runtime."""
    body = {"schema": "ucr-records-only-v1", "record_schema": "ucr-v1",
            "instruments": [], "measured": False, "served": False}
    return {**body, "panel_digest": digest(body)}

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
 atom TEXT PRIMARY KEY, locator TEXT UNIQUE NOT NULL, source TEXT NOT NULL,
 unit TEXT NOT NULL, observed REAL, body BLOB NOT NULL, seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS referents (key TEXT NOT NULL, atom TEXT NOT NULL,
 PRIMARY KEY(key,atom), FOREIGN KEY(atom) REFERENCES records(atom));
CREATE TABLE IF NOT EXISTS slots (
 atom TEXT NOT NULL, lens TEXT NOT NULL, meta BLOB NOT NULL, values_blob BLOB NOT NULL,
 digest TEXT NOT NULL, PRIMARY KEY(atom,lens), FOREIGN KEY(atom) REFERENCES records(atom));
CREATE TABLE IF NOT EXISTS generations (seq INTEGER PRIMARY KEY, hash TEXT UNIQUE NOT NULL, body BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS observation_order ON records(observed,source);
CREATE INDEX IF NOT EXISTS source_locator ON records(source,seq);
CREATE INDEX IF NOT EXISTS referent_atom ON referents(atom,key);
CREATE TABLE IF NOT EXISTS panel (id INTEGER PRIMARY KEY CHECK(id=1), body BLOB NOT NULL);
"""


def locator_key(r):
    return canonical({k: r["locator"][k] for k in ("source", "shard", "row", "span")}).decode()


def observed(r):
    clocks = [f["value"] for f in r["channels"]["fields"]["items"] if f["type"] == "clock" and f["clock_role"] == "observed"]
    require(len(set(clocks)) <= 1, "MNLM_UCR_CLOCK", "ambiguous observation clocks")
    return time_value(clocks[0]) if clocks else None


def validate_history_clock(clock):
    """An explicit source-backed order assertion, never an inferred timestamp."""
    require(isinstance(clock, dict) and set(clock) == {
        "schema", "field", "unit", "scope", "source_manifest_sha256", "evidence"},
        "MNLM_UCR_ORDER_POLICY", "declare field, unit, source manifest, clock domain and evidence")
    require(clock["schema"] == "ucr-ordinal-clock-v1" and clock["scope"] == "source_shard"
            and all(isinstance(clock[k], str) and clock[k] for k in ("field", "unit", "evidence"))
            and is_sha256(clock["source_manifest_sha256"]),
            "MNLM_UCR_ORDER_POLICY", "ordinal clock must bind one retained source contract")
    return clock


def ordinal(r, clock):
    require(r["locator"]["manifest_digest"] == clock["source_manifest_sha256"],
            "MNLM_UCR_ORDER_BINDING", "observation does not bind the declared clock authority")
    fields = [f for f in r["channels"]["fields"]["items"] if f["name"] == clock["field"]]
    require(len(fields) == 1 and fields[0]["type"] == "integer"
            and fields[0]["unit"] == clock["unit"] and type(fields[0]["value"]) is int
            and fields[0]["value"] >= 0 and fields[0]["clock_role"] is None,
            "MNLM_UCR_ORDER_VALUE", "ordinal clock requires one retained nonnegative integer with exact units")
    return fields[0]["value"]


class Store:
    def __init__(self, path, create=False, records_only=None):
        self.path = Path(path)
        require(create or self.path.exists(), "MNLM_UCR_MISSING", str(path))
        require(records_only is None or type(records_only) is bool,
                "MNLM_UCR_STORE_MODE", "records_only must be boolean or omitted")
        # Inspect existing contracts read-only before any journal/schema mutation.
        existing = self.path.exists()
        contract = None
        if existing:
            db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                row = db.execute("SELECT body FROM panel WHERE id=1").fetchone()
                require(row is not None, "MNLM_UCR_PANEL_DRIFT", "missing store contract")
                contract = json.loads(row[0])
                require(isinstance(contract, dict), "MNLM_UCR_PANEL_DRIFT", "invalid store contract")
            except (sqlite3.Error, ValueError, TypeError) as exc:
                raise SenseError("MNLM_UCR_STORE_CONTRACT", "retained database contract is unreadable",
                                 "Restore the retained store from its sealed generation; do not recreate it in place.") from exc
            finally:
                db.close()
            retained_only = contract.get("schema") == "ucr-records-only-v1"
            require(records_only is None or records_only == retained_only,
                    "MNLM_UCR_STORE_MODE", "existing store mode differs; use a new store path")
        else:
            retained_only = records_only is True
        if retained_only:
            expected, self.instruments = records_contract(), []
        else:
            from .instruments import contracts, roster
            expected, self.instruments = roster(), contracts()
        require(not existing or contract == expected, "MNLM_UCR_PANEL_DRIFT",
                "store contract changed; start a new index version")
        self.records_only = retained_only
        self.panel_digest = expected["panel_digest"]
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path) if create else self.path.resolve().as_uri() + "?mode=ro", uri=not create)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=0")  # a busy writer is a refusal, not an unbounded wait
        if create:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript(SCHEMA)
            for table in ("records", "referents", "slots", "generations", "panel"):
                for op in ("UPDATE", "DELETE"):
                    self.db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{op} BEFORE {op} ON {table} BEGIN SELECT RAISE(ABORT,'MNLM_UCR_IMMUTABLE'); END")
            self.db.execute("INSERT OR IGNORE INTO panel VALUES(1,?)", (canonical(expected),))
            self.db.commit()
        row = self.db.execute("SELECT body FROM panel WHERE id=1").fetchone()
        try:
            require(row is not None and json.loads(row[0]) == expected, "MNLM_UCR_PANEL_DRIFT", "store contract changed during open")
            self.verify_chain()
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    @property
    def head(self):
        row = self.db.execute("SELECT hash FROM generations ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def verify_chain(self):
        previous, seq = None, 0
        for n, h, body in self.db.execute("SELECT seq,hash,body FROM generations ORDER BY seq"):
            obj = json.loads(body)
            require(n == seq + 1 and obj["parent"] == previous and digest(obj) == h
                    and obj["panel_digest"] == self.panel_digest,
                    "MNLM_UCR_CHAIN", "generation chain changed")
            previous, seq = h, n
        return seq

    def ingest(self, records, scope, expected_head):
        from .scope import check_source
        from training_scope import POLICY_SHA256
        if self.instruments:
            from .instruments import measure, slot_digest
        require(isinstance(scope, dict) and scope.get("input_sha256") and scope.get("count") is not None,
                "MNLM_UCR_SCOPE", "declare input digest and exact record count")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            require(self.head == expected_head, "MNLM_UCR_STALE_HEAD", "another writer advanced this index; reread before retry")
            seq = self.db.execute("SELECT count(*) FROM generations").fetchone()[0] + 1
            entries, n, duplicates = [], 0, 0
            # All instruments are invariant across this loop and every later source.
            instruments = self.instruments
            for r in records:
                validate_record(r)
                check_source(r["locator"]["source"])
                n += 1
                old = self.db.execute("SELECT atom FROM records WHERE locator=?", (locator_key(r),)).fetchone()
                if old:
                    require(old[0] == r["atom_id"], "MNLM_UCR_IDENTITY_CONFLICT", "same locator, changed observation; publish an explicitly versioned source")
                    self.get(old[0])  # idempotence cannot conceal corrupted retained bytes
                    duplicates += 1
                    continue
                self.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)", (r["atom_id"], locator_key(r), r["locator"]["source"], r["unit"], observed(r), canonical(r), seq))
                for ref in r["referents"]:
                    self.db.execute("INSERT OR IGNORE INTO referents VALUES(?,?)", (ref["key"], r["atom_id"]))
                slots = {}
                for c in instruments:
                    s = measure(r, c["channel"])
                    h = slot_digest(s)
                    meta = canonical({k: v for k, v in s.items() if k != "tokens"})
                    self.db.execute("INSERT INTO slots VALUES(?,?,?,?,?)", (r["atom_id"], c["lens_id"], meta, zlib.compress(s["tokens"].astype("<f4").tobytes()), h))
                    slots[c["lens_id"]] = h
                entries.append([r["atom_id"], slots])
            require(n == scope["count"], "MNLM_UCR_SCOPE", f"declared {scope['count']} records, received {n}")
            body = {"schema": "ucr-generation-v1", "parent": expected_head, "panel_digest": self.panel_digest,
                    "training_scope_sha256": POLICY_SHA256,
                    "scope": scope, "entries_digest": digest(sorted(entries)), "added": len(entries), "duplicates": duplicates}
            h = digest(body)
            self.db.execute("INSERT INTO generations VALUES(?,?,?)", (seq, h, canonical(body)))
            self.db.commit()
            return {"generation": h, "seq": seq, **body}
        except BaseException:
            self.db.rollback()
            raise

    def get(self, atom_id, generation=None):
        row = self.db.execute("SELECT body,seq,locator,source,unit,observed FROM records WHERE atom=?", (atom_id,)).fetchone()
        require(row is not None, "MNLM_UCR_NOT_FOUND", atom_id)
        if generation is not None:
            g = self.db.execute("SELECT seq FROM generations WHERE hash=?", (generation,)).fetchone()
            require(g is not None and row[1] <= g[0], "MNLM_UCR_NOT_IN_GENERATION", atom_id)
        r = validate_record(json.loads(row[0]))
        require(r["atom_id"] == atom_id, "MNLM_UCR_ID", "index and record identity differ")
        require((locator_key(r), r["locator"]["source"], r["unit"], observed(r)) == row[2:],
                "MNLM_UCR_INDEX_DIGEST", "lookup columns disagree with retained observation")
        keys = [v[0] for v in self.db.execute("SELECT key FROM referents WHERE atom=? ORDER BY key", (atom_id,))]
        require(keys == sorted({x["key"] for x in r["referents"]}), "MNLM_UCR_INDEX_DIGEST", "referent index disagrees with retained observation")
        return r

    def find(self, source, shard, row, span=None, generation=None):
        # JSON extraction is not used to scan a source: exact locator prefix selects
        # at most the atom views of one canonical record through the locator index.
        prefix = canonical({"row": row, "shard": shard, "source": source}).decode()[:-1] + ',"span":'
        found = self.db.execute("SELECT atom FROM records WHERE locator>=? AND locator<? ORDER BY locator", (prefix, prefix + "\uffff")).fetchall()
        if generation is not None:
            g = self.db.execute("SELECT seq FROM generations WHERE hash=?", (generation,)).fetchone()
            require(g is not None, "MNLM_UCR_NOT_IN_GENERATION", str(generation))
            found = [a for a in found if self.db.execute("SELECT seq FROM records WHERE atom=?", a).fetchone()[0] <= g[0]]
        records = [self.get(a[0], generation) for a in found]
        if span is not None:
            records = [r for r in records if r["locator"]["span"] == list(span)]
        require(len(records) == 1, "MNLM_UCR_LOCATOR_AMBIGUOUS", f"{source}/{shard}/{row}: {len(records)} matches; supply exact atom span")
        return records[0]

    def read_slot(self, atom, lens):
        require(not self.records_only, "MNLM_UCR_UNMEASURED",
                "observation store has no lens slots; run the separate admitted measurement stage")
        import numpy as np
        from .instruments import slot_digest
        self.get(atom)
        row = self.db.execute("SELECT meta,values_blob,digest FROM slots WHERE atom=? AND lens=?", (atom, lens)).fetchone()
        require(row is not None, "MNLM_UCR_SLOT_MISSING", f"{atom}/{lens}")
        s = json.loads(row[0])
        raw = zlib.decompress(row[1])
        require(len(raw) == s["shape"][0] * s["shape"][1] * 4, "MNLM_UCR_SLOT_SHAPE", atom)
        s["tokens"] = np.frombuffer(raw, dtype="<f4").reshape(s["shape"])
        require(slot_digest(s) == row[2], "MNLM_UCR_SLOT_DIGEST", atom)
        return s

    def context(self, r, channel, generation, excluded_units=(), limit=256, history_clock=None):
        require(channel in ("co_reference", "history"), "MNLM_UCR_CONTEXT", channel)
        require(type(limit) is int and limit > 0, "MNLM_UCR_CONTEXT_BUDGET", "positive candidate bound required")
        g = self.db.execute("SELECT seq FROM generations WHERE hash=?", (generation,)).fetchone()
        require(g is not None, "MNLM_UCR_NOT_IN_GENERATION", str(generation))
        require(canonical(self.get(r["atom_id"], generation)) == canonical(r),
                "MNLM_UCR_VIEW_BINDING", "context target differs from retained observation")
        if history_clock is not None:
            require(channel == "history", "MNLM_UCR_ORDER_POLICY", "ordinal clocks apply to history only")
            validate_history_clock(history_clock)
            at = ordinal(r, history_clock)
        keys = sorted({x["key"] for x in r["referents"]})
        if not keys:
            return []
        marks = ",".join("?" for _ in keys)
        params = [*keys, g[0], r["unit"]]
        clause = "r.source != ?" if channel == "co_reference" else "r.observed < ?"
        if history_clock is not None:
            # Referent and source lookup first; no whole-source JSON/body scan.
            clause = "r.source=? AND json_extract(r.locator,'$.shard')=?"
            params.extend((r["locator"]["source"], r["locator"]["shard"]))
        elif channel == "history":
            at = observed(r)
            if at is None:
                return []
            params.append(at)
        else:
            params.append(r["locator"]["source"])
        if channel == "co_reference" and observed(r) is not None:
            clause += " AND r.observed <= ?"
            params.append(observed(r))
        # CROSS JOIN fixes the index loop order. A source predicate otherwise
        # made SQLite scan the whole source per query (PC-07, #24).
        rows = self.db.execute(f"SELECT DISTINCT r.atom,r.unit FROM referents f CROSS JOIN records r ON r.atom=f.atom WHERE f.key IN ({marks}) AND r.seq<=? AND r.unit!=? AND {clause} ORDER BY r.atom LIMIT ?", (*params, limit + 1)).fetchall()
        require(len(rows) <= limit, "MNLM_UCR_CONTEXT_BUDGET", "context exceeds declared limit; narrow the referent scope")
        excluded = set(excluded_units)
        retained = [self.get(a, generation) for a, u in rows if u not in excluded]
        if history_clock is None:
            return retained
        ordered, seen = [], {at}
        for candidate in retained:
            position = ordinal(candidate, history_clock)
            require(position not in seen, "MNLM_UCR_ORDER_AMBIGUOUS",
                    "clock position is not unique; provide an unambiguous observation order")
            seen.add(position)
            if position < at:
                ordered.append((position, candidate))
        return [candidate for _, candidate in sorted(ordered, key=lambda pair: pair[0])]

    def audit(self):
        """Full physical replay is an explicit acceptance operation, not per-lookup cost."""
        if self.instruments:
            from .instruments import measure, slot_digest
        counts = {"records": 0, "slots": 0, "sources": {}, "channels": {c["channel"]: 0 for c in self.instruments}}
        generation_entries = {}
        for a, seq in self.db.execute("SELECT atom,seq FROM records ORDER BY atom"):
            r, slots = self.get(a), {}
            counts["records"] += 1
            source = r["locator"]["source"]
            counts["sources"][source] = counts["sources"].get(source, 0) + 1
            for c in self.instruments:
                s = self.read_slot(a, c["lens_id"])
                require(slot_digest(s) == slot_digest(measure(r, c["channel"])), "MNLM_UCR_REPLAY", a)
                slots[c["lens_id"]] = slot_digest(s)
                counts["slots"] += 1
                counts["channels"][c["channel"]] += s["kind"] != "Absent"
            generation_entries.setdefault(seq, []).append([a, slots])
        for seq, body in self.db.execute("SELECT seq,body FROM generations ORDER BY seq"):
            b = json.loads(body)
            entries = generation_entries.get(seq, [])
            require(digest(sorted(entries)) == b["entries_digest"] and len(entries) == b["added"], "MNLM_UCR_GENERATION_DIGEST", str(seq))
        require(self.db.execute("SELECT count(*) FROM slots").fetchone()[0] == counts["slots"], "MNLM_UCR_SLOT_COUNT", "extra/missing slot rows")
        require(sum(json.loads(b)["added"] for (b,) in self.db.execute("SELECT body FROM generations")) == counts["records"],
                "MNLM_UCR_GENERATION_DIGEST", "records outside accepted generations")
        return {"head": self.head, "generations": self.verify_chain(), **counts,
                "records_only": self.records_only, "measured": not self.records_only, "served": False}
