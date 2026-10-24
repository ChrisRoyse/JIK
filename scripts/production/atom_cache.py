"""Immutable native row cache; one SQLite transaction per accepted checkpoint.

The stream caller holds the cache's OS writer lock for the job. Database pages
have a fixed cap, with equal rollback-journal space reserved by the input plan.
Only requested entries are decoded; lookup cost is indexed, never a corpus scan.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
from collections import OrderedDict
from functools import wraps
import zipfile

import blake3
from senses.contract import canonical, digest, require, SenseError
from production.native_rows import one, pack, unpack, value_hash, mapped_rows
from production.input_plan import sha

SCHEMA = "atom-cache-v1"


def storage_boundary(fn):
    @wraps(fn)
    def checked(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError, TypeError, EOFError, OverflowError, zipfile.BadZipFile, sqlite3.DatabaseError) as exc:
            raise SenseError("MNLM_CACHE_CORRUPT", "cache structure is unreadable; restore exact accepted bytes") from exc
    return checked


def panel_key(plan):
    roster = {n: {k: i[k] for k in ("spec", "weights_sha256", "revision") if k in i}
              for n, i in plan["instruments"].items()}
    return digest({"purpose": plan["purpose"], "admission": plan.get("panel_blake3"),
                   "training_scope": plan["training_scope_sha256"], "roster": roster})


def selector(plan, name):
    return digest({"panel": panel_key(plan), "instrument": name,
                   "code": plan["code_sha256"], "environment": plan["environment"]})


def input_key(panel, instrument_selector, prepared):
    binding = {k: v for k, v in prepared["binding"].items() if k != "generation"}
    source = digest({"input": prepared["input"], "context": prepared["context"], "binding": binding})
    return {"panel": panel, "selector": instrument_selector,
            "atom": binding["atom"], "input_sha256": source}


def identity(contract):
    return {k: v for k, v in contract.items() if k not in ("ucr_generation", "policies_sha256", "blobs_manifest_sha256")}


def instrument(slot, item):
    native = identity(slot["contract"])
    require(blake3.blake3(canonical(native)).hexdigest()[:32] == slot["lens_id"],
            "MNLM_CACHE_CONTRACT", "instrument identity differs from its frozen contract")
    kind = item["spec"]["kind"] if item["spec"]["module"] == "learned_ucr" else "multi"
    return {"identity": native, "lens_id": slot["lens_id"], "dim": slot["seal"]["dims"], "kind": kind}


class AtomCache:
    @storage_boundary
    def __init__(self, root, config, metadata_limit, native_limit):
        self.root, self.limit, self.native_limit = Path(root), metadata_limit, native_limit
        self.dependencies, self.dependency_bytes = OrderedDict(), 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "index.sqlite", timeout=0)
        try:
            require(config.get("schema") == SCHEMA and type(config.get("max_db_bytes")) is int,
                    "MNLM_CACHE_CONFIG", "frozen cache page budget required")
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            pagesize = self.db.execute("PRAGMA page_size").fetchone()[0]
            pages = config["max_db_bytes"] // pagesize
            require(pages >= 16 and self.db.execute("PRAGMA page_count").fetchone()[0] <= pages,
                    "MNLM_CACHE_BUDGET", "database exceeds frozen cache page budget")
            require(self.db.execute(f"PRAGMA max_page_count={pages}").fetchone()[0] == pages,
                    "MNLM_CACHE_BUDGET", "SQLite could not enforce the cache page limit")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY CHECK(id=1),hash TEXT,body BLOB);
                CREATE TABLE IF NOT EXISTS instruments(key TEXT PRIMARY KEY,hash TEXT NOT NULL,body BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS proofs(hash TEXT PRIMARY KEY,body BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY,hash TEXT NOT NULL,body BLOB NOT NULL,native BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,hash TEXT UNIQUE NOT NULL,body BLOB NOT NULL);
            """)
            for table in ("config", "instruments", "proofs", "entries", "events"):
                for op in ("UPDATE", "DELETE"):
                    self.db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{op} BEFORE {op} ON {table} "
                                    "BEGIN SELECT RAISE(ABORT,'MNLM_CACHE_IMMUTABLE'); END")
            self.db.execute("INSERT OR IGNORE INTO config VALUES(1,?,?)", (digest(config), canonical(config)))
            self.db.commit()
            h, raw = self.db.execute("SELECT hash,body FROM config WHERE id=1").fetchone()
            require(h == digest(config) and json.loads(raw) == config, "MNLM_CACHE_CONFIG", "cache budget/schema changed")
            self.state()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.db.close()

    @storage_boundary
    def state(self):
        row = self.db.execute("SELECT seq,hash,body FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        if row is None:
            return {"seq": 0, "head": None, "entries": 0}
        seq, h, raw = row
        event = json.loads(raw)
        require(digest(event) == h and event["seq"] == seq, "MNLM_CACHE_CORRUPT", "cache head changed")
        return {"seq": seq, "head": h, "entries": event["after_count"]}

    def _record(self, table, key, column="key"):
        cache_key = (table, key, column)
        if cache_key in self.dependencies:
            self.dependencies.move_to_end(cache_key)
            return self.dependencies[cache_key][0]
        row = self.db.execute(f"SELECT hash,length(body) FROM {table} WHERE {column}=?", (key,)).fetchone()
        require(row is not None and row[1] <= self.limit, "MNLM_CACHE_CORRUPT", "missing or oversized cache dependency")
        raw, = self.db.execute(f"SELECT body FROM {table} WHERE {column}=?", (key,)).fetchone()
        body = json.loads(raw)
        require(digest(body) == row[0], "MNLM_CACHE_CORRUPT", "cache dependency bytes changed")
        while self.dependencies and self.dependency_bytes + len(raw) > 4 * self.limit:
            _, (_, size) = self.dependencies.popitem(last=False)
            self.dependency_bytes -= size
        self.dependencies[cache_key] = (body, len(raw))
        self.dependency_bytes += len(raw)
        return body

    @storage_boundary
    def lookup(self, key):
        address = digest(key)
        row = self.db.execute("SELECT hash,length(body),length(native) FROM entries WHERE key=?", (address,)).fetchone()
        if row is None:
            return None
        require(row[1] <= self.limit and row[2] <= self.native_limit, "MNLM_CACHE_CORRUPT", "oversized cache entry")
        raw, native = self.db.execute("SELECT body,native FROM entries WHERE key=?", (address,)).fetchone()
        body = json.loads(raw)
        require(digest(body) == row[0] and body["key"] == key and hashlib.sha256(native).hexdigest() == body["native_sha256"],
                "MNLM_CACHE_CORRUPT", "cached input or native bytes changed; restore exact cache")
        info = self._record("instruments", key["selector"])
        proof = self._record("proofs", body["proof_sha256"], "hash")
        require(blake3.blake3(canonical(info["identity"])).hexdigest()[:32] == info["lens_id"]
                and info["lens_id"] == proof["slot"]["lens_id"] and identity(proof["slot"]["contract"]) == info["identity"]
                and proof["slot"].get("replay_mode") in ("forward", "mixed_forward_and_cache")
                and proof["slot"]["replay"]["byte_identical"] is True and proof["slot"]["replay"]["max_abs"] == 0,
                "MNLM_CACHE_CORRUPT", "cached row has no matching accepted replay proof")
        arrays = unpack(native, info["identity"], body["detail"], info["dim"])
        require(value_hash(arrays) == body["measurement_sha256"], "MNLM_CACHE_CORRUPT", "native layout/value hash differs")
        return arrays, body["detail"], info, {"entry_sha256": row[0], "proof_sha256": body["proof_sha256"], "key": address}

    @storage_boundary
    def accept(self, plan, name, rows, keys, path, checkpoint, *, after_insert=None):
        slot = checkpoint["slot"]
        info = instrument(slot, plan["instruments"][name])
        require(sha(path) == slot["sha256"] and sha(path.with_suffix(".views.json")) == slot["views_sha256"],
                "MNLM_CACHE_CORRUPT", "accepted checkpoint content changed before cache publication")
        views = json.loads(path.with_suffix(".views.json").read_bytes())
        require(len(rows) == len(keys) == len(views["details"]), "MNLM_CACHE_INPUT", "cache row count mismatch")
        proof_hash, sk = digest(checkpoint), selector(plan, name)
        require(all(k["selector"] == sk and k["atom"] == r["atom"] for r, k in zip(rows, keys)),
                "MNLM_CACHE_INPUT", "cache key belongs to another atom/instrument")
        require(len(canonical(checkpoint)) <= self.limit, "MNLM_CACHE_BUDGET", "proof exceeds metadata reservation")
        missing = []
        with mapped_rows(path) as arrays:
            for i, key in enumerate(keys):
                found = self.lookup(key)
                detail = views["details"][i]
                if found is None:
                    require("cache_origin" not in detail and slot.get("replay_mode") in ("forward", "mixed_forward_and_cache"),
                            "MNLM_CACHE_PROVENANCE", "readback cannot replace a missing original forward proof; restore original cache")
                    missing.append((i, key))
                else:
                    values, old_detail, old_info, _ = found
                    require(info == old_info and value_hash(one(arrays, i)) == value_hash(values)
                            and {k: v for k, v in detail.items() if k != "cache_origin"} == old_detail,
                            "MNLM_CACHE_CONFLICT", "checkpoint differs from the accepted measurement for this exact input")
        if not missing:
            return 0
        before = self.state()
        changes = []
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("INSERT OR IGNORE INTO instruments VALUES(?,?,?)", (sk, digest(info), canonical(info)))
            require(self._record("instruments", sk) == info, "MNLM_CACHE_CONFLICT", "instrument changed under one cache selector")
            self.db.execute("INSERT OR IGNORE INTO proofs VALUES(?,?)", (proof_hash, canonical(checkpoint)))
            # Parse/map the shard once; copy only one native row at a time.
            with mapped_rows(path) as arrays:
                for i, key in missing:
                    native_arrays = one(arrays, i)
                    detail = {k: v for k, v in views["details"][i].items() if k != "cache_origin"}
                    native = pack(native_arrays)
                    require(len(native) <= self.native_limit, "MNLM_CACHE_BUDGET", "native row exceeds fixed reservation")
                    unpack(native, info["identity"], detail, info["dim"])
                    body = {"key": key, "proof_sha256": proof_hash, "row": i, "detail": detail,
                            "native_sha256": hashlib.sha256(native).hexdigest(), "measurement_sha256": value_hash(native_arrays)}
                    encoded = canonical(body)
                    require(len(encoded) <= self.limit, "MNLM_CACHE_BUDGET", "cache row provenance exceeds reservation")
                    h = digest(body)
                    self.db.execute("INSERT INTO entries VALUES(?,?,?,?)", (digest(key), h, encoded, native))
                    changes.append([digest(key), h])
                    if after_insert is not None:
                        after_insert(len(changes))
            event = {"seq": before["seq"] + 1, "parent": before["head"], "before_count": before["entries"],
                     "after_count": before["entries"] + len(changes), "proof_sha256": proof_hash, "entries": changes}
            require(len(canonical(event)) <= self.limit, "MNLM_CACHE_BUDGET", "cache event exceeds reservation")
            self.db.execute("INSERT INTO events VALUES(?,?,?)", (event["seq"], digest(event), canonical(event)))
            self.db.commit()
        except BaseException as exc:
            self.db.rollback()
            self.dependencies.clear()
            self.dependency_bytes = 0
            if isinstance(exc, sqlite3.DatabaseError):
                code = "MNLM_CACHE_BUDGET" if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL else "MNLM_CACHE_STORAGE"
                raise SenseError(code, "cache transaction refused; previous accepted rows remain unchanged") from exc
            raise
        return len(changes)
