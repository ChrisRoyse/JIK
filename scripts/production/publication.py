"""Immutable native measurement shards, SQLite acceptance barrier (#16, PC-04/07/35).

One transaction publishes a shard and its provenance-chain event after native
blobs pass cold validation. Content files may survive a crash without a commit;
readers only enumerate committed shards. A fixed plan names every blob and caps
their TOTAL possible bytes, including orphans. Files are copied in 1 MiB chunks.
Engineering replay is a separate purpose and never a production acceptance.
This storage primitive is not the Stage 4 inference runner or an admission test.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

import blake3

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import SenseError, canonical, digest, require
from senses.scope import check_source
from slotio import slot_digest
from slot_layout import check_seal
from text_windows import validate_slot
from production.gate import require_admitted
from training_scope import POLICY_SHA256

HEX = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = """
CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), hash TEXT NOT NULL, body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS shards (key TEXT PRIMARY KEY, hash TEXT NOT NULL, body BLOB NOT NULL, seq INTEGER UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, hash TEXT UNIQUE NOT NULL, body BLOB NOT NULL);
"""


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sync_dir(path):
    if os.name != "nt":
        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def validate_plan(plan):
    require(isinstance(plan, dict), "MNLM_MEASURE_PLAN", "plan must be an object")
    require(plan.get("schema") == "native-publication-plan-v1", "MNLM_MEASURE_PLAN", "unknown schema")
    require(plan.get("purpose") in ("engineering_boundary", "production"), "MNLM_MEASURE_PLAN", "explicit purpose required")
    require(plan.get("storage") == "raw-f32", "MNLM_MEASURE_QUANTIZER", "no quantizer is commissioned")
    require(plan.get("training_scope_sha256") == POLICY_SHA256, "MNLM_MEASURE_SCOPE", "source policy changed; freeze a new plan")
    budget, blobs, shards = plan.get("blob_budget_bytes"), plan.get("blobs"), plan.get("shards")
    require(type(budget) is int and budget > 0 and isinstance(blobs, dict) and bool(blobs)
            and isinstance(shards, dict) and bool(shards), "MNLM_MEASURE_PLAN", "named scope and byte budget required")
    require(all(isinstance(h, str) and HEX.fullmatch(h) and type(n) is int and n > 0 for h, n in blobs.items()),
            "MNLM_MEASURE_PLAN", "invalid content-addressed blob")
    require(sum(blobs.values()) <= budget, "MNLM_MEASURE_STORAGE_BUDGET", "planned blobs exceed fixed storage budget")
    used = set()
    for key, shard in shards.items():
        require(isinstance(shard, dict) and isinstance(key, str) and bool(key) and shard.get("key") == key and type(shard.get("n")) is int
                and shard["n"] > 0 and bool(shard.get("tier")), "MNLM_MEASURE_PLAN", "shard key/count/tier required")
        require(isinstance(shard.get("slots"), dict) and bool(shard["slots"]), "MNLM_MEASURE_PLAN", "separate native slots required")
        used.add(shard.get("rows_sha256"))
        for name, slot in shard["slots"].items():
            require(isinstance(slot, dict) and isinstance(slot.get("seal"), dict), "MNLM_MEASURE_PLAN", "native slot seal required")
            require(isinstance(name, str) and bool(name) and slot["seal"].get("n") == shard["n"] and slot.get("lens_id")
                    and isinstance(slot.get("contract"), dict), "MNLM_MEASURE_PLAN", "slot identity/count required")
            require(all(isinstance(slot.get(k), str) and HEX.fullmatch(slot[k]) for k in ("sha256", "views_sha256")),
                    "MNLM_MEASURE_PLAN", "slot files require full SHA256 addresses")
            used.update((slot["sha256"], slot["views_sha256"]))
    require(used == set(blobs), "MNLM_MEASURE_PLAN", "blob inventory differs from declared shard dependencies")


class Publication:
    def __init__(self, root, plan=None, *, admission_root=None, ledger=None, production=False):
        self.root, self.db = Path(root), None
        if plan is not None:
            validate_plan(plan)
            if production or plan["purpose"] == "production":
                admitted = require_admitted(admission_root, ledger, plan.get("required_native_media"))
                require(plan.get("panel_blake3") == admitted["panel_blake3"], "MNLM_MEASURE_PANEL", "plan panel mismatch")
                # Revised base/tier-plan commissioning has not happened. Do not
                # turn a future Stage 3 flag into an unreviewed Stage 4 schema.
                require(False, "MNLM_STAGE4_SCOPE_UNCOMMISSIONED", "accepted revised atom/tier scope required before production publication")
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "blobs").mkdir(exist_ok=True)
            self.db = sqlite3.connect(self.root / "index.sqlite", timeout=0)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript(SCHEMA)
            for table in ("config", "shards", "events"):
                for op in ("UPDATE", "DELETE"):
                    self.db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{op} BEFORE {op} ON {table} "
                                    "BEGIN SELECT RAISE(ABORT,'MNLM_MEASURE_IMMUTABLE'); END")
            self.db.execute("INSERT OR IGNORE INTO config VALUES(1,?,?)", (digest(plan), canonical(plan)))
            self.db.commit()
        else:
            require((self.root / "index.sqlite").is_file(), "MNLM_MEASURE_MISSING", str(root))
            self.db = sqlite3.connect((self.root / "index.sqlite").resolve().as_uri() + "?mode=ro", uri=True, timeout=0)
        try:
            row = self.db.execute("SELECT hash,body FROM config WHERE id=1").fetchone()
            require(row is not None, "MNLM_MEASURE_PLAN", "missing frozen plan")
            self.plan = json.loads(row[1])
            validate_plan(self.plan)
            self.plan_sha256 = digest(self.plan)
            require(digest(self.plan) == row[0] and (plan is None or digest(plan) == row[0]),
                    "MNLM_MEASURE_PLAN_DRIFT", "store belongs to a different immutable plan")
            if production:
                require(self.plan["purpose"] == "production", "MNLM_STAGE4_BOUNDARY_ONLY", "engineering replay is not production")
                require(False, "MNLM_STAGE4_SCOPE_UNCOMMISSIONED", "production reader requires commissioned atom/tier scope")
            self.verify()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def state(self):
        rows = self.db.execute("SELECT count(*),coalesce(max(seq),0) FROM shards").fetchone()
        head = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return {"shards": rows[0], "seq": rows[1], "head": head[0] if head else None}

    def verify(self):
        own = not self.db.in_transaction
        if own:
            self.db.execute("BEGIN")
        try:
            return self._verify()
        finally:
            if own:
                self.db.rollback()

    def _verify(self):
        previous, seq = None, 0
        query = "SELECT e.seq,e.hash,e.body,s.key,s.hash,s.body,s.seq FROM events e LEFT JOIN shards s ON e.seq=s.seq ORDER BY e.seq"
        for n, h, raw, key, sh, sb, sn in self.db.execute(query):
            body = json.loads(raw)
            require(n == seq + 1 and body.get("parent") == previous and digest(body) == h
                    and body.get("plan_sha256") == self.plan_sha256, "MNLM_MEASURE_CHAIN", "broken publication chain")
            require(key == body["key"] and sh == body["shard_sha256"] and sn == n
                    and digest(json.loads(sb)) == sh and json.loads(sb) == self.plan["shards"].get(key),
                    "MNLM_MEASURE_CHAIN", "ledger/shard/plan disagreement")
            require(body.get("before_count") == seq and body.get("after_count") == n,
                    "MNLM_MEASURE_CHAIN", "shard counts disagree with event")
            previous, seq = h, n
        require(self.db.execute("SELECT count(*) FROM shards").fetchone()[0] == seq,
                "MNLM_MEASURE_CHAIN", "unledgered shard")
        return self.state()

    def blob(self, h):
        require(h in self.plan["blobs"] and HEX.fullmatch(h), "MNLM_MEASURE_BLOB", "unplanned blob")
        return self.root / "blobs" / h

    def check_blob(self, h):
        p = self.blob(h)
        require(p.is_file() and p.stat().st_size == self.plan["blobs"][h] and sha(p) == h,
                "MNLM_MEASURE_BLOB_CHANGED", f"restore exact frozen blob {h}; do not relabel it")
        return p

    def copy_blob(self, h, source):
        dest = self.blob(h)
        if dest.exists():
            return self.check_blob(h)
        tmp = dest.with_name(h + f".tmp.{os.getpid()}")
        require(not tmp.exists(), "MNLM_MEASURE_SCRATCH_EXISTS", str(tmp))
        try:
            total, checksum = 0, hashlib.sha256()
            with Path(source).open("rb") as src, tmp.open("xb") as out:
                while chunk := src.read(1024 * 1024):
                    total += len(chunk)
                    require(total <= self.plan["blobs"][h], "MNLM_MEASURE_BLOB_CHANGED", "source exceeds declared bytes")
                    checksum.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            require(total == self.plan["blobs"][h] and checksum.hexdigest() == h,
                    "MNLM_MEASURE_BLOB_CHANGED", "source changed since preflight")
            try:
                os.link(tmp, dest)  # exclusive, never replace an existing blob
                sync_dir(dest.parent)
            except FileExistsError:
                self.check_blob(h)
            return dest
        finally:
            tmp.unlink(missing_ok=True)  # only this process's exclusively created scratch

    def audit(self, shard):
        checked = set()

        def read(h):
            if h not in checked:
                self.check_blob(h)
                checked.add(h)
            return self.blob(h)

        raw = read(shard["rows_sha256"]).read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        require(len(rows) == shard["n"] and [r.get("i") for r in rows] == list(range(len(rows))),
                "MNLM_MEASURE_ROWS", "observation order/count changed")
        locators = set()
        for r in rows:
            check_source(r["source"])
            key = canonical({k: r[k] for k in ("source", "shard", "row", "span")})
            require(key not in locators, "MNLM_MEASURE_ROWS", "duplicate observation locator")
            locators.add(key)
        for name, slot in shard["slots"].items():
            contract, seal = slot["contract"], slot["seal"]
            require(seal.get("calibration_blake3") == blake3.blake3(raw).hexdigest()
                    and seal.get("contract_blake3") == blake3.blake3(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                    "MNLM_MEASURE_BINDING", f"{name}: slot is bound to different rows/contract")
            measured = slot_digest(read(slot["sha256"]), norm=contract.get("norm"))
            check_seal(measured, seal, name)
            require(measured["kind"] == seal["kind"], "MNLM_MEASURE_BINDING", "native slot kind changed")
            views = json.loads(read(slot["views_sha256"]).read_bytes())
            validate_slot(read(slot["sha256"]), contract, views)
            require(views.get("generation") and views.get("generation") == seal.get("ucr_generation") == contract.get("ucr_generation")
                    and digest(views.get("bindings")) == seal.get("view_bindings_sha256")
                    and digest(views.get("policy")) == seal.get("policy_sha256")
                    and len(views.get("bindings", [])) == len(rows), "MNLM_MEASURE_BINDING", "view generation/policy changed")
        return {"rows": len(rows), "slots": len(shard["slots"]), "blobs": len(checked),
                "purpose": self.plan["purpose"], "stage4_accepted": False}

    def publish(self, key, inputs):
        require(key in self.plan["shards"], "MNLM_MEASURE_SCOPE", "shard outside frozen plan")
        shard = self.plan["shards"][key]
        required = {shard["rows_sha256"]}
        for s in shard["slots"].values():
            required.update((s["sha256"], s["views_sha256"]))
        require(required <= inputs.keys(), "MNLM_MEASURE_INPUT", "missing named input blobs")
        try:
            self.db.execute("BEGIN IMMEDIATE")  # writer lease, fail immediately on contention
        except sqlite3.OperationalError as exc:
            raise SenseError("MNLM_MEASURE_WRITER_BUSY", "one publisher per store; retry after its transaction") from exc
        try:
            before = self.verify()
            if self.db.execute("SELECT 1 FROM shards WHERE key=?", (key,)).fetchone():
                report = self.audit(shard)
                self.db.rollback()
                return {**report, "reused": True, "before": before, "after": before}
            for h in sorted(required):
                self.copy_blob(h, inputs[h])
            report = self.audit(shard)
            n = before["seq"] + 1
            body = {"key": key, "parent": before["head"], "plan_sha256": self.plan_sha256,
                    "shard_sha256": digest(shard), "before_count": before["shards"], "after_count": before["shards"] + 1}
            self.db.execute("INSERT INTO shards VALUES(?,?,?,?)", (key, digest(shard), canonical(shard), n))
            self.db.execute("INSERT INTO events VALUES(?,?,?)", (n, digest(body), canonical(body)))
            self.db.commit()
            return {**report, "reused": False, "before": before, "after": self.verify()}
        except BaseException:
            self.db.rollback()
            raise

    def read(self, key):
        self.db.execute("BEGIN")
        try:
            row = self.db.execute("SELECT body FROM shards WHERE key=?", (key,)).fetchone()
            require(row is not None, "MNLM_MEASURE_NOT_PUBLISHED", "shard has no committed acceptance")
            self.verify()
            body = json.loads(row[0])
            return body, self.audit(body)
        finally:
            self.db.rollback()
