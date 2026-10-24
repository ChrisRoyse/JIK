"""Native canonical population: exact typed channel payloads and observation aliases.

This path preserves audio-only records and distinct outcomes behind identical
surface text. Payload sharing is storage deduplication, never sense admission or
permission to collapse observations. A manifest published last is the barrier.
"""
import argparse
import collections
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.contract import CHANNELS, SenseError, canonical, digest, is_sha256, require, validate_record
from senses.pipeline import file_hash, read_bound
from senses.scope import check_source
from senses.store import locator_key
from training_scope import POLICY_SHA256

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE payloads(channel TEXT NOT NULL,digest TEXT NOT NULL,body BLOB NOT NULL,
 PRIMARY KEY(channel,digest));
CREATE TABLE observations(atom TEXT PRIMARY KEY,source TEXT NOT NULL,unit TEXT NOT NULL,
 locator TEXT UNIQUE NOT NULL,header BLOB NOT NULL,record_sha256 TEXT NOT NULL);
CREATE TABLE channels(atom TEXT NOT NULL,channel TEXT NOT NULL,digest TEXT NOT NULL,
 PRIMARY KEY(atom,channel),FOREIGN KEY(atom) REFERENCES observations(atom),
 FOREIGN KEY(channel,digest) REFERENCES payloads(channel,digest));
CREATE TABLE occurrences(export TEXT NOT NULL,ordinal INTEGER NOT NULL,atom TEXT NOT NULL,
 PRIMARY KEY(export,ordinal),FOREIGN KEY(atom) REFERENCES observations(atom));
CREATE INDEX occurrence_atom ON occurrences(atom);
"""


def inputs(plan):
    require(isinstance(plan, dict) and set(plan) == {"schema", "training_scope_sha256", "exports", "source_rows"}
            and plan.get("schema") == "native-canon-plan-v1" and plan.get("training_scope_sha256") == POLICY_SHA256,
            "MNLM_NATIVE_PLAN", "explicit native plan under the current training scope required")
    exports = plan.get("exports")
    require(isinstance(exports, list) and bool(exports), "MNLM_NATIVE_PLAN", "named exports required")
    require(isinstance(plan.get("source_rows"), dict) and bool(plan["source_rows"])
            and all(isinstance(s, str) and s for s in plan["source_rows"])
            and all(type(n) is int and n > 0 for n in plan["source_rows"].values()),
            "MNLM_NATIVE_PLAN", "declare complete input row counts per source")
    for source in plan["source_rows"]:
        check_source(source)
    manifests = []
    names = set()
    for entry in exports:
        require(isinstance(entry, dict) and set(entry) == {"name", "path", "manifest", "manifest_sha256", "sha256", "bytes", "count"},
                "MNLM_NATIVE_PLAN", "explicit export file, scope and manifest binding required")
        require(isinstance(entry["name"], str) and 0 < len(entry["name"]) <= 80 and entry["name"] not in names
                and all(isinstance(entry[k], str) and entry[k] for k in ("path", "manifest"))
                and all(type(entry[k]) is int and entry[k] > 0 for k in ("bytes", "count"))
                and all(is_sha256(entry[k]) for k in ("sha256", "manifest_sha256")),
                "MNLM_NATIVE_PLAN", "named unique export with exact positive sizes/counts and SHA256 required")
        names.add(entry["name"])
        p, mp = Path(entry["path"]), Path(entry["manifest"])
        require(mp.is_file() and file_hash(mp) == entry.get("manifest_sha256"),
                "MNLM_NATIVE_INPUT_DIGEST", str(mp))
        manifest = json.loads(mp.read_bytes())
        require(manifest.get("schema") == "ucr-export-v1" and type(manifest.get("count")) is int
                and manifest["count"] > 0 and manifest["count"] == entry.get("count")
                and manifest.get("sha256") == entry.get("sha256") and p.is_file()
                and p.stat().st_size == entry.get("bytes"), "MNLM_NATIVE_INPUT_SCOPE", str(p))
        manifests.append(manifest)
    return manifests


def assemble(db, plan, manifests):
    counts = collections.Counter()
    for entry, manifest in zip(plan["exports"], manifests):
        n = 0
        for r in read_bound(entry["path"], entry["sha256"]):
            validate_record(r)
            source = r["locator"]["source"]
            check_source(source)
            counts[source] += 1
            header = {k: v for k, v in r.items() if k not in ("atom_id", "channels")}
            body = canonical(r)
            key = locator_key(r)
            existing = db.execute("SELECT atom,record_sha256 FROM observations WHERE locator=?", (key,)).fetchone()
            h = hashlib.sha256(body).hexdigest()
            require(existing is None or existing == (r["atom_id"], h),
                    "MNLM_UCR_IDENTITY_CONFLICT", "one physical locator carries conflicting observations")
            if existing is None:
                db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?)",
                           (r["atom_id"], source, r["unit"], key, canonical(header), h))
                for channel in CHANNELS:
                    payload = canonical(r["channels"][channel])
                    ph = hashlib.sha256(payload).hexdigest()
                    db.execute("INSERT OR IGNORE INTO payloads VALUES(?,?,?)", (channel, ph, payload))
                    db.execute("INSERT INTO channels VALUES(?,?,?)", (r["atom_id"], channel, ph))
            db.execute("INSERT INTO occurrences VALUES(?,?,?)", (entry["name"], n, r["atom_id"]))
            n += 1
        require(n == manifest["count"], "MNLM_NATIVE_INPUT_SCOPE", f"{entry['name']}: row count mismatch")
    require(dict(counts) == plan["source_rows"], "MNLM_NATIVE_INPUT_SCOPE", "source denominators differ from frozen plan")


def records(db):
    """One ordered join, with a bounded cache for shared small payloads (PC-07)."""
    query = ("SELECT o.atom,o.source,o.unit,o.locator,o.header,o.record_sha256,c.channel,c.digest,p.body "
             "FROM observations o JOIN channels c ON c.atom=o.atom "
             "JOIN payloads p ON p.channel=c.channel AND p.digest=c.digest ORDER BY o.atom,c.channel")
    cache = collections.OrderedDict()
    for atom_id, rows in itertools.groupby(db.execute(query), key=lambda r: r[0]):
        rows = list(rows)  # exactly nine typed slots per observation, never an entire corpus
        require(len(rows) == len(CHANNELS), "MNLM_NATIVE_CHANNEL_MISSING", atom_id)
        r = {**json.loads(rows[0][4]), "atom_id": atom_id, "channels": {}}
        for row in rows:
            channel, ph, body = row[6:]
            require(isinstance(body, bytes) and hashlib.sha256(body).hexdigest() == ph,
                    "MNLM_NATIVE_PAYLOAD_DIGEST", ph)
            key = channel, ph
            if key in cache:
                payload = cache.pop(key)
            else:
                payload = json.loads(body)
            if len(body) <= 65536:
                cache[key] = payload
                if len(cache) > 128:
                    cache.popitem(last=False)
            r["channels"][channel] = copy.deepcopy(payload) if key in cache else payload
        validate_record(r)
        require(hashlib.sha256(canonical(r)).hexdigest() == rows[0][5]
                and (r["locator"]["source"], r["unit"], locator_key(r)) == tuple(rows[0][1:4]),
                "MNLM_NATIVE_OBSERVATION_DIGEST", atom_id)
        check_source(r["locator"]["source"])
        yield r


def census(db):
    require(db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            and not db.execute("PRAGMA foreign_key_check").fetchall(),
            "MNLM_NATIVE_DATABASE", "database structural integrity failed")
    obs = db.execute("SELECT count(*) FROM observations").fetchone()[0]
    refs = db.execute("SELECT count(*) FROM channels").fetchone()[0]
    require(obs > 0 and refs == obs * len(CHANNELS), "MNLM_NATIVE_CHANNEL_MISSING", "incomplete native record")
    sources = dict(db.execute("SELECT source,count(*) FROM observations GROUP BY source"))
    occurrences = db.execute("SELECT count(*) FROM occurrences").fetchone()[0]
    require(db.execute("SELECT atom FROM observations EXCEPT SELECT atom FROM occurrences LIMIT 1").fetchone() is None,
            "MNLM_NATIVE_SCOPE", "observation outside the input population")
    require(db.execute("SELECT channel,digest FROM payloads EXCEPT SELECT channel,digest FROM channels LIMIT 1").fetchone() is None,
            "MNLM_NATIVE_SCOPE", "unreferenced channel payload")
    h = hashlib.sha256()
    n = 0
    for r in records(db):
        h.update(canonical(r) + b"\n")
        n += 1
    require(n == obs, "MNLM_NATIVE_CHANNEL_MISSING", "join omitted observations")
    return {"observations": obs, "occurrences": occurrences, "exact_observation_aliases": occurrences - obs,
            "units": db.execute("SELECT count(DISTINCT unit) FROM observations").fetchone()[0],
            "sources": sources, "channel_references": refs,
            "payloads": {c: {"count": n, "bytes": b} for c, n, b in
                         db.execute("SELECT channel,count(*),sum(length(body)) FROM payloads GROUP BY channel")},
            "records_sha256": h.hexdigest()}


def build(plan, output, fault=None):
    start = time.perf_counter()
    manifests = inputs(plan)
    preflight = time.perf_counter() - start
    output = Path(output)
    require(not output.exists(), "MNLM_NATIVE_OUTPUT_EXISTS", "publish under a new generation directory")
    output.mkdir(parents=True)
    temporary, final = output / "building.sqlite", output / "native.sqlite"
    db = sqlite3.connect(temporary)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(SCHEMA)
    try:
        t = time.perf_counter()
        db.execute("BEGIN IMMEDIATE")
        assemble(db, plan, manifests)
        db.commit()
        assembled = time.perf_counter() - t
    finally:
        db.close()
    t = time.perf_counter()
    cold = sqlite3.connect(temporary.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        readback = census(cold)
    finally:
        cold.close()
    cold_time = time.perf_counter() - t
    t = time.perf_counter()
    binding = [{k: e[k] for k in ("name", "sha256", "manifest_sha256", "bytes", "count")} for e in plan["exports"]]
    identity = {"schema": "native-canon-v1", "inputs": binding, "census": readback,
                "training_scope_sha256": POLICY_SHA256}
    manifest = {**identity, "generation": digest(identity), "database": {"sha256": file_hash(temporary),
                "bytes": temporary.stat().st_size}, "input_manifests": manifests,
                "code_sha256": {p.name: file_hash(p) for p in (Path(__file__),
                    Path(__file__).parents[1] / "senses" / "contract.py")},
                "stage1_complete": False, "stage2_complete": False, "scientific_admission": False,
                "pending": ["substring_and_near_duplicate_annotations", "native_atomization", "final_base_scope"]}
    os.link(temporary, final)
    if fault:
        fault("before_manifest")
    raw = canonical(manifest) + b"\n"
    with (output / "manifest.json").open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    temporary.unlink()
    phases = {"preflight": preflight, "assemble": assembled, "cold_readback": cold_time,
              "publish": time.perf_counter() - t}
    elapsed = time.perf_counter() - start
    receipt = {"acceptance": "author-checked", "generation": manifest["generation"], "census": readback,
               "manifest_sha256": hashlib.sha256(raw).hexdigest(), "seconds": elapsed, "phases": phases,
               "phase_coverage": sum(phases.values()) / elapsed, "served": False}
    with (output / "receipt.json").open("xb") as f:
        f.write(canonical(receipt) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    return receipt


class Population:
    def __init__(self, directory, manifest_sha256):
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        require(path.is_file() and file_hash(path) == manifest_sha256,
                "MNLM_NATIVE_MANIFEST", "complete, externally pinned publication manifest required")
        self.manifest = json.loads(path.read_bytes())
        m = self.manifest
        require(m.get("schema") == "native-canon-v1" and m.get("training_scope_sha256") == POLICY_SHA256,
                "MNLM_NATIVE_SCOPE", "native schema/current training policy mismatch")
        identity = {k: m[k] for k in ("schema", "inputs", "census", "training_scope_sha256")}
        require(digest(identity) == m.get("generation"), "MNLM_NATIVE_MANIFEST", "generation identity differs")
        for source in m["census"]["sources"]:
            check_source(source)
        path = self.directory / "native.sqlite"
        require(path.is_file() and path.stat().st_size == m["database"]["bytes"]
                and file_hash(path) == m["database"]["sha256"], "MNLM_NATIVE_DATABASE_DIGEST", str(path))
        self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.db.execute("BEGIN")

    def close(self):
        self.db.close()

    def records(self):
        yield from records(self.db)

    def audit(self):
        actual = census(self.db)
        require(actual == self.manifest["census"], "MNLM_NATIVE_CENSUS", "physical population differs from manifest")
        return actual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    require(file_hash(args.plan) == args.sha256, "MNLM_NATIVE_PLAN", "plan digest differs")
    print(json.dumps(build(json.loads(args.plan.read_bytes()), args.output)))


if __name__ == "__main__":
    try:
        main()
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)
