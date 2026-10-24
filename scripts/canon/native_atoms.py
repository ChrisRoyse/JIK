"""Source-neutral atomization of a pinned native population (#20, PC-07/35/38).

O(retained bytes + output atoms); one parent at a time, indexed lineage joins.
Observation units and typed channels survive. Derivation/order metadata lives
outside sensory channels: running a segmenter cannot manufacture another sense.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from native_population import Population, file_hash, POLICY_SHA256
from code_atoms import atomise_text
from segment import conserved
from senses.contract import SenseError, canonical, digest, require, seal_record, present
from senses.pipeline import write_records, read_bound
from senses.store import Store

SCHEMA = """
CREATE TABLE lineage(atom TEXT PRIMARY KEY,parent TEXT NOT NULL,ordinal INTEGER NOT NULL,
 start INTEGER NOT NULL,end INTEGER NOT NULL,span_unit TEXT NOT NULL,role INTEGER NOT NULL,
 code_unit INTEGER NOT NULL,source TEXT NOT NULL,shard TEXT NOT NULL,row INTEGER NOT NULL,
 unit TEXT NOT NULL,UNIQUE(parent,ordinal));
CREATE INDEX parent_atoms ON lineage(parent,ordinal);
CREATE TABLE edges(origin TEXT NOT NULL,target TEXT NOT NULL,kind TEXT NOT NULL,
 PRIMARY KEY(origin,target,kind),FOREIGN KEY(origin) REFERENCES lineage(atom),
 FOREIGN KEY(target) REFERENCES lineage(atom));
"""
WHOLE = {"tool_exchange": "character", "recorded_audio": "sample", "row": "character", "triple": "character", "observation": None}


def kind(record):
    values = [f["value"] for f in record["channels"]["structure"]["items"] if f["name"] == "kind"]
    require(len(values) == 1 and values[0] in (*WHOLE, "chapter_text", "prose"),
            "MNLM_NATIVE_ATOM_KIND", "adapter must declare one supported structural kind")
    return values[0]


def split(record):
    k = kind(record)
    channels, loc = record["channels"], record["locator"]
    if k in WHOLE:
        span_unit = WHOLE[k]
        if span_unit is None:
            units = [f for f in channels["structure"]["items"] if f["name"] == "span_unit"]
            require(len(units) == 1 and units[0]["type"] == "category" and bool(units[0]["value"]),
                    "MNLM_NATIVE_ATOM_SPAN", "generic observation adapters must declare their coordinate unit")
            span_unit = units[0]["value"]
        if k == "recorded_audio":
            from audio_encoder import spans
            audio = spans(record)
            require(channels["surface"]["state"] == "absent" and audio
                    and all(a["start"] == loc["span"][0] and a["end"] == loc["span"][1] for a in audio),
                    "MNLM_NATIVE_AUDIO_SPAN", "complete declared audio sample windows required")
        elif k == "tool_exchange":
            require(channels["surface"]["state"] == "present" and len(channels["surface"]["items"]) == 1
                    and len(channels["surface"]["items"][0]["text"]) == loc["span"][1] - loc["span"][0],
                    "MNLM_NATIVE_ATOM_SURFACE", "action missing or source span differs")
        yield record, (loc["span"][0], loc["span"][1], span_unit, 3, 0)
        return
    require(channels["surface"]["state"] == "present" and len(channels["surface"]["items"]) == 1,
            "MNLM_NATIVE_ATOM_SURFACE", "prose atomization requires one retained surface")
    # Copying measurements of an entire parent onto each substring would silently
    # assert alignment that an adapter never supplied. Keep those parents whole.
    require(all(channels[c]["state"] == "absent" for c in ("behaviour", "perception", "history", "co_reference")),
            "MNLM_NATIVE_ATOM_ALIGNMENT", "split prose requires span-aligned evidence; use an observed record unit otherwise")
    text = channels["surface"]["items"][0]["text"]
    require(len(text) == loc["span"][1] - loc["span"][0], "MNLM_NATIVE_ATOM_SPAN", "surface/span mismatch")
    parts, _ = atomise_text(text)
    require(conserved(text, [(a, b, role) for a, b, role, _ in parts]),
            "MNLM_ATOMS_CONSERVATION", record["atom_id"])
    for a, b, role, unit in parts:
        child = copy.deepcopy(record)
        child["locator"]["span"] = [loc["span"][0] + a, loc["span"][0] + b]
        child["channels"]["surface"] = present([{"text": text[a:b]}])
        yield seal_record(child), (*child["locator"]["span"], "character", role, unit)


def emit(population, db):
    for parent in population.records():
        for ordinal, (record, span) in enumerate(split(parent)):
            loc = record["locator"]
            db.execute("INSERT INTO lineage VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (record["atom_id"], parent["atom_id"], ordinal, *span,
                        loc["source"], loc["shard"], loc["row"], record["unit"]))
            yield record


def order_edges(db):
    # Observed order only within one source shard and sibling unit, never a
    # fabricated timestamp or a cross-source audio/transcript alignment.
    rows = db.execute("SELECT atom,source,shard,unit FROM lineage ORDER BY source,shard,unit,row,start,end,atom")
    for _, group in itertools.groupby(rows, key=lambda r: r[1:]):
        previous = None
        for row in group:
            if previous:
                db.execute("INSERT INTO edges VALUES(?,?,'observed_order')", (previous, row[0]))
            previous = row[0]


def audit(population, store, db):
    """Cold source/atom/lineage closure, without running the segmenter again."""
    require(db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            and not db.execute("PRAGMA foreign_key_check").fetchall(), "MNLM_NATIVE_ATOM_DATABASE", "invalid lineage store")
    parents = atoms = 0
    by_kind, source_counts = {}, {}
    atom_ids = hashlib.sha256()
    for parent in population.records():
        parents += 1
        rows = db.execute("SELECT atom,ordinal,start,end,span_unit,source,shard,row,unit FROM lineage WHERE parent=? ORDER BY ordinal",
                          (parent["atom_id"],)).fetchall()
        require(bool(rows) and [r[1] for r in rows] == list(range(len(rows))),
                "MNLM_NATIVE_ATOM_CLOSURE", "missing or unordered parent atoms")
        k, cursor = kind(parent), parent["locator"]["span"][0]
        expected_unit = WHOLE.get(k, "character")
        if expected_unit is None:
            units = [f["value"] for f in parent["channels"]["structure"]["items"] if f["name"] == "span_unit"]
            require(len(units) == 1 and bool(units[0]), "MNLM_NATIVE_ATOM_SPAN", "coordinate unit missing")
            expected_unit = units[0]
        reconstructed = []
        for row in rows:
            child = store.get(row[0])
            loc = child["locator"]
            require((row[2], row[3]) == tuple(loc["span"]) and row[2] == cursor and row[3] > cursor
                    and row[4] == expected_unit and row[5:] == (loc["source"], loc["shard"], loc["row"], child["unit"]),
                    "MNLM_ATOMS_CONSERVATION", "span/locator/order changed")
            cursor = row[3]
            if k in WHOLE:
                require(child == parent and len(rows) == 1, "MNLM_NATIVE_ATOM_CLOSURE", "observed record unit changed")
            else:
                expected = copy.deepcopy(parent)
                start = row[2] - parent["locator"]["span"][0]
                end = row[3] - parent["locator"]["span"][0]
                text = parent["channels"]["surface"]["items"][0]["text"][start:end]
                expected["channels"]["surface"] = present([{"text": text}])
                expected["locator"]["span"] = list(loc["span"])
                require(child == seal_record(expected), "MNLM_NATIVE_ATOM_CLOSURE", "child evidence differs from its parent")
                reconstructed.append(child["channels"]["surface"]["items"][0]["text"])
            atoms += 1
            by_kind[k] = by_kind.get(k, 0) + 1
            source_counts[loc["source"]] = source_counts.get(loc["source"], 0) + 1
            atom_ids.update(bytes.fromhex(child["atom_id"]))
        require(cursor == parent["locator"]["span"][1], "MNLM_ATOMS_CONSERVATION", "parent span not fully covered")
        if reconstructed:
            require("".join(reconstructed).encode() == parent["channels"]["surface"]["items"][0]["text"].encode(),
                    "MNLM_ATOMS_CONSERVATION", "parent UTF-8 bytes not conserved")
    require(atoms == db.execute("SELECT count(*) FROM lineage").fetchone()[0]
            == store.db.execute("SELECT count(*) FROM records").fetchone()[0],
            "MNLM_NATIVE_ATOM_CLOSURE", "unbound lineage or records")
    # Compare edge set with one adjacency walk; no all-pairs graph.
    previous, scope, n = None, None, 0
    for atom, source, shard, unit in db.execute("SELECT atom,source,shard,unit FROM lineage ORDER BY source,shard,unit,row,start,end,atom"):
        current = source, shard, unit
        if scope == current:
            require(db.execute("SELECT 1 FROM edges WHERE origin=? AND target=? AND kind='observed_order'", (previous, atom)).fetchone(),
                    "MNLM_NATIVE_ATOM_ORDER", "missing observed adjacency")
            n += 1
        scope, previous = current, atom
    require(n == db.execute("SELECT count(*) FROM edges").fetchone()[0], "MNLM_NATIVE_ATOM_ORDER", "extra order edge")
    return {"parents": parents, "atoms": atoms, "source_counts": source_counts, "kinds": by_kind,
            "order_edges": n, "atom_identity_sha256": atom_ids.hexdigest(), "ucr_generation": store.head}


def build(source, manifest_sha256, output, fault=None, near=None, near_sha256=None,
          substrings=None, substrings_sha256=None):
    start = time.perf_counter()
    population = Population(source, manifest_sha256)
    output = Path(output)
    require(not output.exists(), "MNLM_NATIVE_OUTPUT_EXISTS", "choose a new atom generation directory")
    output.mkdir(parents=True)
    db = sqlite3.connect(output / "lineage.sqlite")
    store = None
    try:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.executescript(SCHEMA)
        require((near is None) == (near_sha256 is None), "MNLM_NATIVE_NEAR_MANIFEST", "near path and digest must be supplied together")
        annotation = None
        if near is not None:
            from native_annotations import retain
            annotation = retain(population, near, near_sha256, output, manifest_sha256)
        require((substrings is None) == (substrings_sha256 is None), "MNLM_NATIVE_SUBSTRING_MANIFEST",
                "substring path and digest must be supplied together")
        substring_annotation = None
        if substrings is not None:
            from native_substring_readback import retain as retain_substrings
            substring_annotation = retain_substrings(population, substrings, substrings_sha256, output, manifest_sha256)
        t = time.perf_counter()
        preflight = t - start
        binding = {"kind": "native_atoms", "parent_generation": population.manifest["generation"],
                   "parent_manifest_sha256": manifest_sha256, "training_scope_sha256": POLICY_SHA256}
        export = write_records(output / "atoms.ucr.jsonl", emit(population, db), binding)
        order_edges(db)
        db.commit()
        emitted = time.perf_counter() - t
        t = time.perf_counter()
        store = Store(output / "atoms.sqlite", create=True, records_only=True)
        store.ingest(read_bound(output / "atoms.ucr.jsonl", export["sha256"]),
                     {"input_sha256": export["sha256"], "count": export["count"], **binding}, expected_head=None)
        store.close()
        store = Store(output / "atoms.sqlite")
        retained = time.perf_counter() - t
        t = time.perf_counter()
        result = audit(population, store, db)
        readback = time.perf_counter() - t
    finally:
        if store:
            store.close()
        db.close()
        population.close()
    t = time.perf_counter()
    names = ("lineage.sqlite", "atoms.sqlite", "atoms.ucr.jsonl", "atoms.ucr.manifest.json")
    if annotation:
        names += ("near.sqlite", "near.manifest.json")
    if substring_annotation:
        names += ("substrings.sqlite", "substrings.manifest.json")
    files = {name: {"sha256": file_hash(output / name), "bytes": (output / name).stat().st_size} for name in names}
    base = Path(__file__).resolve().parents[1]
    code = {n: file_hash(base / n) for n in ("canon/native_atoms.py", "canon/native_annotations.py", "canon/code_atoms.py", "canon/fences.py", "canon/segment.py",
            "senses/contract.py", "senses/pipeline.py", "senses/store.py", "lens/audio_encoder.py")}
    code["canon/native_substring_readback.py"] = file_hash(base / "canon/native_substring_readback.py")
    manifest = {"schema": "native-atoms-v1", "binding": binding, "census": result, "files": files, "code_sha256": code,
                "substring_annotations": substring_annotation,
                "near_annotations": annotation, "stage2_scope_complete": True, "full_stage2_complete": False, "scientific_admission": False}
    if fault:
        fault("before_manifest")
    with (output / "manifest.json").open("xb") as f:
        f.write(canonical(manifest) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    phases = {"preflight": preflight, "atomize": emitted, "retain": retained, "cold_readback": readback,
              "publish": time.perf_counter() - t}
    elapsed = time.perf_counter() - start
    receipt = {"acceptance": "author-checked", "census": result, "manifest_sha256": file_hash(output / "manifest.json"),
               "seconds": elapsed, "phases": phases, "phase_coverage": sum(phases.values()) / elapsed, "served": False}
    (output / "receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def open_atoms(directory, manifest_sha256):
    directory = Path(directory)
    mp = directory / "manifest.json"
    require(mp.is_file() and file_hash(mp) == manifest_sha256, "MNLM_NATIVE_ATOM_MANIFEST", "pinned completion manifest required")
    m = json.loads(mp.read_bytes())
    require(m["schema"] == "native-atoms-v1" and m["binding"]["training_scope_sha256"] == POLICY_SHA256,
            "MNLM_NATIVE_ATOM_MANIFEST", "schema or policy mismatch")
    names = ("lineage.sqlite", "atoms.sqlite", "atoms.ucr.jsonl", "atoms.ucr.manifest.json")
    if m.get("near_annotations"):
        names += ("near.sqlite", "near.manifest.json")
    if m.get("substring_annotations"):
        names += ("substrings.sqlite", "substrings.manifest.json")
    for name in names:
        p = directory / name
        require(p.is_file() and p.stat().st_size == m["files"][name]["bytes"] and file_hash(p) == m["files"][name]["sha256"],
                "MNLM_NATIVE_ATOM_DIGEST", name)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--near", type=Path)
    ap.add_argument("--near-sha256")
    ap.add_argument("--substrings", type=Path)
    ap.add_argument("--substrings-sha256")
    args = ap.parse_args()
    try:
        print(json.dumps(build(args.population, args.sha256, args.output, near=args.near, near_sha256=args.near_sha256,
                               substrings=args.substrings, substrings_sha256=args.substrings_sha256)))
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)
