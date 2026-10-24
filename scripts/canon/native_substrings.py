"""Global exact-substring annotations over native surface items (#22).

Observation identities/outcomes are never collapsed. One deterministic corpus
of unique typed surface items feeds the existing reference suffix executable.
The completion manifest pins the population, tool, work bytes and span map.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_population import Population, file_hash, POLICY_SHA256
from senses.contract import canonical, require, SenseError, is_sha256
from substring_ranges import resolve_groups
from suffix_protocol import normalize_collect, read_ranges

THRESHOLD = 200
CORPUS_BUDGET = 64 * 1024 * 1024  # bounded one-part build; larger scopes require the existing merged path
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE payloads(digest TEXT PRIMARY KEY,items INTEGER NOT NULL);
CREATE TABLE segments(id INTEGER PRIMARY KEY,payload TEXT NOT NULL,item INTEGER NOT NULL,
 start INTEGER NOT NULL,n INTEGER NOT NULL,sha256 TEXT NOT NULL,
 UNIQUE(payload,item),FOREIGN KEY(payload) REFERENCES payloads(digest));
CREATE TABLE spans(origin INTEGER NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,
 target INTEGER NOT NULL,dup_start INTEGER NOT NULL,sha256 TEXT NOT NULL,
 PRIMARY KEY(origin,start,end,target,dup_start),FOREIGN KEY(origin) REFERENCES segments(id),
 FOREIGN KEY(target) REFERENCES segments(id));
"""


def surfaces(population):
    for ph, body in population.db.execute("SELECT digest,body FROM payloads WHERE channel='surface' ORDER BY digest"):
        require(hashlib.sha256(body).hexdigest() == ph, "MNLM_NATIVE_PAYLOAD_DIGEST", ph)
        channel = json.loads(body)
        yield ph, channel["items"]


def corpus(population, db, out):
    offsets, n, size = [0], 0, 0
    # Budget from existing bytes before writing or launching a process; no discarded prefix corpus.
    expected = sum(len(item["text"].encode()) + 1 for _, items in surfaces(population) for item in items)
    require(0 < expected <= CORPUS_BUDGET, "MNLM_NATIVE_SUBSTRING_BUDGET",
            f"{expected} bytes; one-part cap {CORPUS_BUDGET}; use a separately scoped merged build")
    with (out / "corpus.bin").open("xb") as f:
        for ph, items in surfaces(population):
            db.execute("INSERT INTO payloads VALUES(?,?)", (ph, len(items)))
            for i, item in enumerate(items):
                raw = item["text"].encode("utf-8")
                db.execute("INSERT INTO segments VALUES(?,?,?,?,?,?)",
                           (n, ph, i, size, len(raw), hashlib.sha256(raw).hexdigest()))
                f.write(raw + b"\0")
                n += 1
                size += len(raw) + 1
                offsets.append(size)
        f.flush()
        os.fsync(f.fileno())
    require(size == expected, "MNLM_NATIVE_SUBSTRING_SCOPE", "surface population changed during corpus assembly")
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    return np.asarray(offsets, dtype=np.int64)


def execute(tool, out):
    data, cache = out / "corpus.bin", out / "cache"
    cache.mkdir()
    calls = [([str(tool), "make", "--data-file", str(data)], "make.log"),
             ([str(tool), "self-similar", "--data-file", str(data), "--length-threshold", str(THRESHOLD),
               "--cache-dir", str(cache), "--num-threads", "1"], "similar.log"),
             ([str(tool), "collect", "--data-file", str(data), "--cache-dir", str(cache),
               "--length-threshold", str(THRESHOLD)], "collect.txt")]
    timings = {}
    for argv, name in calls:
        t = time.perf_counter()
        with (out / name).open("xb") as stdout, (out / (name + ".stderr")).open("xb") as stderr:
            result = subprocess.run(argv, stdout=stdout, stderr=stderr,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        require(result.returncode == 0, "MNLM_NATIVE_SUFFIX_TOOL", f"{name}: exit {result.returncode}; inspect retained logs")
        timings[name] = time.perf_counter() - t
    normalize_collect(out)
    return timings


def annotate(db, out, offsets):
    ranges = read_ranges(out / "ranges.txt")
    spans, stats = resolve_groups(out / "corpus.bin", offsets, THRESHOLD, out / "span-windows.sqlite")
    data = np.memmap(out / "corpus.bin", dtype=np.uint8, mode="r")
    for origin, s, e, target, ds in spans.tolist():
        start, end = int(offsets[origin]) + s, int(offsets[origin]) + e
        dup = int(offsets[target]) + ds
        raw = bytes(data[start:end])
        raw.decode("utf-8")
        require(raw == bytes(data[dup:dup + end - start]), "MNLM_NATIVE_SUBSTRING_BYTES", "resolved span not equal")
        db.execute("INSERT INTO spans VALUES(?,?,?,?,?,?)", (origin, start - int(offsets[origin]),
                   end - int(offsets[origin]), target, dup - int(offsets[target]), hashlib.sha256(raw).hexdigest()))
    return {"collected_ranges": len(ranges), **stats}


def build(source, parent_sha256, tool, tool_sha256, output, fault=None):
    start = time.perf_counter()
    require(is_sha256(tool_sha256) and file_hash(tool) == tool_sha256,
            "MNLM_NATIVE_SUFFIX_TOOL", "externally pinned suffix executable required")
    pop = Population(source, parent_sha256)
    output = Path(output).resolve()
    require(not output.exists(), "MNLM_NATIVE_OUTPUT_EXISTS", "new immutable substring generation required")
    output.mkdir(parents=True)
    db = sqlite3.connect(output / "substrings.sqlite")
    phases = {"preflight": time.perf_counter() - start}
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.executescript(SCHEMA)
        t = time.perf_counter()
        offsets = corpus(pop, db, output)
        db.commit()
        corpus_sha = file_hash(output / "corpus.bin")
        phases["corpus"] = time.perf_counter() - t
        t = time.perf_counter()
        tool_times = execute(Path(tool).resolve(), output)
        require(file_hash(tool) == tool_sha256 and file_hash(output / "corpus.bin") == corpus_sha,
                "MNLM_NATIVE_SUBSTRING_CHANGED", "tool or corpus changed during suffix execution")
        phases["suffix"] = time.perf_counter() - t
        t = time.perf_counter()
        resolution = annotate(db, output, offsets)
        db.commit()
        phases["resolve"] = time.perf_counter() - t
        t = time.perf_counter()
        from native_substring_readback import audit
        result = audit(pop, db, THRESHOLD)
        phases["cold_coordinate_readback"] = time.perf_counter() - t
    finally:
        db.close()
        pop.close()
    t = time.perf_counter()
    names = [p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()]
    # The completion manifest cannot become durable ahead of tool-created data.
    for name in names:
        with (output / name).open("r+b") as f:
            os.fsync(f.fileno())
    if os.name != "nt":
        for directory in (output / "cache", output):
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    files = {n: {"sha256": file_hash(output / n), "bytes": (output / n).stat().st_size} for n in names}
    manifest = {"schema": "native-substrings-v1", "parent_manifest_sha256": parent_sha256,
                "training_scope_sha256": POLICY_SHA256, "threshold_bytes": THRESHOLD,
                "tool_sha256": tool_sha256, "files": files, "census": result, "resolution": resolution,
                "observations_removed": 0, "full_stage1_complete": False, "scientific_admission": False,
                "code_sha256": {n: file_hash(Path(__file__).with_name(n)) for n in
                    ("native_substrings.py", "native_substring_readback.py", "substring_ranges.py", "suffix_protocol.py")}}
    if fault:
        fault("before_manifest")
    with (output / "manifest.json").open("xb") as f:
        f.write(canonical(manifest) + b"\n"); f.flush(); os.fsync(f.fileno())
    if os.name != "nt":
        fd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    phases["publish"] = time.perf_counter() - t
    elapsed = time.perf_counter() - start
    receipt = {"acceptance": "author-checked", "census": result, "resolution": resolution,
               "manifest_sha256": file_hash(output / "manifest.json"), "phases": phases,
               "tool_seconds": tool_times, "seconds": elapsed, "phase_coverage": sum(phases.values()) / elapsed}
    (output / "receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--tool", type=Path, required=True)
    ap.add_argument("--tool-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        print(json.dumps(build(args.population, args.sha256, args.tool, args.tool_sha256, args.output)))
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)
