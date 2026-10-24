"""Typed near-surface annotations; never merge observations or their outcomes.

Reuses minhash-v1 and the existing 0.6 / 32 representative contract. Cost:
O(unique surface shingles * 112 + U * 14 log U + U * 32 * 112), O(U * 112)
memory. U=2,865 payloads in the approved base, including explicit absence.
These approximate links are provenance annotations, not extra senses or bits.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_population import Population, file_hash, POLICY_SHA256
from minhash import shingle_hashes, signature, band_keys, PERMS, BANDS, CONTRACT_ID
from dedup import NEAR_FLOOR, NEAR_K
from senses.contract import canonical, require, SenseError


def links(signatures, valid):
    """Component candidates from band stars, then bounded kept representatives."""
    n = len(signatures)
    bands = band_keys(signatures)
    ids = np.flatnonzero(valid)
    origins, targets = [], []
    for band in range(BANDS):
        ordered = ids[np.argsort(bands[ids, band], kind="stable")]
        groups = collections.defaultdict(list)
        for i in ordered:
            groups[int(bands[i, band])].append(int(i))
        for group in groups.values():
            origins.extend(group[1:])
            targets.extend([group[0]] * (len(group) - 1))
    graph = coo_matrix((np.ones(len(origins), dtype=np.int8), (origins, targets)), shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)
    groups = collections.defaultdict(list)
    for i in ids:
        groups[int(labels[i])].append(int(i))
    matches = []
    for group in groups.values():
        kept = []
        for i in group:
            if not kept:
                kept.append(i)
                continue
            agreements = (signatures[kept] == signatures[i]).mean(axis=1)
            j = int(np.argmax(agreements))
            if agreements[j] >= NEAR_FLOOR:
                matches.append((i, kept[j], float(agreements[j])))
            elif len(kept) < NEAR_K:
                kept.append(i)
    return matches, {"band_links": len(origins), "components": int(ncomp), "valid": int(valid.sum())}


def build(source, manifest_sha256, output):
    start = time.perf_counter()
    pop = Population(source, manifest_sha256)
    output = Path(output)
    require(not output.exists(), "MNLM_NATIVE_OUTPUT_EXISTS", "new near-annotation generation required")
    output.mkdir(parents=True)
    db = sqlite3.connect(output / "near.sqlite")
    try:
        db.executescript("CREATE TABLE signatures(digest TEXT PRIMARY KEY,signature BLOB NOT NULL,valid INTEGER NOT NULL);"
                         "CREATE TABLE links(origin TEXT PRIMARY KEY,target TEXT NOT NULL,agreement REAL NOT NULL,kind TEXT NOT NULL);")
        t = time.perf_counter()
        preflight = t - start
        hashes, signatures, valid = [], [], []
        for ph, body in pop.db.execute("SELECT digest,body FROM payloads WHERE channel='surface' ORDER BY digest"):
            require(hashlib.sha256(body).hexdigest() == ph, "MNLM_NATIVE_PAYLOAD_DIGEST", ph)
            payload = json.loads(body)
            text = "\n".join(item["text"] for item in payload["items"])
            shingles = shingle_hashes(text)
            sg = signature(shingles)
            hashes.append(ph)
            signatures.append(sg)
            valid.append(bool(len(shingles)))
            db.execute("INSERT INTO signatures VALUES(?,?,?)", (ph, sg.astype("<u4").tobytes(), bool(len(shingles))))
        signed = time.perf_counter() - t
        t = time.perf_counter()
        matches, stats = links(np.asarray(signatures, dtype=np.uint32), np.asarray(valid, dtype=bool))
        for i, j, agreement in matches:
            db.execute("INSERT INTO links VALUES(?,?,?,'near_surface_signature')", (hashes[i], hashes[j], agreement))
        db.commit()
        matched = time.perf_counter() - t
    finally:
        db.close()
        pop.close()
    t = time.perf_counter()
    # Cold signature/relationship readback and edge digest, not the return count.
    db = sqlite3.connect((output / "near.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    h, n = hashlib.sha256(), 0
    try:
        for a, b, score, sa, sb in db.execute("SELECT l.origin,l.target,l.agreement,a.signature,b.signature FROM links l "
                "JOIN signatures a ON a.digest=l.origin JOIN signatures b ON b.digest=l.target ORDER BY l.origin"):
            actual = float((np.frombuffer(sa, dtype='<u4') == np.frombuffer(sb, dtype='<u4')).mean())
            require(actual == score and actual >= NEAR_FLOOR and a != b,
                    "MNLM_NATIVE_NEAR_IDENTITY", "invalid signature relationship")
            h.update(canonical([a, b, score]) + b"\n")
            n += 1
        require(n == db.execute("SELECT count(*) FROM links").fetchone()[0], "MNLM_NATIVE_NEAR_IDENTITY", "unbound relationship")
    finally:
        db.close()
    manifest = {"schema": "native-near-v1", "parent_manifest_sha256": manifest_sha256,
                "training_scope_sha256": POLICY_SHA256, "contract": CONTRACT_ID, "floor": NEAR_FLOOR, "representatives": NEAR_K,
                "payloads": len(hashes), "links": n, **stats, "links_sha256": h.hexdigest(),
                "database_sha256": file_hash(output / "near.sqlite"), "observations_removed": 0,
                "scope": "approximate surface links only; no observation merging or exhaustive similarity claim",
                "code_sha256": {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("minhash.py"), Path(__file__).with_name("dedup.py"))},
                "stage1_complete": False, "scientific_admission": False}
    with (output / "manifest.json").open("xb") as f:
        f.write(canonical(manifest) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    phases = {"preflight": preflight, "sign": signed, "match": matched, "cold_readback_publish": time.perf_counter() - t}
    elapsed = time.perf_counter() - start
    receipt = {"acceptance": "author-checked", "manifest_sha256": file_hash(output / "manifest.json"),
               "seconds": elapsed, "phases": phases, "phase_coverage": sum(phases.values()) / elapsed, "served": False}
    (output / "receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population", type=Path, required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        print(json.dumps(build(args.population, args.sha256, args.output)))
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)
