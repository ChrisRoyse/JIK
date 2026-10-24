"""Read/copy pinned native near annotations without importing a model or hasher."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct

from native_population import file_hash, POLICY_SHA256
from senses.contract import canonical, require


def retain(population, source, manifest_sha256, output, parent_manifest):
    source, output = Path(source), Path(output)
    mp = source / "manifest.json"
    require(mp.is_file() and file_hash(mp) == manifest_sha256, "MNLM_NATIVE_NEAR_MANIFEST", "pinned near manifest required")
    m = json.loads(mp.read_bytes())
    require(m["schema"] == "native-near-v1" and m["training_scope_sha256"] == POLICY_SHA256
            and m["parent_manifest_sha256"] == parent_manifest and m["observations_removed"] == 0
            and m["contract"] == "minhash-v1" and m["floor"] == 0.6 and m["representatives"] == 32,
            "MNLM_NATIVE_NEAR_SCOPE", "annotations must bind this exact population and retain all observations")
    dbpath = source / "near.sqlite"
    require(file_hash(dbpath) == m["database_sha256"], "MNLM_NATIVE_NEAR_DIGEST", "annotation bytes changed")
    db = sqlite3.connect(dbpath.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        actual = set()
        for ph, sg, valid in db.execute("SELECT digest,signature,valid FROM signatures"):
            require(isinstance(sg, bytes) and len(sg) == 448 and valid in (0, 1),
                    "MNLM_NATIVE_NEAR_IDENTITY", "minhash-v1 signature geometry changed")
            actual.add(ph)
        expected = {r[0] for r in population.db.execute("SELECT digest FROM payloads WHERE channel='surface'")}
        require(actual == expected and len(actual) == m["payloads"], "MNLM_NATIVE_NEAR_SCOPE", "surface population differs")
        h, n = hashlib.sha256(), 0
        for a, b, score, kind, sa, sb, va, vb in db.execute("SELECT l.origin,l.target,l.agreement,l.kind,a.signature,b.signature,a.valid,b.valid "
                "FROM links l JOIN signatures a ON a.digest=l.origin JOIN signatures b ON b.digest=l.target ORDER BY l.origin"):
            agreement = sum(x == y for x, y in zip(struct.unpack('<112I', sa), struct.unpack('<112I', sb))) / 112
            require(kind == "near_surface_signature" and a != b and va == vb == 1 and score == agreement and score >= m["floor"],
                    "MNLM_NATIVE_NEAR_IDENTITY", "unbound or invalid near-surface relationship")
            h.update(canonical([a, b, score]) + b"\n")
            n += 1
        require(n == m["links"] == db.execute("SELECT count(*) FROM links").fetchone()[0]
                and h.hexdigest() == m["links_sha256"], "MNLM_NATIVE_NEAR_IDENTITY", "link census differs")
    finally:
        db.close()
    require(not (output / "near.sqlite").exists() and not (output / "near.manifest.json").exists(),
            "MNLM_NATIVE_OUTPUT_EXISTS", "annotations are immutable; use a new publication directory")
    for src, name in ((dbpath, "near.sqlite"), (mp, "near.manifest.json")):
        with src.open('rb') as reader, (output / name).open('xb') as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
    require(file_hash(output / "near.sqlite") == m["database_sha256"]
            and file_hash(output / "near.manifest.json") == manifest_sha256,
            "MNLM_NATIVE_NEAR_DIGEST", "annotation changed during retention")
    return {"manifest_sha256": manifest_sha256, "links": n, "links_sha256": m["links_sha256"], "observations_removed": 0}
