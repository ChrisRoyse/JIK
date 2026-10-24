"""Graph channel sense for the A2 record anchors (kg-context-masked-v1, #12/#15/#25): what the entity's OTHER
Wikidata statements say about it, a different input channel from its description text.

Rules (predeclared before any reading; recorded in the manifest):
  masked properties  M = the family's anchor properties (summary.json) + their Wikidata inverses (INVERSE) +
                     P31 and P279 (the sampling class) - no masked statement in either direction is read;
  out-features       for each kept out-statement (s=entity, p, o): hashed (p) and hashed (p, class of o),
                     where class of o = o's own first P31 (the target's identity is never a feature);
  in-features        for each kept in-statement (o=entity): hashed (p) and hashed (p, class of s);
  scalars            log1p of kept out-degree, in-degree, distinct out/in properties; plus one explicit
                     absence indicator (1 when the entity has no kept statement), so absence is typed, not zero;
  layout             signed feature hashing into HASH dims per block, log1p counts, then a unit vector.
An entity with no kept statement is recorded as absent (the assay refuses absent rows: they are counted).
Cost: two streaming passes over the triples (classes, then the entities' kept statements), O(statements).
"""
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish

HASH = 512
INVERSE = {"P2175": "P2176", "P2176": "P2175", "P527": "P361", "P361": "P527", "P150": "P131", "P131": "P150",
           "P749": "P355", "P355": "P749", "P127": "P1830", "P1830": "P127", "P737": "P738", "P738": "P737",
           "P1001": "P1001", "P279": "P279", "P31": "P31", "P1365": "P1366", "P1366": "P1365",
           "P155": "P156", "P156": "P155", "P40": "P22", "P22": "P40", "P26": "P26", "P3373": "P3373"}
SHARDS = ("wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test")
NAME = "kg-context-masked-v1"


def lines(path):
    with path.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        yield from io.TextIOWrapper(zr, encoding="utf-8")


def h(*parts):
    d = hashlib.blake2b("\x00".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(d[:6], "big") % HASH, 1.0 if d[6] & 1 else -1.0


def main():
    canon, anchor, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    summ = json.loads((anchor / "summary.json").read_bytes())
    rows = [json.loads(l) for l in (anchor / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    ents = [r["entity"] for r in rows]; idx = {e: i for i, e in enumerate(ents)}
    masked = set(summ["properties"]) | {INVERSE[p] for p in summ["properties"] if p in INVERSE} | {"P31", "P279"}
    cls = {}
    for sh in SHARDS:
        for line in lines(canon / f"{sh}.jsonl.zst"):
            if '"p": "P31"' in line:
                f = json.loads(line)["f"]
                cls.setdefault(f["s"], f["o"])
    X = np.zeros((len(ents), 4 * HASH + 5), np.float32)  # last dim: explicit absence indicator (typed, not a zero vector)
    deg = np.zeros((len(ents), 4), np.float32)
    outp, inp = [set() for _ in ents], [set() for _ in ents]
    for sh in SHARDS:
        for line in lines(canon / f"{sh}.jsonl.zst"):
            f = json.loads(line)["f"]
            s, p, o = f["s"], f["p"], f["o"]
            if p in masked:
                continue
            if s in idx:
                i = idx[s]
                j, g = h("out", p); X[i, j] += g
                j, g = h("out", p, cls.get(o, "?")); X[i, HASH + j] += g
                deg[i, 0] += 1; outp[i].add(p)
            if o in idx:
                i = idx[o]
                j, g = h("in", p); X[i, 2 * HASH + j] += g
                j, g = h("in", p, cls.get(s, "?")); X[i, 3 * HASH + j] += g
                deg[i, 1] += 1; inp[i].add(p)
    deg[:, 2] = [len(x) for x in outp]; deg[:, 3] = [len(x) for x in inp]
    X[:, : 4 * HASH] = np.sign(X[:, : 4 * HASH]) * np.log1p(np.abs(X[:, : 4 * HASH]))
    X[:, 4 * HASH:4 * HASH + 4] = np.log1p(deg)
    absent = (deg[:, 0] + deg[:, 1]) == 0
    X[absent, -1] = 1.0
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(nrm > 0, X / np.where(nrm > 0, nrm, 1), 0)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{NAME}.npy"
    if dest.exists():
        raise SystemExit(f"MNLM_PILOT_EXISTS: {dest}")
    np.save(dest, X.astype(np.float16))
    publish(out / f"manifest-{NAME}.json", {"schema": "graph-sense-v1", "name": NAME, "masked_properties": sorted(masked),
        "inverse_map": INVERSE, "hash_dims_per_block": HASH, "absent": int(absent.sum()), "rows": len(ents),
        "anchor_rows_sha256": summ["rows_sha256"], "slot_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"rows": len(ents), "absent": int(absent.sum()), "masked": len(masked),
                      "mean_kept_out": float(deg[:, 0].mean()), "mean_kept_in": float(deg[:, 1].mean())}))


if __name__ == "__main__":
    main()
