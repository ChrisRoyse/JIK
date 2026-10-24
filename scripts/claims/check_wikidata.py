"""Executable claim checker over the Wikidata5m triples as canonicalised in the corpus (operator review recommendation 4).

Source of truth: <out>/canon/<MNLM_CANON_VERSION>/wikidata5m/wikidata5m_transductive_{train,valid,test}.jsonl.zst, one
canonical record per triple: {"b3", "f": {"o","p","s","split"}, "kind": "row", "n", "row", "text": "Q<s>\\tP<p>\\tQ<o>"}
(scripts/canon/contracts.py `_wikidata5m_members`). The triple id of a record is `<shard>#<row>`.

Cost (PC discipline, issue #5). N = triples over the three shards (20.6 M train + ~5 k valid + ~5 k test; the build reads
the count back from the shards and cross-checks it against each shard's sealed rows.npy). Build: one streamed pass,
O(N) canonical-record parses, then one O(N log N) lexsort; held as five int arrays (~25 bytes per triple, ~0.5 GB) and cached as
.npy under the cache directory, keyed by the BLAKE3 of each shard file (a digest change rebuilds; the cache is never
trusted without that check). Per assertion: O(log N) binary search on the sorted (subject, property) key plus
O(k) over the k objects of that pair, and O(log n_shard) per cited reference. Invariant across the assertion loop: the
index; nothing re-reads a shard per assertion.

Open-world discipline: Wikidata5m carries no property constraints, so an alternate object
is unsupported, never contradicted merely because one other object appears in this subset.
Only a matching positive statement contradicts a negated claim. The measured share of multi-valued pairs per property is
still reported with every verdict. Wikidata5m holds no qualifiers, no time, no
literals and no negative statements: an assertion that depends on any of them is never `supported` here."""
from __future__ import annotations

# This corpus carries no constraints, qualifiers, validity intervals or negations.
# A hard-coded interpretation of a property is not evidence of exclusivity.
FUNCTIONAL = {}

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import blake3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lens"))
from assertion import Assertion, ClaimError, EvidenceRef, Verdict, parse_assertion  # noqa: E402
from lenscommon import Paths  # noqa: E402

SOURCE = "wikidata5m"
SHARDS = ["wikidata5m_transductive_train", "wikidata5m_transductive_valid", "wikidata5m_transductive_test"]
_REC = re.compile(rb'"row": (\d+), "text": "Q(\d+)\\tP(\d+)\\tQ(\d+)"')
P_BITS = 16
CACHE_FILES = ("key", "o", "shard", "row", "ents", "rels", "multi_p", "multi_share")


def _shard_path(paths: Paths, shard: str) -> Path:
    return paths.canon(SOURCE) / f"{shard}.jsonl.zst"


def shard_digests(paths: Paths) -> dict[str, str]:
    manifest_path = paths.canon(SOURCE) / "manifest.jsonl"
    if not manifest_path.exists():
        raise ClaimError("MNLM_CLAIM_SOURCE_UNSEALED", f"missing {manifest_path}")
    manifest = {r["shard"]: r for r in (json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip())}
    out = {}
    for sh in SHARDS:
        p = _shard_path(paths, sh)
        if not p.exists():
            raise ClaimError("MNLM_CLAIM_SHARD_MISSING", f"{p} not found (MNLM_CANON_VERSION={paths.canon_version})")
        h = blake3.blake3(max_threads=blake3.blake3.AUTO)
        h.update_mmap(str(p))
        out[sh] = h.hexdigest()
        if sh not in manifest or manifest[sh].get("out_blake3") != out[sh] or manifest[sh].get("out_bytes") != p.stat().st_size:
            raise ClaimError("MNLM_CLAIM_SOURCE_UNSEALED", f"{sh}: bytes differ from canonical manifest; revalidate source")
        from claim_cache import file_hash
        sidecar = paths.canon(SOURCE) / f"{sh}.rows.npy"
        if not sidecar.is_file():
            raise ClaimError("MNLM_CLAIM_SOURCE_UNSEALED", f"missing {sidecar}")
        out[f"{sh}.rows"] = file_hash(sidecar)
    out["manifest"] = file_hash(manifest_path)
    return out


def _parse_shard(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rows, s, p, o of one canonical shard; any record the pattern does not parse refuses (no silent drop)."""
    import zstandard
    from array import array

    rows, s, p, o = array("q"), array("q"), array("q"), array("q")
    bad = []
    with path.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr:
        for line in io.BufferedReader(zr, buffer_size=1 << 22):
            m = _REC.search(line)
            if m is None:
                if line.strip():
                    bad.append(line[:160])
                continue
            rec = json.loads(line)
            text = f"Q{int(m.group(2))}\tP{int(m.group(3))}\tQ{int(m.group(4))}"
            fields = rec.get("f", {})
            if (rec.get("row") != int(m.group(1)) or rec.get("text") != text
                    or [fields.get(k) for k in ("s", "p", "o")] != text.split("\t")
                    or rec.get("b3") != blake3.blake3(text.encode()).digest(length=16).hex()):
                raise ClaimError("MNLM_CLAIM_RECORD_MISMATCH", f"{path.name}: canonical fields/text/digest disagree")
            rows.append(int(m.group(1)))
            s.append(int(m.group(2)))
            p.append(int(m.group(3)))
            o.append(int(m.group(4)))
    if bad:
        raise ClaimError("MNLM_CLAIM_INDEX_UNPARSED", f"{path.name}: {len(bad)} records not of the form Q\\tP\\tQ, first {bad[:2]!r}")
    r = np.frombuffer(rows, np.int64).copy()
    if len(r) and np.any(np.diff(r) <= 0):
        raise ClaimError("MNLM_CLAIM_INDEX_ROW_ORDER", f"{path.name}: rows are not strictly increasing")
    return r, np.frombuffer(s, np.int64).copy(), np.frombuffer(p, np.int64).copy(), np.frombuffer(o, np.int64).copy()


class WikidataIndex:
    """The sorted (subject<<16 | property, object) index plus per-shard row tables for resolving cited references."""

    def __init__(self, arrays: dict, per_shard: dict, meta: dict):
        self.key, self.o, self.shard, self.row = arrays["key"], arrays["o"], arrays["shard"], arrays["row"]
        self.ents, self.rels = arrays["ents"], arrays["rels"]
        self.multi_share = dict(zip(arrays["multi_p"].tolist(), arrays["multi_share"].tolist()))
        self.per_shard = per_shard  # shard -> (rows, s, p, o)
        self.meta = meta
        self.generation = blake3.blake3(json.dumps(meta["shard_blake3"], sort_keys=True).encode()).hexdigest()

    # ---------------------------------------------------------------- lookups, all O(log N)
    def objects(self, s: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if s < 1 or s >= (1 << (63 - P_BITS)) or p < 1 or p >= (1 << P_BITS):
            return self.o[:0], self.shard[:0], self.row[:0]
        k = (s << P_BITS) | p
        lo, hi = np.searchsorted(self.key, k, "left"), np.searchsorted(self.key, k, "right")
        return self.o[lo:hi], self.shard[lo:hi], self.row[lo:hi]

    def has_entity(self, q: int) -> bool:
        i = np.searchsorted(self.ents, q)
        return bool(i < len(self.ents) and self.ents[i] == q)

    def has_relation(self, p: int) -> bool:
        i = np.searchsorted(self.rels, p)
        return bool(i < len(self.rels) and self.rels[i] == p)

    def resolve(self, ref: EvidenceRef) -> tuple[int, int, int] | None:
        """The (s, p, o) at a cited shard#row, or None when the reference does not resolve."""
        if ref.source != SOURCE:
            raise ClaimError("MNLM_CLAIM_FOREIGN_SOURCE", f"evidence cites source {ref.source!r}; this checker reads {SOURCE} only")
        t = self.per_shard.get(ref.shard)
        if t is None:
            return None
        rows, s, p, o = t
        i = np.searchsorted(rows, ref.row)
        if i >= len(rows) or rows[i] != ref.row:
            return None
        return int(s[i]), int(p[i]), int(o[i])


def build_index(paths: Paths, cache: Path) -> WikidataIndex:
    t0 = time.time()
    digests = shard_digests(paths)
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "meta.json"
    from claim_cache import load_cache, write_cache
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("shard_blake3") == digests:
            arrays, per = load_cache(cache, meta, CACHE_FILES, SHARDS)
            meta["loaded_from_cache_secs"] = round(time.time() - t0, 2)
            return WikidataIndex(arrays, per, meta)
    per, counts = {}, {}
    for sh in SHARDS:
        r, s, p, o = _parse_shard(_shard_path(paths, sh))
        sealed_rows = np.load(paths.canon(SOURCE) / f"{sh}.rows.npy", allow_pickle=False)
        if not np.array_equal(sealed_rows, r):
            raise ClaimError("MNLM_CLAIM_INDEX_COUNT", f"{sh}: parsed row identities differ from rows.npy")
        if len(p) and int(p.max()) >= (1 << P_BITS):
            raise ClaimError("MNLM_CLAIM_INDEX_PROPERTY_RANGE", f"{sh}: property id {int(p.max())} exceeds {P_BITS} bits")
        per[sh], counts[sh] = (r, s, p, o), len(r)
    t_parse = time.time() - t0
    s = np.concatenate([per[sh][1] for sh in SHARDS])
    p = np.concatenate([per[sh][2] for sh in SHARDS])
    o = np.concatenate([per[sh][3] for sh in SHARDS])
    shard = np.concatenate([np.full(counts[sh], i, np.int8) for i, sh in enumerate(SHARDS)])
    row = np.concatenate([per[sh][0] for sh in SHARDS]).astype(np.int32)
    key = (s << P_BITS) | p
    order = np.lexsort((o, key))
    key, o, shard, row = key[order], o[order], shard[order], row[order]
    del order
    ents = np.unique(np.concatenate([s, o]))
    rels = np.unique(p)
    del s, p
    # distinct (key, o) pairs, then per key the number of distinct objects; per property the share of keys with > 1
    first = np.ones(len(key), bool)
    first[1:] = (key[1:] != key[:-1]) | (o[1:] != o[:-1])
    dkey = key[first]
    ukey, nobj = np.unique(dkey, return_counts=True)
    kp = ukey & ((1 << P_BITS) - 1)
    tot = np.bincount(kp, minlength=1 << P_BITS)
    multi = np.bincount(kp, weights=(nobj > 1).astype(np.float64), minlength=1 << P_BITS)
    multi_p = np.flatnonzero(tot)
    multi_share = multi[multi_p] / tot[multi_p]
    arrays = {"key": key, "o": o, "shard": shard, "row": row, "ents": ents, "rels": rels, "multi_p": multi_p, "multi_share": multi_share}
    meta = {"source": SOURCE, "canon_dir": str(paths.canon(SOURCE)), "shards": SHARDS, "shard_blake3": digests, "triples_per_shard": counts,
            "triples": int(sum(counts.values())), "distinct_sp_pairs": int(len(ukey)), "distinct_triples": int(first.sum()),
            "entities": int(len(ents)), "relations": int(len(rels)), "parse_secs": round(t_parse, 1), "build_secs": round(time.time() - t0, 1)}
    write_cache(cache, arrays, per, meta)
    return WikidataIndex(arrays, per, meta)


def _qid(x: str) -> int | None:
    return int(x[1:]) if re.fullmatch(r"Q[1-9]\d*", x) else None


def check(a: Assertion, idx: WikidataIndex) -> Verdict:
    """One typed assertion against the source. Order: cited references, then conditions the source cannot carry, then
    the triple itself. Never `supported` without a source row holding (s, p, o)."""
    a = parse_assertion(a.to_json())  # direct dataclass callers must obey the same schema
    s, p, o = int(a.subject[1:]), int(a.relation[1:]), _qid(a.object)
    cited, all_hold = [], bool(a.evidence)
    for ref in a.evidence:
        t = idx.resolve(ref)
        holds = t is not None and o is not None and t == (s, p, o)
        cited.append({"triple_id": ref.triple_id, "resolves": t is not None, "holds": holds,
                      "found": None if t is None else f"Q{t[0]}\tP{t[1]}\tQ{t[2]}"})
        all_hold &= holds
    objs, shs, rws = idx.objects(s, p)
    distinct = sorted(set(objs.tolist()))
    rows_hold = [f"{SHARDS[int(shs[i])]}#{int(rws[i])}" for i in range(len(objs)) if o is not None and int(objs[i]) == o]
    cond = {"world": "open; no exclusivity constraints available", "functional": False, "source_generation": idx.generation,
            "property_multi_valued_share": idx.multi_share.get(p), "subject_in_source": idx.has_entity(s), "relation_in_source": idx.has_relation(p),
            "object_in_source": None if o is None else idx.has_entity(o)}

    def v(outcome: str, reason: str) -> Verdict:
        return Verdict(outcome, a.to_json(), reason, cited, rows_hold, [f"Q{x}" for x in distinct[:50]], cond)

    broken = any(not c["resolves"] for c in cited)
    if broken:
        return v("evidence_mismatch", "CITED_REF_BROKEN")
    if a.qualifiers or a.time is not None:
        if broken or (a.evidence and not all_hold and a.polarity == "asserted"):
            return v("evidence_mismatch", "CITED_REF_BROKEN_OR_LACKS_TRIPLE")
        return v("unsupported_missing", "SOURCE_HAS_NO_QUALIFIERS_OR_TIME")
    if o is None:
        return v("evidence_mismatch", "CITED_REF_BROKEN") if broken else v("unsupported_missing", "SOURCE_HOLDS_NO_LITERALS")
    if a.polarity == "negated":
        if rows_hold:
            return v("contradicted", "SOURCE_HOLDS_NEGATED_TRIPLE")
        if broken:
            return v("evidence_mismatch", "CITED_REF_BROKEN")
        return v("unsupported_missing", "OPEN_WORLD_SOURCE_HOLDS_NO_NEGATIVES")
    if all_hold:
        return v("supported", "CITED_ROWS_HOLD_TRIPLE")
    if a.evidence:
        return v("evidence_mismatch", "CITED_REF_BROKEN" if broken else "CITED_ROW_LACKS_TRIPLE")
    if rows_hold:
        return v("supported", "CHECKER_FOUND_ROW_FOR_HYPOTHESIS")
    if distinct:
        return v("unsupported_missing", "SOURCE_HOLDS_OTHER_OBJECTS_NOT_FUNCTIONAL")
    if not cond["subject_in_source"]:
        return v("unsupported_missing", "SUBJECT_ABSENT")
    if not cond["relation_in_source"]:
        return v("unsupported_missing", "RELATION_ABSENT")
    return v("unsupported_missing", "SUBJECT_HAS_NO_STATEMENT_OF_RELATION")


def default_paths() -> Paths:
    os.environ.setdefault("MNLM_CANON_VERSION", "p0micro-v1")
    return Paths(root=Path("/zfs/archive/mnlm"), out=Path("/zfs/hot/mnlm"), version="p0-v1")


def main() -> int:
    ap = argparse.ArgumentParser(description="check typed assertions (JSONL) against the canonical Wikidata5m triples")
    ap.add_argument("--assertions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path.home() / "diag" / "claims" / "cache")
    args = ap.parse_args()
    idx = build_index(default_paths(), args.cache)
    n = 0
    tmp = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        with args.assertions.open(encoding="utf-8") as fi, tmp.open("w", encoding="utf-8", newline="\n") as fo:
            for line in fi:
                if line.strip():
                    fo.write(json.dumps(check(parse_assertion(json.loads(line)), idx).to_json(), sort_keys=True) + "\n")
                    n += 1
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(tmp, args.out)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps({"ok": True, "assertions": n, "out": str(args.out), "index": idx.meta}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ClaimError as e:
        print(json.dumps({"ok": False, "code": e.code, "message": str(e)}), file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"ok": False, "code": "MNLM_CLAIM_INPUT_INVALID", "message": str(e)}), file=sys.stderr)
        sys.exit(2)
