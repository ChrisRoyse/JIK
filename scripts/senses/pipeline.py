"""Stage 2 UCR export, transactional ingestion and physical readback (#15).

Run with python -I scripts/senses/pipeline.py --help. Inputs are always named,
digest-pinned files. No recursive discovery and no implicit whole-corpus scans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canon"))
from atomise import AtomError
from senses.adapters import atom, native, prepare
from senses.contract import SenseError, canonical, digest, require
from senses.store import Store


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_bound(path, sha):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for line in f:
            h.update(line)
            require(bool(line.strip()), "MNLM_UCR_INPUT", "blank line in bound JSONL")
            yield json.loads(line)
    require(h.hexdigest() == sha, "MNLM_UCR_INPUT_DIGEST", str(path))


def write_records(path, records, binding):
    path = Path(path)
    require(not path.exists() and not path.with_suffix(".manifest.json").exists(), "MNLM_UCR_OUTPUT_EXISTS", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    h, n = hashlib.sha256(), 0
    try:
        with tmp.open("xb") as f:
            for r in records:
                data = canonical(native(r)) + b"\n"
                f.write(data)
                h.update(data)
                n += 1
            f.flush()
            os.fsync(f.fileno())
        code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        manifest = {"schema": "ucr-export-v1", "sha256": h.hexdigest(), "count": n, "input": binding, "code_sha256": code}
        # Exclusive publication: another writer cannot be silently overwritten
        # between the initial existence check and this acceptance boundary.
        os.link(tmp, path)
        with path.with_suffix(".manifest.json").open("xb") as mf:
            mf.write(canonical(manifest) + b"\n")
            mf.flush()
            os.fsync(mf.fileno())
        return manifest
    finally:
        tmp.unlink(missing_ok=True)


def export_canonical(plan):
    """Read the accepted Stage 2 spans, not a fresh segmentation of the corpus.

Cost O(compressed shard bytes + selected atoms), metadata resident once per shard.
Plan explicitly names canonical and atom files plus SHA256s and manifest records.
"""
    import zstandard
    import io
    import blake3
    from atoms_publication import check_contract, check_arrays
    class BoundReader(io.RawIOBase):
        def __init__(self, raw):
            self.raw, self.b3, self.sha = raw, blake3.blake3(), hashlib.sha256()

        def readable(self):
            return True

        def readinto(self, buffer):
            n = self.raw.readinto(buffer)
            self.b3.update(memoryview(buffer)[:n])
            self.sha.update(memoryview(buffer)[:n])
            return n

    for sh in plan["shards"]:
        require(file_hash(sh["atoms"]) == sh["atoms_sha256"], "MNLM_UCR_INPUT_DIGEST", sh["atoms"])
        am = sh["atom_manifest"]
        check_contract(am)
        arrays = check_arrays(sh["atoms"], am)
        rows, starts, ends, roles = (arrays[k] for k in ("row", "start", "end", "role"))
        require(am.get("source") == sh["source"] and am.get("shard") == sh["shard"] and not am.get("error"),
                "MNLM_UCR_ATOMS_DIGEST", "Stage 2 manifest does not bind these spans")
        require(len(rows) == len(starts) == len(ends) == len(roles) and bool((rows[1:] >= rows[:-1]).all()), "MNLM_UCR_ATOMS", "atom columns/row order")
        i, previous = 0, -1
        from senses.referents import ReferentMap
        refs = ReferentMap(sh["referents"], sh["referents_digest"]) if "referents" in sh else None
        with Path(sh["canon"]).open("rb") as f:
            bound = BoundReader(f)
            with zstandard.ZstdDecompressor().stream_reader(bound, closefd=False) as zr:
                for line in io.TextIOWrapper(zr, encoding="utf-8"):
                    rec = json.loads(line)
                    require(type(rec["row"]) is int and rec["row"] > previous, "MNLM_UCR_ATOMS", "canonical row order")
                    previous = rec["row"]
                    if i == len(rows):
                        continue  # finish the bound stream before publication
                    require(rec["row"] <= rows[i], "MNLM_UCR_ATOMS", "atom row absent from canonical stream")
                    if rec["row"] != rows[i]:
                        continue
                    p = prepare({"source": sh["source"], "shard": sh["shard"], "record": rec, "source_manifest": sh["manifest"]}, sh.get("adapter"))
                    if refs:
                        p = refs.apply(p)
                    end = 0
                    while i < len(rows) and rows[i] == rec["row"]:
                        require(int(starts[i]) == end and end < int(ends[i]) <= len(rec["text"]),
                                "MNLM_UCR_ATOMS_CONSERVATION", "atom spans must tile the retained text")
                        end = int(ends[i])
                        yield atom(p, int(starts[i]), int(ends[i]), int(roles[i]))
                        i += 1
                    require(end == len(rec["text"]), "MNLM_UCR_ATOMS_CONSERVATION", "atom spans omit retained text")
            require(bound.b3.hexdigest() == sh["manifest"]["out_blake3"], "MNLM_UCR_CANON_DIGEST", sh["canon"])
            if sh.get("canon_sha256"):
                require(bound.sha.hexdigest() == sh["canon_sha256"], "MNLM_UCR_INPUT_DIGEST", sh["canon"])
        require(i == len(rows), "MNLM_UCR_ATOMS", "canonical stream ended before all atoms")


def ingest_file(store, path, manifest, expected_head):
    require(manifest.get("schema") == "ucr-export-v1", "MNLM_UCR_MANIFEST", "export manifest required")
    return store.ingest(read_bound(path, manifest["sha256"]),
                        {"input_sha256": manifest["sha256"], "count": manifest["count"], "export": manifest}, expected_head)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    adapt = sub.add_parser("adapt", help="adapt a digest-pinned canonical proof envelope file")
    adapt.add_argument("--input", required=True)
    adapt.add_argument("--sha256", required=True)
    adapt.add_argument("--adapter", help="declarative adapter JSON for a new source")
    adapt.add_argument("--output", required=True)
    export = sub.add_parser("export", help="export an explicit plan of accepted Stage 2 atom shards")
    export.add_argument("--plan", required=True)
    export.add_argument("--sha256", required=True)
    export.add_argument("--output", required=True)
    stage = sub.add_parser("stage2", help="export existing accepted Stage 2 source shards")
    stage.add_argument("--root", required=True)
    stage.add_argument("--out", required=True)
    stage.add_argument("--version", required=True)
    stage.add_argument("--sources", required=True)
    stage.add_argument("--output", required=True)
    ingest = sub.add_parser("ingest", help="retain native UCR; optionally run the diagnostic instruments")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--manifest", required=True)
    ingest.add_argument("--store", required=True)
    ingest.add_argument("--records-only", action="store_true", default=None,
                        help="create a source store without lens slots; never counts as measurement")
    ingest.add_argument("--expected-head", required=True, help="current generation hash, or empty for first ingestion")
    audit = sub.add_parser("audit", help="physically reread and replay every accepted channel and slot")
    audit.add_argument("--store", required=True)
    args = ap.parse_args()
    if args.command == "adapt":
        mapping = json.loads(Path(args.adapter).read_text()) if args.adapter else None
        records = (atom(prepare(e, mapping)) for e in read_bound(args.input, args.sha256))
        out = write_records(args.output, records, {"kind": "bounded_canonical_record_views", "sha256": args.sha256, "adapter": mapping})
    elif args.command == "export":
        require(file_hash(args.plan) == args.sha256, "MNLM_UCR_INPUT_DIGEST", args.plan)
        out = write_records(args.output, export_canonical(json.loads(Path(args.plan).read_text())),
                            {"kind": "explicit_atom_shards", "complete_stage2_claim": False, "plan_sha256": args.sha256})
    elif args.command == "stage2":
        from senses.stage2 import export_stage2
        from common import Paths
        out = export_stage2(Paths(args.root, args.out, args.version), args.output, args.sources.split(","))
    else:
        store = Store(args.store, create=args.command == "ingest",
                      records_only=getattr(args, "records_only", None))
        try:
            out = store.audit() if args.command == "audit" else ingest_file(store, args.input, json.loads(Path(args.manifest).read_text()), args.expected_head or None)
        finally:
            store.close()
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SenseError, AtomError) as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)
