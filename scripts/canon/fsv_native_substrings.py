"""Native substring FSV: source bytes, complete group coverage, atom retention, refusals."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_population import Population, file_hash
from native_substring_readback import inspect
from native_atoms import build, open_atoms
from senses.contract import SenseError, canonical


def files(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}


def run(args):
    start = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    ph, sh = file_hash(args.population / "manifest.json"), file_hash(args.substrings / "manifest.json")
    pop = Population(args.population, ph)
    before = file_hash(args.population / "native.sqlite")
    checks, phases = [], {}
    def check(name, ok, **detail):
        checks.append({"boundary": name, "pass": bool(ok), **detail})
    t = time.perf_counter()
    m = inspect(pop, args.substrings, sh, ph)
    # Independent restatement from source JSON, not the producer's coordinate reader.
    texts = {}
    for digest, body in pop.db.execute("SELECT digest,body FROM payloads WHERE channel='surface'"):
        for i, item in enumerate(json.loads(body)["items"]):
            texts[digest, i] = item["text"].encode()
    db = sqlite3.connect((args.substrings / "substrings.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    segments = {r[0]: r[1:] for r in db.execute("SELECT id,payload,item,start,n FROM segments")}
    spans = list(db.execute("SELECT origin,start,end,target,dup_start,sha256 FROM spans ORDER BY origin,start,end,target,dup_start"))
    equal = all(texts[segments[a][:2]][s:e] == texts[segments[b][:2]][k:k+e-s]
                and hashlib.sha256(texts[segments[a][:2]][s:e]).hexdigest() == h and e-s >= 200
                for a, s, e, b, k, h in spans)
    check("all_source_spans_equal_within_items", equal and len(spans) == m["census"]["spans"], spans=len(spans))
    offsets = [segments[i][2] for i in sorted(segments)]
    corpus = (args.substrings / "corpus.bin").read_bytes()
    check("complete_ordered_source_projection", corpus == b"".join(texts[segments[i][:2]] + b"\0" for i in sorted(segments)))
    # Decode the reference tool's packed streams independently; the resolver's
    # scratch database alone cannot prove that it visited every occurrence.
    width = (args.substrings / "corpus.bin.table.bin").stat().st_size // len(corpus)
    expected_windows, excluded = [], 0
    for fp in sorted((args.substrings / "cache").glob("dups_corpus.bin_*")):
        raw = fp.read_bytes()
        sizes = fp.with_name(fp.name.replace("dups_", "sizes_", 1)).read_bytes()
        cursor = 0
        for i in range(0, len(sizes), width):
            n = int.from_bytes(sizes[i:i+width], "little")
            positions = [int.from_bytes(raw[j:j+width], "little") for j in range(cursor, cursor+n*width, width)]
            cursor += n*width
            valid = []
            for pos in positions:
                doc = bisect.bisect_right(offsets, pos)-1
                assert corpus[pos:pos+200] == corpus[positions[0]:positions[0]+200]
                if pos+200 <= offsets[doc] + segments[doc][3]:
                    valid.append((pos, doc))
                else:
                    excluded += 1
            if len(valid) >= 2:
                target, td = min(valid)
                expected_windows.extend((d, td, target-p, p, p+200) for p, d in valid if p != target)
        assert cursor == len(raw)
    # Every valid duplicate window must land in an annotation with the same source/target displacement.
    intervals = {}
    for a, s, e, b, k, _ in spans:
        delta = offsets[b] + k - offsets[a] - s
        intervals.setdefault((a, b, delta), []).append((offsets[a]+s, offsets[a]+e))
    windows = sqlite3.connect((args.substrings / "span-windows.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    retained_windows = list(windows.execute("SELECT origin,target,delta,start,end FROM windows ORDER BY origin,target,delta,start,end"))
    check("reference_groups_fully_accounted_for", sorted(expected_windows) == retained_windows
          and excluded == m["resolution"]["boundary_crossing"],
          windows=len(expected_windows), excluded_boundary_crossings=excluded)
    covered, total = 0, 0
    for a, b, delta, s, e in windows.execute("SELECT origin,target,delta,start,end FROM windows"):
        covered += any(x <= s and e <= y for x, y in intervals.get((a, b, delta), []))
        total += 1
    windows.close()
    check("all_valid_duplicate_windows_accounted_for", covered == total == m["resolution"]["windows"],
          windows=total, covered=covered, utf8_dropped=m["resolution"]["dropped_short"])
    db.close()
    phases["independent_coordinate_readback"] = time.perf_counter() - t
    t = time.perf_counter()
    receipt = build(args.population, ph, out / "atoms", near=args.near,
                    near_sha256=file_hash(args.near / "manifest.json") if args.near else None,
                    substrings=args.substrings, substrings_sha256=sh)
    atom_manifest = open_atoms(out / "atoms", receipt["manifest_sha256"])
    check("native_atom_publication_retains_annotations", atom_manifest["substring_annotations"]["spans"] == len(spans)
          and file_hash(out / "atoms/substrings.sqlite") == m["files"]["substrings.sqlite"]["sha256"])
    if args.prior_atoms:
        previous = json.loads((args.prior_atoms / "manifest.json").read_bytes())
        check("annotations_do_not_change_atoms_or_order", atom_manifest["census"] == previous["census"],
              atoms=atom_manifest["census"]["atoms"], parents=atom_manifest["census"]["parents"])
    check("all_original_observations_preserved", atom_manifest["census"]["parents"] == pop.manifest["census"]["observations"]
          and file_hash(args.population / "native.sqlite") == before)
    phases["native_atom_consumer"] = time.perf_counter() - t
    t = time.perf_counter()

    def control(name, mutate, code, reseal=False):
        target = out / name
        target.mkdir()
        for file in ("manifest.json", "substrings.sqlite"):
            shutil.copyfile(args.substrings / file, target / file)
        mutate(target)
        pin = sh
        if reseal:
            body = json.loads((target / "manifest.json").read_bytes())
            body["files"]["substrings.sqlite"] = {"sha256": file_hash(target / "substrings.sqlite"),
                                                   "bytes": (target / "substrings.sqlite").stat().st_size}
            (target / "manifest.json").write_bytes(canonical(body) + b"\n")
            pin = file_hash(target / "manifest.json")
        error = None
        try:
            inspect(pop, target, pin, ph)
        except SenseError as exc:
            error = exc.code
        check(name, error == code, code=error)

    control("missing_completion", lambda p: (p / "manifest.json").unlink(), "MNLM_NATIVE_SUBSTRING_MANIFEST")
    def corrupt(p):
        f = p / "substrings.sqlite"; raw = f.read_bytes(); f.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    control("corrupt_physical_map", corrupt, "MNLM_NATIVE_SUBSTRING_DIGEST")
    def parent(p):
        f=p / "manifest.json"; x=json.loads(f.read_bytes()); x["parent_manifest_sha256"]="f"*64; f.write_bytes(canonical(x))
    control("wrong_parent", parent, "MNLM_NATIVE_SUBSTRING_SCOPE", reseal=True)
    def schema(p):
        f = p / "manifest.json"; x = json.loads(f.read_bytes()); del x["schema"]; f.write_bytes(canonical(x))
    control("malformed_manifest", schema, "MNLM_NATIVE_SUBSTRING_INVALID", reseal=True)
    def sql(p, command):
        d=sqlite3.connect(p / "substrings.sqlite"); d.execute(command); d.commit(); d.close()
    control("cross_item_boundary", lambda p: sql(p, "UPDATE spans SET end=100000000 WHERE rowid=(SELECT min(rowid) FROM spans)"),
            "MNLM_NATIVE_SUBSTRING_COORDINATES", reseal=True)
    control("wrong_duplicate_bytes", lambda p: sql(p, "UPDATE spans SET sha256='bad' WHERE rowid=(SELECT min(rowid) FROM spans)"),
            "MNLM_NATIVE_SUBSTRING_BYTES", reseal=True)
    control("missing_item", lambda p: sql(p, "DELETE FROM segments WHERE id=(SELECT max(id) FROM segments WHERE id NOT IN (SELECT origin FROM spans) AND id NOT IN (SELECT target FROM spans))"),
            "MNLM_NATIVE_SUBSTRING_SCOPE", reseal=True)
    original = (out / "atoms/substrings.sqlite").read_bytes()
    (out / "atoms/substrings.sqlite").write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    error = None
    try:
        open_atoms(out / "atoms", receipt["manifest_sha256"])
    except SenseError as exc:
        error = exc.code
    check("atom_consumer_rejects_corrupt_annotation", error == "MNLM_NATIVE_ATOM_DIGEST", code=error)
    (out / "atoms/substrings.sqlite").write_bytes(original)
    check("source_unchanged_after_controls", file_hash(args.population / "native.sqlite") == before)
    pop.close()
    phases["negative_controls"] = time.perf_counter() - t
    elapsed = time.perf_counter() - start
    report = {"acceptance": "author-checked only", "full_scope": False, "scientific_admission": False,
              "checks": checks, "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "census": m["census"], "resolution": m["resolution"], "atom_census": atom_manifest["census"],
              "seconds": elapsed, "phases": phases, "phase_coverage": sum(phases.values()) / elapsed,
              "parent_manifest_sha256": ph, "substring_manifest_sha256": sh}
    (out / "report.json").write_bytes(canonical(report) + b"\n")
    (out / "SHA256.json").write_bytes(canonical(files(out)) + b"\n")
    print(json.dumps({k:report[k] for k in ("passed", "failed", "seconds", "phase_coverage")}))
    return int(report["failed"] > 0)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("population", "substrings", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--near", type=Path)
    ap.add_argument("--prior-atoms", type=Path)
    sys.exit(run(ap.parse_args()))
