"""Boundary probes on a small EXTRACTED REAL canonical subset (#12).

Real subset provenance comes from full_source_readback.py after its complete
source replay. Mutations below are explicit adversarial controls, never data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import blake3
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.full_source_readback import cold_publications
from senses.trajectory_population import publish

PRODUCER = Path(__file__).parents[1]/"text/restore_anchor_text.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, data):
    path.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    def check(name, ok):
        results.append({"name": name, "pass": bool(ok)})
    def run(name, src=a.fixture, manifest_sha=None):
        dest = a.output/name
        cp = subprocess.run([sys.executable, "-I", str(PRODUCER), "--canon", str(src/"canon"),
                             "--records", str(src/"records"), "--chunked", str(src/"chunked"),
                             "--output", str(dest), "--canonical-manifest-sha256",
                             manifest_sha or sha(src/"canon/manifest.jsonl")], capture_output=True, text=True, encoding="utf-8")
        (a.output/(name+".log")).write_text(cp.stdout+cp.stderr, encoding="utf-8")
        report = json.loads((dest/"report.json").read_bytes()) if (dest/"report.json").exists() else None
        return cp, dest, report
    def clone(name):
        dest = a.output/(name+"-input")
        shutil.copytree(a.fixture, dest)
        return dest
    def rebind_source(root, body):
        compressed = zstandard.ZstdCompressor().compress(body)
        (root/"canon/wikidata5m_text.jsonl.zst").write_bytes(compressed)
        p = root/"canon/manifest.jsonl"
        m = json.loads(p.read_bytes())
        m.update(docs=len(body.splitlines()), out_bytes=len(compressed), out_blake3=blake3.blake3(compressed).hexdigest())
        put(p, m)
    before = {str(p.relative_to(a.fixture)): sha(p) for p in a.fixture.rglob("*") if p.is_file()}
    cp, actual, report = run("real")
    check("B1 real CLI completes", cp.returncode == 0 and report["complete"])
    _, expected, _, stats = cold_publications(actual, a.fixture/"records", a.fixture/"chunked")
    check("B2 original labels/order and every published span read back", True)
    # Independent subset read: actual canonical bodies must equal all restored text.
    raw = zstandard.ZstdDecompressor().decompress((a.fixture/"canon/wikidata5m_text.jsonl.zst").read_bytes())
    canon = [json.loads(l) for l in raw.split(b"\n") if l]
    check("B2 full canonical observations equal restored records",
          all(expected[r["f"]["entity_id"]][0] == r["text"] for r in canon) and len(canon) == len(expected))
    check("B2 real recovered tails exercised", any(len(t)>1500 for t, _ in expected.values()))
    check("B2 real Unicode line separator retained", any("\u2028" in t or "\u2029" in t for t, _ in expected.values()))
    cp, _, _ = run("real")
    check("B3 existing publication refuses", cp.returncode != 0 and "MNLM_RESTORE_EXISTS" in cp.stderr)
    cp, _, _ = run("wrong-manifest", manifest_sha="0"*64)
    check("B1 wrong pinned manifest refuses", cp.returncode != 0 and "MNLM_RESTORE_MANIFEST_HASH" in cp.stderr)
    bad = clone("wrong-stream")
    manifest = json.loads((bad/"canon/manifest.jsonl").read_bytes())
    manifest["out_blake3"] = "0"*64
    put(bad/"canon/manifest.jsonl", manifest)
    cp, _, rep = run("wrong-stream", bad)
    check("B1 changed compressed stream digest refuses before commit",
          cp.returncode != 0 and "MNLM_ATOMS_CANON_CHANGED" in cp.stderr and rep is None)
    family = "biomedical"
    for name, edit, code in (
        ("missing-entity", lambda lines: lines[1:], "MNLM_RESTORE_MISSING_ENTITY"),
        ("duplicate-entity", lambda lines: lines+[lines[0]], "MNLM_RESTORE_DUPLICATE_SOURCE")):
        bad = clone(name)
        lines = [l for l in raw.split(b"\n") if l]
        q = json.loads(lines[0])["f"]["entity_id"]
        affected = next(f.name for f in (bad/"records").iterdir()
                        if q in {json.loads(l)["entity"] for l in (f/"rows.jsonl").read_text(encoding="utf-8").split("\n") if l})
        rebind_source(bad, b"\n".join(edit(lines))+b"\n")
        cp, _, rep = run(name, bad)
        check("B1 "+name+" refuses affected family; valid siblings persist",
              cp.returncode != 0 and not rep["complete"] and code in rep["errors"][affected]
              and len(rep["families"]) == 4)
    bad = clone("changed-old-text")
    rp = bad/"records"/family/"rows.jsonl"
    rows = [json.loads(l) for l in rp.read_text(encoding="utf-8").split("\n") if l]
    rows[0]["text"] += " invalid old text"
    rp.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    for p, key in ((bad/"records"/family/"summary.json", "rows_sha256"),
                   (bad/"chunked"/family/"summary.json", "parent_rows_sha256")):
        j = json.loads(p.read_bytes()); j[key] = sha(rp); put(p, j)
    cp, _, rep = run("changed-old-text", bad)
    check("B1 bound but non-source historical description refuses",
          cp.returncode != 0 and "MNLM_RESTORE_NOT_HISTORICAL_CUT" in rep["errors"][family]
          and len(rep["families"]) == 4)
    corrupt = a.output/"changed-label-output"
    shutil.copytree(actual, corrupt)
    rp = corrupt/"records"/family/"rows.jsonl"
    rows = [json.loads(l) for l in rp.read_text(encoding="utf-8").split("\n") if l]
    key = next(iter(rows[0]["y"]))
    rows[0]["y"][key] += 1
    rp.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    s = json.loads((corrupt/"records"/family/"summary.json").read_bytes()); s["rows_sha256"] = sha(rp)
    put(corrupt/"records"/family/"summary.json", s)
    r = json.loads((corrupt/"report.json").read_bytes()); r["families"][family]["rows_sha256"] = sha(rp)
    put(corrupt/"report.json", r)
    try:
        cold_publications(corrupt, a.fixture/"records", a.fixture/"chunked")
        ok = False
    except ValueError as exc:
        ok = "MNLM_FULL_READBACK_LABELS" in str(exc)
    check("B3 altered label with recomputed hashes fails cold readback", ok)
    check("B3 fixture source bytes unchanged", all(sha(a.fixture/name) == h for name,h in before.items()))
    out = {"schema": "fsv-full-source-v1", "checks": results, "passed": sum(x["pass"] for x in results),
           "total": len(results), "real_counts": stats, "fixture_files": before,
           "acceptance": "author-checked", "production_scope": "real extracted subset; full-source coverage requires separate full readback"}
    publish(a.output/"fsv.json", out)
    print(json.dumps({k:out[k] for k in ("passed","total","real_counts")}))
    if out["passed"] != out["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
