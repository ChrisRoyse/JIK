"""Full native surface-index readback and typed annotation refusal controls."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_population import Population, file_hash
from native_annotations import retain
from minhash import shingle_hashes, signature
from senses.contract import canonical, SenseError


def run(popdir, mh, output):
    output.mkdir(parents=True, exist_ok=False)
    root = output / "near"
    p = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("native_near.py")),
                        "--population", str(popdir), "--sha256", mh, "--output", str(root)], capture_output=True,
                       text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    (output / "build.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    checks = []

    def check(name, ok):
        checks.append({"boundary": name, "pass": bool(ok)})

    def refuses(name, fn, expected):
        code = None
        try:
            fn()
        except SenseError as e:
            code = e.code
        check(name, code == expected)

    pop = Population(popdir, mh)
    before = file_hash(popdir / "native.sqlite")
    nm = file_hash(root / "manifest.json")
    dest = output / "retained"
    dest.mkdir()
    bound = retain(pop, root, nm, dest, mh)
    check("native_consumer_retains_exact_annotations", file_hash(dest / "near.sqlite") == file_hash(root / "near.sqlite")
          and bound["observations_removed"] == 0)
    db = sqlite3.connect((root / "near.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    values = {ph: (sg, valid) for ph, sg, valid in db.execute("SELECT digest,signature,valid FROM signatures")}
    actual = 0
    same = True
    for ph, raw in pop.db.execute("SELECT digest,body FROM payloads WHERE channel='surface'"):
        payload = json.loads(raw)
        shingles = shingle_hashes("\n".join(item["text"] for item in payload["items"]))
        sg, valid = values[ph]
        same &= sg == signature(shingles).astype('<u4').tobytes() and valid == bool(len(shingles))
        actual += 1
    check("every_signature_rederived_from_retained_payload", same and actual == len(values) == 2865)
    check("near_links_are_annotations_not_row_deletions", bound["links"] == 325 and pop.audit()["observations"] == 4146)
    check("unobserved_surface_cannot_have_near_links", not db.execute(
        "SELECT 1 FROM links l JOIN signatures s ON s.digest=l.origin OR s.digest=l.target WHERE s.valid=0 LIMIT 1").fetchone())
    db.close()
    refuses("wrong_parent_generation_refuses", lambda: retain(pop, root, nm, dest, '0' * 64), "MNLM_NATIVE_NEAR_SCOPE")
    corrupt = output / "corrupt"
    shutil.copytree(root, corrupt)
    db = sqlite3.connect(corrupt / "near.sqlite")
    db.execute("UPDATE links SET agreement=0 WHERE rowid=(SELECT min(rowid) FROM links)")
    db.commit()
    db.close()
    refuses("corrupt_annotation_file_refuses", lambda: retain(pop, corrupt, nm, dest, mh), "MNLM_NATIVE_NEAR_DIGEST")
    m = json.loads((corrupt / "manifest.json").read_bytes())
    m["database_sha256"] = file_hash(corrupt / "near.sqlite")
    (corrupt / "manifest.json").write_bytes(canonical(m) + b"\n")
    refuses("repinned_wrong_similarity_refuses", lambda: retain(pop, corrupt, file_hash(corrupt / "manifest.json"), dest, mh),
            "MNLM_NATIVE_NEAR_IDENTITY")
    check("source_population_unchanged", file_hash(popdir / "native.sqlite") == before)
    pop.close()
    report = {"acceptance": "author-checked", "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks), "near_manifest_sha256": nm}
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    (output / "SHA256.json").write_bytes(canonical({str(p.relative_to(output)): file_hash(p)
            for p in sorted(output.rglob('*')) if p.is_file()}) + b"\n")
    print(json.dumps({k: report[k] for k in ("passed", "failed", "near_manifest_sha256")}))
    return int(report["failed"] != 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--population', type=Path, required=True)
    ap.add_argument('--sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sys.exit(run(args.population, args.sha256, args.output))
