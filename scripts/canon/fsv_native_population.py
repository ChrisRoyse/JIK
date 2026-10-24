"""Full approved native-population proof and corruption/publication controls (#20)."""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_population import Population, build, census, file_hash, POLICY_SHA256
from senses.contract import canonical, SenseError


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    plan = {"schema": "native-canon-plan-v1", "training_scope_sha256": POLICY_SHA256,
            "source_rows": {}, "exports": []}
    original = {}
    input_counts = collections.Counter()
    payloads = {c: set() for c in ("surface", "fields", "relations", "history", "provenance",
                                  "co_reference", "structure", "behaviour", "perception")}
    for name in ("trajectories.ucr.jsonl", "audio.ucr.jsonl"):
        p, mp = source / name, (source / name).with_suffix(".manifest.json")
        m = json.loads(mp.read_bytes())
        assert file_hash(p) == m["sha256"]
        plan["exports"].append({"name": name, "path": str(p), "manifest": str(mp), "sha256": m["sha256"],
                                "manifest_sha256": file_hash(mp), "count": m["count"], "bytes": p.stat().st_size})
        n = 0
        with p.open("rb") as f:
            for line in f:
                r = json.loads(line)
                original[r["atom_id"]] = hashlib.sha256(canonical(r)).hexdigest()
                input_counts[r["locator"]["source"]] += 1
                for c, value in r["channels"].items():
                    payloads[c].add(hashlib.sha256(canonical(value)).hexdigest())
                n += 1
        assert n == m["count"]
    plan["source_rows"] = dict(input_counts)
    planfile = output / "plan.json"
    planfile.write_bytes(canonical(plan) + b"\n")
    cmd = [sys.executable, "-I", str(Path(__file__).with_name("native_population.py")),
           "--plan", str(planfile), "--sha256", file_hash(planfile), "--output", str(output / "canonical")]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    (output / "build.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    checks = []

    def check(name, ok, **details):
        checks.append({"boundary": name, "pass": bool(ok), **details})

    def refuses(name, call, code):
        actual = None
        try:
            call()
        except SenseError as exc:
            actual = exc.code
        check(name, actual == code, code=actual)

    root = output / "canonical"
    mh = file_hash(root / "manifest.json")
    pop = Population(root, mh)
    audit = pop.audit()
    check("complete_source_denominators", audit["observations"] == len(original)
          and audit["sources"] == dict(input_counts), census=audit)
    check("typed_payload_dedup_matches_independent_counts",
          {c: v["count"] for c, v in audit["payloads"].items()} == {c: len(s) for c, s in payloads.items()})
    # Independent SQL reconstruction: bypass production Population.records/census.
    db = sqlite3.connect((root / "native.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    readback = {}
    for atom, header in db.execute("SELECT atom,header FROM observations ORDER BY atom"):
        ch = {c: json.loads(b) for c, b in db.execute(
            "SELECT c.channel,p.body FROM channels c JOIN payloads p ON p.channel=c.channel AND p.digest=c.digest WHERE c.atom=?",
            (atom,))}
        r = {**json.loads(header), "atom_id": atom, "channels": ch}
        readback[atom] = hashlib.sha256(canonical(r)).hexdigest()
    check("every_source_observation_reconstructed_exactly", readback == original)
    collisions = db.execute("SELECT s.digest FROM channels s JOIN channels b ON s.atom=b.atom "
                            "WHERE s.channel='surface' AND b.channel='behaviour' GROUP BY s.digest "
                            "HAVING count(DISTINCT b.digest)>1").fetchall()
    check("same_text_different_outcomes_preserved", len(collisions) == 212, groups=len(collisions))
    check("audio_without_surface_is_preserved", db.execute("SELECT count(*) FROM observations WHERE source='web-audio-mpj'").fetchone()[0] == 2)
    db.close()
    pop.close()
    before = file_hash(root / "native.sqlite")
    pop = Population(root, mh)
    r = next(pop.records())
    r["channels"]["structure"]["items"].clear()
    check("reader_caller_mutation_cannot_change_retained_state", pop.audit() == audit)
    pop.close()
    check("readback_preserves_database_bytes", before == file_hash(root / "native.sqlite"))
    duplicate = copy.deepcopy(plan)
    duplicate["exports"].append({**duplicate["exports"][0], "name": "repeat-trajectory-export"})
    duplicate["source_rows"]["swe-agent-trajectories"] *= 2
    repeated = build(duplicate, output / "duplicate-input")
    check("global_exact_aliases_keep_all_provenance_occurrences", repeated["census"]["exact_observation_aliases"] == 4142
          and repeated["census"]["observations"] == audit["observations"]
          and repeated["census"]["records_sha256"] == audit["records_sha256"]
          and repeated["census"]["payloads"] == audit["payloads"])
    excluded = copy.deepcopy(plan)
    excluded["source_rows"]["cuyahoga-eighth-district"] = 1
    refuses("excluded_source_refuses_before_output", lambda: build(excluded, output / "excluded"), "MNLM_SOURCE_EXCLUDED")
    check("excluded_output_absent", not (output / "excluded").exists())
    wrong = copy.deepcopy(plan)
    wrong["source_rows"]["swe-agent-trajectories"] -= 1
    refuses("partial_scope_cannot_publish", lambda: build(wrong, output / "wrong-scope"), "MNLM_NATIVE_INPUT_SCOPE")
    check("partial_scope_has_no_accepted_manifest", not (output / "wrong-scope" / "manifest.json").exists())
    crash = subprocess.run([sys.executable, "-I", str(Path(__file__)), "--crash-plan", str(planfile),
                            "--output", str(output / "crash")], capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    check("actual_process_death_at_publication_barrier", crash.returncode == 77
          and (output / "crash" / "native.sqlite").is_file())
    refuses("unaccepted_physical_database_is_not_readable", lambda: Population(output / "crash", "0" * 64), "MNLM_NATIVE_MANIFEST")
    refuses("no_overwrite_of_interrupted_generation", lambda: build(plan, output / "crash"), "MNLM_NATIVE_OUTPUT_EXISTS")
    for name, sql, expected in (
        ("changed_scalar", "UPDATE payloads SET body=replace(CAST(body AS TEXT),'sequence_index','sequence_indeX') WHERE channel='fields'",
         "MNLM_NATIVE_PAYLOAD_DIGEST"),
        ("missing_channel", "DELETE FROM channels WHERE rowid=(SELECT min(rowid) FROM channels)", "MNLM_NATIVE_CHANNEL_MISSING"),
    ):
        dest = output / name
        shutil.copytree(root, dest)
        db = sqlite3.connect(dest / "native.sqlite")
        db.execute(sql)
        db.commit()
        db.close()
        refuses(name + "_outer_binding", lambda d=dest: Population(d, mh), "MNLM_NATIVE_DATABASE_DIGEST")
        # Deliberately repin the corrupted container to test internal source
        # identities too; this is a negative control, never a trusted publication.
        manifest = json.loads((dest / "manifest.json").read_bytes())
        manifest["database"] = {"sha256": file_hash(dest / "native.sqlite"), "bytes": (dest / "native.sqlite").stat().st_size}
        (dest / "manifest.json").write_bytes(canonical(manifest) + b"\n")
        p = Population(dest, file_hash(dest / "manifest.json"))
        try:
            refuses(name + "_internal_identity", p.audit, expected)
        finally:
            p.close()
    manifest = json.loads((root / "manifest.json").read_bytes())
    check("bounded_storage_proof_cannot_claim_stage_completion", manifest["stage1_complete"] is False
          and manifest["stage2_complete"] is False and manifest["scientific_admission"] is False)
    report = {"acceptance": "author-checked", "scope": "all approved bounded native exports", "checks": checks,
              "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "seconds": time.perf_counter() - start, "population_manifest_sha256": mh,
              "source_unchanged": all(file_hash(e["path"]) == e["sha256"] for e in plan["exports"])}
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    hashes = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "SHA256.json").write_bytes(canonical(hashes) + b"\n")
    print(json.dumps({k: report[k] for k in ("passed", "failed", "seconds", "population_manifest_sha256")}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--crash-plan", type=Path)
    args = ap.parse_args()
    if args.crash_plan:
        build(json.loads(args.crash_plan.read_bytes()), args.output, fault=lambda _: os._exit(77))
    else:
        assert args.input
        sys.exit(run(args.input, args.output))
