"""Real Stage 1 -> Stage 2 -> UCR -> retained store, with publication fault controls.

Uses an existing sealed 36-record identity generation and a pinned 64-record
OpenCodeInstruct capture; no invented training data or scientific admission.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import blake3
import numpy as np
import zstandard

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
from common import Paths, iter_canon_docs
from atomise import atomise_shard, plan_jobs, prepare_tree
from atoms_publication import accepted, begin, canonical, contract, identity, sha, finish
from senses.pipeline import export_canonical, write_records, ingest_file
from senses.stage2 import export_stage2
from senses.store import Store

CAPTURE = "7ab91b1bb0adf14eec126a7297e7f9aa274308d9a8fc8da43dffbd49e0744dbe"


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def run(args):
    t0 = time.perf_counter()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    p = Paths(root / "root", root / "published", "seal-proof")
    # Reuse only the real Stage 1 tree; older atom output is deliberately not copied.
    for sub in ("canon", "sig", "dedup", "sa"):
        shutil.copytree(args.accepted / sub, p.out / sub)
    before = inventory(p.out)
    checks = []
    def check(name, ok, **detail):
        checks.append(dict(boundary=name, passed=bool(ok), **detail))
    def refusal(name, action, code, watched=None, output=None):
        old = inventory(watched) if watched else None
        error = None
        try:
            action()
        except (Exception, SystemExit) as exc:
            error = str(exc)
        ok = error is not None and code in error
        if watched:
            ok = ok and inventory(watched) == old
        if output:
            ok = ok and not output.exists() and not output.with_suffix(".manifest.json").exists()
        check(name, ok, error=error)

    # Actual CLI includes planning, worker, ledger, completion and optional UCR publication.
    output = root / "identity.ucr.jsonl"
    cmd = [sys.executable, "-I", str(Path(__file__).with_name("atomise.py")), "--root", str(p.root),
           "--out", str(p.out), "--version", p.version, "--sources", "identity-v1", "--workers", "1",
           "--ucr-output", str(output)]
    proc = subprocess.run(cmd, capture_output=True)
    (root / "producer.log").write_bytes(proc.stdout + proc.stderr)
    check("actual_producer_cli", proc.returncode == 0, returncode=proc.returncode)
    if proc.returncode:
        raise RuntimeError((proc.stdout + proc.stderr).decode(errors="replace"))
    adir = p.out / "atoms" / p.version
    seal, selected = accepted(p)
    jobs, stats = plan_jobs(p, ["identity-v1"])
    check("stage1_bytes_unchanged", all(sha(p.out / rel) == h for rel, h in before.items()))
    em = json.loads(output.with_suffix(".manifest.json").read_bytes())
    persisted = [json.loads(l) for l in output.read_bytes().splitlines()]
    # Independent span readback against the canonical text, without the production validator.
    expected_text = []
    for sh in selected:
        docs = {r["row"]: r["text"] for r in iter_canon_docs(Path(sh["canon"]).parent, sh["shard"])}
        with np.load(sh["atoms"], allow_pickle=False) as z:
            previous, end = None, 0
            for row, start, stop in zip(z["row"], z["start"], z["end"]):
                row, start, stop = int(row), int(start), int(stop)
                if row != previous:
                    assert previous is None or end == len(docs[previous])
                    end, previous = 0, row
                assert start == end < stop <= len(docs[row])
                expected_text.append(docs[row][start:stop])
                end = stop
            assert previous is None or end == len(docs[previous])
    actual_text = [r["channels"]["surface"]["items"][0]["text"] for r in persisted]
    check("canonical_atoms_export_cold_readback", expected_text == actual_text and len(persisted) == em["count"],
          atoms=em["count"], docs=sum(r["docs"] for r in json.loads((adir / "plan.json").read_bytes())["jobs"]))
    store_path = root / "records.sqlite"
    store = Store(store_path, create=True, records_only=True)
    receipt = ingest_file(store, output, em, None)
    store.close()
    store = Store(store_path)
    audit = store.audit()
    check("persisted_store_cold_reopen", audit["records"] == len(persisted) == receipt["added"]
          and all(store.get(r["atom_id"]) == r for r in persisted), records=audit["records"], slots=audit["slots"])
    store.close()

    def mutate(name, path, transform, code):
        raw = path.read_bytes()
        if transform is None:
            path.unlink()
        else:
            path.write_bytes(transform(raw))
        try:
            dest = root / (name + ".jsonl")
            refusal(name, lambda: export_stage2(p, dest, ["identity-v1"]), code, p.out, dest)
        finally:
            path.write_bytes(raw)

    mutate("missing_completion", adir / "complete.json", None, "MNLM_ATOMS_UNSEALED")
    mutate("missing_expected_plan", adir / "plan.json", None, "MNLM_ATOMS_UNSEALED")
    mutate("duplicate_manifest", adir / "manifest.jsonl", lambda b: b + b, "MNLM_ATOMS_SEAL")
    mutate("incomplete_manifest", adir / "manifest.jsonl", lambda b: b"", "MNLM_ATOMS_SEAL")
    mutate("stale_parser_manifest", adir / "manifest.jsonl",
           lambda b: canonical({**json.loads(b), "code_atomizer": "stale"}) + b"\n", "MNLM_ATOMS_SEAL")
    mutate("changed_summary", adir / "atoms.summary.json", lambda b: b + b" ", "MNLM_ATOMS_SEAL")
    mutate("missing_summary", adir / "atoms.summary.json", None, "MNLM_ATOMS_UNSEALED")
    mutate("changed_array", Path(selected[0]["atoms"]), lambda b: b[:-1] + bytes([b[-1] ^ 1]), "MNLM_ATOMS_")
    mutate("missing_array", Path(selected[0]["atoms"]), None, "MNLM_ATOMS_ARRAYS")
    mutate("changed_stage1", p.dedup / "spans.npy", lambda b: b[:-1] + bytes([b[-1] ^ 1]), "MNLM_DEDUP_BYTES_MISMATCH")
    mutate("changed_canonical", Path(selected[0]["canon"]), lambda b: b + zstandard.ZstdCompressor().compress(b""),
           "MNLM_UCR_CANON_DIGEST")
    dest = root / "missing-source.jsonl"
    refusal("missing_requested_export_source", lambda: export_stage2(p, dest, ["identity-v1", "opencodeinstruct"]),
            "MNLM_UCR_SOURCE_MISSING", p.out, dest)
    refusal("missing_requested_producer_source", lambda: plan_jobs(p, ["identity-v1", "opencodeinstruct"]),
            "MNLM_ATOMS_SOURCE_MISSING", p.out)
    refusal("existing_tree_refused", lambda: prepare_tree(adir, False), "MNLM_ATOMS_TREE_EXISTS", p.out)
    # Exercise final publication's exact expected/completed set, independently of seal hash checks.
    complete_bytes, plan_bytes = (adir / "complete.json").read_bytes(), (adir / "plan.json").read_bytes()
    (adir / "complete.json").unlink()
    changed = json.loads(plan_bytes)
    changed["jobs"].append({**changed["jobs"][0], "shard": "missing-job"})
    changed["id"] = identity({k: v for k, v in changed.items() if k != "id"})
    (adir / "plan.json").write_bytes(canonical(changed))
    refusal("missing_job_cannot_publish", lambda: finish(p), "MNLM_ATOMS_INCOMPLETE", p.out)
    check("failed_publication_has_no_complete_marker", not (adir / "complete.json").exists())
    (adir / "plan.json").write_bytes(plan_bytes)
    (adir / "complete.json").write_bytes(complete_bytes)

    # The same real 64 records used for the pre-fix reproducer, split into two jobs.
    assert sha(args.capture) == CAPTURE
    records = [json.loads(l)["record"] for l in args.capture.read_bytes().splitlines()]
    assert len(records) == 64
    proof = root / "code-proof"
    proof.mkdir()
    code_plan, code_jobs = {"shards": []}, []
    for index in range(2):
        shard, rows = f"part{index}", records[index*32:(index+1)*32]
        cp = proof / (shard + ".jsonl.zst")
        cp.write_bytes(zstandard.ZstdCompressor().compress(b"".join(canonical(r) + b"\n" for r in rows)))
        cm = dict(source="opencodeinstruct", shard=shard, out_blake3=blake3.blake3(cp.read_bytes()).hexdigest(),
                  docs=32, scope="retained-canonical-proof-repack", parent_sha256=CAPTURE)
        job = dict(source="opencodeinstruct", shard=shard, canon_dir=str(proof), out_dir=str(proof / "atoms"),
                   unique_rows=[r["row"] for r in rows], cuts={}, canon_blake3=cm["out_blake3"])
        am = atomise_shard(job)
        ap = proof / "atoms" / (shard + ".atoms.npz")
        code_plan["shards"].append(dict(source=job["source"], shard=shard, canon=str(cp), atoms=str(ap),
                                       atoms_sha256=sha(ap), manifest=cm, atom_manifest=am))
        code_jobs.append(job)
    dest = root / "code.ucr.jsonl"
    cm = write_records(dest, export_canonical(code_plan), {"kind": "bounded_stage2_proof", "capture": CAPTURE})
    check("two_real_code_shards_complete", cm["count"] == 1439, atoms=cm["count"], docs=64)
    badjob = {**code_jobs[0], "out_dir": str(root / "bad-job"), "unique_rows": code_jobs[0]["unique_rows"] + [999999999]}
    refusal("missing_selected_row", lambda: atomise_shard(badjob), "MNLM_ATOMS_ROWS_MISSING", proof)
    check("missing_row_leaves_no_atom_output", not Path(badjob["out_dir"]).exists())
    badjob = {**code_jobs[0], "unique_rows": code_jobs[0]["unique_rows"] * 2}
    refusal("duplicate_selected_row", lambda: atomise_shard(badjob), "MNLM_ATOMS_ROWS", proof)
    badjob = {**code_jobs[0], "canon_blake3": "0" * 64}
    refusal("writer_canonical_binding", lambda: atomise_shard(badjob), "MNLM_ATOMS_CANON_CHANGED", proof)
    bad = copy.deepcopy(code_plan)
    bad["shards"][0]["atom_manifest"]["code_atomizer"] = "stale"
    dest = root / "stale-lowlevel.jsonl"
    refusal("lowlevel_stale_parser", lambda: write_records(dest, export_canonical(bad), {}),
            "MNLM_ATOMS_CONTRACT", proof, dest)
    bad = copy.deepcopy(code_plan)
    bad["shards"][0]["atom_manifest"]["writer_sha256"] = "stale"
    dest = root / "stale-writer.jsonl"
    refusal("lowlevel_stale_writer", lambda: write_records(dest, export_canonical(bad), {}),
            "MNLM_ATOMS_CONTRACT", proof, dest)
    sh = copy.deepcopy(code_plan["shards"][0])
    with np.load(sh["atoms"], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["start"][0] += 1
    ap = root / "gapped.atoms.npz"
    np.savez(ap, **arrays)
    sh["atoms"], sh["atoms_sha256"] = str(ap), sha(ap)
    h = blake3.blake3()
    for k in ("row", "start", "end", "role", "unit", "cut_of"):
        h.update(arrays[k].tobytes())
    sh["atom_manifest"]["arrays_blake3"] = h.hexdigest()
    dest = root / "gapped-export.jsonl"
    refusal("bound_but_gapped_spans", lambda: write_records(dest, export_canonical({"shards": [sh]}), {}),
            "MNLM_UCR_ATOMS_CONSERVATION", proof, dest)
    # Recovery consumes restored immutable state; no repair or reseal by the reader.
    dest = root / "replay.ucr.jsonl"
    replay = export_stage2(p, dest, ["identity-v1"])
    check("restored_generation_replays_exact_bytes", replay["sha256"] == em["sha256"] and accepted(p)[0] == seal)
    report = dict(acceptance="author-checked only", scientific_admission=False, checks=checks,
        passed=sum(c["passed"] for c in checks), failed=sum(not c["passed"] for c in checks),
        seconds=time.perf_counter()-t0, producer_scope=stats, parser=contract(), capture_sha256=CAPTURE,
        artifacts=dict(identity_export=sha(output), completion=seal["id"], code_export=cm["sha256"]))
    (root / "report.json").write_bytes(canonical(report) + b"\n")
    (root / "SHA256.json").write_bytes(canonical(inventory(root)) + b"\n")
    print(json.dumps({k: report[k] for k in ("passed", "failed", "seconds")}))
    return bool(report["failed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accepted", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    sys.exit(run(ap.parse_args()))
