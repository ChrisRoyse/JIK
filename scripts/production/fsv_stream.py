"""Physical streaming proof over retained, operator-approved observations (#16).

Runs the real CLI, kills a process after durable checkpoints, reopens outputs
independently, and exercises rejection boundaries. No synthetic source records.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import canonical, digest, CHANNELS
from production.input_plan import sha, code_hashes, environment, POLICY_SHA256
from production.stream import run, paths
from lenscommon import LEASE


def historical_scope(store, plan, policies, rows, out, check):
    from senses.store import Store
    from senses.pipeline import ingest_file
    from senses.contract import SenseError
    from universal import ChannelReader
    source = Store(store)
    historical = out / "historical.sqlite"
    retained = Store(historical, create=True, records_only=True)
    batches = []
    for i in range(2):
        rs = [json.loads(line) for line in (rows / f"deterministic-{i}.jsonl").read_bytes().splitlines()]
        path = out / f"historical-input-{i}.jsonl"
        path.write_bytes(b"".join(canonical(source.get(r["atom"], plan["generation"])) + b"\n" for r in rs))
        manifest = {"schema": "ucr-export-v1", "sha256": sha(path), "count": 4,
                    "source_generation": plan["generation"], "selected_atoms": [r["atom"] for r in rs]}
        path.with_suffix(".manifest.json").write_bytes(canonical(manifest))
        batches.append((path, manifest, rs))
    source.close()
    ingest_file(retained, batches[0][0], batches[0][1], None)
    old = retained.head
    p = copy.deepcopy(plan)
    p["generation"] = old
    p["shards"] = {"deterministic-0": copy.deepcopy(plan["shards"]["deterministic-0"])}
    p["shards"]["deterministic-0"]["instruments"] = ["ucr-fields-v1"]
    p["instruments"] = {"ucr-fields-v1": p["instruments"]["ucr-fields-v1"]}
    first = run(p, historical, policies, rows, out / "historical-job")
    ingest_file(retained, batches[1][0], batches[1][1], old)
    latest = retained.head
    retained.close()
    second = run(p, historical, policies, rows, out / "historical-job")
    check("historical_plan_survives_continual_append", old != latest and first["after"] == second["after"]
          and second["checkpoints_reused"] == 1 and second["loaded_instruments"] == [],
          old_generation=old, new_generation=latest)
    reader = ChannelReader(historical, policies, generation=old)
    code = None
    try:
        reader.read(batches[1][2][0], reader.policy("stream-boundary"))
    except SenseError as exc:
        code = exc.code
    finally:
        reader.close()
    check("future_observation_absent_from_frozen_scope", code == "MNLM_UCR_LOCATOR_AMBIGUOUS", error=code)
    (out / "historical.receipt.json").write_bytes(canonical({"plan": p, "before_append": first, "after_append": second}))


def prepare(store, blobs, out):
    from universal import specs
    from catalogue import by_name
    from learned import snapshot_dir, weights_sha256
    from audio_encoder import SPEC
    db = sqlite3.connect(Path(store).resolve().as_uri() + "?mode=ro", uri=True)
    records, units = [], set()
    try:
        census = dict(db.execute("SELECT source,count(*) FROM records GROUP BY source"))
        generation = db.execute("SELECT hash FROM generations ORDER BY seq DESC LIMIT 1").fetchone()[0]
        for raw, in db.execute("SELECT body FROM records ORDER BY atom"):
            record = json.loads(raw)
            text = "\n".join(i["text"] for i in record["channels"]["surface"]["items"])
            if record["locator"]["source"] == "swe-agent-trajectories":
                if len(units) >= 8 or record["unit"] in units or not 10 < len(text) <= 2000:
                    continue
                units.add(record["unit"])
            elif record["locator"]["source"] not in ("web-audio-mpj", "ebible-english"):
                continue
            records.append({**record["locator"], "atom": record["atom_id"], "text": text})
    finally:
        db.close()
    assert len(records) == 12 and len(units) == 8, "minimum proof scope changed; investigate before using it"
    rows_root = out / "rows"
    rows_root.mkdir()
    policy = {"schema": "ucr-view-v1", "question": "stream-boundary", "allow_channels": list(CHANNELS),
              "deny_fields": [], "deny_relations": [], "blocked_units": [], "label_aliases": [],
              "rationale": "Retention/inference boundary only; no label fitting or information admission."}
    policies = out / "policies.json"
    policies.write_bytes(canonical({"stream-boundary": policy}))
    instruments = {s["name"]: {"spec": s, "weights_sha256": s["weights_sha256"], "tier": "deterministic",
                                "slot_budget_bytes": 1 << 20} for s in specs() if s["channel"] != "perception"}
    # Raw acoustic perception replaces the redundant frame-feature hash lane in
    # this minimum proof. Its 6,224 retained frame tokens exceeded the 1 MiB
    # diagnostic reservation in run2; the quota and total budget stay unchanged.
    for name, tier, cap in (("ucr-all-minilm-l6-v2", "cheap-semantic", 1 << 20),
                            ("ucr-opensearch-sparse-doc-v2", "panel", 1 << 20),
                            ("ucr-answerai-colbert-small", "multi", 2 << 20),
                            (SPEC["name"], "multi", 12 << 20)):
        spec = SPEC if name == SPEC["name"] else by_name(name)
        snapshot = snapshot_dir(spec["repo"])
        instruments[name] = {"spec": spec, "weights_sha256": weights_sha256(snapshot)[0], "revision": snapshot.name,
                             "tier": tier, "slot_budget_bytes": cap}
    plan = {"schema": "tier-input-plan-v1", "purpose": "engineering_boundary", "storage": "raw-f32",
            "training_scope_sha256": POLICY_SHA256, "generation": generation, "policies_sha256": sha(policies),
            "blobs_manifest_sha256": sha(blobs), "code_sha256": code_hashes(), "environment": environment(),
            "batch_rows": 2, "slot_budget_bytes": 16 << 20, "metadata_budget_bytes": 1 << 20,
            "blob_budget_bytes": 256 << 20, "work_budget_bytes": 1 << 30, "instruments": instruments, "shards": {}}
    for tier in sorted({i["tier"] for i in instruments.values()}):
        for n in range(3):
            key = f"{tier}-{n}"
            rows = [{**r, "i": i} for i, r in enumerate(records[n * 4:(n + 1) * 4])]
            path = rows_root / (key + ".jsonl")
            path.write_bytes(b"".join(canonical(r) + b"\n" for r in rows))
            plan["shards"][key] = {"n": len(rows), "rows_sha256": sha(path), "rows_bytes": path.stat().st_size,
                                    "family": "stream-boundary", "tier": tier,
                                    "instruments": sorted(nm for nm, i in instruments.items() if i["tier"] == tier)}
    (out / "plan.json").write_bytes(canonical(plan))
    (out / "input-census.json").write_bytes(canonical({"census": census, "selected": records, "generation": generation}))
    return plan, policies, rows_root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--blobs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--crash-child", action="store_true")
    a = ap.parse_args()
    out = a.out.resolve()
    if a.crash_child:
        run(json.loads((out / "plan.json").read_bytes()), a.store, out / "policies.json", out / "rows", out / "job",
            blobs_path=a.blobs, after_native=lambda n: os._exit(94) if n == 2 else None)
        raise RuntimeError("crash boundary did not run")
    out.mkdir(parents=True, exist_ok=False)
    plan, policies, rows = prepare(a.store, a.blobs, out)
    checks, started = [], time.monotonic()

    def check(name, ok, **evidence):
        checks.append({"name": name, "pass": bool(ok), **evidence})
        print(json.dumps(checks[-1]), flush=True)

    cli_script = Path(__file__).with_name("stream.py")

    def cli(label, chosen=plan, output=None, row_dir=rows, policy_path=policies):
        pp, receipt = out / (label + ".plan.json"), out / (label + ".receipt.json")
        pp.write_bytes(canonical(chosen))
        argv = [sys.executable, "-I", str(cli_script), "--plan", str(pp), "--store", str(a.store),
                "--policies", str(policy_path), "--rows", str(row_dir), "--out", str(output or out / "job"),
                "--blobs", str(a.blobs), "--receipt", str(receipt)]
        result = subprocess.run(argv, capture_output=True)
        (out / (label + ".stdout")).write_bytes(result.stdout)
        (out / (label + ".stderr")).write_bytes(result.stderr)
        report = json.loads(receipt.read_bytes()) if receipt.exists() else None
        return result, report

    def refused(label, chosen, code, **kwargs):
        target = out / label
        result, _ = cli(label, chosen, target, **kwargs)
        check(label, result.returncode == 2 and code.encode() in result.stdout and not target.exists(),
              exit=result.returncode, expected=code)

    bad = copy.deepcopy(plan)
    bad["purpose"] = "production"
    refused("unadmitted_before_output", bad, "MNLM_STAGE4_ADMISSION_MISSING")
    bad = copy.deepcopy(plan)
    bad["work_budget_bytes"] = 1
    refused("unreserved_before_output", bad, "MNLM_MEASURE_STORAGE_BUDGET")
    bad = copy.deepcopy(plan)
    bad["generation"] = "0" * 64
    refused("stale_generation", bad, "MNLM_UCR_NOT_IN_GENERATION")
    bad = copy.deepcopy(plan)
    bad["shards"]["duplicate"] = copy.deepcopy(next(iter(bad["shards"].values())))
    shutil.copyfile(rows / (next(iter(plan["shards"])) + ".jsonl"), rows / "duplicate.jsonl")
    refused("duplicate_work", bad, "MNLM_STREAM_DUPLICATE")
    bad = copy.deepcopy(plan)
    bad["code_sha256"]["production/stream.py"] = "0" * 64
    refused("changed_code", bad, "MNLM_STREAM_CODE_DRIFT")
    # Physical process death after two fsynced, cold-audited checkpoints.
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "--store", str(a.store),
               "--blobs", str(a.blobs), "--out", str(out), "--crash-child"]
    crash = subprocess.run(command, capture_output=True)
    (out / "crash.stdout").write_bytes(crash.stdout)
    (out / "crash.stderr").write_bytes(crash.stderr)
    cps = list((out / "job" / "slots").glob("*/*.checkpoint.json"))
    check("process_death_durable_checkpoints", crash.returncode == 94 and len(cps) == 2
          and not (out / "job" / "publication").exists(), exit=crash.returncode, checkpoints=len(cps))
    old = {str(p): sha(p) for p in cps}
    result, first = cli("resume")
    check("resume_cli", result.returncode == 0, exit=result.returncode)
    if first is None:
        raise RuntimeError(f"stream failed; inspect {out / 'resume.stderr'}")
    check("resume_reuses_complete_work", first["checkpoints_reused"] == 2 and first["checkpoints_computed"] == 34
          and all(sha(p) == h for p, h in old.items()), receipt=first)
    check("unaccepted_native_files_recovered", len(first["recovered_unaccepted_files"]) == 2)
    result, second = cli("cached")
    check("cached_no_model_no_gpu_no_mutation", result.returncode == 0 and second["loaded_instruments"] == []
          and second["gpu_peak_allocated_bytes"] == 0 and second["checkpoints_reused"] == 36
          and second["before"] == second["after"] == first["after"])
    check("gpu_lease_released", not LEASE.exists())
    # Independent storage readback, no Publication/slot_digest reader.
    root = out / "job" / "publication"
    db = sqlite3.connect((root / "index.sqlite").as_uri() + "?mode=ro", uri=True)
    native_kinds, bytes_ok, shapes_ok, absence_ok, exact_atoms = set(), True, True, True, True
    import numpy as np
    from senses.contract import digest as chain_digest
    previous, seq = None, 0
    for sn, sh, body in db.execute("SELECT seq,hash,body FROM events ORDER BY seq"):
        event = json.loads(body)
        check("physical_chain_" + str(sn), sn == seq + 1 and chain_digest(event) == sh and event["parent"] == previous)
        previous, seq = sh, sn
    for key, body in db.execute("SELECT key,body FROM shards ORDER BY key"):
        shard = json.loads(body)
        source = (root / "blobs" / shard["rows_sha256"]).read_bytes()
        declared = [json.loads(r) for r in source.splitlines()]
        exact_atoms &= source == (rows / (key + ".jsonl")).read_bytes()
        for name, slot in shard["slots"].items():
            path = root / "blobs" / slot["sha256"]
            views = json.loads((root / "blobs" / slot["views_sha256"]).read_bytes())
            bytes_ok &= sha(path) == slot["sha256"]
            exact_atoms &= [b["atom"] for b in views["bindings"]] == [r["atom"] for r in declared]
            with np.load(path, allow_pickle=False) as z:
                keys = set(z.files)
                kind = "multi" if "tokens" in keys else "sparse" if "indptr" in keys else "dense"
                native_kinds.add(kind)
                shapes_ok &= z["absent"].shape == (4,) and z["absent"].dtype == bool
                values = z["tokens"] if kind == "multi" else z["data"] if kind == "sparse" else z["X"]
                shapes_ok &= values.dtype == np.float32 and bool(np.isfinite(values).all())
                if name == "ucr-whisper-tiny-en-audio":
                    absence_ok &= z["absent"].tolist() == [r["source"] != "web-audio-mpj" for r in declared]
    db.close()
    check("physical_native_bytes_shapes", bytes_ok and shapes_ok and native_kinds == {"dense", "sparse", "multi"})
    check("physical_source_bindings_and_audio_absence", exact_atoms and absence_ok)
    check("engineering_cannot_claim_stage4", first["stage4_accepted"] is False and first["served"] is False)
    check("phase_coverage", first["timing_coverage"] >= .90 and second["timing_coverage"] >= .90)
    # Corruption is not a cache miss: a completed slot must fail, with no repair.
    corrupt = out / "corrupt-job"
    shutil.copytree(out / "job", corrupt)
    victim = next((corrupt / "slots").glob("*/*.npz"))
    with victim.open("r+b") as f:
        f.seek(-1, 2)
        f.write(b"!")
    before = sha(victim)
    result, _ = cli("corrupt", output=corrupt)
    check("corrupt_complete_checkpoint_refused_unchanged", result.returncode != 0 and sha(victim) == before
          and b"MNLM_STREAM_CHECKPOINT" in result.stdout)
    historical_scope(a.store, plan, policies, rows, out, check)
    report = {"acceptance": "author-checked", "stage4_accepted": False, "checks": checks,
              "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "wall_seconds": time.monotonic() - started, "plan_sha256": digest(plan)}
    (out / "report.json").write_bytes(canonical(report))
    files = {str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.rglob("*"))
             if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
    (out / "SHA256.json").write_bytes(canonical(files))
    print(json.dumps({"passed": report["passed"], "failed": report["failed"]}), flush=True)
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
