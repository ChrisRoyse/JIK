"""Real retained slots -> publication -> crash/restart/cold read; engineering only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.publication import Publication, sha, POLICY_SHA256, validate_plan
from senses.contract import canonical, digest, SenseError
from production.gate import require_admitted
from lenscommon import LensError


def build(registry):
    reg = json.loads(registry.read_bytes())
    plan = {"schema": "native-publication-plan-v1", "purpose": "engineering_boundary", "storage": "raw-f32",
            "training_scope_sha256": POLICY_SHA256, "blob_budget_bytes": 500_000,
            "registry_sha256": sha(registry), "blobs": {}, "shards": {}}
    inputs = {}

    def add(path):
        h = sha(path)
        inputs[h], plan["blobs"][h] = str(path), path.stat().st_size
        return h

    for name, entry in reg.items():
        for family, rec in entry["families"].items():
            path = Path(rec["slot"])
            rows = path.parent.parent.parent / "cal" / (family + ".jsonl")
            key = "native-boundary/" + family
            shard = plan["shards"].setdefault(key, {"key": key, "tier": "native-boundary",
                                                   "n": rec["n"], "rows_sha256": add(rows), "slots": {}})
            shard["slots"][name] = {"lens_id": entry["lens_id"], "contract": entry["contract"],
                                    "seal": {k: v for k, v in rec.items() if k != "slot"},
                                    "sha256": add(path), "views_sha256": add(path.with_suffix(".views.json"))}
    return plan, inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crash", action="store_true")
    args = ap.parse_args()
    root = Path(args.out)
    if args.crash:
        plan = json.loads((root / "plan.json").read_bytes())
        inputs = json.loads((root / "inputs.json").read_bytes())
        store = Publication(root / "store", plan)
        # Real process death at SQLite's acceptance barrier, no production hook.
        store.db.set_trace_callback(lambda sql: os._exit(91) if sql == "COMMIT" else None)
        store.publish(sorted(plan["shards"])[1], inputs)
        raise AssertionError("crash boundary not reached")
    root.mkdir(parents=True, exist_ok=False)
    start, checks = time.monotonic(), []
    plan, inputs = build(Path(args.registry))
    for name, obj in (("plan.json", plan), ("inputs.json", inputs)):
        (root / name).write_bytes(canonical(obj))
    sources = {p: sha(p) for p in set(inputs.values())}

    def check(name, ok, **detail):
        checks.append({"name": name, "pass": bool(ok), **detail})
        (root / "checks.json").write_bytes(canonical(checks))
        if not ok:
            raise AssertionError(name)

    def refuses(name, fn, expected):
        try:
            fn()
        except (SenseError, LensError, RuntimeError, sqlite3.DatabaseError) as exc:
            code = getattr(exc, "code", str(exc).split(":")[0])
            check(name, code == expected, code=code)
            return
        check(name, False)

    small = copy.deepcopy(plan)
    small["blob_budget_bytes"] = sum(plan["blobs"].values()) - 1
    refuses("budget_before_output", lambda: Publication(root / "overbudget", small), "MNLM_MEASURE_STORAGE_BUDGET")
    check("no_budget_output_created", not (root / "overbudget").exists())
    refuses("malformed_plan_typed", lambda: validate_plan([]), "MNLM_MEASURE_PLAN")
    malformed = copy.deepcopy(plan)
    malformed["shards"][next(iter(malformed["shards"]))]["slots"] = ["flattened"]
    refuses("flattened_slot_manifest_refuses", lambda: validate_plan(malformed), "MNLM_MEASURE_PLAN")
    # Existing frozen artifact is real; this command must not trust its old claims.
    old = Path(__file__).resolve().parents[2] / "config" / "panel-p0-v10.json"
    gate_root = root / "old-panel"
    gate_root.mkdir()
    (gate_root / "panel.json").write_bytes(old.read_bytes())
    refuses("historical_panel_unadmitted", lambda: require_admitted(gate_root, root / "missing-ledger"),
            "MNLM_STAGE4_PANEL_UNADMITTED")
    production = copy.deepcopy(plan)
    production["purpose"] = "production"
    refuses("production_before_output", lambda: Publication(root / "production", production, admission_root=gate_root,
                                                            ledger=root / "missing-ledger"), "MNLM_STAGE4_PANEL_UNADMITTED")
    check("no_production_output_created", not (root / "production").exists())
    store = Publication(root / "store", plan)
    first, second = sorted(plan["shards"])
    before = store.state()
    result = store.publish(first, inputs)
    check("atomic_shard_plus_event", before["shards"] == 0 and result["after"]["shards"] == 1
          and result["after"]["seq"] == 1 and result["rows"] == 16 and result["slots"] == 4
          and result["stage4_accepted"] is False, before=before, after=result["after"])
    replay = store.publish(first, inputs)
    check("repeat_reuses_exact_publication", replay["reused"] and replay["before"] == replay["after"] == result["after"])
    refuses("immutable_manifest", lambda: store.db.execute("UPDATE shards SET hash='bad'"), "MNLM_MEASURE_IMMUTABLE")
    store.db.rollback()
    other = Publication(root / "store", plan)
    store.db.execute("BEGIN IMMEDIATE")
    refuses("competing_writer", lambda: other.publish(second, inputs), "MNLM_MEASURE_WRITER_BUSY")
    store.db.rollback()
    other.close()
    store.close()
    crashed = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--out", str(root), "--crash"],
                             capture_output=True, text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    check("process_died_at_commit", crashed.returncode == 91, exit_code=crashed.returncode, stderr=crashed.stderr)
    cold = Publication(root / "store")
    check("crash_did_not_publish", cold.state() == result["after"], state=cold.state())
    refuses("uncommitted_shard_invisible", lambda: cold.read(second), "MNLM_MEASURE_NOT_PUBLISHED")
    uncommitted_h = next(iter(plan["shards"][second]["slots"].values()))["sha256"]
    check("durable_orphan_is_not_acceptance", cold.check_blob(uncommitted_h).is_file())
    cold.close()
    store = Publication(root / "store", plan)
    restarted = store.publish(second, inputs)
    check("restart_accepts_once", restarted["after"]["shards"] == 2 and restarted["after"]["seq"] == 2)
    store.close()
    cold = Publication(root / "store")
    for key in sorted(plan["shards"]):
        body, stats = cold.read(key)
        check("cold_scope_" + key, body == plan["shards"][key] and stats["stage4_accepted"] is False)
        for name, slot in body["slots"].items():
            path = cold.blob(slot["sha256"])
            with np.load(path, allow_pickle=False) as saved, np.load(inputs[slot["sha256"]], allow_pickle=False) as original:
                check("native_replay_" + key + "/" + name,
                      saved.files == original.files and all(saved[k].dtype == original[k].dtype
                      and saved[k].shape == original[k].shape and saved[k].tobytes() == original[k].tobytes() for k in saved.files))
    cold.close()
    refuses("boundary_cannot_be_opened_as_production", lambda: Publication(root / "store", production=True),
            "MNLM_STAGE4_BOUNDARY_ONLY")
    altered = copy.deepcopy(plan)
    altered["registry_sha256"] = "0" * 64
    refuses("plan_drift", lambda: Publication(root / "store", altered), "MNLM_MEASURE_PLAN_DRIFT")
    cold = Publication(root / "store")
    target = cold.blob(uncommitted_h)
    original = target.read_bytes()
    changed = bytearray(original)
    changed[-1] ^= 1
    target.write_bytes(changed)  # explicit corruption control of proof-owned copy
    try:
        refuses("corrupted_blob_refuses", lambda: cold.read(second), "MNLM_MEASURE_BLOB_CHANGED")
    finally:
        target.write_bytes(original)
    check("corruption_did_not_rewrite_chain", cold.state() == restarted["after"])
    cold.close()
    # Rebind a real slot to wrong contract in a separate negative-control store.
    wrong = copy.deepcopy(plan)
    next(iter(wrong["shards"][first]["slots"].values()))["contract"]["norm"] = "wrong"
    rejected = Publication(root / "wrong-binding", wrong)
    state = rejected.state()
    refuses("slot_contract_binding", lambda: rejected.publish(first, inputs), "MNLM_MEASURE_BINDING")
    check("binding_failure_atomic", rejected.state() == state)
    rejected.close()
    # Independent SQLite/JSON walk, without Publication.verify().
    db = sqlite3.connect((root / "store" / "index.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    parent, n = None, 0
    for seq, h, raw in db.execute("SELECT seq,hash,body FROM events ORDER BY seq"):
        obj = json.loads(raw)
        hashed = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        check(f"independent_chain_{seq}", hashed == h and obj["parent"] == parent and seq == n + 1
              and obj["before_count"] == n and obj["after_count"] == seq)
        parent, n = h, seq
    db.close()
    check("source_artifacts_unchanged", sources == {p: sha(p) for p in sources})
    files = [p for base in (root, root / "store", root / "store" / "blobs", root / "wrong-binding", root / "wrong-binding" / "blobs", gate_root)
             for p in base.iterdir() if p.is_file() and not p.name.endswith(("-shm", "-wal"))]
    report = {"schema": "publication-fsv-v1", "checks": checks, "author_checked": all(c["pass"] for c in checks),
              "independently_accepted": False, "stage4_accepted": False, "purpose": "engineering_boundary",
              "plan_sha256": digest(plan), "source_before": sources, "source_after": {p: sha(p) for p in sources},
              "content_blob_bytes_measured": sum(plan["blobs"].values()), "blob_budget_bytes": plan["blob_budget_bytes"],
              "code_sha256": {n: sha(Path(__file__).parent / n) for n in ("publication.py", "gate.py", "fsv_publication.py")},
              "store_file_bytes_measured": {p.name: p.stat().st_size for p in (root / "store").iterdir() if p.is_file()},
              "wall_seconds_measured": time.monotonic() - start, "retention_owner": "ChrisRoyse/MNLM#16"}
    (root / "report.json").write_bytes(canonical(report))
    files.append(root / "report.json")
    (root / "SHA256.json").write_bytes(canonical({str(p.relative_to(root)): sha(p) for p in files}))
    print(json.dumps({"checks": len(checks), "all_pass": report["author_checked"], "stage4_accepted": False,
                      "bytes": report["content_blob_bytes_measured"], "report": str(root / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
