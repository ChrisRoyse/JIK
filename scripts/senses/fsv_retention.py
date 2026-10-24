"""Cold, complete retention audit of approved native exports, without inference.

PC-04/07/35: O(serialized bytes + N log N), two bounded exports, no root scan.
Negative-control stores are separate from the retained source store. Author check
only: this is not scientific lens admission, global dedup or continual training.
"""
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
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from senses.contract import CHANNELS, SenseError, canonical, digest, seal_record
from senses.pipeline import file_hash, ingest_file, read_bound
from senses.store import Store, records_contract


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checks, timings, inputs = [], {}, []

    def check(name, ok, **evidence):
        checks.append({"boundary": name, "pass": bool(ok), **evidence})
        if not ok:
            raise AssertionError(name)

    def refuses(name, call, code):
        try:
            call()
        except SenseError as exc:
            check(name, exc.code == code, error=exc.envelope())
        else:
            check(name, False, expected=code)

    def cli(name, *argv):
        # -S proves retention and cold auditing need no site packages/models.
        command = [sys.executable, "-I", "-S", str(Path(__file__).with_name("pipeline.py")), *map(str, argv)]
        at = time.perf_counter()
        result = subprocess.run(command, capture_output=True)
        timings[name] = time.perf_counter() - at
        (args.output / (name + ".stdout.json")).write_bytes(result.stdout)
        (args.output / (name + ".stderr.txt")).write_bytes(result.stderr)
        check(name + "_cli", result.returncode == 0, returncode=result.returncode, command=command)
        return json.loads(result.stdout)

    path = args.output / "observations.sqlite"
    head, generations = None, []
    for filename in ("trajectories.ucr.jsonl", "audio.ucr.jsonl"):
        source = args.exports / filename
        mp = source.with_suffix(".manifest.json")
        manifest = json.loads(mp.read_bytes())
        inputs.append({"path": str(source), "bytes": source.stat().st_size,
                       "manifest_sha256": file_hash(mp), "manifest": manifest})
        result = cli(filename.split(".")[0], "ingest", "--records-only", "--store", path,
                     "--input", source, "--manifest", mp, "--expected-head", head or "")
        check(filename + "_generation", result["parent"] == head and result["added"] == manifest["count"]
              and result["duplicates"] == 0, before=head, after=result["generation"], added=result["added"])
        head = result["generation"]
        generations.append(head)
    physical = cli("cold-audit", "audit", "--store", path)
    expected = sum(x["manifest"]["count"] for x in inputs)
    check("retention_not_measurement", physical["records"] == expected and physical["slots"] == 0
          and physical["records_only"] and not physical["measured"] and not physical["served"], audit=physical)

    # Independent SQL + raw JSONL byte comparison, not Store.get or adapter output.
    at = time.perf_counter()
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    samples, units, channel_counts, compared = [], {}, dict.fromkeys(CHANNELS, 0), 0
    for source in inputs:
        sha = hashlib.sha256()
        with Path(source["path"]).open("rb") as stream:
            count = 0
            for line in stream:
                sha.update(line)
                record = json.loads(line)
                row = db.execute("SELECT body,source,unit FROM records WHERE atom=?", (record["atom_id"],)).fetchone()
                check_ok = row is not None and row[0] + b"\n" == line and row[1] == record["locator"]["source"] and row[2] == record["unit"]
                if not check_ok:
                    check("independent_byte_conservation", False, atom=record["atom_id"])
                kind = record["locator"]["source"]
                units.setdefault(kind, set()).add(record["unit"])
                for channel in CHANNELS:
                    channel_counts[channel] += record["channels"][channel]["state"] == "present"
                if count == 0 or source["manifest"]["count"] == 4:
                    samples.append(record)
                count += 1
        check(Path(source["path"]).name + "_full_byte_conservation",
              count == source["manifest"]["count"] and sha.hexdigest() == source["manifest"]["sha256"], count=count, sha256=sha.hexdigest())
        compared += count
    check("no_unexpected_rows", db.execute("SELECT count(*) FROM records").fetchone()[0] == compared)
    check("no_orphan_slots", db.execute("SELECT count(*) FROM slots").fetchone()[0] == 0)
    check("physical_contract", json.loads(db.execute("SELECT body FROM panel").fetchone()[0]) == records_contract())
    db.close()
    timings["independent_readback"] = time.perf_counter() - at

    at = time.perf_counter()
    before = file_hash(path)
    refuses("mode_mismatch_before_write", lambda: Store(path, create=True, records_only=False), "MNLM_UCR_STORE_MODE")
    check("mode_mismatch_unchanged_bytes", file_hash(path) == before, before=before, after=file_hash(path))
    with patch("senses.instruments.roster", side_effect=AssertionError("instrument runtime accessed")), \
         patch("senses.instruments.measure", side_effect=AssertionError("unexpected inference")):
        retained = Store(path)
        check("runtime_independent_cold_audit", retained.audit() == physical)
        refuses("unmeasured_slot_refusal", lambda: retained.read_slot(samples[0]["atom_id"], "any"), "MNLM_UCR_UNMEASURED")
        old = retained.get(samples[0]["atom_id"], generations[0])
        check("old_generation_still_readable", canonical(old) == canonical(samples[0]))
        refuses("new_source_not_in_old_generation", lambda: retained.get(samples[-1]["atom_id"], generations[0]), "MNLM_UCR_NOT_IN_GENERATION")
        retained.close()

    # Exercise the actual measurement consumer with channel-preserving policies.
    from lens.universal import ChannelReader
    policy = {"schema": "ucr-view-v1", "question": "retention-audit", "allow_channels": list(CHANNELS),
              "deny_fields": [], "deny_relations": [], "blocked_units": [], "label_aliases": [],
              "rationale": "Input retention proof only; no outcome or admission assay."}
    pp = args.output / "retention-policy.json"
    pp.write_bytes(canonical({"retention-audit": policy}))
    reader = ChannelReader(path, pp)
    for record in samples:
        surface = "\n".join(item["text"] for item in record["channels"]["surface"]["items"])
        view, _, binding = reader.read({**record["locator"], "text": surface}, reader.policy("retention-audit"))
        check("consumer_" + record["atom_id"], view["channels"] == record["channels"] and reader.generation == head,
              source=record["locator"]["source"], binding=binding)
    reader.close()
    from senses.views import masked
    denied = {**policy, "allow_channels": []}
    check("policy_exclusion_distinct_from_missing", all(c["reason"] == "question_policy_excluded" for c in masked(samples[0], denied)["channels"].values()))
    redacted = {**policy, "deny_fields": [x["name"] for x in samples[0]["channels"]["fields"]["items"]]}
    check("policy_redaction_distinct_from_missing", masked(samples[0], redacted)["channels"]["fields"]["reason"] == "question_policy_redacted")

    controls = args.output / "negative-controls.sqlite"
    store = Store(controls, create=True, records_only=True)
    first_input = inputs[0]
    bad = {**first_input["manifest"], "sha256": "0" * 64}
    refuses("wrong_digest_rolls_back_full_export", lambda: ingest_file(store, first_input["path"], bad, None), "MNLM_UCR_INPUT_DIGEST")
    check("no_prefix_after_wrong_digest", store.head is None and store.audit()["records"] == 0)
    bad = {**first_input["manifest"], "count": first_input["manifest"]["count"] + 1}
    refuses("wrong_count_rolls_back_full_export", lambda: ingest_file(store, first_input["path"], bad, None), "MNLM_UCR_SCOPE")
    check("no_prefix_after_wrong_count", store.head is None and store.audit()["records"] == 0)
    excluded = copy.deepcopy(samples[0])
    excluded["locator"]["source"] = "cuyahoga-eighth-district"
    seal_record(excluded)
    refuses("cuyahoga_exclusion", lambda: store.ingest([excluded], {"input_sha256": digest(excluded), "count": 1}, None), "MNLM_SOURCE_EXCLUDED")
    check("excluded_source_no_state", store.head is None and store.audit()["records"] == 0)
    one = {"input_sha256": digest(samples[0]), "count": 1, "kind": "retention-control"}
    store.ingest([samples[0]], one, None)
    prior = store.head
    refuses("stale_head", lambda: store.ingest([samples[0]], one, None), "MNLM_UCR_STALE_HEAD")
    check("stale_head_unchanged", store.head == prior)
    store.ingest([samples[0]], one, prior)
    check("idempotent_record_retention", store.audit()["records"] == 1 and store.audit()["slots"] == 0
          and canonical(store.get(samples[0]["atom_id"])) == canonical(samples[0]))
    store.close()

    diagnostic = Store(args.output / "legacy-diagnostic.sqlite", create=True)
    diagnostic.ingest([samples[0]], one, None)
    check("legacy_measured_store", diagnostic.audit()["slots"] == 9 and not diagnostic.records_only)
    diagnostic.close()
    with patch("senses.instruments.roster", return_value={"panel_digest": "changed"}):
        refuses("legacy_roster_drift_still_refused", lambda: Store(args.output / "legacy-diagnostic.sqlite"), "MNLM_UCR_PANEL_DRIFT")
    corrupt = args.output / "tampered.sqlite"
    shutil.copyfile(controls, corrupt)
    db = sqlite3.connect(corrupt)
    db.execute("DROP TRIGGER immutable_records_UPDATE")
    altered = copy.deepcopy(samples[0])
    altered["unit"] = "tampered-unit"
    db.execute("UPDATE records SET body=?", (canonical(altered),))
    db.commit()
    db.close()
    tampered = Store(corrupt)
    refuses("retained_byte_tamper", tampered.audit, "MNLM_UCR_ID")
    tampered.close()
    bad_contract = args.output / "invalid-contract.sqlite"
    shutil.copyfile(controls, bad_contract)
    db = sqlite3.connect(bad_contract)
    db.execute("DROP TRIGGER immutable_panel_UPDATE")
    db.execute("UPDATE panel SET body=?", (b"{invalid",))
    db.commit()
    db.close()
    bad_before = file_hash(bad_contract)
    refuses("unreadable_contract_refused", lambda: Store(bad_contract, create=True, records_only=True), "MNLM_UCR_STORE_CONTRACT")
    check("unreadable_contract_not_recreated", file_hash(bad_contract) == bad_before)
    timings["boundaries_and_controls"] = time.perf_counter() - at
    check("production_source_unchanged_by_controls", file_hash(path) == before)
    report = {"acceptance": "author-checked", "kind": "observation_retention", "inputs": inputs,
              "records": compared, "units": {k: len(v) for k, v in units.items()}, "present_channels": channel_counts,
              "checks": checks, "passed": len(checks), "failed": 0, "timings_seconds": timings,
              "wall_seconds": time.perf_counter() - started, "store_sha256": file_hash(path), "head": head,
              "scientific_admission": False, "stage4_complete": False, "continual_model_learning": False,
              "limitations": ["complete bounded native exports, not all acquired trajectories", "no global near dedup",
                              "channel presence is not independent-sense admission", "records-only contract retains observations without model slots"]}
    report["code_sha256"] = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    report["timed_fraction"] = sum(timings.values()) / report["wall_seconds"]
    (args.output / "report.json").write_bytes(canonical(report) + b"\n")
    files = {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "SHA256.json").write_bytes(canonical(files) + b"\n")
    print(json.dumps({k: report[k] for k in ("acceptance", "records", "passed", "failed", "wall_seconds", "head")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
