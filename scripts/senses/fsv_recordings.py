"""Real new-source transfer, modality preservation and oracle-exclusion evidence."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.adapters import atom, prepare, typed_fields
from senses.contract import SenseError, canonical, field, present, seal_record, validate_record
from senses.instruments import contracts, measure, roster, slot_digest
from senses.pipeline import file_hash, ingest_file, read_bound, write_records
from senses.recorded import trajectory_records, selected_rows
from senses.store import Store


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, ok, **data):
        checks.append({"boundary": name, "pass": bool(ok), **data})

    def refuses(name, call, code):
        error = None
        try:
            call()
        except SenseError as exc:
            error = exc.code
        check(name, error == code, error=error)

    capture = next(r for r in json.loads((args.input / "capture.json").read_text()) if r["source"] == "wikidata5m")
    old = [atom(prepare(e)) for e in read_bound(args.input / "wikidata5m.jsonl", capture["sha256"])][:16]
    sets = {"baseline": old}
    for name, filename in (("trajectory", "trajectories.ucr.jsonl"), ("audio", "audio.ucr.jsonl")):
        path = args.recordings / filename
        manifest = json.loads(path.with_suffix(".manifest.json").read_bytes())
        records = list(read_bound(path, manifest["sha256"]))
        check(name + "_complete_export_readback", len(records) == manifest["count"], rows=len(records))
        sets[name] = records[:32] if name == "trajectory" else records
    store = Store(args.output / "channels.sqlite", create=True)
    panel = roster()["panel_digest"]
    for name, records in sets.items():
        old_bytes = dict(store.db.execute("SELECT atom,body FROM records"))
        old_slots = {(a, n): d for a, n, d in store.db.execute("SELECT atom,lens,digest FROM slots")}
        path = args.output / (name + ".ucr.jsonl")
        manifest = write_records(path, records, {"scope": "smallest modality proof; not admission", "source": name})
        before = store.head
        ingest_file(store, path, manifest, store.head)
        audit = store.audit()
        check(name + "_physical_generation", store.head != before and audit["records"] == len(old_bytes) + len(records),
              before=before, after=store.head, audit=audit)
        check(name + "_unchanged_instruments", roster()["panel_digest"] == panel)
        check(name + "_retained_old_records", all(canonical(store.get(a)) == data for a, data in old_bytes.items()), count=len(old_bytes))
        check(name + "_retained_old_slots", all(slot_digest(store.read_slot(a, n)) == d for (a, n), d in old_slots.items()), count=len(old_slots))
    import soundfile as sf
    for r in sets["audio"]:
        if r["locator"]["source"] != "web-audio-mpj":
            continue
        name = r["locator"]["shard"]
        raw = args.trajectory.parents[2] / "web-audio-mpj" / name
        info = sf.info(raw)
        bychannel = {}
        contiguous = True
        for item in r["channels"]["perception"]["items"]:
            values = {f["name"]: f["value"] for f in item["measurements"]}
            k = values["audio_channel"]
            contiguous = contiguous and values["sample_start"] == bychannel.get(k, 0)
            bychannel[k] = values["sample_start"] + values["sample_count"]
        lens_id = next(c["lens_id"] for c in contracts() if c["channel"] == "perception")
        slot = store.read_slot(r["atom_id"], lens_id)
        check(name + "_complete_native_frames", contiguous and len(bychannel) == info.channels and all(n == info.frames for n in bychannel.values())
              and len(slot["tokens"]) == len(r["channels"]["perception"]["items"]),
              decoded_samples=info.frames, retained_frames=len(slot["tokens"]))
        check(name + "_audio_has_no_text_embedding", r["channels"]["surface"]["state"] == "absent")
    first = sets["trajectory"][0]
    check("recorded_behaviour_present", measure(first, "behaviour")["kind"] == "Multi")
    check("generated_action_distinct", first["channels"]["surface"]["items"][0]["text"] !=
          first["channels"]["behaviour"]["items"][0]["measurements"][0]["value"])
    check("no_fabricated_execution_clock", all(f["type"] != "clock" for r in sets["trajectory"] for f in r["channels"]["fields"]["items"]))
    import pyarrow as pa
    import pyarrow.parquet as pq
    row_index = int(first["locator"]["shard"].rsplit("#trajectory=", 1)[1])
    rawrow = next(selected_rows(pq.ParquetFile(args.trajectory), [row_index]))[1]
    control = args.output / "control-only.parquet"
    rawrow["target"] = "CONTROL_ORACLE_TARGET"
    rawrow["eval_logs"] = "CONTROL_ORACLE_MUST_NOT_BE_A_SENSE"
    rawrow["generated_patch"] = "CONTROL_PATCH_MUST_NOT_BE_A_SENSE"
    rawrow["exit_status"] = "CONTROL_OUTCOME"
    pq.write_table(pa.Table.from_pylist([rawrow]), control)
    altered = list(trajectory_records(control, file_hash(control), {"negative_control": True}, limit=1))
    check("oracle_perturbation_no_feature_effect", all(measure(first, c)["tokens"].tobytes() == measure(altered[0], c)["tokens"].tobytes()
                                                    for c in (x["channel"] for x in contracts())))
    check("exact_integer_retained", typed_fields({"n": "9007199254740993"}, {"n": "integer"})[0]["value"] == 9007199254740993)
    refuses("noninteger_rejected", lambda: typed_fields({"n": "1.25"}, {"n": "integer"}), "MNLM_UCR_FIELD")
    for name, change, code in (("bad_channel_shape", lambda r: r["channels"].update(fields=None), "MNLM_UCR_CHANNEL"),
                              ("bad_locator_shape", lambda r: r["locator"].update(span=None), "MNLM_UCR_LOCATOR"),
                              ("empty_reference", lambda r: r["referents"][0].update(key=""), "MNLM_UCR_REFERENT")):
        bad = copy.deepcopy(first)
        change(bad)
        head = store.head
        refuses(name, lambda: store.ingest([bad], {"input_sha256": "control", "count": 1}, head), code)
        check(name + "_no_mutation", store.head == head)
    store.close()
    cold = Store(args.output / "channels.sqlite")
    final = cold.audit()
    check("cold_reopen_audit", final["records"] == 52 and final["slots"] == 468, audit=final)
    cold.close()
    report = {"acceptance": "author-checked", "checks": checks, "passed": sum(c["pass"] for c in checks),
              "failed": sum(not c["pass"] for c in checks), "instrument_panel": panel,
              "limitations": ["bounded adapter/retention proof; not continual model learning", "no Shannon-conditional admission",
                              "tool responses source-recorded, not independently executed", "chapter correspondence, not verified transcription"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
