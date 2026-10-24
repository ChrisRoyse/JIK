"""Independent SQLite/tensor/source reader; does not import the session producer."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_SESSION_READBACK_"+code)


def pinned(path, pin):
    check(file_sha(path) == pin, "FILE:"+str(path))
    return json.loads(Path(path).read_bytes())


def array(value, shape):
    check(set(value) == {"shape", "dtype", "sha256", "base64"}
          and value["shape"] == list(shape) and value["dtype"] == "float32-le", "TENSOR_LAYOUT")
    raw = base64.b64decode(value["base64"], validate=True)
    check(len(raw) == math.prod(shape)*4 and hashlib.sha256(raw).hexdigest() == value["sha256"],
          "TENSOR_BYTES")
    result = np.frombuffer(raw, dtype="<f4").reshape(shape)
    check(np.isfinite(result).all(), "NONFINITE")
    return result


def run(a):
    started = time.perf_counter()
    report = pinned(a.run/"report.json", a.pin)
    original = pinned(a.inputs/"report.json", report["input_pin"])
    bank = pinned(a.bank/"bank.json", report["bank_pin"])
    previous = pinned(a.previous/"report.json", report["previous_pin"])
    cp = json.loads((a.previous/"initial/manifest.json").read_bytes())
    check(digest(cp) == report["checkpoint"] == previous["initial"], "CHECKPOINT")
    check(file_sha(a.previous/"initial/state.pt") == cp["sha256"], "CHECKPOINT_BYTES")
    payload = torch.load(a.previous/"initial/state.pt", map_location="cpu", weights_only=True)
    check(cp["binding"] == report["binding"] and cp["standard"] == report["standard"], "BINDING")
    for key, value in report["code"].items():
        check(file_sha(a.repo/"scripts"/key) == value, "CODE:"+key)
    for key, value in report["files"].items():
        check(file_sha(a.run/key) == value, "PUBLICATION:"+key)
    for spec in cp["instruments"]:
        definition = json.loads(spec["definition_json"])
        check(digest(definition) == spec["identity"]
              and definition["measurement_contract"] == bank["instruments"][spec["name"]], "INSTRUMENT")
    byte_count, readings = 0, 0
    storage_roots = [a.bank.resolve(), *(p.resolve() for p in a.shared_bank)]
    for lane, entries in bank["readings"].items():
        for entry in entries:
            path = (a.bank/lane/entry["file"]).resolve()
            check(any(path.is_relative_to(root) for root in storage_roots), "NATIVE_STORAGE:"+str(path))
            check(file_sha(path) == entry["sha256"], "NATIVE_BYTES:"+str(path))
            byte_count += path.stat().st_size
            readings += 1
    phases = dict(source_seconds=time.perf_counter()-started)
    tick = time.perf_counter()
    db = sqlite3.connect(f"file:{(a.run/'session.sqlite').as_posix()}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT contract FROM metadata").fetchall()
        check(len(rows) == 1, "CONTRACT_COUNT")
        contract = json.loads(rows[0][0])
        check(encode(contract) == rows[0][0] and digest(contract) == report["after"]["retained"]["contract"],
              "CONTRACT_PIN")
        check(contract["checkpoint"] == report["checkpoint"]
              and contract["input_binding"] == report["binding"] and contract["flags"] == report["flags"]
              and all(v is False for v in contract["flags"].values()), "CONTRACT_SCOPE")
        states = json.loads(json.dumps(report["before"]["contexts"]))
        calls = json.loads((a.run/"calls.json").read_bytes())
        previous_head = "0"*64
        observed, prediction_bytes, seen_payloads = [], 0, set()
        for expected_seq, row in enumerate(db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq")):
            seq, prior, raw, pin = row
            body = json.loads(raw)
            check(seq == expected_seq and prior == previous_head and encode(body) == raw
                  and pin == digest(dict(seq=seq, previous=prior, body=body)), "CHAIN")
            check(body["kind"] == "native_episode_advance" and body["flags"] == report["flags"], "EVENT")
            r = body["request"]
            context, index = r["context"], r["index"]
            moment = original["moments"][index]
            check(r["episode"] == moment["id"] and r["source"] == moment["source"]
                  and r["event_time"] == moment["event_time"] and r["prior_head"] == prior
                  and r["input_binding"] == report["binding"] and r["origin"] == "source_observation"
                  and r["purpose"] == "engineering_boundary", "SOURCE_JOIN")
            check(context in states and r["sequence"] == states[context]["sequence"]
                  and r["previous_event"] == states[context]["last_event"]
                  and r["phase"] == r["sequence"] % 5 < 3, "CONTEXT_PHASE")
            raw_payload = db.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()
            check(raw_payload is not None and len(raw_payload[0]) <= cp["config"]["checkpoint_bytes"], "PAYLOAD")
            p = json.loads(raw_payload[0])
            check(encode(p) == raw_payload[0] and digest(p) == body["payload"], "PAYLOAD_PIN")
            array(p["world"], payload["world"].shape[1:])
            array(p["identity"], payload["identity"].shape[1:])
            check(p["sequence"] == r["sequence"]+1 and p["last_event"] == digest(r), "STATE_TRANSITION")
            if r["phase"] == 2:
                check(set(p["prediction"]) == set(cp["world_rows"]) and len(p["regions"]) == 1, "PREDICTION_ROSTER")
                for i, spec in enumerate(cp["instruments"]):
                    value = p["prediction"][spec["name"]]
                    check(value["instrument"] == spec["identity"], "PREDICTION_IDENTITY")
                    owner = cp["world_rows"][spec["name"]].index(p["regions"][0])
                    mask = payload["model"][f"world_padding_{i}"][owner]
                    shape = (int((~mask).sum()), spec["dims"])
                    arr = array({k: v for k, v in value.items() if k != "instrument"}, shape)
                    check(np.all(np.abs(np.linalg.norm(arr, axis=1)-1) < 1e-4), "PREDICTION_NORM")
                    prediction_bytes += arr.nbytes
            else:
                check(p["prediction"] is None, "UNEXPECTED_PREDICTION")
            states[context] = dict(sequence=p["sequence"], last_event=p["last_event"],
                world=p["world"]["sha256"], identity=p["identity"]["sha256"])
            check(states == calls[seq]["after"]["contexts"]
                  and pin == calls[seq]["after"]["retained"]["head"], "PER_OPERATION_READBACK")
            observed.append(dict(sequence=seq, context=context, phase=r["phase"],
                                 payload_bytes=len(raw_payload[0])))
            seen_payloads.add(body["payload"])
            previous_head = pin
        check(previous_head == report["after"]["retained"]["head"]
              and states == report["after"]["contexts"] and len(observed) == report["operations"], "FINAL_STATE")
        check(seen_payloads == {r[0] for r in db.execute("SELECT hash FROM payloads")}, "ORPHAN_PAYLOADS")
    finally:
        db.close()
    # Read the fault artifacts themselves, independently of the witness's PASSs.
    def artifact(name):
        connection = sqlite3.connect(f"file:{(a.run/name).as_posix()}?mode=ro", uri=True)
        try:
            return dict(events=connection.execute(
                "SELECT seq,previous,body,hash FROM events ORDER BY seq").fetchall(),
                payloads=dict(connection.execute("SELECT hash,data FROM payloads")),
                triggers=connection.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'").fetchall())
        finally:
            connection.close()

    primary = artifact("session.sqlite")
    rollback, fault = artifact("rollback.sqlite"), artifact("append-fault.sqlite")
    check(rollback["events"] == fault["events"] == primary["events"][:3]
          and rollback["payloads"] == fault["payloads"]
          and len(fault["payloads"]) == 3 and not rollback["triggers"]
          and len(fault["triggers"]) == 1 and fault["triggers"][0][0] == "injected"
          and "injected append failure" in fault["triggers"][0][1], "ABORT_ROLLBACK")
    truncated, corrupt = artifact("truncated.sqlite"), artifact("corrupt.sqlite")
    check(truncated["events"] == primary["events"][:-1]
          and truncated["payloads"] == primary["payloads"], "TRUNCATION_CONTROL")
    check(corrupt["events"] == primary["events"] and len(corrupt["payloads"]) == 6
          and sum(hashlib.sha256(raw).hexdigest() != pin
                  for pin, raw in corrupt["payloads"].items()) == 1, "CORRUPTION_CONTROL")
    physical_controls = dict(rollback_events=3, failed_append_events=3, failed_append_payloads=3,
                             truncated_events=5, corrupt_payloads=1)
    phases["journal_seconds"] = time.perf_counter()-tick
    result = dict(acceptance="author-checked", source_report_sha256=a.pin,
        flags=report["flags"], readings=readings, original_native_bytes=byte_count,
        storage_roots=[str(p) for p in storage_roots],
        operations=observed, prediction_native_bytes=prediction_bytes,
        physical_controls=physical_controls,
        checkpoint=report["checkpoint"], final_head=previous_head,
        phases=phases, seconds=time.perf_counter()-started)
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    check(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    with a.output.open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "inputs", "bank", "previous", "repo", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--shared-bank", type=Path, action="append", default=[],
                   help="Explicit retained bank root for hash-checked shared reading files.")
    run(p.parse_args())
