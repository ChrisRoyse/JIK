"""Read exact condition fields, native membership and forecast bindings without model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.window_readback import readback as read_windows


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def run(root, conditions_root, raw_root, connected):
    start = time.perf_counter()
    setup = json.loads((root/"setup.json").read_bytes())
    population, native = root/"population", Path(setup["condition_native"])
    raw = (conditions_root/"condition-commitment.json").read_bytes()
    commit_pin = sha(raw)
    assert commit_pin == (conditions_root/"conditions-COMMITTED").read_text()
    commitment = json.loads(raw)
    raw = (conditions_root/"condition-inputs.json").read_bytes()
    assert sha(raw) == commitment["inputs"]
    inputs = json.loads(raw)["inputs"]
    ref = commitment["program_source"]
    shard = raw_root/ref["shard"]
    assert sha(shard.read_bytes()) == ref["source_sha256"]
    table = pq.ParquetFile(shard)
    offset = 0
    source = None
    for group in range(table.metadata.num_row_groups):
        count = table.metadata.row_group(group).num_rows
        if offset <= ref["row"] < offset+count:
            source = table.read_row_group(group, columns=["instance_id", "trajectory"]).slice(ref["row"]-offset,1).to_pylist()[0]
            break
        offset += count
    assert source is not None and source["instance_id"] == ref["task"]
    text = source["trajectory"][ref["turn"]]["text"].encode()
    assert sha(text) == ref["text_sha256"]
    program = text[slice(*commitment["program_span"])]
    rows = pq.read_table(population/"condition/chunks/atoms.parquet").to_pylist()
    fields = json.loads((population/"fields.json").read_bytes())
    probe = json.loads((population/"probe.json").read_bytes())
    assert sha((population/"report.json").read_bytes()) == setup["conditions"]
    for key, value in inputs.items():
        assert sha(canonical(value)) == key and value["program"].encode() == program
        members = [r for r in fields if r["condition"] == key]
        expected = [("program", value["program"]), *[("wheel_filename", n) for n in sorted(value["wheels"])]]
        assert len(members) == len(expected)
        for index, (member, (field, text)) in enumerate(zip(members, expected)):
            assert member["ordinal"] == index and member["field"] == field
            assert member["unit"] != probe["unit"] and rows[member["unit"]]["text"] == text
            assert member["text_sha256"] == sha(text.encode())
    assert len(rows) == 4 and len(fields) == 5 and len(inputs) == 2
    source_end = time.perf_counter()
    windows = read_windows(native, population)
    manifest = json.loads((native/"report.json").read_bytes())
    arrays = {}
    for name in manifest["lenses"]:
        with np.load(native/name/"slots.npz", allow_pickle=False) as z:
            arrays[name] = z["vectors"].copy(), z["offsets"].copy()
    native_end = time.perf_counter()
    snapshot = json.loads((connected/"snapshot.json").read_bytes())
    bindings, requests, forecasts = {}, [], []
    db = sqlite3.connect(f"file:{connected/'experience.sqlite3'}?mode=ro", uri=True)
    payloads = dict(db.execute("SELECT hash,data FROM payloads"))
    head = "0"*64
    count = 0
    for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
        body = json.loads(raw)
        assert previous == head and seq == count and canonical(body) == raw
        assert sha(canonical(dict(seq=seq,previous=previous,body=body))) == pin
        if body["kind"] == "execution_conditions":
            bound = body["conditions"]
            b = bound["native"]
            key = b["condition"]
            assert b["commitment"] == commit_pin and b["inputs"] == commitment["inputs"]
            assert b["population"] == setup["conditions"] and b["native"] == sha((native/"report.json").read_bytes())
            for name, (values, ptr) in arrays.items():
                cursor = 0
                for member in [r for r in fields if r["condition"] == key]:
                    u = member["unit"]
                    size = int(ptr[u+1]-ptr[u])
                    assert b["fields"][name][member["ordinal"]] == dict(**member,window_start=cursor,window_end=cursor+size)
                    cursor += size
                assert cursor > 0 and len(b["fields"][name]) == len([r for r in fields if r["condition"] == key])
            bindings[body["context"],body["action"]] = (seq,bound)
        elif body["kind"] == "advance":
            req = body["request"]
            if "execution_conditions" in req:
                before, binding = bindings[req["context"],req["turn"]]
                assert before < seq and req["phase"] in (1,2) and req["execution_conditions"] == binding
                requests.append(seq)
                if req["phase"] == 2:
                    payload = json.loads(payloads[body["payload"]])
                    assert sha(canonical(payload)) == body["payload"] and payload["forecast"]["request"] == req
                    forecasts.append(sha(canonical(payload["forecast"])))
        head, count = pin, count+1
    db.close()
    assert len(bindings) == 2 and len(requests) == 4 and len(forecasts) == 2
    assert head == snapshot["head"] and count == snapshot["events"]
    training = json.loads((connected/"training.json").read_bytes())
    assert len(training) == 2 and all(r["exact_optimizer_recovery"] for r in training)
    end = time.perf_counter()
    phases = dict(source=source_end-start, native=native_end-source_end, journal=end-native_end)
    wall = time.perf_counter()-start
    result = dict(complete=True, source_program_bytes=len(program), exact_field_instances=len(fields),
        native_readback=windows, bindings=len(bindings), conditioned_forecasts=forecasts,
        journal_events=count, journal_head=head, model_implementation_imported=False,
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall, acceptance="author-checked only")
    assert result["phase_coverage"] >= .9
    with (connected/"condition-readback.json").open("xb") as f:
        f.write(canonical(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "conditions", "raw", "connected"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    r = run(a.root, a.conditions, a.raw, a.connected)
    print(json.dumps({k:r[k] for k in ("complete","source_program_bytes","bindings","journal_events","wall_seconds")}))
