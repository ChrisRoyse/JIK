"""Independent bytes/SQL readback of execution conditions; no model/checker calls."""
import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import pyarrow.parquet as pq


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run(root, raw_root):
    start = time.perf_counter()
    report = json.loads((root/"report.json").read_bytes())
    obs_root = root/"observations"
    raw = (obs_root/"manifest.json").read_bytes()
    assert sha(raw) == report["observation_pin"] == (obs_root/"COMMITTED").read_text()
    manifest = json.loads(raw)
    for name, pin in manifest["files"].items():
        raw = (obs_root/name).read_bytes()
        assert len(raw) == pin["bytes"] and sha(raw) == pin["sha256"]
    raw = (obs_root/"condition-commitment.json").read_bytes()
    assert sha(raw) == report["commitment"] == manifest["condition_commitment"]
    assert sha(raw) == (obs_root/"conditions-COMMITTED").read_text()
    commitment = json.loads(raw)
    raw = (obs_root/"condition-inputs.json").read_bytes()
    assert sha(raw) == commitment["inputs"]
    inputs = json.loads(raw)["inputs"]
    assert commitment["prediction_present"] is False and commitment["model_consumption_established"] is False
    source_ref = commitment["program_source"]
    shard = raw_root/source_ref["shard"]
    assert sha(shard.read_bytes()) == source_ref["source_sha256"]
    table = pq.ParquetFile(shard)
    offset = 0
    row = None
    for i in range(table.metadata.num_row_groups):
        size = table.metadata.row_group(i).num_rows
        if offset <= source_ref["row"] < offset+size:
            row = table.read_row_group(i, columns=["instance_id", "trajectory"]).slice(
                source_ref["row"]-offset, 1).to_pylist()[0]
            break
        offset += size
    assert row is not None and row["instance_id"] == source_ref["task"]
    text = row["trajectory"][source_ref["turn"]]["text"].encode()
    assert sha(text) == source_ref["text_sha256"]
    left, right = commitment["program_span"]
    program = text[left:right]
    phases = dict(input_and_source=time.perf_counter()-start)
    tick = time.perf_counter()
    outcomes = {}
    for key, entry in manifest["observations"].items():
        assert sha(canonical(entry)) == key
        condition = inputs[entry["condition"]]
        assert sha(canonical(condition)) == entry["condition"]
        assert set(condition) == {"schema", "program", "program_sha256", "image", "wheels",
                                  "harness_sha256", "mode", "scope"}
        assert condition["program"].encode() == program == (obs_root/entry["program"]).read_bytes()
        assert sha(program) == condition["program_sha256"] == entry["program_sha256"]
        runtime = {p.name:sha(p.read_bytes()) for p in (obs_root/entry["runtime"]).iterdir() if p.is_file()}
        assert runtime == condition["wheels"] == entry["wheels"]
        execution = obs_root/entry["execution"]
        receipt = json.loads((execution/"execution.json").read_bytes())
        before = json.loads((execution/"before.json").read_bytes())
        after = json.loads((execution/"after.json").read_bytes())
        assert before["Id"] == after["Id"] == receipt["container_id"]
        assert before["Image"] == after["Image"] == receipt["image"] == condition["image"]
        assert not after["State"]["Running"] and after["State"]["ExitCode"] == receipt["exitcode"] == 0
        assert stamp(commitment["committed_at"]) < stamp(after["State"]["StartedAt"]) < stamp(after["State"]["FinishedAt"])
        for state in (before, after):
            h = state["HostConfig"]
            assert h["NetworkMode"] == "none" and h["ReadonlyRootfs"] and not h["Privileged"]
            assert h["Memory"] == h["MemorySwap"] == 512*1024*1024 and h["NanoCpus"] == 1_000_000_000
            assert h["CapDrop"] == ["ALL"] and state["Config"]["User"] == "65534:65534"
        result = receipt["result"]
        assert result["program_sha256"] == condition["program_sha256"] and result["wheels"] == runtime
        assert result["harness_sha256"] == sha((obs_root/"process_inside.py").read_bytes()) == condition["harness_sha256"]
        envelope = [line.split(b"=", 1)[1] for line in (execution/"stdout.bin").read_bytes().splitlines()
                    if line.startswith(b"MNLM_EXECUTION_RESULT=")]
        assert len(envelope) == 1 and json.loads(envelope[0]) == result
        captured = {}
        for stream in ("stdout", "stderr"):
            value = base64.b64decode(result[stream]["base64"], validate=True)
            assert len(value) == result[stream]["bytes"] and sha(value) == result[stream]["sha256"]
            captured[stream] = value.decode()
        assert report["readings"][key]["captured"] == captured
        assert report["readings"][key]["returncode"] == result["returncode"]
        outcomes[key] = dict(returncode=result["returncode"], condition=entry["condition"],
                             captured=captured, container=after["Id"])
    phases["actual_execution"] = time.perf_counter()-tick
    tick = time.perf_counter()
    saved = json.loads((root/"state.json").read_bytes())
    db = sqlite3.connect(f"file:{root/'experience.sqlite3'}?mode=ro", uri=True)
    assert db.execute("SELECT contract FROM metadata").fetchall() == [(canonical(saved["contract"]),)]
    previous = "0"*64
    memory_head = "0"*64
    episodes, scopes = {}, {}
    count = 0
    for sequence, prior, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
        body = json.loads(raw)
        assert sequence == count and prior == previous and raw == canonical(body)
        assert sha(canonical(dict(seq=sequence, previous=prior, body=body))) == pin
        rec = body["record"]
        assert rec["sequence"] == count and rec["previous"] == memory_head
        assert rec["sha256"] == sha(canonical({k:v for k,v in rec.items() if k != "sha256"}))
        event = rec["event"]
        if event["op"] == "intake":
            ep = event["episode"]
            assert ep["id"] == sha(canonical({k:v for k,v in ep.items() if k != "id"}))
            episodes[ep["id"]] = ep
            scopes[ep["id"]] = None
            if ep["claim_kind"] == "execution_precondition":
                assert ep["content"] == inputs[ep["source"]["condition"]]
            elif ep["claim_kind"] == "event_occurrence":
                observation = ep["source"]["observation"]
                assert ep["content"] == outcomes[observation]["captured"]
                assert ep["evidence"] == [report["links"][observation]["condition"]]
        elif event["op"] == "check_source":
            receipt = event["result"]
            assert receipt["id"] == sha(canonical({k:v for k,v in receipt.items() if k != "id"}))
        elif event["op"] == "promote":
            kind = episodes[event["episode"]]["claim_kind"]
            scopes[event["episode"]] = ("precommitted_execution_conditions" if kind == "execution_precondition"
                                          else "observed_process_in_pinned_runtime")
        elif event["op"] == "invalidate":
            affected = {event["episode"]}
            for ep in episodes.values():
                if set(ep["evidence"]) & affected:
                    affected.add(ep["id"])
            assert set(event["affected"]) == affected
            for key in affected:
                scopes[key] = None
        else:
            raise AssertionError("unexpected memory operation")
        count += 1
        previous, memory_head = pin, rec["sha256"]
    db.close()
    assert previous == saved["journal_head"] and memory_head == saved["memory"]["head"]
    assert scopes == {k:v["grounded_scope"] for k,v in saved["memory"]["states"].items()}
    assert count == 13 and len(episodes) == 4
    phases["independent_sql_memory"] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    result = dict(complete=True, report_sha256=sha((root/"report.json").read_bytes()),
        manifest_files=len(manifest["files"]), source_program_bytes=len(program), conditions=len(inputs),
        outcomes=outcomes, memory_events=count, grounded_scopes=scopes,
        phases=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
        implementation_modules_imported=False, prediction_present=False,
        evidence_status="author_checked_independent_code_path")
    assert result["phase_coverage"] >= .9
    (root/"independent.json").write_bytes(canonical(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    a = p.parse_args()
    r = run(a.root, a.raw)
    print(json.dumps({k:r[k] for k in ("complete", "source_program_bytes", "conditions", "memory_events", "wall_seconds")}))
