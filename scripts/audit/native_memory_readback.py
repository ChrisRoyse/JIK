"""Independent SQL/hash/coordinate readback; does not call the memory writer/checker."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.perception_corpus import pinned, save, sha


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_NATIVE_MEMORY_READBACK_"+code)


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def pin(value):
    return hashlib.sha256(raw(value)).hexdigest()


def ledger(path, retained, expected):
    db = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
    need(db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "SQL_INTEGRITY")
    contract_raw, = db.execute("SELECT contract FROM metadata").fetchone()
    contract = json.loads(contract_raw)
    need(raw(contract) == contract_raw and pin(contract) == retained["contract"]
         and contract["evidence"] == expected["evidence_pin"], "CONTRACT")
    for name, code_pin in contract["code"].items():
        need(sha(Path(__file__).resolve().parents[1]/"claims"/name) == code_pin, "CODE")
    episodes, states, children = {}, {}, {}
    head, memory_head, count = "0"*64, "0"*64, 0
    for seq, previous, encoded, recorded in db.execute(
            "SELECT seq,previous,body,hash FROM events ORDER BY seq"):
        body = json.loads(encoded)
        need(raw(body) == encoded and seq == count and previous == head
             and pin(dict(seq=seq, previous=previous, body=body)) == recorded, "EVENT_HASH")
        record, command = body["record"], body["command"]
        need(record["sequence"] == count and record["previous"] == memory_head
             and pin({k: v for k, v in record.items() if k != "sha256"}) == record["sha256"],
             "MEMORY_HASH")
        event, op = record["event"], command["op"]
        need(all(event[k] == v for k, v in command.items()), "COMMAND")
        if op == "intake":
            ep = command["episode"]
            key = ep["id"]
            need(pin({k: v for k, v in ep.items() if k != "id"}) == key
                 and key not in episodes and all(p in episodes for p in ep["evidence"]), "EPISODE")
            episodes[key] = ep
            states[key] = dict(active=True, grounded_scope=None, review="unassessed", receipt=None)
            for parent in ep["evidence"]:
                children.setdefault(parent, set()).add(key)
        else:
            key = command["episode"]
            state, ep = states[key], episodes[key]
            if op in ("check_source", "promote"):
                need(state["active"] and all(states[p]["active"]
                     and states[p]["grounded_scope"] == contract["scope"]
                     and states[p]["review"] == "source_checked"
                     for p in ep["evidence"]), "DEPENDENT_SUPPORT")
            if op == "check_source":
                r = dict(checker="native-recording-evidence-v1", evidence_pin=expected["evidence_pin"],
                    episode=key, claim_kind=ep["claim_kind"], scope=contract["scope"],
                    world_validity="unassessed", physical_sync_calibrated=False,
                    scientific_admission=False, training_ready=False)
                receipt = dict(id=pin(r), **r)
                need(ep["claim_kind"] in ("measurement", "derived_relation")
                     and event["result"] == receipt, "SCOPED_RECEIPT")
                state.update(receipt=receipt, review="source_checked")
            elif op == "promote":
                need(state["receipt"] and command["receipt"] == state["receipt"]["id"], "PROMOTION")
                state.update(grounded_scope=contract["scope"], review="source_checked")
            elif op == "invalidate":
                pending, affected = [key], set()
                while pending:
                    item = pending.pop()
                    if item not in affected:
                        affected.add(item)
                        pending.extend(children.get(item, ()))
                need(event["affected"] == sorted(affected), "DEPENDENCY_CLOSURE")
                for item in affected:
                    states[item].update(grounded_scope=None, receipt=None, review="needs_review")
            elif op == "retire":
                need(state["active"] and state["grounded_scope"] is None, "RETIRE")
                state["active"] = False
            elif op == "reactivate":
                need(not state["active"], "REACTIVATE")
                state["active"] = True
            else:
                need(False, "OPERATION")
        head, memory_head, count = recorded, record["sha256"], count+1
    need(head == retained["head"] and memory_head == retained["memory_head"], "RETAINED_HEAD")
    need(db.execute("SELECT count(*) FROM payloads").fetchone() == (0,), "NO_DUPLICATE_PAYLOADS")
    db.close()
    return episodes, states, count


def main(a):
    started = time.perf_counter()
    report = pinned(a.root/"report.json", a.report_pin)
    path = a.root/"memory.sqlite"
    need(sha(path) == report["journal_sha256"], "JOURNAL_BYTES")
    episodes, states, count = ledger(path, report["retained"], report)
    need(states == report["snapshot"]["states"]
         and report["snapshot"]["grounded_world_claims"] == [], "FINAL_STATE")
    rollback = pinned(a.root/"boundary-withdrawal.json", sha(a.root/"boundary-withdrawal.json"))
    _, old_states, old_count = ledger(a.root/"before-withdrawal.sqlite",
                                    report["rollback_retained"], report)
    need(old_states == rollback["before"]["states"]
         and sha(a.root/"before-withdrawal.sqlite") == report["rollback_sha256"], "ROLLBACK")
    phases = dict(journals=time.perf_counter()-started)
    tick = time.perf_counter()
    bank, inputs, corpus = map(Path, (report["bank"], report["inputs"], report["corpus"]))
    bank_report = pinned(bank/"report.json", report["report_pin"])
    input_report = pinned(inputs/"report.json", bank_report["input_pin"])
    bank_values = {m: pinned(bank/m/"bank.json", proof["bank_pin"])
                   for m, proof in bank_report["banks"].items()}
    sources = {s["clip"]: s for s in input_report["sources"]}
    totals, coordinates = {}, []
    for readout in report["readouts"]:
        moment = readout["moment"]
        need(episodes[moment["id"]] == moment, "PERSISTED_MOMENT")
        clip, t = readout["clip"], Fraction(*moment["content"]["at"])
        source = sources[clip]
        root = corpus/"clips"/clip
        sr = pinned(root/"report.json", source["episode_pin"])
        need(sr["complete"] and not sr["errors"], "SOURCE_REPORT")
        for name in (f"public/{clip}.json", f"probes/{clip}.json", f"media/{clip}.mp4"):
            need(sha(root/name) == sr["files"][name]["sha256"]
                 and (root/name).stat().st_size == sr["files"][name]["bytes"], "SOURCE_BYTES")
        graph = json.loads((root/"public"/(clip+".json")).read_bytes())
        need(moment["source"]["clock_id"] == graph["clock"]["id"]
             and moment["source"]["split_group"] == graph["split_group"]
             and moment["source"]["source"] == source["source"]
             and moment["content"]["measured_alignment_error"] is None
             and not moment["content"]["physical_sync_calibrated"]
             and moment["content"]["text"] == "absent_no_attested_timed_text", "SOURCE_SCOPE")
        per_clip = []
        available = []
        for member in moment["content"]["members"]:
            medium = member["medium"]
            measurement = episodes[member["measurement"]]
            d = source["media"][medium]
            prefix_root = inputs/d["root"]
            prefix = pinned(prefix_root/"manifest.json", d["pin"])
            need((prefix_root/"COMMITTED").read_text() == d["pin"], "PREFIX_COMMIT")
            records = prefix["records"][medium]
            for rec in records:
                need(sha(prefix_root/rec["file"]) == rec["sha256"]
                     and (prefix_root/rec["file"]).stat().st_size == rec["bytes"], "NATIVE_BYTES")
            covering = [r["node"] for r in records
                        if Fraction(*r["node"]["span"][0]) <= t < Fraction(*r["node"]["span"][1])]
            need(len(covering) == 1 and member["coordinate"]["node"] == covering[0], "OVERLAP")
            node = covering[0]
            last_available = max(Fraction(*r["node"]["available_after"]) for r in records)
            available.append(last_available)
            need(Fraction(*measurement["event_time"]["available_after"]) == last_available
                 and member["context"] == measurement["event_time"], "CONTEXT_AVAILABILITY")
            if medium == "audio":
                rate = prefix["formats"]["audio"]["sample_rate"]
                offset = (t-Fraction(*node["span"][0]))*rate
                sample = offset.numerator//offset.denominator
                earlier = sum(r["node"]["sample_count"] for r in records
                              if r["node"]["ordinal"] < node["ordinal"])
                need(member["coordinate"]["sample_in_block"] == sample
                     and member["coordinate"]["sample_in_prefix"] == earlier+sample
                     and Fraction(*member["coordinate"]["sample_time"]) ==
                         Fraction(*node["span"][0])+Fraction(sample, rate), "EXACT_SAMPLE")
            refs = []
            for name, entries in bank_values[medium]["readings"].items():
                for entry in entries:
                    if entry["input"]["clip"] != clip:
                        continue
                    ref = pin(dict(medium=medium, name=name, entry=entry))
                    value_path = bank/medium/entry["file"]
                    size = entry["shape"][0]*entry["shape"][1]*4
                    need(sha(value_path) == entry["sha256"]
                         and value_path.stat().st_size == size, "READING_BYTES")
                    per_clip.append(dict(reference=ref, medium=medium, instrument=name,
                        shape=entry["shape"], sha256=entry["sha256"], bytes=size))
                    refs.append(ref)
                    totals[medium, name] = totals.get((medium, name), 0)+size
            need(set(refs) == set(measurement["content"]["readings"])
                 and len(refs) == len(measurement["content"]["readings"]), "COMPLETE_TYPED_READINGS")
        need(Fraction(*moment["event_time"]["available_after"]) == max(available), "MOMENT_AVAILABILITY")
        need(sorted(per_clip, key=lambda r: r["reference"]) ==
             sorted(readout["rows"], key=lambda r: r["reference"]), "CONSUMER_READBACK")
        coordinates.append(dict(clip=clip, at=moment["content"]["at"],
            available_after=moment["event_time"]["available_after"],
            members=moment["content"]["members"], readings=len(per_clip)))
    phases["source_readings_coordinates"] = time.perf_counter()-tick
    need(sha(path) == report["journal_sha256"], "READBACK_MUTATED")
    result = dict(author_checked=True, witness_pin=a.report_pin, events=count,
        rollback_events=old_count, episodes=len(episodes), exact_coordinates=coordinates,
        bytes_by_instrument={m+"/"+n: b for (m, n), b in totals.items()},
        total_bytes=sum(totals.values()), phase_coverage=None, phases=phases,
        scientific_admission=False, training_ready=False, physical_sync_calibrated=False,
        timed_text_present=False, code_pin=sha(Path(__file__)), seconds=time.perf_counter()-started)
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    result_pin = save(a.output, result)
    print(json.dumps(dict(report_pin=result_pin, events=count, episodes=len(episodes),
        bytes=result["total_bytes"], phase_coverage=result["phase_coverage"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report-pin", required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
