"""Read withdrawal closure, repaired windows, forecasts and weights from persisted bytes."""
import base64
import json
import sqlite3

import numpy as np

from audit.fsv_kernel_route import independent_score
from audit.tool_execution_readback import readback as read_journal
from audit.tool_growth_readback import checkpoint, manifest, fingerprint
from claims.codata_source import check, sha
from claims.episodic_memory import digest


def readback(inputs, native, root, observations):
    snapshot = json.loads((root/"recovery-snapshot.json").read_bytes())
    journal = read_journal(root/"recovery.sqlite3",inputs,snapshot,observations,native)
    with sqlite3.connect(f"file:{(root/'experience.sqlite3').as_posix()}?mode=ro",uri=True) as db:
        prefix = db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq").fetchall()
    with sqlite3.connect(f"file:{(root/'recovery.sqlite3').as_posix()}?mode=ro",uri=True) as db:
        rows = db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq").fetchall()
        payloads = {pin:json.loads(raw) for pin,raw in db.execute("SELECT hash,data FROM payloads")}
        contract = json.loads(db.execute("SELECT contract FROM metadata").fetchone()[0])
    check(rows[:len(prefix)] == prefix, "RECOVERY_READ_PREFIX")
    bodies = [json.loads(row[2]) for row in rows]
    repair_rows = [(i,b) for i,b in enumerate(bodies) if b["kind"] == "repair_bank"]
    check(len(repair_rows) == 1, "RECOVERY_READ_COUNT")
    index, repair = repair_rows[0]
    prior = next(b for b in reversed(bodies[:index]) if b["kind"] == "grow")
    old = checkpoint(root/"growth"/prior["name"]/"model",prior["checkpoint"])
    new = checkpoint(root/"growth"/repair["name"]/"model",repair["checkpoint"])
    old_kernel = manifest(root/"growth"/prior["name"]/"kernel",prior["kernel"])
    kernel = manifest(root/"growth"/repair["name"]/"kernel",repair["kernel"])
    invalid = set()
    for b in bodies[:index]:
        if b["kind"] == "invalidate":
            invalid.update(b["record"]["event"]["affected"])
    removed = {}
    # All newly grown supports come from persisted growth events, not live session objects.
    supports = {}
    for b in bodies[:index]:
        if b["kind"] == "grow":
            supports.update({u:{k:v for k,v in s.items() if k != "forecast"} for u,s in b["supports"].items()})
    for u,support in supports.items():
        affected = sorted(invalid.intersection(support.values()))
        if affected:
            removed[u] = dict(support=support,invalid=affected)
    check(repair["removed"] == removed and len(removed) == 2 and len(invalid) == 2,
          "RECOVERY_READ_WITHDRAWAL")
    retained = [u for u in prior["units"] if str(u) not in removed]
    check(repair["units"] == retained == inputs.world_units
          and kernel["pairs"] == {u:v for u,v in old_kernel["pairs"].items() if u not in removed}
          and kernel["plan"] == old_kernel["plan"], "RECOVERY_READ_ROSTER")
    source = digest(dict(parent=old[0]["world_source"],retired=removed,units=retained))
    check(repair["source"] == source == kernel["source"] == new[0]["world_source"]
          and new[0]["world"] == repair["world"] and old[0]["world"] == repair["old_world"]
          and new[0]["standard"] == old[0]["standard"] == repair["standard"],
          "RECOVERY_READ_IDENTITIES")
    check(new[2].keys() == old[2].keys() and
          all(np.array_equal(v,old[2][k]) for k,v in new[2].items()) and
          all(np.array_equal(v,old[3][k]) for k,v in new[3].items() if not k.startswith("world_")),
          "RECOVERY_READ_PARAMETERS")
    counts = dict(retained_windows=0,calibration_pairs=0,new_contexts=0,post_repair_forecasts=0,
                  retired_regions=len(removed),canceled_forecasts=len(repair["canceled"]))
    recalls = {}
    refs = [inputs.population.lookup[e["task"],t] for e in inputs.population.source.episodes
            if e["status"] == "ready" and e["partition"] == "development" for t in (6,10)]
    for i,spec in enumerate(inputs.specs):
        check(new[0]["world_rows"][spec.name] == [f"{u}:{inputs.refs[u]['text_sha256']}" for u in retained],
              "RECOVERY_READ_TYPED_ROWS")
        values, ptr = inputs.arrays[spec.name]
        for j,u in enumerate(retained):
            actual = new[3][f"world_values_{i}"][j,~new[3][f"world_padding_{i}"][j]]
            check(np.array_equal(actual,values[ptr[u]:ptr[u+1]]), "RECOVERY_READ_WINDOWS")
            counts["retained_windows"] += len(actual)
        saved = kernel["calibration"][spec.name]
        check([r["reference"] for r in saved["rows"]] == refs, "RECOVERY_READ_QUERIES")
        scores = []
        for r in saved["rows"]:
            u = r["reference"]["unit"]
            direct = {k:independent_score(values[ptr[u]:ptr[u+1]],
                      values[ptr[p["action"]["unit"]]:ptr[p["action"]["unit"]+1]])
                      for k,p in kernel["pairs"].items()}
            exact = sorted(direct,key=lambda k:(-direct[k],k))[:kernel["plan"]["top_k"]]
            hits = r["route"]["hits"]
            recall = len(set(exact)&{k for k,_ in hits})/len(exact)
            check(exact == r["exact"] and r["recall"] == recall
                  and all(abs(direct[k]-v) < 2e-12 for k,v in hits), "RECOVERY_READ_RECALL")
            scores.append(recall)
            counts["calibration_pairs"] += len(direct)
        recalls[spec.name] = sum(scores)/len(scores)
        check(saved["recall"] == recalls[spec.name] and saved["passes"] and recalls[spec.name] >= .95,
              "RECOVERY_READ_GATE")
    hidden, context_tasks = {}, dict(contract["contexts"])
    records, pending, invalidated, dirty = {}, {}, set(), set()
    repaired = False
    for b in bodies:
        payload = payloads.get(b.get("payload"),{})
        if b["kind"] == "start_context":
            c = b["context"]
            check(c not in context_tasks and b["task"] not in context_tasks.values(), "RECOVERY_READ_CONTEXT")
            check(b["state"] == digest({c:dict(sequence=0,last_event=None,hidden=fingerprint(new[2]["initial"]))})
                  and b["standard"] == repair["standard"]
                  and b["world"] == (repair["world"] if repaired else repair["old_world"]),
                  "RECOVERY_READ_INITIAL_STATE")
            context_tasks[c] = b["task"]
            counts["new_contexts"] += 1
        if b["kind"] == "advance":
            c = b["request"]["context"]
            check(c not in dirty and b["request"]["task"] == context_tasks[c], "RECOVERY_READ_DIRTY_ACTIVITY")
            hidden[c] = dict(sequence=b["request"]["sequence"]+1,last_event=digest(b["request"]),
                             raw=base64.b64decode(b["hidden"],validate=True))
        if "forecast" in payload:
            f = payload["forecast"]
            key = digest(f)
            records[key] = f
            pending[f["request"]["context"]] = key
            if repaired:
                desc = f["region_context"]["definition"]
                check(f["world"] == repair["world"] and f["routing"]["kernel"] == repair["kernel"]
                      and not {int(u) for u in removed}.intersection(desc["units"])
                      and not invalid.intersection(f["bank_support"].values()),
                      "RECOVERY_READ_FUTURE_MEMBERSHIP")
                counts["post_repair_forecasts"] += 1
        if "assessment" in payload:
            pending.pop(b["request"]["context"])
        if b["kind"] == "invalidate":
            affected = set(b["record"]["event"]["affected"])
            for k,f in records.items():
                support = {*f["support"].values(),*f["bank_support"].values(),
                           *(e["episode"] for e in f["region_context"]["definition"].get("observed_echoes",{}).values())}
                ep = snapshot["process_observations"]["forecasts"].get(k,{}).get("episode")
                if affected.intersection(support|{ep}):
                    invalidated.add(k)
                    dirty.add(f["request"]["context"])
        if b["kind"] == "repair_bank":
            check(dirty == set(b["dirty"]), "RECOVERY_READ_DIRTY_SCOPE")
            states = {}
            for i,c in enumerate(new[1]["contexts"]):
                h = hidden[c["context"]]
                check(new[4][i].tobytes() == h["raw"] and c["sequence"] == h["sequence"]
                      and c["last_event"] == h["last_event"], "RECOVERY_READ_CONTEXT_RETAINED")
                states[c["context"]] = dict(sequence=c["sequence"],last_event=c["last_event"],
                                            hidden=fingerprint(new[4][i]))
            check(digest(states) == b["context_state"], "RECOVERY_READ_CONTEXT_PIN")
            for c,k in b["canceled"].items():
                check(pending.pop(c) == k and k in invalidated and
                      snapshot["forecasts"][k]["status"] == "canceled", "RECOVERY_READ_CANCELLATION")
            repaired = True
        if repaired and b["kind"] == "candidate":
            check(not invalidated.intersection(b["train"]+b["evaluate"]) and "continuation" not in b,
                  "RECOVERY_READ_TRAINING_SCOPE")
    check(not pending and context_tasks == snapshot["contexts"] and dirty == set(snapshot["dirty"])
          and invalidated == {k for k,v in snapshot["forecasts"].items() if v["invalidated"]},
          "RECOVERY_READ_FINAL_SCOPE")
    check(snapshot["region_units"] == retained and snapshot["kernel"] == repair["kernel"]
          and snapshot["active_model"] == repair["checkpoint"], "RECOVERY_READ_FINAL_BANK")
    return dict(complete=True,journal=journal,counts=counts,recall=recalls,prefix_events=len(prefix),
                frozen_parameters=True,retained_contexts=True,scientific_admission=False)
