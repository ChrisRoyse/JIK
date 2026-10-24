"""Independent SQL/native-array/checkpoint readback of observed region growth."""
import base64
import json
import sqlite3

import numpy as np

from audit.fsv_kernel_route import independent_score
from audit.tool_growth_readback import checkpoint, manifest, fingerprint
from claims.codata_source import check
from claims.episodic_memory import digest


def readback(inputs, native, root, parent):
    initial = json.loads((parent/"run.json").read_bytes())
    old = checkpoint(parent/"initial", initial["initial"])
    source_id = inputs.source_id
    bank = list(inputs.world_units)
    pair_map = {str(r["unit"]): dict(response=r, action=inputs.population.lookup[r["task"], r["turn"]-1])
                for r in inputs.population.initial_bank()}
    kernel = manifest(root/"initial-kernel", (root/"initial-kernel"/"COMMITTED").read_text())
    current_kernel = (root/"initial-kernel"/"COMMITTED").read_text()
    current_checkpoint = initial["initial"]
    observed, records, hidden, bound, rows, scopes = {}, {}, {}, {}, dict(enumerate(inputs.refs)), {}
    promoted, episodes = set(), {}
    counts = dict(growths=0, retained_windows=0, added_windows=0, direct_calibration_pairs=0,
                  forecasts=0, forecasts_with_observed_regions=0, selected_observed_regions=0)
    versions = []
    native_refs = [json.loads(raw) for raw in (native.population.root/"references.jsonl").read_bytes().splitlines()]
    native_arrays = {}
    for spec in inputs.specs:
        with np.load(native.native/spec.name/"slots.npz", allow_pickle=False) as data:
            native_arrays[spec.name] = data["vectors"].copy(), data["offsets"].copy()

    def windows(name, unit):
        if unit not in bound:
            x, ptr = inputs.arrays[name]
            return x[ptr[unit]:ptr[unit+1]]
        refs = [r for r in native_refs if r["observation"] == bound[unit]["observation"] and r["unit"] is not None]
        x, ptr = native_arrays[name]
        check(refs and all(r["unit"] != native.population.probe["unit"] for r in refs),
              "PROCESS_GROWTH_READ_PROBE")
        return np.concatenate([x[ptr[r["unit"]]:ptr[r["unit"]+1]] for r in refs])

    def calibrate(value):
        check(value["pairs"] == pair_map and value["source"] == source_id
              and value["instruments"] == {s.name:s.identity for s in inputs.specs},
              "PROCESS_GROWTH_READ_KERNEL")
        expected_refs = [inputs.population.lookup[e["task"], t]
                         for e in inputs.population.source.episodes
                         if e["status"] == "ready" and e["partition"] == "development" for t in (6,10)]
        recalls = {}
        for spec in inputs.specs:
            saved = value["calibration"][spec.name]
            check([r["reference"] for r in saved["rows"]] == expected_refs, "PROCESS_GROWTH_READ_QUERIES")
            total = []
            for r in saved["rows"]:
                scores = {key:independent_score(windows(spec.name,r["reference"]["unit"]),
                          windows(spec.name,pair["action"]["unit"])) for key,pair in pair_map.items()}
                exact = sorted(scores, key=lambda k:(-scores[k], k))[:value["plan"]["top_k"]]
                hits = r["route"]["hits"]
                recall = len(set(exact) & {k for k,_ in hits})/len(exact)
                check(exact == r["exact"] and recall == r["recall"]
                      and all(abs(scores[k]-v) < 2e-12 for k,v in hits), "PROCESS_GROWTH_READ_RECALL")
                total.append(recall)
                counts["direct_calibration_pairs"] += len(scores)
            recalls[spec.name] = sum(total)/len(total)
            check(recalls[spec.name] == saved["recall"] and saved["passes"]
                  and recalls[spec.name] >= .95, "PROCESS_GROWTH_READ_GATE")
        return recalls

    calibrate(kernel)
    with sqlite3.connect(f"file:{(root/'experience.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        payloads = {pin:json.loads(raw) for pin,raw in db.execute("SELECT hash,data FROM payloads")}
        for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
            b = json.loads(raw)
            p = payloads.get(b.get("payload"), {})
            events = [b["record"]] if b["kind"] == "memory" else b.get("memory", [])
            for r in events:
                e = r["event"]
                if e["op"] == "intake":
                    episodes[e["episode"]["id"]] = e["episode"]
                if e["op"] == "check_source":
                    scopes[e["episode"]] = e["result"]["scope"]
                if e["op"] == "promote":
                    promoted.add(e["episode"])
            if b["kind"] == "advance":
                req = b["request"]
                hidden[req["context"]] = dict(sequence=req["sequence"]+1, last_event=digest(req),
                                             raw=base64.b64decode(b["hidden"], validate=True))
            if "forecast" in p:
                f = p["forecast"]
                records[digest(f)] = f
                desc = f["region_context"]["definition"]
                check(f["routing"]["kernel"] == current_kernel and desc["source"] == source_id
                      and f["world"] == old[0]["world"], "PROCESS_GROWTH_READ_CUTOFF")
                eligible = [bank[i] for i in f["routing"]["allowed"]]
                private = [r["reference"]["unit"] for r in desc["private"]]
                check(desc["units"] == list(dict.fromkeys(eligible+private))
                      and all(u < len(inputs.refs) or u in bound for u in desc["units"])
                      and int(f["region"].split(":")[0]) in desc["units"], "PROCESS_GROWTH_READ_REGIONS")
                for u in bound:
                    check(f["bank_support"][str(u)] == bound[u]["episode"], "PROCESS_GROWTH_READ_SUPPORT")
                counts["forecasts"] += 1
                counts["forecasts_with_observed_regions"] += bool(set(desc["units"]) & set(bound))
                counts["selected_observed_regions"] += int(f["region"].split(":")[0]) in bound
            if "execution" in p:
                entry = p["execution"]
                key = next(k for k,f in records.items() if f["request"]["context"] == b["request"]["context"]
                           and f["request"]["turn"]+1 == b["request"]["turn"])
                observed[key] = entry
            if b["kind"] != "grow":
                continue
            check(b["old_kernel"] == current_kernel and b["old_checkpoint"] == current_checkpoint
                  and b["old_world"] == old[0]["world"], "PROCESS_GROWTH_READ_PRIOR")
            k = manifest(root/"growth"/b["name"]/"kernel", b["kernel"])
            additions = b["units"][len(bank):]
            check(b["units"][:len(bank)] == bank and len(additions) == len(b["keys"]), "PROCESS_GROWTH_READ_APPEND")
            for unit,key in zip(additions,b["keys"]):
                f, entry = records[key], observed[key]
                response = k["pairs"][str(unit)]["response"]
                support = b["supports"][str(unit)]
                check(unit == len(rows) and f["request"]["reference"]["partition"] == "train"
                      and support == dict(action=f["support"][str(f["request"]["turn"])],
                                          response=entry["episode"], forecast=key)
                      and support["action"] in promoted and support["response"] in promoted
                      and scopes[support["response"]] == "observed_process_in_pinned_runtime"
                      and response == dict(kind="captured_process_output", unit=unit,
                          observation=entry["observation"], native=entry["native"],
                          target=episodes[entry["episode"]]["source"]["reexecution_of"]),
                      "PROCESS_GROWTH_READ_OCCURRENCE")
                bound[unit] = dict(observation=entry["observation"], episode=entry["episode"])
                rows[unit] = dict(text_sha256=digest([r for r in native_refs if r["observation"] == entry["observation"]]))
                pair_map[str(unit)] = dict(action=f["request"]["reference"], response=response)
                source_id = digest(dict(parent=source_id, addition=response))
            new = checkpoint(root/"growth"/b["name"]/"model", b["checkpoint"])
            check(new[0]["world_source"] == source_id and new[0]["world"] == b["world"]
                  and new[0]["standard"] == old[0]["standard"]
                  and new[2].keys() == old[2].keys()
                  and all(np.array_equal(v,old[2][k]) for k,v in new[2].items())
                  and all(np.array_equal(v,old[3][k]) for k,v in new[3].items() if not k.startswith("world_")),
                  "PROCESS_GROWTH_READ_FROZEN")
            for i,spec in enumerate(inputs.specs):
                check(new[0]["world_rows"][spec.name] == [f"{u}:{rows[u]['text_sha256']}" for u in b["units"]],
                      "PROCESS_GROWTH_READ_ROWS")
                for j,u in enumerate(b["units"]):
                    actual = new[3][f"world_values_{i}"][j,~new[3][f"world_padding_{i}"][j]]
                    check(np.array_equal(actual,windows(spec.name,u)), "PROCESS_GROWTH_READ_NATIVE")
                    counts["added_windows" if u in additions else "retained_windows"] += len(actual)
            states = {}
            for i,c in enumerate(new[1]["contexts"]):
                check(new[4][i].tobytes() == hidden[c["context"]]["raw"]
                      and c["sequence"] == hidden[c["context"]]["sequence"]
                      and c["last_event"] == hidden[c["context"]]["last_event"], "PROCESS_GROWTH_READ_STATE")
                states[c["context"]] = dict(sequence=c["sequence"],last_event=c["last_event"],hidden=fingerprint(new[4][i]))
            check(digest(states) == b["context_state"], "PROCESS_GROWTH_READ_STATE_PIN")
            versions.append(dict(name=b["name"],units=b["units"],recall=calibrate(k),source=source_id))
            counts["growths"] += 1
            old, bank, current_kernel, current_checkpoint = new, b["units"], b["kernel"], b["checkpoint"]
    snapshot = json.loads((root/"snapshot.json").read_bytes())
    check(snapshot["region_units"] == bank and snapshot["kernel"] == current_kernel
          and snapshot["active_model"] == current_checkpoint, "PROCESS_GROWTH_READ_FINAL")
    return dict(complete=True,counts=counts,versions=versions,scientific_admission=False,
                predictive_capability=False)
