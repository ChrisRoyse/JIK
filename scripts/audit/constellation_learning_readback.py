"""Raw SQLite, checkpoint tensor, and scalar score readback of learning trials."""
import io
import json
from pathlib import Path
import sqlite3

import torch

from audit.constellation_readback import decode, cosine, edit_distance
from claims.codata_source import sha
from claims.episodic_memory import canonical, digest, ZERO


def readback(root, inputs, expected):
    root = Path(root)
    db = sqlite3.connect(f"file:{(root/'science.sqlite3').as_posix()}?mode=ro", uri=True)
    head, count, decisions, commits = ZERO, 0, [], {}
    for seq, previous, raw, pin in db.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
        body = json.loads(raw)
        assert seq == count and previous == head and canonical(body) == raw
        assert pin == digest(dict(seq=seq, previous=previous, body=body))
        if body["kind"] == "prediction_commit":
            commits[body["id"]] = body["record"]
        if body["kind"] == "constellation_learning_decision":
            assert body["prior_head"] == head and body["result"]["selected"] is False
            decisions.append(body)
        head, count = pin, count+1
    db.close()
    assert head == expected["head"] and count == expected["events"]
    assert decisions == expected["learning_history"] and len(decisions) == 2
    reports = []
    for d in decisions:
        directory = root/"candidates"/d["candidate"]
        raw = (directory/"manifest.json").read_bytes()
        assert sha(raw) == d["pin"] == (directory/"COMMITTED").read_text()
        m = json.loads(raw)
        assert m["step"] == 8 and digest(m["binding"]) == d["binding"]
        assert m["parent"] == expected["active_model"]
        raw = (directory/"training.pt").read_bytes()
        assert len(raw) == m["bytes"] and sha(raw) == m["sha256"]
        opt = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")["optimizer"]
        assert len(opt["state"]) == len(m["names"]) == 24
        assert all(float(s["step"]) == 8 and torch.isfinite(s["exp_avg"]).all()
                   and torch.isfinite(s["exp_avg_sq"]).all() for s in opt["state"].values())
        trial = json.loads((root/(d["candidate"]+".json")).read_bytes())
        assert trial["final_fingerprint"] == trial["resumed_fingerprint"] == m["fingerprint"]
        assert trial["trace"][4:] == trial["resumed_trace"]
        assert trial["initial_fingerprint"]["model"] != trial["final_fingerprint"]["model"]
        initial_dir = root/"candidates"/(d["candidate"]+"-initial")/"model"
        states = []
        for model_dir in (initial_dir, directory/"model"):
            world = json.loads((model_dir/"manifest.json").read_bytes())
            raw = (model_dir/"world.pt").read_bytes()
            assert sha(raw) == world["payload_sha256"]
            extra = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
            core = torch.load(model_dir/"core"/"weights.pt", weights_only=True, map_location="cpu")
            states.append({**{"core."+k: v for k, v in core.items()}, **extra})
        old, new = states
        assert old.keys() == new.keys()
        changed = [n for n in old if not torch.equal(old[n], new[n])]
        assert changed and all(n.startswith(("query.", "place.")) for n in changed)
        assert all(any(n.startswith(branch+"."+spec.name+".") for n in changed)
                   for branch in ("query", "place") for spec in inputs.specs)
        for label in ("baseline", "assessment"):
            outputs = trial[label]
            assert digest(outputs) == d[label]
            for key, o in outputs.items():
                r = commits[key]
                for name, saved in o["assessment"]["instruments"].items():
                    a, ptr = inputs.arrays[name]
                    unit = inputs.unit("row", r["target_edition"], r["quantity"])
                    target = [list(map(float, v)) for v in a[ptr[unit]:ptr[unit+1]]]
                    old = r["baseline_unit"]
                    persistence = [list(map(float, v)) for v in a[ptr[old]:ptr[old+1]]]
                    predicted = decode(o["predictions"][name])
                    error = edit_distance([[cosine(x, y) for y in predicted] for x in target])
                    reference = edit_distance([[cosine(x, y) for y in persistence] for x in target])
                    assert abs(error-saved["prediction_loss"]) < 1e-12
                    assert abs(reference-saved["persistence_loss"]) < 1e-12
        for key, lanes in d["result"]["comparisons"].items():
            for name, v in lanes.items():
                assert v["candidate"] == trial["assessment"][key]["assessment"]["instruments"][name]["prediction_loss"]
                assert v["parent"] == trial["baseline"][key]["assessment"]["instruments"][name]["prediction_loss"]
                assert v["better_than_parent"] == (v["candidate"] < v["parent"])
                assert v["better_than_persistence"] == (v["candidate"] < v["persistence"])
        qualified = all(v["better_than_parent"] and v["better_than_persistence"]
                        for lanes in d["result"]["comparisons"].values() for v in lanes.values())
        assert qualified == d["result"]["metric_gate_passed"]
        reports.append(dict(candidate=d["candidate"], optimizer_states=len(opt["state"]),
                            steps=8, exact_recovery=True, changed_parameters=changed,
                            frozen_tensors_equal=True, metric_gate_passed=qualified))
    return dict(events=count, head=head, candidates=reports, selected=0,
                full_native_scalar_readback=True, acceptance="author-checked")
