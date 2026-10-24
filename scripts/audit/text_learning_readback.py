"""Independent persisted-byte readback of text forecast order and frozen/update boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from senses.contract import canonical, digest
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need
from text.constellation_input import Corpus


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def journal(path, retained, expected):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    head, pending, count, lanes, revealed = "0"*64, None, 0, 0, []
    try:
        contract = json.loads(connection.execute("SELECT contract FROM metadata").fetchone()[0])
        need(digest(contract) == retained["contract"], "READBACK_JOURNAL_CONTRACT")
        for seq, previous, raw, pin in connection.execute("SELECT seq,previous,body,hash FROM events ORDER BY seq"):
            body = json.loads(raw)
            need(seq == count and previous == head and raw == canonical(body)
                 and pin == digest(dict(seq=seq, previous=previous, body=body)), "READBACK_JOURNAL_CHAIN")
            value = connection.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()
            need(value is not None and sha(value[0]) == body["payload"], "READBACK_JOURNAL_PAYLOAD")
            payload = json.loads(value[0])
            need(value[0] == canonical(payload), "READBACK_JOURNAL_CANONICAL")
            if body["kind"] == "source_prediction":
                need(pending is None and "target" not in payload and "label" not in payload,
                     "READBACK_EARLY_TARGET")
                pending = (pin, body["request"]["left"], payload)
                lanes += len(payload["predictions"])
                need(payload["prior"]["sequence"] == payload["after"]["sequence"] == 0,
                     "READBACK_FABRICATED_ECHO")
                need(np.isclose(np.exp(payload["probabilities"]).sum(), 1., atol=1e-6), "READBACK_LOGPROB")
            else:
                need(body["kind"] == "source_reveal" and pending is not None
                     and body["prediction"] == pending[0] == previous and body["left"] == pending[1],
                     "READBACK_REVEAL_ORDER")
                need(payload["region_nll_bits"] == -pending[2]["probabilities"][payload["label"]]/np.log(2),
                     "READBACK_LOGSCORE")
                revealed.append(dict(left=body["left"], **payload))
                pending = None
            head, count = pin, count+1
        need(pending is None and head == retained["head"] and count == 394 and lanes == 197*6,
             "READBACK_JOURNAL_COVERAGE")
        need(digest(revealed) == expected["outcomes"], "READBACK_OUTCOMES")
        return dict(events=count, committed_predictions=count//2, typed_predicted_slots=lanes,
                    head=head, source_outcomes=digest(revealed))
    finally:
        connection.close()


def checkpoint(root, pin):
    raw = (root/"manifest.json").read_bytes()
    manifest = json.loads(raw)
    need(sha(raw) == pin == (root/"COMMITTED").read_text(), "READBACK_CHECKPOINT_PIN")
    optimizer_raw = (root/"training.pt").read_bytes()
    need(len(optimizer_raw) == manifest["bytes"] and sha(optimizer_raw) == manifest["sha256"], "READBACK_OPTIMIZER")
    model_raw = (root/"model"/"manifest.json").read_bytes()
    model_manifest = json.loads(model_raw)
    need(sha(model_raw) == manifest["model"] == (root/"model"/"COMMITTED").read_text(), "READBACK_MODEL_PIN")
    need(file_sha(root/"model"/"state.pt") == model_manifest["sha256"], "READBACK_MODEL_BYTES")
    state = torch.load(root/"model"/"state.pt", weights_only=True, map_location="cpu")["model"]
    optimizer = torch.load(root/"training.pt", weights_only=True, map_location="cpu")["optimizer"]
    steps = [int(s["step"]) for s in optimizer["state"].values()]
    need((not steps if manifest["step"] == 0 else len(steps) == len(manifest["names"]))
         and all(s == manifest["step"] for s in steps), "READBACK_OPTIMIZER_CURSOR")
    need(manifest["flags"]["serving"] is False and model_manifest["serving"] is False, "READBACK_SERVING")
    return manifest, state


def run(a):
    need(file_sha(a.run/"report.json") == a.pin, "READBACK_REPORT_PIN")
    report = json.loads((a.run/"report.json").read_bytes())
    journals = {n: journal(a.run/(n+".db"), v["retained"], v) for n, v in report["tapes"].items()}
    if report["schema"] == "text-prefix-region-replay-v1":
        prefix_readback(a, report, journals)
        return
    checkpoints = {n: checkpoint(a.run/n, pin) for n, pin in report["checkpoints"].items()}
    base = checkpoints["initial-training"][1]
    changes = {}
    for name in ("cycle-1", "cycle-2"):
        manifest, after = checkpoints[name]
        need(set(after) == set(base), "READBACK_MODEL_ROSTER")
        changed = [n for n in base if not torch.equal(base[n], after[n])]
        need(changed and all(n.startswith(("query.", "place.")) for n in changed), "READBACK_FROZEN_DRIFT")
        changes[name] = dict(optimizer_steps=manifest["step"], changed_tensors=changed,
                             unchanged_frozen_tensors=len(base)-len(changed))
    need(checkpoints["cycle-2"][0]["step"] == 1520
         and checkpoints["cycle-2"][0]["parent"] == report["checkpoints"]["cycle-1"], "READBACK_PARENT")
    publish(a.output, dict(schema="text-learning-independent-byte-readback-v1", report=a.pin,
        journals=journals, changes=changes, checks="committed_source_order; optimizer_cursor; actual_tensor_differences",
        acceptance="author-checked only", scientific_admission=False, served=False))
    print(json.dumps(dict(report_sha256=file_sha(a.output), predictions=sum(j["committed_predictions"] for j in journals.values()),
                          final_optimizer_steps=1520)))


def prefix_readback(a, report, journals):
    need(a.parent is not None and file_sha(a.parent/"report.json") == report["parent"], "READBACK_PREFIX_PARENT")
    parent = json.loads((a.parent/"report.json").read_bytes())
    corpus = Corpus(a.corpus, a.corpus_pin)
    rows = []
    for index, doc in enumerate(corpus.documents):
        if doc["domain"] in ("general", "legal", "science"):
            rows.extend(corpus.document(index))
    base = report["data"]["base"]
    need(len(rows) == 203 and base == parent["data"], "READBACK_PREFIX_SOURCE")
    groups = {i: g for g in base["groups"] for i in g["indices"]}
    checks, identical = 0, {}
    for name in journals:
        with sqlite3.connect(f"file:{a.run/(name+'.db')}?mode=ro", uri=True) as db:
            for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
                body = json.loads(raw)
                if body["kind"] != "source_prediction":
                    continue
                left = body["request"]["left"]
                group = groups[left]
                indices = list(dict.fromkeys(base["references"]+group["indices"][:group["indices"].index(left)+1]))
                payload = json.loads(db.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()[0])
                definition = payload["private_region"]["definition"]
                need(definition["rows"] == [rows[i]["atom_id"] for i in indices]
                     and definition["left"] == left and definition["context"] == group["context"]
                     and definition["through"] == body["request"]["through"] == rows[left]["char_end"]
                     and body["request"]["source"] == rows[left]["atom_id"]
                     and payload["region"] in definition["rows"], "READBACK_PREFIX_FUTURE_OR_CROSS_CONTEXT")
                checks += 1
        target = a.run/(name+"-model")
        raw = (target/"manifest.json").read_bytes()
        need(sha(raw) == report["snapshots"][name] == (target/"COMMITTED").read_text(), "READBACK_PREFIX_SNAPSHOT")
        manifest = json.loads(raw)
        need(file_sha(target/"state.pt") == manifest["sha256"], "READBACK_PREFIX_MODEL_BYTES")
        old_root = a.parent/"initial-model" if name == "baseline" else a.parent/name/"model"
        if name == "baseline":
            old_pin = parent["initial_model"]
        else:
            training_raw = (a.parent/name/"manifest.json").read_bytes()
            need(sha(training_raw) == parent["checkpoints"][name], "READBACK_PREFIX_PARENT_CHECKPOINT")
            old_pin = json.loads(training_raw)["model"]
        old_raw = (old_root/"manifest.json").read_bytes()
        need(sha(old_raw) == old_pin and file_sha(old_root/"state.pt") == json.loads(old_raw)["sha256"],
             "READBACK_PREFIX_PARENT_BYTES")
        old = torch.load(old_root/"state.pt", weights_only=True, map_location="cpu")["model"]
        new = torch.load(target/"state.pt", weights_only=True, map_location="cpu")["model"]
        need(set(old) == set(new) and all(torch.equal(old[n], new[n]) for n in old), "READBACK_HIDDEN_RETRAINING")
        identical[name] = len(old)
    publish(a.output, dict(schema="text-prefix-independent-byte-readback-v1", run=a.pin, parent=report["parent"],
        journals=journals, source_prefix_checks=checks, identical_model_tensors=identical,
        scientific_admission=False, served=False, acceptance="author-checked only"))
    print(json.dumps(dict(report_sha256=file_sha(a.output), prefix_checks=checks, exact_weights=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-pin")
    run(parser.parse_args())
