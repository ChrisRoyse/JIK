"""Separate persisted-byte audit: real sources, complete slots, dynamics and standard."""
import argparse
import base64
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from audit.text_learning_readback import checkpoint, journal, sha
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need
from text.constellation_input import Corpus
from text.corpus_bank import Bank
from senses.contract import digest


def semantic_book(binding, base, values):
    book = binding["book"]
    need(digest(book) == binding["book_pin"] and book["fit"] == base["references"], "SEMANTIC_READBACK_BOOK")
    fit = book["fit"]
    def gap(i, j):
        return sum(distance(values[i][n], values[j][n]) for n in values[i])
    chosen, best = [], {i: float("inf") for i in fit}
    candidate = fit[0]
    for _ in range(book["regions"]):
        chosen.append(candidate)
        for i in fit:
            best[i] = -1 if i in chosen else min(best[i], gap(i, candidate))
        candidate = max(fit, key=best.__getitem__)
    codes = [min(range(len(chosen)), key=lambda j: gap(i, chosen[j])) for i in range(len(values))]
    need(chosen == book["exemplars"] and codes == book["codes"], "SEMANTIC_READBACK_ASSIGNMENTS")
    need(np.bincount([codes[i] for i in fit], minlength=book["regions"]).tolist() == book["train_occupancy"],
         "SEMANTIC_READBACK_OCCUPANCY")
    return chosen


def distance(a, b):
    costs = 1-np.clip(a@b.T, -1, 1)
    table = np.zeros((len(a)+1, len(b)+1), dtype=np.float64)
    table[:, 0], table[0, :] = np.arange(len(a)+1), np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            table[i, j] = min(table[i-1, j]+1, table[i, j-1]+1, table[i-1, j-1]+costs[i-1, j-1])
    return float(table[-1, -1]/max(len(a), len(b)))


def source_scores(a, report):
    corpus = Corpus(a.corpus, a.corpus_pin)
    bank = Bank(a.bank, a.bank_pin, corpus)
    sources, offset = [], 0
    for index, doc in enumerate(corpus.documents):
        rows = corpus.document(index)
        if doc["domain"] in ("general", "legal", "science"):
            sources.extend((row, offset+j) for j, row in enumerate(rows))
        offset += len(rows)
    need(len(sources) == 203, "DYNAMICS_READBACK_POPULATION")
    binding = report["data"]
    if binding["schema"] == "text-association-learning-v1":
        binding = binding["base"]
    base = binding
    causal = base["schema"] == "text-source-causal-regions-v1"
    semantic = base["schema"] == "text-semantic-region-sequence-v1"
    while "groups" not in base:
        base = base["base"]
    groups = {i: g for g in base["groups"] for i in g["indices"]}
    values = {}
    for index, source in enumerate(sources):
        values[index] = {}
        for name, (array, _) in bank.constellation(*source).items():
            array = array.astype(np.float64)
            values[index][name] = array/np.linalg.norm(array, axis=1, keepdims=True)
    fixed = semantic_book(binding, base, values) if semantic else None
    if report["data"]["schema"] == "text-association-learning-v1":
        association_targets(a.run, report["data"], values)
    counts = dict(prefixes=0, native_scores=0, source_successors=0)
    for name in report["tapes"]:
        with sqlite3.connect(f"file:{a.run/(name+'.db')}?mode=ro", uri=True) as db:
            pending = None
            for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
                body = json.loads(raw)
                payload = json.loads(db.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()[0])
                if body["kind"] == "source_prediction":
                    left = body["request"]["left"]
                    group = groups[left]
                    shared = [i for i in base["references"] if not causal or i not in group["indices"]]
                    indices = list(dict.fromkeys(shared+group["indices"][:group["indices"].index(left)+1]))
                    if semantic:
                        indices = fixed
                    definition = payload["private_region"]["definition"]
                    need(definition["rows"] == [sources[i][0]["atom_id"] for i in indices]
                         and definition["left"] == left and definition["context"] == group["context"]
                         and definition["through"] == body["request"]["through"] == sources[left][0]["char_end"]
                         and body["request"]["source"] == sources[left][0]["atom_id"]
                         and payload["region"] in definition["rows"], "DYNAMICS_READBACK_PREFIX")
                    pending = payload
                    counts["prefixes"] += 1
                else:
                    right = body["right"]
                    current, target = sources[left][0], sources[right][0]
                    need(body["left"] == left and current["sequence"] == target["sequence"]
                         and current["pos"]+1 == target["pos"] and current["byte_end"] == target["byte_start"]
                         and payload["target"] == target["atom_id"], "DYNAMICS_READBACK_SUCCESSOR")
                    if semantic:
                        need(payload["region_vocabulary"] == fixed
                             and payload["label"] == binding["book"]["codes"][right], "SEMANTIC_READBACK_TARGET_SHIFT")
                    need(set(pending["predictions"]) == set(values[right]), "DYNAMICS_READBACK_ROSTER")
                    for lane, slot in pending["predictions"].items():
                        raw = base64.b64decode(slot["base64"], validate=True)
                        need(sha(raw) == slot["sha256"] and slot["dtype"] == "float32-le", "DYNAMICS_READBACK_SLOT")
                        predicted = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(slot["shape"])
                        predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
                        metric = payload["instruments"][lane]
                        need(abs(distance(values[right][lane], predicted)-metric["prediction"]) < 1e-12
                             and abs(distance(values[right][lane], values[left][lane])-metric["persistence"]) < 1e-12
                             and len(predicted) == metric["predicted_windows"]
                             and len(values[right][lane]) == metric["target_windows"], "DYNAMICS_READBACK_SCORE")
                        counts["native_scores"] += 1
                    pending = None
                    counts["source_successors"] += 1
    need(counts == dict(prefixes=591, native_scores=3546, source_successors=591), "DYNAMICS_READBACK_COVERAGE")
    return counts


def association_targets(root, binding, values):
    matrix = np.load(root/"teacher.npy", allow_pickle=False)
    need(matrix.dtype == np.float32 and list(matrix.shape) == binding["teacher_shape"]
         and sha(matrix.astype("<f4").tobytes()) == binding["teacher_sha256"], "ASSOCIATION_READBACK_TARGET_BYTES")
    for lane, name in enumerate(binding["lanes"]):
        for q, left in enumerate(binding["nodes"]):
            for r, right in enumerate(binding["train_nodes"]):
                need(matrix[lane, q, r] == np.float32(1-distance(values[left][name], values[right][name])),
                     "ASSOCIATION_READBACK_NATIVE_TARGET")


def run(a):
    torch.set_num_threads(32)
    need(file_sha(a.run/"report.json") == a.pin, "DYNAMICS_READBACK_PIN")
    report = json.loads((a.run/"report.json").read_bytes())
    journals = {n: journal(a.run/(n+".db"), r["retained"], r) for n, r in report["tapes"].items()}
    checkpoints = {n: checkpoint(a.run/n, pin) for n, pin in report["checkpoints"].items()}
    base = checkpoints["initial-training"][1]
    changes = {}
    steps = (8, 16) if report["schema"] == "text-association-learning-witness-v1" else (16, 40)
    for name, step in zip(("cycle-1", "cycle-2"), steps):
        manifest, after = checkpoints[name]
        allowed = set(manifest["names"])
        changed = [n for n in base if not torch.equal(base[n], after[n])]
        need(set(after) == set(base) and changed and set(changed) <= allowed
             and not any(n.startswith("core.identity.") for n in changed), "DYNAMICS_READBACK_FROZEN")
        for prefix in ("core.projectors.", "core.temporal_projectors.", "core.update.", "core.mix.", "query.", "place."):
            need(any(n.startswith(prefix) for n in changed), "DYNAMICS_READBACK_TRAINABLE:"+prefix)
        need(manifest["step"] == step, "DYNAMICS_READBACK_CURSOR")
        changes[name] = dict(optimizer_steps=step, changed_tensors=changed,
            exact_identity_tensors=sum(n.startswith("core.identity.") for n in base),
            frozen_pin=manifest["frozen"])
    need(checkpoints["cycle-1"][0]["parent"] == report["checkpoints"]["initial-training"]
         and checkpoints["cycle-2"][0]["parent"] == report["checkpoints"]["cycle-1"], "DYNAMICS_READBACK_PARENT")
    counts = source_scores(a, report)
    publish(a.output, dict(schema="text-dynamics-independent-byte-readback-v1", run=a.pin,
        journals=journals, changes=changes, source_counts=counts, scientific_admission=False,
        served=False, acceptance="author-checked only"))
    print(json.dumps(dict(report_sha256=file_sha(a.output), counts=counts)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("run", "corpus", "bank", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("pin", "corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
