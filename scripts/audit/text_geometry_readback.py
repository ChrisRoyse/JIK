"""Independent source/cap calibration and actual predicted-window containment audit."""
import argparse
import base64
import json
import math
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from audit.text_learning_readback import checkpoint
from claims.codata_source import sha
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def run(a):
    torch.set_num_threads(32)
    need(file_sha(a.run/"report.json") == a.pin, "GEOMETRY_READBACK_PIN")
    report = json.loads((a.run/"report.json").read_bytes())
    geometry = report["data"]["geometry"]
    semantic = report["data"]["base"]
    book = semantic["book"]
    corpus = Corpus(a.corpus, a.corpus_pin)
    bank = Bank(a.bank, a.bank_pin, corpus)
    rows, groups, fit, offset = [], {}, [], 0
    for doc_index, doc in enumerate(corpus.documents):
        records = corpus.document(doc_index)
        if doc["domain"] in ("general", "legal", "science"):
            for j, record in enumerate(records):
                index = len(rows)
                rows.append((record, offset+j))
                groups[index] = ("train" if doc["ordinal"] == 0 else "development", doc["domain"])
                if doc["ordinal"] == 0:
                    fit.append(index)
        offset += len(records)
    need(fit == geometry["fit"] == book["fit"] and len(fit) == 107 and len(rows) == 203,
         "GEOMETRY_READBACK_TRAIN_ONLY")
    values = {name: [] for name in geometry["lanes"]}
    for source in rows:
        for name, (array, _) in bank.constellation(*source).items():
            need(array.shape == (1, 768), "GEOMETRY_READBACK_NATIVE_WINDOW")
            v = array[0].astype(np.float64)
            values[name].append(v/np.linalg.norm(v))
    radius = np.asarray(geometry["angles"], np.float32)
    need(radius.shape == (6, 8) and sha(radius.astype("<f4").tobytes()) == geometry["native_sha256"],
         "GEOMETRY_READBACK_BYTES")
    rebuilt = np.zeros((6, 8), np.float64)
    for lane, name in enumerate(geometry["lanes"]):
        for index in fit:
            region = book["codes"][index]
            angle = math.acos(np.clip(values[name][index]@values[name][book["exemplars"][region]], -1, 1))
            rebuilt[lane, region] = max(rebuilt[lane, region], angle)
    # Outward rounding is at most 1.5 float32 ULP; acos near one also has
    # square-root sensitivity to the independent dot-product rounding.
    tolerance = np.maximum(1e-7, 2*np.spacing(radius))
    need(np.all(np.abs(rebuilt-radius) <= tolerance) and np.all(rebuilt <= radius+1e-7), "GEOMETRY_READBACK_FIT")
    for name, pin in report["checkpoints"].items():
        tensor = checkpoint(a.run/name, pin)[1]["placement_angles"].numpy()
        need(np.array_equal(tensor, radius), "GEOMETRY_READBACK_FROZEN")
    control = report["representation_controls"]["geometry"]
    need(file_sha(a.run/control["artifact"]) == control["sha256"], "GEOMETRY_READBACK_CONTROL_PIN")
    analytic = np.load(a.run/control["artifact"], allow_pickle=False)
    need(len(analytic["indices"]) == 642, "GEOMETRY_READBACK_CONTROL_COUNT")
    max_error = 0.
    for j, (lane, index, region) in enumerate(zip(analytic["lanes"], analytic["indices"], analytic["regions"])):
        name = geometry["lanes"][lane]
        need(index in fit and region == book["codes"][index], "GEOMETRY_READBACK_CONTROL_SOURCE")
        for key, wanted in (("base", values[name][book["exemplars"][region]]), ("target", values[name][index])):
            need(np.max(np.abs(analytic[key][j]-wanted)) < 2e-7, "GEOMETRY_READBACK_CONTROL_VECTOR")
        predicted = analytic["predicted"][j].astype(np.float64)
        predicted /= np.linalg.norm(predicted)
        error = 1-predicted@values[name][index]
        max_error = max(max_error, float(error))
    need(max_error < 2e-6, "GEOMETRY_READBACK_REACHABILITY")
    row_ids = [rows[i][0]["atom_id"] for i in book["exemplars"]]
    counts, maximum_cap_excess, summaries = 0, 0., {}
    for stage in report["tapes"]:
        scores = {}
        with sqlite3.connect(f"file:{a.run/(stage+'.db')}?mode=ro", uri=True) as db:
            pending = None
            for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
                body = json.loads(raw)
                payload = json.loads(db.execute("SELECT data FROM payloads WHERE hash=?", (body["payload"],)).fetchone()[0])
                if body["kind"] == "source_prediction":
                    pending = payload
                    left = body["request"]["left"]
                    continue
                need(pending is not None and body["left"] == left, "GEOMETRY_READBACK_PAIR")
                region = row_ids.index(pending["region"])
                for lane, name in enumerate(geometry["lanes"]):
                    slot = pending["predictions"][name]
                    raw_vector = base64.b64decode(slot["base64"], validate=True)
                    need(sha(raw_vector) == slot["sha256"] and slot["shape"] == [1, 768], "GEOMETRY_READBACK_PREDICTION")
                    prediction = np.frombuffer(raw_vector, dtype="<f4").astype(np.float64)
                    norm = np.linalg.norm(prediction)
                    need(abs(norm-1) < 2e-6, "GEOMETRY_READBACK_UNIT_SPHERE")
                    prediction /= norm
                    dot = prediction@values[name][book["exemplars"][region]]
                    cap = radius[lane, region] if geometry["mode"] == "derived" else math.asin(.1)
                    excess = math.cos(float(cap))-dot
                    maximum_cap_excess = max(maximum_cap_excess, float(excess))
                    need(excess < 2e-6, "GEOMETRY_READBACK_CAP_ESCAPE")
                    target = values[name][body["right"]]
                    error = float(1-np.clip(prediction@target, -1, 1))
                    persistence = float(1-np.clip(values[name][left]@target, -1, 1))
                    scores.setdefault((*groups[left], name), []).append((error, persistence))
                    counts += 1
                pending = None
        summaries[stage] = []
        for (split, domain, name), items in scores.items():
            prediction, persistence = np.mean(items, axis=0)
            old = report["tapes"][stage]["summary"][split][domain]["instruments"][name]
            need(abs(prediction-old["prediction"]) < 1e-12 and abs(persistence-old["persistence"]) < 1e-12,
                 "GEOMETRY_READBACK_SUMMARY")
            summaries[stage].append(dict(split=split, domain=domain, lane=name, prediction=float(prediction),
                                        persistence=float(persistence), count=len(items)))
    need(counts == 3546, "GEOMETRY_READBACK_COVERAGE")
    publish(a.output, dict(source=a.pin, mode=geometry["mode"], native_predictions=counts,
        fit_atoms=len(fit), calibrated_caps=48, stored_bytes=radius.nbytes,
        max_calibration_numeric_difference=float(np.max(np.abs(rebuilt-radius))),
        max_cap_cosine_excess=maximum_cap_excess, analytic_reachability_error=max_error,
        radii_radians=radius.tolist(), summaries=summaries, learned_capability=False,
        acceptance="author-checked only", scientific_admission=False))
    print(json.dumps(dict(sha256=file_sha(a.output), native_predictions=counts, max_cap_excess=maximum_cap_excess)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("run", "corpus", "bank", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("pin", "corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
