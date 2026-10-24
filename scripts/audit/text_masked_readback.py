"""Independent NumPy readback of frozen source geometry and masked relationship scores."""
import argparse
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
import torch

from claims.codata_source import check, sha
from model.contracts import ModelConfig
from model.text_sequence_data import SequenceData
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def logprob(a):
    shifted = a-np.max(a, axis=-1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def metrics(native, values, groups, fit):
    scaled = native.astype(np.float64)/.1
    raw = scaled[:, fit].copy()
    for j in range(len(fit)):
        raw[:, j, j] = -np.inf
    prior = np.exp(logprob(raw)).mean(axis=1)
    result = {}
    for group in groups:
        if group["split"] != "development":
            continue
        ids = group["indices"]
        lp, lq = logprob(scaled[:, ids]), logprob(values[:, ids].astype(np.float64)/.1)
        p = np.exp(lp)
        kl = np.sum(p*(lp-lq), axis=-1)/np.log(2)
        pk = np.sum(p*(lp-np.log(prior[:, None])), axis=-1)/np.log(2)
        target = np.argsort(-scaled[:, ids], axis=-1, kind="stable")[..., :5]
        ranked = np.argsort(-values[:, ids], axis=-1, kind="stable")[..., :5]
        recall = (target[..., :, None] == ranked[..., None, :]).any(axis=-1).mean(axis=-1)
        gain = np.sum(p*(lq-np.roll(lq, 1, axis=1)), axis=-1)/np.log(2)
        neighbors = []
        for lane in range(len(native)):
            other = np.delete(scaled, lane, axis=0)[:, ids].sum(axis=0)
            chosen = np.argmax(other, axis=-1)
            nearest = logprob(scaled[lane, fit][chosen])
            neighbors.append(np.sum(p[lane]*(lp[lane]-nearest), axis=-1)/np.log(2))
        result[group["domain"]] = [dict(kl_bits=float(kl[j].mean()),
            prior_kl_bits=float(pk[j].mean()), neighbor_baseline_kl_bits=float(neighbors[j].mean()),
            recall_at_5=float(recall[j].mean()), correct_vs_shuffled_bits=float(gain[j].mean()),
            prediction_std=float(values[j, ids].astype(np.float64).std(axis=0).mean()))
            for j in range(len(native))]
    return result


def run(a):
    corpus = Corpus(a.corpus, a.corpus_pin)
    bank = Bank(a.bank, a.bank_pin, corpus)
    data = SequenceData(corpus, bank, ModelConfig())
    fit, names = data.refs, [s.name for s in data.inputs.specs]
    full = {n: np.stack([data.array(n, i)[0] for i in range(len(data.sources))]) for n in names}
    native = np.stack([np.clip(full[n]@full[n][fit].T, -1, 1).astype(np.float32) for n in names])
    check(native.shape == (6, 203, 107), "MASKED_READBACK_POPULATION")
    reconstructed, report_pins, checkpoint_states = {}, {}, {}
    worst = 0.
    for arm in ("correct-v2", "shuffled-v2"):
        root = a.root/arm
        report = json.loads((root/"report.json").read_bytes())
        report_pins[arm] = file_sha(root/"report.json")
        check(report["data"]["train_nodes"] == fit and report["data"]["nodes"] == list(range(203))
              and report["data"]["lanes"] == names
              and report["data"]["source_atoms"] == [r[0]["atom_id"] for r in data.sources],
              "MASKED_READBACK_SOURCE")
        target = np.load(root/"teacher.npy", allow_pickle=False)
        check(np.array_equal(native, target), "MASKED_READBACK_NATIVE_GEOMETRY")
        arm_result, checkpoint_states[arm] = {}, {}
        for name, pin in report["checkpoints"].items():
            cp = root/name
            check(file_sha(cp/"manifest.json") == pin, "MASKED_READBACK_CHECKPOINT_PIN")
            manifest = json.loads((cp/"manifest.json").read_bytes())
            check(file_sha(cp/"state.pt") == manifest["state_sha256"], "MASKED_READBACK_CHECKPOINT_BYTES")
            state = torch.load(cp/"state.pt", map_location="cpu", weights_only=True)
            model = state["model"]
            check(np.array_equal(model["landmark_ids"].numpy(), fit), "MASKED_READBACK_LANDMARK_ORDER")
            for lane, n in enumerate(names):
                check(np.array_equal(model[f"keys_{lane}"].numpy(), full[n][fit].astype(np.float32)),
                      "MASKED_READBACK_FROZEN_KEYS")
            checkpoint_states[arm][name] = model
            check({int(s["step"].item()) for s in state["optimizer"]["state"].values()} ==
                  (set() if name == "initial" else {8 if name == "cycle-1" else 16}),
                  "MASKED_READBACK_OPTIMIZER_CURSOR")
            values = np.load(root/(name+".npy"), allow_pickle=False)
            check(values.shape == native.shape and np.isfinite(values).all(), "MASKED_READBACK_LOGITS")
            independently = metrics(native, values, data.groups, fit)
            arm_result[name] = {}
            for domain, rows in independently.items():
                arm_result[name][domain] = {n: rows[j] for j, n in enumerate(names)}
                for j, n in enumerate(names):
                    for k, v in rows[j].items():
                        difference = abs(v-report["metrics"][name][domain][n][k])
                        worst = max(worst, difference)
                        check(difference <= 1e-10, "MASKED_READBACK_SCORE")
        reconstructed[arm] = arm_result
        cold = json.loads((a.root/(arm+"-cold")/"report.json").read_bytes())
        check(cold["source"] == report_pins[arm] and cold["exact"] and cold["optimizer_exact"]
              and cold["rollback"], "MASKED_READBACK_COLD")
    first = checkpoint_states["correct-v2"]["initial"]
    control = checkpoint_states["shuffled-v2"]["initial"]
    check(set(first) == set(control) and all(torch.equal(v, control[n]) for n, v in first.items()),
          "MASKED_READBACK_MATCHED_INITIAL")
    comparisons = []
    for domain, rows in reconstructed["correct-v2"]["cycle-2"].items():
        for n, after in rows.items():
            before = reconstructed["correct-v2"]["initial"][domain][n]
            shuffled = reconstructed["shuffled-v2"]["cycle-2"][domain][n]
            comparisons.append(dict(domain=domain, instrument=n, **after,
                improvement_over_initial_bits=before["kl_bits"]-after["kl_bits"],
                improvement_over_prior_bits=after["prior_kl_bits"]-after["kl_bits"],
                improvement_over_shuffled_training_bits=shuffled["kl_bits"]-after["kl_bits"]))
    summary = {k: dict(wins=sum(r[k] > 0 for r in comparisons), cells=len(comparisons),
                      mean=float(np.mean([r[k] for r in comparisons]))) for k in
        ("improvement_over_initial_bits", "improvement_over_prior_bits", "improvement_over_shuffled_training_bits")}
    result = dict(schema="masked-relationship-independent-readback-v1", report_pins=report_pins,
        native_comparisons=native.size, prediction_values_read=int(native.size*3*2),
        metric_max_difference=worst, matched_initial_tensors=len(first),
        summary=summary, comparisons=comparisons, metrics=reconstructed,
        all_source_atoms_including_six_tails=True, frozen_keys_exact=True,
        scope="development_native_relationship_prediction_not_D2_information_or_grounded_understanding",
        acceptance="author-checked only", scientific_admission=False, goal_complete=False)
    publish(a.output, result)
    print(json.dumps(dict(sha256=file_sha(a.output), summary=summary, metric_max_difference=worst)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("root", "corpus", "bank", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
