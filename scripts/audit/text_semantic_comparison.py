"""Read matched semantic-region experiments and train-only categorical baselines."""
import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from audit.text_learning_readback import checkpoint, journal
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need


def measures(path, groups):
    by_index = {i: g for g in groups for i in g["indices"]}
    rows, pending = {}, None
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
            event = json.loads(raw)
            payload = json.loads(db.execute("SELECT data FROM payloads WHERE hash=?",
                (event["payload"],)).fetchone()[0])
            if event["kind"] == "source_prediction":
                pending = payload
            else:
                group = by_index[event["left"]]
                need(payload["correct_region"] == (int(np.argmax(pending["probabilities"])) == payload["label"]),
                     "SEMANTIC_COMPARISON_ACCURACY")
                if group["split"] == "development":
                    rows.setdefault(group["domain"], []).append(payload)
                pending = None
    result = {}
    for domain, values in rows.items():
        result[domain] = dict(count=len(values),
            region_nll_bits=float(np.mean([v["region_nll_bits"] for v in values])),
            region_accuracy=float(np.mean([v["correct_region"] for v in values])),
            instruments={n: {k: float(np.mean([v["instruments"][n][k] for v in values]))
                for k in ("prediction", "persistence")} for n in values[0]["instruments"]})
    return result


def read(root, pin):
    need(file_sha(root/"report.json") == pin, "SEMANTIC_COMPARISON_PIN")
    report = json.loads((root/"report.json").read_bytes())
    source = report["data"]
    need(source["schema"] == "text-semantic-region-sequence-v1", "SEMANTIC_COMPARISON_SCHEMA")
    while "groups" not in source:
        source = source["base"]
    groups = source["groups"]
    codes = report["data"]["book"]["codes"]
    size = report["data"]["book"]["regions"]
    results = {}
    for stage in ("cycle-1", "cycle-2"):
        expected = report["tapes"][stage]
        journal(root/(stage+".db"), expected["retained"], expected)
        measured_rows = measures(root/(stage+".db"), groups)
        need(measured_rows == expected["summary"]["development"], "SEMANTIC_COMPARISON_SUMMARY")
        domains = ("general", "legal") if stage == "cycle-1" else ("general", "legal", "science")
        unigram, bigram = np.ones(size), np.ones((size, size))
        for g in groups:
            if g["split"] == "train" and g["domain"] in domains:
                for a, b in zip(g["indices"], g["indices"][1:]):
                    unigram[codes[b]] += 1
                    bigram[codes[a], codes[b]] += 1
        unigram /= unigram.sum()
        bigram /= bigram.sum(1, keepdims=True)
        per_domain = {}
        for g in groups:
            if g["split"] == "development":
                pairs = list(zip(g["indices"], g["indices"][1:]))
                measured = measured_rows[g["domain"]]
                per_domain[g["domain"]] = dict(count=len(pairs), model_nll_bits=measured["region_nll_bits"],
                    model_accuracy=measured["region_accuracy"],
                    unigram_nll_bits=math.fsum(-math.log2(unigram[codes[b]]) for _, b in pairs)/len(pairs),
                    bigram_nll_bits=math.fsum(-math.log2(bigram[codes[a], codes[b]]) for a, b in pairs)/len(pairs),
                    copy_accuracy=sum(codes[a] == codes[b] for a, b in pairs)/len(pairs),
                    native=measured["instruments"])
        results[stage] = per_domain
    initial = checkpoint(root/"initial-training", report["checkpoints"]["initial-training"])[1]
    return report, initial, results


def run(a):
    torch.set_num_threads(32)
    recurrent, base, rstats = read(a.recurrent, a.recurrent_pin)
    transformer, changed, tstats = read(a.transformer, a.transformer_pin)
    need(not recurrent["data"]["transformer"] and transformer["data"]["transformer"]
         and recurrent["data"]["book"] == transformer["data"]["book"], "SEMANTIC_COMPARISON_BOOK")
    need(set(base) < set(changed) and all(torch.equal(v, changed[n]) for n, v in base.items()),
         "SEMANTIC_COMPARISON_INITIAL_WEIGHTS")
    added = sorted(set(changed)-set(base))
    need(all(n.startswith("core.sequence_transformer.") for n in added), "SEMANTIC_COMPARISON_SCOPE")
    need(recurrent["optimizer_steps"] == transformer["optimizer_steps"] == 40
         and recurrent["supervised_transitions"] == transformer["supervised_transitions"] == 1520,
         "SEMANTIC_COMPARISON_EXPOSURES")
    result = dict(schema="native-semantic-sequence-comparison-v1", recurrent=a.recurrent_pin,
        transformer=a.transformer_pin, shared_initial_tensors=len(base), added_tensors=added,
        book=transformer["data"]["book_pin"], baselines="train_only_laplace_one_unigram_and_bigram; hard_copy_accuracy",
        recurrent_metrics=rstats, transformer_metrics=tstats,
        attention_gain_bits={s: {d: rstats[s][d]["model_nll_bits"]-tstats[s][d]["model_nll_bits"]
            for d in tstats[s]} for s in tstats}, scientific_admission=False,
        conditional_lens_information=False, acceptance="author-checked only")
    publish(a.output, result)
    print(json.dumps(dict(sha256=file_sha(a.output), shared_initial_tensors=len(base), added_tensors=len(added))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("recurrent", "transformer", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("recurrent-pin", "transformer-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
