"""Independent lower bounds for the actual bounded native placement head."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from audit.text_dynamics_readback import distance, semantic_book
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def lower_bound(target, reference):
    # For ||r|| <= .1, normalize(unit_reference+r) lies within asin(.1).
    # Relax each edit match independently: this can only LOWER achievable error.
    angles = np.arccos(np.clip(target@reference.T, -1, 1))
    costs = 1-np.cos(np.maximum(angles-math.asin(.1), 0))
    table = np.zeros((len(target)+1, len(reference)+1), np.float64)
    table[:, 0], table[0, :] = np.arange(len(target)+1), np.arange(len(reference)+1)
    for i in range(1, len(target)+1):
        for j in range(1, len(reference)+1):
            table[i, j] = min(table[i-1, j]+1, table[i, j-1]+1, table[i-1, j-1]+costs[i-1, j-1])
    return float(table[-1, -1]/max(len(target), len(reference)))


def run(a):
    need(file_sha(a.run/"report.json") == a.pin, "PLACEMENT_REPORT_PIN")
    report = json.loads((a.run/"report.json").read_bytes())
    binding = report["data"]
    need(binding["schema"] == "text-semantic-region-sequence-v1", "PLACEMENT_SCHEMA")
    base = binding
    while "groups" not in base:
        base = base["base"]
    source = Path(__file__).parents[1]/"model/constellation_world.py"
    need("residual = .1*residual / torch.linalg.vector_norm" in source.read_text(), "PLACEMENT_IMPLEMENTATION")
    corpus = Corpus(a.corpus, a.corpus_pin)
    bank = Bank(a.bank, a.bank_pin, corpus)
    values, offset = {}, 0
    for index, doc in enumerate(corpus.documents):
        rows = corpus.document(index)
        if doc["domain"] in ("general", "legal", "science"):
            for j, row in enumerate(rows):
                values[len(values)] = {n: x.astype(np.float64)/np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)
                    for n, (x, _) in bank.constellation(row, offset+j).items()}
        offset += len(rows)
    prototypes = semantic_book(binding, base, values)
    codes = binding["book"]["codes"]
    rows, work = [], 0
    for group in base["groups"]:
        if group["split"] != "development":
            continue
        for left, right in zip(group["indices"], group["indices"][1:]):
            for lane, target in values[right].items():
                fine = {i: lower_bound(target, values[i][lane]) for i in base["references"]}
                work += len(fine)
                coarse = min(fine[i] for i in prototypes)
                persistence = distance(target, values[left][lane])
                rows.append(dict(domain=group["domain"], left=left, right=right, lane=lane,
                    persistence=persistence, coarse_oracle_floor=coarse,
                    fitted_label_floor=fine[prototypes[codes[right]]], full_train_oracle_floor=min(fine.values()),
                    cannot_beat_persistence=coarse >= persistence))
    summary = {}
    for domain in ("general", "legal", "science"):
        summary[domain] = {}
        for lane in values[0]:
            subset = [r for r in rows if r["domain"] == domain and r["lane"] == lane]
            means = {n: math.fsum(r[n] for r in subset)/len(subset) for n in
                ("persistence", "coarse_oracle_floor", "fitted_label_floor", "full_train_oracle_floor")}
            summary[domain][lane] = dict(**means, count=len(subset),
                individual_impossible=sum(r["cannot_beat_persistence"] for r in subset),
                mean_impossible=means["coarse_oracle_floor"] >= means["persistence"])
    result = dict(schema="native-placement-reachability-audit-v1", report=a.pin, model_code=file_sha(source),
        radius=.1, maximum_angle_radians=math.asin(.1), source_native_comparisons=work,
        interpretation="optimistic_geometric_lower_bound; each_lane_may_choose_its_own_best_region",
        summary=summary, rows=rows, acceptance="author-checked only", scientific_admission=False)
    publish(a.output, result)
    print(json.dumps(dict(sha256=file_sha(a.output), comparisons=work,
        mean_impossible=sum(v["mean_impossible"] for d in summary.values() for v in d.values()))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("run", "corpus", "bank", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("pin", "corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
