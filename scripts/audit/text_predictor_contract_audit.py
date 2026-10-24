"""Read the prediction vocabulary and its reuse from authenticated source tapes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.text_learning_readback import journal
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need


def inspect(root, pin):
    need(file_sha(root/"report.json") == pin, "PREDICTOR_AUDIT_REPORT_PIN")
    report = json.loads((root/"report.json").read_bytes())
    base = report["data"]
    while "groups" not in base:
        base = base["base"]
    groups = {i: g for g in base["groups"] for i in g["indices"]}
    tape = report["tapes"]["baseline"]
    checked = journal(root/"baseline.db", tape["retained"], tape)
    counts = {s: Counter() for s in ("train", "development")}
    sizes = {s: [] for s in counts}
    own = Counter()
    vocabularies = set()
    with sqlite3.connect(f"file:{root/'baseline.db'}?mode=ro", uri=True) as db:
        for raw, in db.execute("SELECT body FROM events ORDER BY seq"):
            body = json.loads(raw)
            if body["kind"] != "source_reveal":
                continue
            payload = json.loads(db.execute("SELECT data FROM payloads WHERE hash=?",
                                           (body["payload"],)).fetchone()[0])
            group = groups[body["left"]]
            split, refs = group["split"], payload["region_vocabulary"]
            target = refs[payload["label"]]
            counts[split][target] += 1
            sizes[split].append(len(refs))
            vocabularies.add(tuple(refs))
            own[split] += target in group["indices"]
    summary = {}
    for split, values in counts.items():
        summary[split] = dict(transitions=sum(values.values()), distinct_target_references=len(values),
            target_occurrence_histogram=dict(sorted(Counter(values.values()).items())),
            own_document_target_references=own[split], reference_count_range=[min(sizes[split]), max(sizes[split])],
            target_counts={str(k): v for k, v in sorted(values.items())})
    summary["development"]["targets_seen_as_training_labels"] = sum(
        v for k, v in counts["development"].items() if k in counts["train"])
    need(summary["train"]["transitions"] == 104 and summary["development"]["transitions"] == 93,
         "PREDICTOR_AUDIT_COVERAGE")
    return dict(report=pin, journals=checked, counts=summary, distinct_reference_vocabularies=len(vocabularies),
                training_documents=sum(g["split"] == "train" for g in base["groups"]),
                baseline_development=report["tapes"]["baseline"]["summary"]["development"],
                final_development=report["tapes"]["cycle-2"]["summary"]["development"])


def run(a):
    started = time.perf_counter()
    before, after = inspect(a.before, a.before_pin), inspect(a.after, a.after_pin)
    need(before["baseline_development"] == after["baseline_development"], "PREDICTOR_BASELINE_CHANGED")
    changes = {}
    for domain, final in after["final_development"].items():
        old = before["final_development"][domain]
        changes[domain] = dict(region_nll_bits_before=old["region_nll_bits"],
            region_nll_bits_after=final["region_nll_bits"],
            per_instrument={k: dict(before=old["instruments"][k]["prediction"], **v)
                            for k, v in final["instruments"].items()})
    code = Path(__file__).resolve().parents[1]/"model"
    paths = ("recognising_lm.py", "recognition.py", "constellation_lm.py", "native_identity_world.py",
             "constellation_world.py", "text_sequence_data.py", "text_context_regions.py", "text_causal_regions.py")
    publish(a.output, dict(schema="text-predictor-contract-audit-v1", before=before, after=after,
        baseline_development_identical=True, final_comparison=changes,
        inspected_code={n: file_sha(code/n) for n in paths},
        interpretation="passage-reference targets; not a reusable semantic-codebook validation",
        scientific_admission=False, acceptance="author-checked only",
        main_function_seconds=time.perf_counter()-started))
    compact = lambda x: {s: {k: v for k, v in row.items() if k != "target_counts"}
                         for s, row in x["counts"].items()}
    print(json.dumps(dict(sha256=file_sha(a.output), before=compact(before), after=compact(after))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("before", "after", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("before-pin", "after-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
