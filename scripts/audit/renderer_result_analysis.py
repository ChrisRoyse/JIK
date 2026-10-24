"""Paired document sampling intervals and exact-text novelty, not composition."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_comparison_readback import inspect
from audit.renderer_population_readback import sha, check
from senses.trajectory_population import publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "population", "conditioning", "historical", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--historical-sha256", required=True)
    a = p.parse_args(); check(not a.output.exists(), "RESULT_OUTPUT_EXISTS")
    inspect(a.comparison, a.population, a.conditioning, a.report_sha256)
    check(sha(a.historical/"report.json") == a.historical_sha256, "RESULT_HISTORY_PIN")
    hist = json.loads((a.historical/"report.json").read_bytes())
    source = a.historical/"observed.jsonl.gz"
    check(sha(source) == hist["files"][source.name]["sha256"], "RESULT_HISTORY_BYTES")
    population = [json.loads(line) for line in (a.population/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    arms = ("true", "gaussian", "predicted")
    scores = {arm: [json.loads(line) for line in (a.comparison/arm/"scores.jsonl").read_text().splitlines()]
              for arm in arms}
    renders = [dict(arm=arm, **json.loads(line)) for arm in arms
               for line in (a.comparison/arm/"renders.jsonl").read_text(encoding="utf-8").splitlines()]
    a.output.mkdir(parents=True); samples = {}; findings = {}
    for si, subset in enumerate(("all_evaluation", "exact_unseen_evaluation")):
        selected = [i for i, row in enumerate(population) if row["split"] == "evaluation" and
                    (si == 0 or not row["seen_in_predictor_training"])]
        check(len(selected) > 2, "RESULT_UNDERPOWERED")
        ix = np.random.default_rng(20261019+si).integers(0, len(selected), size=(2000, len(selected)), dtype=np.uint16)
        samples[subset+"__indices"] = ix
        weights = np.asarray([len(population[i]["atoms"][-1]["text"].encode()) for i in selected])
        bpb = {}; rows_by_arm = {}
        for arm in arms:
            by_id = {r["row"]: r for r in scores[arm] if r["route"] == "predicted" and r["split"] == "evaluation"}
            values = np.asarray([by_id[i]["byte_nll_bits"] for i in selected])
            rows_by_arm[arm] = values
            bpb[arm] = float(values.sum()/weights.sum())
        comparisons = {}
        for base in ("true", "gaussian"):
            delta = rows_by_arm["predicted"]-rows_by_arm[base]
            boot = delta[ix].sum(1)/weights[ix].sum(1)
            samples[subset+"__predicted_minus_"+base] = boot
            comparisons[base] = dict(delta_bits_per_byte=float(delta.sum()/weights.sum()),
                                     percentile95=np.percentile(boot, [2.5, 97.5]).tolist())
        findings[subset] = dict(documents=len(selected), target_bytes=int(weights.sum()),
                               predicted_input_bpb=bpb, predicted_minus_baseline=comparisons,
                               intervals="paired document bootstrap; one training seed; not entity or seed uncertainty")
    normalize = lambda text: " ".join(text.lower().split())
    wanted = {normalize(r["text"]): [] for r in renders if r["status"] == "complete_unasserted"}
    # A real training-target positive control must be found in the pinned history.
    control = normalize(next(r["atoms"][-1]["text"] for r in population if r["split"] == "train"))
    wanted.setdefault(control, []); historical_atoms = 0
    with gzip.open(source, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line); historical_atoms += len(record["texts"])
            for text in record["texts"]:
                key = normalize(text)
                if key in wanted:
                    wanted[key].append(dict(history=record["history"], domain=record["domain"], doc=record["doc"]))
    check(bool(wanted[control]) and historical_atoms == hist["counts"]["unique_atoms"], "RESULT_NOVELTY_POSITIVE_CONTROL")
    novelty = []
    for r in renders:
        complete = r["status"] == "complete_unasserted"
        hits = wanted[normalize(r["text"])] if complete else None
        novelty.append(dict(arm=r["arm"], route=r["route"], row=r["row"], status=r["status"],
                            matches_observed_historical_atom=bool(hits) if complete else None,
                            historical_matches=hits, exact_target=r["exact_target"],
                            text_difference_is_composition=False, asserted=False))
    summary = {}
    for arm in arms:
        summary[arm] = {}
        for route in ("true", "predicted"):
            part = [r for r in novelty if r["arm"] == arm and r["route"] == route]
            summary[arm][route] = dict(rows=len(part), statuses=dict(Counter(r["status"] for r in part)),
                complete=sum(r["status"] == "complete_unasserted" for r in part),
                exact_target=sum(r["exact_target"] for r in part),
                exact_historical_matches=sum(r["matches_observed_historical_atom"] is True for r in part), asserted=0)
    np.savez(a.output/"resamples.npz", **samples)
    publish(a.output/"render-novelty.json", novelty)
    check(sha(source) == hist["files"][source.name]["sha256"], "RESULT_HISTORY_DRIFT")
    publish(a.output/"report.json", dict(schema="renderer-result-analysis-v1", acceptance="author-checked",
            comparison_sha256=a.report_sha256, historical_sha256=a.historical_sha256,
            findings=findings, render_summary=summary, normalized_historical_atoms_scanned=historical_atoms,
            historical_match_rule="Unicode lower + whitespace collapse; exact complete retained atom only",
            historical_positive_control_matches=len(wanted[control]), resamples_sha256=sha(a.output/"resamples.npz"),
            novelty_sha256=sha(a.output/"render-novelty.json"), guard_evaluated=False,
            independent_anchors_evaluated=False, scientific_admission=False, composition_success=False,
            limitations=["short bounded diagnostic; 800 steps; one training seed", "exact text novelty is not new meaning",
                         "no semantic near-copy or anchor correctness certificate", "refused outputs have no novelty verdict"]))
    print(json.dumps(dict(findings=findings, render_summary=summary)))


if __name__ == "__main__":
    main()
