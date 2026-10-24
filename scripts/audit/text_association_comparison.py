"""Matched input-scale comparison; association diagnostics are never D2 bits."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from audit.text_learning_readback import checkpoint
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need


def report(path, pin):
    need(file_sha(path) == pin, "ASSOCIATION_COMPARISON_PIN")
    return json.loads(path.read_bytes())


def run(a):
    torch.set_num_threads(32)
    raw = report(a.raw/"report.json", a.raw_pin)
    scaled = report(a.scaled/"report.json", a.scaled_pin)
    first = report(a.raw_readback/"report.json", a.raw_readback_pin)
    second = report(a.scaled_readback/"report.json", a.scaled_readback_pin)
    need(first["source"] == a.raw_pin and second["source"] == a.scaled_pin, "ASSOCIATION_COMPARISON_SOURCE")
    change = "association_representation" if "association_representation" in scaled["data"] else "content_input"
    if "geometry" in scaled["data"]:
        change = "geometry"
        need(raw["data"][change]["mode"] == "fixed" and scaled["data"][change]["mode"] == "derived",
             "GEOMETRY_COMPARISON_MODE")
        need({k: v for k, v in raw["data"][change].items() if k != "mode"}
             == {k: v for k, v in scaled["data"][change].items() if k != "mode"}, "GEOMETRY_COMPARISON_RECIPE")
    elif "regularizer" in scaled["data"]:
        change = "regularizer"
        need(raw["data"][change]["strength"] == 0. and scaled["data"][change]["strength"] == .02,
             "SIGREG_COMPARISON_STRENGTH")
        need({k: v for k, v in raw["data"][change].items() if k != "strength"}
             == {k: v for k, v in scaled["data"][change].items() if k != "strength"},
             "SIGREG_COMPARISON_RECIPE")
    need({k: v for k, v in raw["data"].items() if k != change}
         == {k: v for k, v in scaled["data"].items() if k != change}
         and raw["recipe"] == scaled["recipe"] and raw["training_node_exposures"] == scaled["training_node_exposures"] == 1520
         and raw["optimizer_steps"] == scaled["optimizer_steps"] == 16, "ASSOCIATION_COMPARISON_MATCH")
    before = checkpoint(a.raw/"initial-training", raw["checkpoints"]["initial-training"])[1]
    after = checkpoint(a.scaled/"initial-training", scaled["checkpoints"]["initial-training"])[1]
    need(set(before) == set(after) and all(torch.equal(v, after[n]) for n, v in before.items()),
         "ASSOCIATION_COMPARISON_INITIAL_TENSORS")
    rows = []
    for domain in ("general", "legal", "science"):
        for lane, b in first["metrics"]["cycle-2"][domain].items():
            c = second["metrics"]["cycle-2"][domain][lane]
            rows.append(dict(domain=domain, lane=lane,
                raw_kl_bits=b["kl_bits"], scaled_kl_bits=c["kl_bits"], improvement_bits=b["kl_bits"]-c["kl_bits"],
                raw_recall_at_5=b["neighbor_recall_at_5"], scaled_recall_at_5=c["neighbor_recall_at_5"],
                raw_gain_over_uniform_bits=b["gain_over_uniform_bits"], scaled_gain_over_uniform_bits=c["gain_over_uniform_bits"],
                raw_advantage_over_shuffled_bits=b["shuffled_correspondence_kl_bits"]-b["kl_bits"],
                scaled_advantage_over_shuffled_bits=c["shuffled_correspondence_kl_bits"]-c["kl_bits"]))
    result = dict(schema="typed-association-method-comparison-v1", change=change, raw=a.raw_pin, scaled=a.scaled_pin,
        raw_readback=a.raw_readback_pin, scaled_readback=a.scaled_readback_pin, initial_tensors_identical=len(before),
        rows=rows, improved_kl=sum(r["improvement_bits"] > 0 for r in rows),
        improved_recall=sum(r["scaled_recall_at_5"] > r["raw_recall_at_5"] for r in rows),
        input_scale=scaled["representation_controls"]["input_scale"],
        scientific_admission=False, conditional_lens_information=False, successful_acquisition_demonstrated=False,
        acceptance="author-checked only")
    if change == "geometry":
        result["native_prediction"] = [dict(domain=domain, lane=lane,
            fixed=r["prediction"], derived=scaled["tapes"]["cycle-2"]["summary"]["development"][domain]["instruments"][lane]["prediction"],
            persistence=r["persistence"])
            for domain, values in raw["tapes"]["cycle-2"]["summary"]["development"].items()
            for lane, r in values["instruments"].items()]
    publish(a.output, result)
    print(json.dumps(dict(sha256=file_sha(a.output), initial_tensors_identical=len(before),
        improved_kl=result["improved_kl"], improved_recall=result["improved_recall"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("raw", "scaled", "raw-readback", "scaled-readback", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("raw-pin", "scaled-pin", "raw-readback-pin", "scaled-readback-pin"):
        p.add_argument("--"+n, required=True)
    run(p.parse_args())
