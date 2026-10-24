"""Cold read of matched training orders, complete renders and per-byte scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import read, sha, check
from model.renderer_comparison import read_conditioning, ARMS
from senses.trajectory_population import publish


def decoded(tokens):
    # Independent byte parser; no use of the producing decoder.
    if 257 not in tokens:
        return "MNLM_SURFACE_NO_EOS", None
    content = tokens[:tokens.index(257)]
    if len(content) > 512:
        return "MNLM_SURFACE_OUTPUT_CAPACITY", None
    if any(not isinstance(t, int) or t < 0 or t > 255 for t in content):
        return "MNLM_SURFACE_INVALID_SYMBOL", None
    try:
        text = bytes(content).decode("utf-8")
    except UnicodeDecodeError:
        return "MNLM_SURFACE_INVALID_UTF8", None
    if not text.strip():
        return "MNLM_SURFACE_EMPTY_OUTPUT", None
    return "complete_unasserted", text


def inspect(root, population, conditioning, pin):
    check(sha(root/"report.json") == pin, "COMPARISON_PIN")
    report = json.loads((root/"report.json").read_bytes())
    check(report["schema"] == "renderer-comparison-v1" and tuple(report["arms"].keys()) != (), "COMPARISON_SCHEMA")
    check(set(report["arms"]) == set(ARMS), "COMPARISON_ARMS")
    check(all(report.get(k) is False for k in ("guard_evaluated", "independent_anchors_evaluated", "scientific_admission", "composition_success", "serving_ready", "final_test")), "COMPARISON_CLAIMS")
    read(population, report["population_sha256"])
    read_conditioning(conditioning, report["conditioning_sha256"], population, report["population_sha256"])
    names = {"gpu-before.txt", "initial.pt", "batches.npy"} | {a+"/"+f for a in ARMS for f in ("renderer.pt", "scores.jsonl", "renders.jsonl", "training.jsonl")}
    check(set(report["files"]) == names, "COMPARISON_FILES")
    for name, meta in report["files"].items():
        check(sha(root/name) == meta["sha256"] and (root/name).stat().st_size == meta["bytes"], "COMPARISON_FILE")
    cfg = report["config"]
    check({k: cfg[k] for k in ("width", "layers", "heads", "batch", "seed", "lr", "capacity", "gaussian_sigma")} ==
          dict(width=256, layers=4, heads=8, batch=32, seed=25, lr=3e-4, capacity=512, gaussian_sigma=.3), "COMPARISON_CONFIG")
    rows = [json.loads(s) for s in (population/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    train = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"])
    ev = np.asarray([i for i, r in enumerate(rows) if r["split"] != "train"])
    batches = np.load(root/"batches.npy", allow_pickle=False)
    check(np.array_equal(batches, np.random.default_rng(25).choice(train, size=(cfg["steps"], 32), replace=True)), "COMPARISON_ORDER")
    render_ix = [i for i in ev if rows[i]["split"] == "evaluation"][:16]
    result = {}
    for arm in ARMS:
        log = [json.loads(s) for s in (root/arm/"training.jsonl").read_text().splitlines()]
        expected_steps = sorted(set(range(1, cfg["steps"]+1, 100)) | {cfg["steps"]})
        check([x["step"] for x in log] == expected_steps and all(x["arm"] == arm and np.isfinite(x["loss"]) and x["loss"] >= 0 for x in log), "COMPARISON_TRAIN_LOG")
        scores = [json.loads(s) for s in (root/arm/"scores.jsonl").read_text().splitlines()]
        check(len(scores) == 4*len(ev), "COMPARISON_SCORE_ROWS")
        metrics = {s: {} for s in ("calibration", "evaluation")}
        for route in ("true", "predicted", "other_true", "other_predicted"):
            part = [r for r in scores if r["route"] == route]
            check([r["row"] for r in part] == ev.tolist(), "COMPARISON_SCORE_ORDER")
            expected_source = np.roll(ev, 1) if route.startswith("other_") else ev
            check([r["conditioning_row"] for r in part] == expected_source.tolist(), "COMPARISON_DERANGEMENT")
            for row in part:
                target = rows[row["row"]]
                check(row["split"] == target["split"] and row["bytes"] == len(target["atoms"][-1]["text"].encode()), "COMPARISON_BYTES")
                check(all(np.isfinite(row[k]) and row[k] >= 0 for k in ("byte_nll_bits", "eos_nll_bits")), "COMPARISON_NLL")
            for split in metrics:
                group = [r for r in part if r["split"] == split]
                total_bytes = sum(r["bytes"] for r in group)
                metrics[split][route] = dict(rows=len(group), bytes=total_bytes,
                    bits_per_byte=sum(r["byte_nll_bits"] for r in group)/total_bytes,
                    mean_eos_bits=sum(r["eos_nll_bits"] for r in group)/len(group))
        check(metrics == report["arms"][arm]["metrics"], "COMPARISON_METRICS")
        generated = [json.loads(s) for s in (root/arm/"renders.jsonl").read_text(encoding="utf-8").splitlines()]
        check([(r["route"], r["row"]) for r in generated] == [(s, int(i)) for s in ("true", "predicted") for i in render_ix], "COMPARISON_RENDER_ORDER")
        for r in generated:
            status, text = decoded(r["tokens"])
            check(status == r["status"] and text == r["text"] and r["asserted"] is False, "COMPARISON_RENDER_STATUS")
            check(r["target"] == rows[r["row"]]["atoms"][-1]["text"] and r["exact_target"] == (text == r["target"]), "COMPARISON_RENDER_TARGET")
        check(dict(Counter(r["status"] for r in generated)) == report["arms"][arm]["render_statuses"], "COMPARISON_RENDER_COUNTS")
        check(sha(root/arm/"renderer.pt") == report["arms"][arm]["checkpoint_sha256"] and sha(root/"initial.pt") == report["arms"][arm]["initialization_sha256"], "COMPARISON_CHECKPOINT")
        result[arm] = metrics
    check(.9 <= report["phase_coverage"] <= 1.001 and abs(sum(report["phases_seconds"].values())/report["wall_seconds"]-report["phase_coverage"]) < 1e-10, "COMPARISON_TIMING")
    return dict(acceptance="author-checked", report_sha256=pin, arms=result,
                all_score_rows_recounted=True, all_render_bytes_reparsed=True, scientific_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "population", "conditioning", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--report-sha256", required=True)
    a = p.parse_args(); check(not a.output.exists(), "COMPARISON_OUTPUT_EXISTS")
    result = inspect(a.comparison, a.population, a.conditioning, a.report_sha256)
    publish(a.output, result); print(json.dumps(result))


if __name__ == "__main__":
    main()
