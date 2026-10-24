"""Separate scalar metric/selection/partition reconstruction from persisted probe bytes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from senses.contract import canonical, digest, require
from senses.forecast import load_bundle, put
from senses.pipeline import file_hash


def measured(path, expected_truth, claimed):
    with np.load(path, allow_pickle=False) as z:
        p, y = z["probabilities"], z["truth"]
    require(np.array_equal(y, expected_truth) and p.shape == (len(y), 7, 3),
            "MNLM_FORECAST_READBACK_LABELS", "test or validation outcomes/order changed")
    losses, correct = [0.0] * 7, [0] * 7
    for i, row in enumerate(p):
        for j, distribution in enumerate(row):
            require(all(math.isfinite(float(x)) and x > 0 for x in distribution)
                    and abs(math.fsum(float(x) for x in distribution) - 1) < 1e-6,
                    "MNLM_FORECAST_READBACK_PROB", "invalid persisted probabilities")
            losses[j] += -math.log2(float(distribution[int(y[i, j])]))
            correct[j] += max(range(3), key=lambda k: distribution[k]) == y[i, j]
    loss, accuracy = math.fsum(losses) / (len(y) * 7), sum(correct) / (len(y) * 7)
    require(abs(loss - claimed["mean_log_loss_bits"]) < 1e-12 and abs(accuracy - claimed["accuracy"]) < 1e-12
            and all(abs(a / len(y) - b) < 1e-12 for a, b in zip(losses, claimed["coordinate_log_loss_bits"]))
            and all(abs(a / len(y) - b) < 1e-12 for a, b in zip(correct, claimed["coordinate_accuracy"])),
            "MNLM_FORECAST_READBACK_METRIC", "claimed metric differs from scalar recomputation")
    return loss


def run(bundle, probe, out):
    m, data = load_bundle(bundle)
    inventory = json.loads((probe / "SHA256.json").read_bytes())
    for name, sha in inventory.items():
        path = (probe / name).resolve()
        require(path.is_relative_to(probe.resolve()) and file_hash(path) == sha,
                "MNLM_FORECAST_READBACK_DIGEST", "sealed file changed or escaped evidence directory")
    report = json.loads((probe / "report.json").read_bytes())
    require(all(report[k] is False for k in ("stage3_admitted", "stage4_accepted", "meaning_asserted",
                                           "conditional_information_measured", "independent_acceptance")),
            "MNLM_FORECAST_READBACK_CLAIM", "diagnostic was incorrectly promoted")
    labels = data["labels.jsonl"]
    expected_ids = {s: [r["atom"] for r in labels if r["split"] == s] for s in ("train", "validation", "test")}
    truth = {s: np.asarray([[v + 1 for v in r["sign"]] for r in labels if r["split"] == s], dtype=np.int64)
             for s in expected_ids}
    files, checkpoints = 0, 0
    for name in ("real", "permuted"):
        root = probe / name
        selection = json.loads((root / "selection.json").read_bytes())
        mapping = json.loads((root / "training-label-map.json").read_bytes())
        require(mapping["predictor_ids"] == expected_ids["train"] and len(mapping["label_source_ids"]) == len(expected_ids["train"])
                and set(mapping["label_source_ids"]) == set(expected_ids["train"]),
                "MNLM_FORECAST_READBACK_SPLIT", "held-out or duplicate group used for training")
        order = np.random.default_rng(43).permutation(len(expected_ids["train"])) if name == "permuted" else range(len(expected_ids["train"]))
        require(mapping["label_source_ids"] == [expected_ids["train"][int(i)] for i in order],
                "MNLM_FORECAST_READBACK_PERMUTATION", "label control is not the declared whole-group permutation")
        require(selection["test_used_for_selection"] is False and len(selection["history"]) == 5,
                "MNLM_FORECAST_READBACK_SELECTION", "undeclared selection protocol")
        scores = []
        for entry in selection["history"]:
            epoch = entry["epoch"]
            scores.append(measured(root / f"validation-{epoch}.npz", truth["validation"], entry["validation"]))
            files += 1
            cp = root / f"epoch-{epoch}"
            cm = json.loads((cp / "manifest.json").read_bytes())
            require(digest(cm) == (cp / "COMMITTED").read_text() == entry["checkpoint_sha256"]
                    and file_hash(cp / "weights.pt") == cm["weights_sha256"] and cm["input_sha256"] == report["input_sha256"],
                    "MNLM_FORECAST_READBACK_CHECKPOINT", "wrong state or checkpoint lineage")
            checkpoints += 1
        best = min(range(5), key=lambda i: scores[i]) + 1
        require(best == selection["best_epoch"] == report[name]["best_epoch"],
                "MNLM_FORECAST_READBACK_SELECTION", "checkpoint did not minimize validation loss")
        require(json.loads((root / "test-ids.json").read_bytes()) == expected_ids["test"],
                "MNLM_FORECAST_READBACK_SPLIT", "test row identity/order differs")
        measured(root / "test.npz", truth["test"], report[name]["test"])
        files += 1
    for name, score in report["real"]["ablations"].items():
        measured(probe / "real" / ("ablate-" + name + ".npz"), truth["test"], score)
        files += 1
    measured(probe / "baseline.npz", truth["test"], report["baseline"])
    files += 1
    with np.load(probe / "baseline.npz", allow_pickle=False) as z:
        require(np.array_equal(z["training_truth"], truth["train"]), "MNLM_FORECAST_READBACK_BASELINE", "wrong baseline population")
        for j in range(7):
            for c in range(3):
                probability = (sum(int(v == c) for v in truth["train"][:, j]) + 1) / (len(truth["train"]) + 3)
                require(bool((z["probabilities"][:, j, c] == probability).all()),
                        "MNLM_FORECAST_READBACK_BASELINE", "baseline was fitted on held-out labels")
    result = {"acceptance": "author-checked only", "files_rehashed": len(inventory), "metric_arrays_recomputed": files,
              "checkpoints_rehashed": checkpoints, "splits": m["splits"], "partition_and_selection_rederived": True,
              "report_sha256": file_hash(probe / "report.json"), "bundle_manifest_sha256": file_hash(bundle / "manifest.json"),
              "real_beats_marginal_log_loss": report["real"]["test"]["mean_log_loss_bits"] < report["baseline"]["mean_log_loss_bits"],
              "real_beats_permuted_log_loss": report["real"]["test"]["mean_log_loss_bits"] < report["permuted"]["test"]["mean_log_loss_bits"],
              "conditional_information_measured": False, "stage3_admitted": False, "stage4_accepted": False}
    put(out, canonical(result))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("bundle", "probe", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    a = ap.parse_args()
    run(a.bundle, a.probe, a.out)
