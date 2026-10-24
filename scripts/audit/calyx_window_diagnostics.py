"""Reference point/null and D1 diagnostics on cold-read full-window senses (#12).

Joint distances use the maximum of typed-slot Chebyshev matrices. MI differences
are finite-sample diagnostics, never conditional certification or admission.
"""
import argparse
from itertools import combinations
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.calibration_population import sha, require
from text.calyx_bits import unit, maxsim_profiles, chebyshev_matrix, raw_bits, plain, K
from senses.trajectory_population import publish

ROSTER = ["e5-base-v2", "all-mpnet-base-v2", "pubmedbert-embeddings", "finbert-mean", "specter2-base", "codebert-mean"]
SEEDS, PERMUTATIONS, PAIRS = (7, 17, 27, 37, 47), 199, 4096


def point(distance, labels):
    for i in range(len(labels)):
        same = np.flatnonzero(labels == labels[i])
        require(np.count_nonzero(distance[i,same] == 0)-1 < K, "MNLM_WINDOW_BITS_DEGENERATE")
    return raw_bits(distance, labels)


def maxsim(a, b):
    scores = a@b.T
    return min(scores.max(1).mean(), scores.max(0).mean())


def run(slots, calibration, fsv, output):
    begin = time.perf_counter()
    require(not output.exists(), "MNLM_WINDOW_BITS_EXISTS")
    raw = (slots/"report.json").read_bytes()
    report = json.loads(raw)
    proof_raw = fsv.read_bytes(); proof = json.loads(proof_raw)
    require(proof["passed"] == proof["total"] and proof["cold_readback"]["report_sha256"] == sha(raw), "MNLM_WINDOW_BITS_FSV")
    require(report["complete"] and not report["errors"] and set(report["lenses"]) == set(ROSTER), "MNLM_WINDOW_BITS_ROSTER")
    calraw = (calibration/"report.json").read_bytes()
    require(sha(calraw) == report["calibration_report_sha256"], "MNLM_WINDOW_BITS_CALIBRATION")
    cal = json.loads(calraw)
    matrices, offsets = {}, {}
    for name in ROSTER:
        lens_raw = (slots/name/"report.json").read_bytes()
        require(sha(lens_raw) == report["lenses"][name]["report_sha256"], "MNLM_WINDOW_BITS_LENS")
        native = (slots/name/"slots.npz").read_bytes()
        require(sha(native) == json.loads(lens_raw)["files"]["slots.npz"]["sha256"], "MNLM_WINDOW_BITS_SLOTS")
        with np.load(io.BytesIO(native)) as z:
            matrices[name] = unit(z["vectors"].astype(np.float64)); offsets[name] = z["offsets"].copy()
    populations, by_row, cursor = {}, {}, 0
    for family, spec in sorted(cal["families"].items()):
        if not spec["ready"]:
            continue
        summary_raw = (calibration/family/"summary.json").read_bytes(); summary = json.loads(summary_raw)
        require(sha(summary_raw) == spec["summary_sha256"], "MNLM_WINDOW_BITS_SUMMARY")
        ar = (calibration/family/"anchors.json").read_bytes()
        cr = (calibration/family/"chunks/atoms.parquet").read_bytes()
        require(sha(ar) == summary["anchors_sha256"] and sha(cr) == summary["chunks_sha256"], "MNLM_WINDOW_BITS_INPUT_HASH")
        populations[family] = json.loads(ar)
        for chunk in pq.read_table(io.BytesIO(cr), columns=["row"]).column("row").to_pylist():
            by_row.setdefault((family, chunk), []).append(cursor); cursor += 1
    require(cursor == report["inputs"], "MNLM_WINDOW_BITS_COUNT")
    # Null instruments are explicit controls. Generate once, not per question.
    randoms = {s: unit(np.random.default_rng(s).normal(size=(cursor, 768))) for s in SEEDS}
    output.mkdir(parents=True)
    loaded = time.perf_counter()
    findings, errors = {}, []
    for family, questions in populations.items():
        for prop, anchor in questions.items():
            name = family+"-"+prop
            start = time.perf_counter()
            try:
                rows = anchor["rows"]; y = np.asarray(anchor["labels"])
                require(len(rows) == len(set(rows)) == len(y) and len(y) >= 500, "MNLM_WINDOW_BITS_POWER")
                upper = np.stack(np.triu_indices(len(y), 1), axis=1)
                pairs = upper[np.random.default_rng(20261019).choice(len(upper), PAIRS, replace=False)]
                distances, readings, standalone, artifacts = {}, {}, {}, {"labels": y, "pairs": pairs}
                for sense in ROSTER:
                    sets = [np.concatenate([matrices[sense][offsets[sense][c]:offsets[sense][c+1]]
                                            for c in by_row[(family,r)]]) for r in rows]
                    profiles = maxsim_profiles(sets, rows)
                    d = chebyshev_matrix(profiles)
                    distances[sense] = d
                    standalone[sense] = point(d, y)
                    readings[sense] = np.asarray([maxsim(sets[a], sets[b]) for a,b in pairs])
                    artifacts[sense+"__distance"] = d
                    artifacts[sense+"__pair_readings"] = readings[sense]
                pair_corr = {a+"|"+b: float(spearmanr(readings[a], readings[b]).statistic) for a,b in combinations(ROSTER,2)}
                require(all(np.isfinite(c) for c in pair_corr.values()), "MNLM_WINDOW_BITS_D1_UNDEFINED")
                prepared = time.perf_counter()
                joint, previous, increments = None, 0.0, []
                for sense in ROSTER:
                    joint = distances[sense] if joint is None else np.maximum(joint, distances[sense])
                    current = point(joint, y)
                    increments.append({"lens": sense, "joint_raw_bits": current, "incremental_raw_bits": current-previous})
                    previous = current
                drop = {sense: previous-point(np.maximum.reduce([distances[n] for n in ROSTER if n != sense]), y) for sense in ROSTER}
                duplicate_gain = point(np.maximum(joint, distances[ROSTER[0]]), y)-previous
                require(duplicate_gain == 0, "MNLM_WINDOW_BITS_DUPLICATE_GAIN")
                permutations = np.asarray([np.random.default_rng(20261019+i).permutation(y) for i in range(PERMUTATIONS)])
                permuted = {sense: np.asarray([raw_bits(distances[sense], p) for p in permutations]) for sense in ROSTER}
                random_scores = {}
                for seed, random in randoms.items():
                    profiles = maxsim_profiles([random[by_row[(family,r)]] for r in rows], rows)
                    random_scores[str(seed)] = point(chebyshev_matrix(profiles), y)
                end = time.perf_counter()
                for sense in ROSTER:
                    artifacts[sense+"__null_raw_bits"] = permuted[sense]
                artifacts["permuted_labels"] = permutations
                np.savez(output/(name+".npz"), **artifacts)
                counts = np.unique(y, return_counts=True)[1]; probs = counts/len(y)
                finding = {"n": len(y), "classes": anchor["classes"], "entropy_bits": float(-np.sum(probs*np.log2(probs))),
                           "standalone_raw_bits": standalone, "d1_spearman": pair_corr, "fixed_order_incremental": increments,
                           "joint_raw_bits": previous, "drop_one_raw_bits": drop, "duplicate_added_raw_bits": duplicate_gain,
                           "random_control_raw_bits": random_scores,
                           "label_nulls": {sense: {"n": PERMUTATIONS, "at_or_above_point_floor": int(np.sum(permuted[sense] >= .05)),
                                                  "max_raw_bits": float(permuted[sense].max()),
                                                  "tail_probability": float((1+np.sum(permuted[sense] >= standalone[sense]))/(PERMUTATIONS+1))}
                                           for sense in ROSTER},
                           "artifact_sha256": sha((output/(name+".npz")).read_bytes()),
                           "timing": {"profiles_pairs_seconds": prepared-start, "joint_null_seconds": end-prepared,
                                      "write_seconds": time.perf_counter()-end}}
                findings[name] = finding
                publish(output/(name+".json"), plain(finding))
                print(json.dumps({"question": name, "standalone": standalone, "joint": previous,
                                  "random_controls": random_scores, "label_null_crossings": {n:r["at_or_above_point_floor"] for n,r in finding["label_nulls"].items()}}), flush=True)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append({"question": name, "error": str(exc)})
    result = {"schema": "calyx-window-point-diagnostics-v1", "source_report_sha256": sha(raw), "fsv_sha256": sha(proof_raw),
              "roster": ROSTER, "findings": findings, "errors": errors, "complete": not errors,
              "scientific_admission": False, "conditional_confidence_measured": False, "served": False,
              "estimand": "reference profile MI point estimates and joint-minus-panel differences; finite-sample diagnostics",
              "rules": {"pairs": PAIRS, "permutations": PERMUTATIONS, "random_seeds": SEEDS, "point_floor": .05},
              "code_sha256": sha(Path(__file__).read_bytes()), "reference_code_sha256": sha(Path(__file__).parents[1].joinpath("text/calyx_bits.py").read_bytes()),
              "timing": {"load_seconds": loaded-begin, "wall_seconds": time.perf_counter()-begin}}
    publish(output/"report.json", plain(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("slots", "calibration", "fsv", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args()
    result = run(a.slots, a.calibration, a.fsv, a.output)
    if not result["complete"]:
        raise SystemExit(2)
