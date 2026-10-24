"""Independent numerical replay of persisted window diagnostics; no admission."""
import argparse
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.special import digamma
from scipy.stats import rankdata

ROSTER = ["e5-base-v2", "all-mpnet-base-v2", "pubmedbert-embeddings",
          "finbert-mean", "specter2-base", "codebert-mean"]
SEEDS = (7, 17, 27, 37, 47)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(ok, suffix):
    if not ok:
        raise ValueError("MNLM_DIAGNOSTIC_READBACK_"+suffix)


def equal(actual, expected, suffix, tolerance=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
          and np.allclose(a, b, rtol=0, atol=tolerance), suffix)


def ross(distance, labels):
    """Separate vectorized count implementation; deliberately no producer import."""
    n = len(labels)
    d = np.array(distance, dtype=np.float64, copy=True)
    check(d.shape == (n,n) and np.isfinite(d).all() and (d >= 0).all(), "DISTANCE")
    same = labels[:,None] == labels[None,:]
    sizes = same.sum(axis=1)
    check((sizes > 3).all(), "SUPPORT")
    np.fill_diagonal(d, np.inf)
    radii = np.partition(np.where(same, d, np.inf), 2, axis=1)[:,2]
    check((radii > 0).all(), "DEGENERATE")
    neighbours = d <= radii[:,None]
    terms = digamma(n) + digamma((neighbours & same).sum(axis=1))
    terms -= digamma(sizes) + digamma(neighbours.sum(axis=1))
    return float(terms.mean()/np.log(2))


def normalized(v):
    n = np.sqrt(np.sum(v*v, axis=1, keepdims=True))
    check(np.isfinite(v).all() and (n > 0).all(), "VECTOR")
    return v/n


def similarity(a, b):
    cosines = a @ b.T
    return min(np.max(cosines, axis=0).mean(), np.max(cosines, axis=1).mean())


def geometry(sets, pairs=None):
    landmarks = np.arange(24)*len(sets)//24
    profiles = normalized(np.asarray([[similarity(a, sets[j]) for j in landmarks] for a in sets]))
    # Reference intentionally quantizes profiles before distance computation.
    p = profiles.astype(np.float32)
    d = np.max(np.abs(p[:,None,:]-p[None,:,:]), axis=2)
    readings = None if pairs is None else np.asarray([similarity(sets[a], sets[b]) for a,b in pairs])
    return d, readings


def load_inputs(slots, calibration, fsv):
    raw = (slots/"report.json").read_bytes(); report = json.loads(raw)
    proof_raw = fsv.read_bytes(); proof = json.loads(proof_raw)
    check(proof["passed"] == proof["total"] == 16 and
          proof["cold_readback"]["report_sha256"] == sha(raw), "FSV")
    check(report["complete"] and not report["errors"] and set(report["lenses"]) == set(ROSTER), "ROSTER")
    calraw = (calibration/"report.json").read_bytes(); cal = json.loads(calraw)
    check(sha(calraw) == report["calibration_report_sha256"], "CALIBRATION")
    matrices, offsets, anchors, rows = {}, {}, {}, {}
    for sense in ROSTER:
        lr = (slots/sense/"report.json").read_bytes()
        check(sha(lr) == report["lenses"][sense]["report_sha256"], "LENS")
        native = (slots/sense/"slots.npz").read_bytes()
        check(sha(native) == json.loads(lr)["files"]["slots.npz"]["sha256"], "NATIVE")
        with np.load(io.BytesIO(native), allow_pickle=False) as z:
            matrices[sense] = normalized(z["vectors"].astype(np.float64))
            offsets[sense] = z["offsets"].copy()
    cursor = 0
    for family, spec in sorted(cal["families"].items()):
        if not spec["ready"]:
            continue
        raw_summary = (calibration/family/"summary.json").read_bytes(); sm = json.loads(raw_summary)
        check(sha(raw_summary) == spec["summary_sha256"], "SUMMARY")
        ar = (calibration/family/"anchors.json").read_bytes()
        cr = (calibration/family/"chunks/atoms.parquet").read_bytes()
        check(sha(ar) == sm["anchors_sha256"] and sha(cr) == sm["chunks_sha256"], "SOURCE")
        anchors[family] = json.loads(ar)
        for row in pq.read_table(io.BytesIO(cr), columns=["row"])["row"].to_pylist():
            rows.setdefault((family,row), []).append(cursor); cursor += 1
    check(cursor == report["inputs"] == 6552, "COUNT")
    return {"report_sha256": sha(raw), "fsv_sha256": sha(proof_raw), "matrices": matrices,
            "offsets": offsets, "anchors": anchors, "rows": rows, "inputs": cursor}


def readback(source, inputs):
    begin = time.perf_counter()
    raw = (source/"report.json").read_bytes(); report = json.loads(raw)
    check(report["complete"] and not report["errors"] and report["roster"] == ROSTER, "PUBLICATION")
    check(all(report[k] is False for k in ("scientific_admission", "conditional_confidence_measured", "served")), "CLAIMS")
    check(report["source_report_sha256"] == inputs["report_sha256"] and
          report["fsv_sha256"] == inputs["fsv_sha256"], "BINDING")
    check(report["rules"] == {"pairs": 4096, "permutations": 199, "random_seeds": list(SEEDS),
                              "point_floor": .05}, "RULES")
    expected = {f+"-"+p for f,qs in inputs["anchors"].items() for p in qs}
    check(set(report["findings"]) == expected, "QUESTIONS")
    random = {s: normalized(np.random.default_rng(s).normal(size=(inputs["inputs"],768))) for s in SEEDS}
    summaries = {}
    for family, questions in inputs["anchors"].items():
        for prop, anchor in questions.items():
            key = family+"-"+prop; finding = report["findings"][key]
            check(json.loads((source/(key+".json")).read_bytes()) == finding, "REPORT_COPY")
            stored = (source/(key+".npz")).read_bytes()
            check(sha(stored) == finding["artifact_sha256"], "ARTIFACT_HASH")
            with np.load(io.BytesIO(stored), allow_pickle=False) as z:
                cache = {k: z[k].copy() for k in z.files}
            y = np.asarray(anchor["labels"]); n = len(y)
            check(n == finding["n"] == 504 and finding["classes"] == anchor["classes"], "POPULATION")
            equal(cache["labels"], y, "LABELS", 0)
            upper = np.stack(np.triu_indices(n,1), axis=1)
            pairs = upper[np.random.default_rng(20261019).choice(len(upper),4096,replace=False)]
            equal(cache["pairs"], pairs, "PAIRS", 0)
            permutations = np.asarray([np.random.default_rng(20261019+i).permutation(y) for i in range(199)])
            equal(cache["permuted_labels"], permutations, "PERMUTATIONS", 0)
            c = np.unique(y, return_counts=True)[1]/n
            equal(finding["entropy_bits"], -np.sum(c*np.log2(c)), "ENTROPY")
            distances, readings, standalone = {}, {}, {}
            for sense in ROSTER:
                v, offsets = inputs["matrices"][sense], inputs["offsets"][sense]
                sets = [np.vstack([v[offsets[k]:offsets[k+1]] for k in inputs["rows"][(family,r)]])
                        for r in anchor["rows"]]
                d, pair = geometry(sets, pairs)
                equal(cache[sense+"__distance"], d, "GEOMETRY", 0)
                equal(cache[sense+"__pair_readings"], pair, "PAIR_GEOMETRY")
                distances[sense], readings[sense] = d, pair
                standalone[sense] = ross(d,y)
                equal(finding["standalone_raw_bits"][sense], standalone[sense], "POINT")
                nulls = np.asarray([ross(d, labels) for labels in permutations])
                equal(cache[sense+"__null_raw_bits"], nulls, "NULL_VALUES")
                claims = finding["label_nulls"][sense]
                equal([claims["n"], claims["at_or_above_point_floor"], claims["max_raw_bits"], claims["tail_probability"]],
                      [199, np.sum(nulls >= .05), nulls.max(), (1+np.sum(nulls >= standalone[sense]))/200], "NULL_CLAIMS")
            for a,b in combinations(ROSTER,2):
                correlation = np.corrcoef(rankdata(readings[a]), rankdata(readings[b]))[0,1]
                equal(finding["d1_spearman"][a+"|"+b], correlation, "D1")
            joint = np.zeros_like(distances[ROSTER[0]]); previous = 0.0
            check(len(finding["fixed_order_incremental"]) == len(ROSTER), "INCREMENTS")
            for sense, claim in zip(ROSTER, finding["fixed_order_incremental"]):
                joint = np.maximum(joint, distances[sense]); now = ross(joint,y)
                check(claim["lens"] == sense, "ORDER")
                equal([claim["joint_raw_bits"], claim["incremental_raw_bits"]], [now, now-previous], "INCREMENT")
                previous = now
            equal(finding["joint_raw_bits"], previous, "JOINT")
            for sense in ROSTER:
                without = np.maximum.reduce([distances[k] for k in ROSTER if k != sense])
                equal(finding["drop_one_raw_bits"][sense], previous-ross(without,y), "DROP_ONE")
            equal(finding["duplicate_added_raw_bits"], ross(np.maximum(joint, distances[ROSTER[0]]),y)-previous, "DUPLICATE")
            for seed, v in random.items():
                d,_ = geometry([v[inputs["rows"][(family,r)]] for r in anchor["rows"]])
                equal(finding["random_control_raw_bits"][str(seed)], ross(d,y), "RANDOM")
            summaries[key] = {"observations": n, "null_scores_replayed": 199*6,
                              "d1_pairs_rebuilt_per_lens": len(pairs), "all_native_geometry_rebuilt": True,
                              "joint_below_best_standalone": previous < max(standalone.values()),
                              "null_point_floor_crossings": {s: finding["label_nulls"][s]["at_or_above_point_floor"] for s in ROSTER}}
    return {"report_sha256": sha(raw), "source_report_sha256": inputs["report_sha256"], "questions": summaries,
            "all_numerical_claims_replayed": True, "scientific_admission": False,
            "acceptance": "author-checked", "wall_seconds": time.perf_counter()-begin}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("diagnostic", "slots", "calibration", "fsv", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args()
    result = readback(a.diagnostic, load_inputs(a.slots, a.calibration, a.fsv))
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, indent=2)
    print(json.dumps(result))
