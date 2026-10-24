"""Standalone Calyx/D1 diagnostics for a cold-read commissioned candidate.

No joint-minus-panel estimator, conditional certification, or sense admission.
Existing public pair scores are reused by content hash on the same population.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_diagnostic_readback import load_inputs, ROSTER, SEEDS, sha, check
from audit.commissioned_window_readback import read_measurement, digest
from audit.calyx_paired_roots import evaluate
from audit.calyx_window_diagnostics import maxsim, point
from text.calyx_bits import unit, maxsim_profiles, chebyshev_matrix, raw_bits
from senses.trajectory_population import publish

SCHEMA = "commissioned-candidate-diagnostic-v1"
RULES = dict(seeds=list(SEEDS), resamples=999, pairs=4096, label_permutations=199,
             landmarks=24, d1_ceiling=.6, standalone_floor=.05, conditional_information_measured=False)


def sources(a):
    cold = read_measurement(a.publication, a.plan, a.calibration, a.instrument)
    replay = json.loads(a.gpu_replay.read_bytes())
    check(replay["source_report_sha256"] == cold["report_sha256"] and replay["lens_id"] == cold["lens_id"]
          and replay["all_replayed_vectors_bitwise_equal"] is True and replay["scientific_admission"] is False, "CANDIDATE_REPLAY")
    check(digest(a.gpu_replay.parent/"replayed.npz") == replay["artifact_sha256"], "CANDIDATE_REPLAY_ARTIFACT")
    inputs = load_inputs(a.public_native, a.calibration, a.public_fsv)
    raw = (a.public_diagnostic/"report.json").read_bytes(); public = json.loads(raw)
    check(sha(raw) == a.public_diagnostic_sha256 and public["complete"] and not public["errors"]
          and public["source_report_sha256"] == inputs["report_sha256"] and public["scientific_admission"] is False, "CANDIDATE_PUBLIC")
    candidate = unit(np.load(a.publication/"vectors.npy", allow_pickle=False).astype(np.float64))
    offsets = np.load(a.plan/"offsets.npy", allow_pickle=False)
    check(len(offsets) == inputs["inputs"]+1, "CANDIDATE_POPULATION")
    return cold, inputs, public, candidate, offsets


def run(a):
    check(not a.output.exists(), "CANDIDATE_EXISTS"); started = time.perf_counter()
    cold, inputs, public, candidate, offsets = sources(a)
    a.output.mkdir(parents=True); loaded = time.perf_counter(); findings = {}; errors = []
    for family, questions in inputs["anchors"].items():
        for prop, anchor in questions.items():
            name = family+"-"+prop; begin = time.perf_counter()
            try:
                raw = (a.public_diagnostic/(name+".npz")).read_bytes()
                check(sha(raw) == public["findings"][name]["artifact_sha256"], "CANDIDATE_PUBLIC_CACHE")
                with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                    cache = {k: z[k].copy() for k in z.files}
                y = np.asarray(anchor["labels"]); pairs = cache["pairs"]
                check(np.array_equal(y, cache["labels"]) and pairs.shape == (4096, 2), "CANDIDATE_LABELS")
                sets = [np.vstack([candidate[offsets[c]:offsets[c+1]] for c in inputs["rows"][(family, row)]]) for row in anchor["rows"]]
                profile = maxsim_profiles(sets, anchor["rows"]); distance = chebyshev_matrix(profile)
                point(distance, y)  # Reference tie/support refusal stays in force.
                readings = np.asarray([maxsim(sets[x], sets[z]) for x, z in pairs])
                correlations = {sense: float(spearmanr(readings, cache[sense+"__pair_readings"]).statistic) for sense in ROSTER}
                check(all(np.isfinite(v) for v in correlations.values()), "CANDIDATE_D1_UNDEFINED")
                geometry_done = time.perf_counter()
                key = sha(distance.tobytes()); definitions = dict(standalone=[key, None], duplicate=[key, key])
                ix, values, stats = evaluate(y, {key: distance}, definitions)
                nulls = np.asarray([raw_bits(distance, labels) for labels in cache["permuted_labels"]])
                counted = time.perf_counter()
                np.savez(a.output/(name+".npz"), profile=profile, distance=distance, pair_readings=readings, pairs=pairs,
                         labels=y, indices=ix, samples=values[key], null_scores=nulls, permuted_labels=cache["permuted_labels"])
                result = dict(n=len(y), source_artifact_sha256=sha(raw), artifact_sha256=digest(a.output/(name+".npz")),
                              d1_spearman=correlations, below_d1_ceiling=[s for s in ROSTER if correlations[s] <= .6],
                              above_d1_ceiling=[s for s in ROSTER if correlations[s] > .6],
                              standalone=stats["standalone"], duplicate_control=stats["duplicate"],
                              label_nulls=dict(n=len(nulls), max_raw_bits=float(nulls.max()),
                                   point_floor_crossings=int(np.sum(nulls >= .05)),
                                   tail_probability=float((1+np.sum(nulls >= stats["standalone"]["raw_point"]))/(1+len(nulls)))),
                              timing=dict(geometry_seconds=geometry_done-begin, estimate_seconds=counted-geometry_done,
                                          write_seconds=time.perf_counter()-counted))
                findings[name] = result; publish(a.output/(name+".json"), result)
                print(json.dumps(dict(question=name, d1=correlations, standalone=result["standalone"]["raw_point"],
                                      standalone_lower=result["standalone"]["outer_low"])), flush=True)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(dict(question=name, error=str(exc)))
    elapsed = time.perf_counter()-started
    result = dict(schema=SCHEMA, candidate_report_sha256=cold["report_sha256"], candidate_lens_id=cold["lens_id"],
                  public_diagnostic_sha256=a.public_diagnostic_sha256, gpu_replay_sha256=digest(a.gpu_replay),
                  rules=RULES, findings=findings, errors=errors, complete=not errors, scientific_admission=False,
                  conditional_information_certified=False, served=False, code_sha256=digest(Path(__file__)),
                  limitations=["calibration-only A2 population", "no IID/near-duplicate certificate", "no D2 certificate",
                               "standalone signal and pairwise distinctness do not establish an admitted sense"],
                  timing=dict(load_seconds=loaded-started, wall_seconds=elapsed))
    publish(a.output/"report.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "plan", "calibration", "instrument", "gpu-replay", "public-native", "public-fsv", "public-diagnostic", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--public-diagnostic-sha256", required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args(); result = run(args)
    print(json.dumps(dict(complete=result["complete"], errors=result["errors"], timing=result["timing"])))
    if not result["complete"]:
        raise SystemExit(2)
