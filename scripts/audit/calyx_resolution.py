"""Fixed sampling-resolution experiment, plus real-profile index equivalence.

Artificial known-law boundaries are not production observations. Five seeds and
all predeclared sample sizes are reported, including failures; nothing here
certifies native conditional information or promotes a panel. #12, PC-07/16/37.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text.calyx_indexed import indexed_counts, raw_bits, rounding_boundary

SEEDS = (7, 17, 27, 37, 47)
SIZES = (504, 2000, 8192, 32768, 131072)
KINDS = ("duplicate", "frequency_80", "frequency_512", "independent", "positive")
ROSTER = ("e5-base-v2", "all-mpnet-base-v2", "pubmedbert-embeddings",
          "finbert-mean", "specter2-base", "codebert-mean")
SOURCE_PIN = "ca92b06492e502a5ba32d06fe19e5ca7494a3074d5675d750dfd5afd4f47b561"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def save(path, arrays):
    with Path(path).open("xb") as f:
        np.savez(f, **arrays)
    return sha(path)


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_CALYX_RESOLUTION_"+code)


def reference(blocks, y, rows):
    """Independent full scan, no index, radius translation or producer counts."""
    blocks = [np.asarray(b, dtype=np.float32) for b in blocks]
    c, radii = [], []
    for i in rows:
        distances = np.maximum.reduce([np.abs(b-b[i]).max(axis=1) for b in blocks])
        distances[i] = np.inf
        same = y == y[i]
        r = np.sort(distances[same])[2]
        need(r > 0 and np.isfinite(r), "REFERENCE_RADIUS")
        c.append([np.sum((distances <= r) & same), same.sum(), np.sum(distances <= r)])
        radii.append(r)
    return np.asarray(c, dtype=np.int64), np.asarray(radii, dtype=np.float32)


def evaluate(path, blocks, y, mode):
    begin = time.perf_counter()
    artifacts = dict(labels=y, **{f"block_{i}": b for i, b in enumerate(blocks)})
    comparisons = {"panel": blocks[:-1], "joint": blocks} if mode == "gain" else {"single": blocks}
    values, timings = {}, {}
    for name, selected in comparisons.items():
        tick = time.perf_counter()
        counts, radius = indexed_counts(selected, y)
        timings[name+"_index_seconds"] = time.perf_counter()-tick
        rows = np.arange(len(y)) if len(y) <= 504 else np.unique(np.linspace(0, len(y)-1, 16, dtype=int))
        tick = time.perf_counter()
        expected, expected_radius = reference(selected, y, rows)
        need(np.array_equal(counts[rows], expected) and np.array_equal(radius[rows], expected_radius), "EXACT_COUNTS")
        timings[name+"_independent_seconds"] = time.perf_counter()-tick
        artifacts.update({name+"_counts": counts, name+"_radii": radius, name+"_readback_rows": rows})
        values[name] = raw_bits(counts)
    if mode == "gain":
        values["gain"] = values["joint"]-values["panel"]
    tick = time.perf_counter()
    pin = save(path, artifacts)
    with np.load(path, allow_pickle=False) as z:
        for name in comparisons:
            need(raw_bits(z[name+"_counts"]) == values[name], "COLD_FORMULA")
    timings["publication_seconds"] = time.perf_counter()-tick
    timings["wall_seconds"] = time.perf_counter()-begin
    return dict(file=path.name, sha256=pin, n=len(y), dimensions=[b.shape[1] for b in blocks],
                mode=mode, bits=values, checked_rows=len(rows), timing=timings)


def boundaries():
    results = {}
    def refused(name, blocks, y, code):
        try:
            indexed_counts(blocks, y)
        except ValueError as exc:
            need(str(exc) == "MNLM_CALYX_INDEX_"+code, "WRONG_REFUSAL")
            results[name] = str(exc)
        else:
            raise ValueError("MNLM_CALYX_RESOLUTION_MISSING_REFUSAL:"+name)
    y = np.repeat([0, 1], 8); x = np.arange(16, dtype=float)[:, None]
    refused("constant", [np.zeros_like(x)], y, "DEGENERATE")
    bad = x.copy(); bad[0, 0] = np.nan
    refused("nan", [bad], y, "INPUT")
    refused("support", [x[:6]], np.repeat([0, 1], 3), "CLASS_SUPPORT")
    refused("misaligned", [x[:-1]], y, "INPUT")
    # Exact halfway ties of both parities, subnormals and ordinary magnitudes.
    radius = np.array([1, np.nextafter(np.float32(1), np.float32(2)),
                       np.nextafter(np.float32(0), np.float32(1)), 0.125], dtype=np.float32)
    midpoint = (radius.astype(float)+np.nextafter(radius, np.float32(np.inf)).astype(float))/2
    threshold = rounding_boundary(radius)
    for delta in (-1, 0, 1):
        probe = midpoint if delta == 0 else np.nextafter(midpoint, delta*np.inf)
        need(np.array_equal(probe.astype(np.float32) <= radius, probe <= threshold), "ROUNDING")
    results["float32_halfway"] = True
    return results


def run(source, output):
    start = time.perf_counter()
    need(sha(source/"report.json") == SOURCE_PIN, "SOURCE_PIN")
    report = json.loads((source/"report.json").read_bytes())
    need(report["complete"] and not report["errors"], "SOURCE_REPORT")
    output.mkdir(parents=True, exist_ok=False)
    write(output/"intent.json", dict(seeds=SEEDS, sizes=SIZES, kinds=KINDS, k=3,
        source=str(source), source_sha256=SOURCE_PIN, floor_bits=.05,
        owner="ChrisRoyse/MNLM#12", scientific_admission=False))
    checks = boundaries(); real = []; controls = []
    for question in ("science-P123", "biomedical-P1057", "biomedical-P1995"):
        path = source/(question+".npz")
        need(sha(path) == report["findings"][question]["artifact_sha256"], "SOURCE_ARRAYS")
        with np.load(path, allow_pickle=False) as data:
            y = data["labels"].copy()
            profiles = [data[name+"__profile"].copy() for name in ROSTER]
        need(len(y) == 504 and all(x.shape == (504, 24) for x in profiles), "SOURCE_SHAPE")
        for i, name in enumerate(ROSTER):
            result = evaluate(output/(question+"-"+name+".npz"), profiles[:i+1], y,
                              "gain" if i else "single")
            result.update(question=question, candidate=name, source_sha256=sha(path))
            real.append(result)
        print(json.dumps(dict(real_question=question, exact_reference_counts=True)), flush=True)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        z = rng.uniform(-1, 1, (max(SIZES), 2))
        independent = rng.uniform(-1, 1, (max(SIZES), 1))
        outcomes = rng.integers(0, 2, max(SIZES))
        for n in SIZES:
            zz = z[:n]
            for kind in KINDS:
                if kind == "duplicate":
                    x, y, truth = zz.copy(), (zz[:, 0] > 0).astype(int), 0
                elif kind.startswith("frequency"):
                    value = np.sin(int(kind.split("_")[1])*zz[:, 0])
                    x, y, truth = np.column_stack((value, zz[:, 1])), (value > 0).astype(int), 0
                elif kind == "independent":
                    x, y, truth = independent[:n], outcomes[:n], 0
                else:
                    x, y, truth = independent[:n], (independent[:n, 0] > 0).astype(int), 1
                result = evaluate(output/f"control-{seed}-{n}-{kind}.npz", [zz, x], y, "gain")
                result.update(seed=seed, kind=kind, true_cmi_bits=truth, artificial_boundary=True)
                controls.append(result)
            print(json.dumps(dict(seed=seed, n=n, gains={r["kind"]: r["bits"]["gain"] for r in controls[-5:]})), flush=True)
    result = dict(schema="calyx-indexed-resolution-v1", source_sha256=SOURCE_PIN,
        code_sha256={str(p.relative_to(Path(__file__).parents[1])): sha(p) for p in
            (Path(__file__), Path(__file__).parents[1]/"text/calyx_indexed.py")},
        controls=controls, real=real, boundaries=checks, seeds=SEEDS, sizes=SIZES,
        scientific_admission=False, conditional_information_certified=False, served=False,
        wall_seconds=time.perf_counter()-start, complete=True)
    write(output/"report.json", result)
    write(output/"manifest.json", {"report_sha256": sha(output/"report.json"), "intent_sha256": sha(output/"intent.json")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
