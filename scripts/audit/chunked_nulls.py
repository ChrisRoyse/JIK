"""Replay and diagnose existing Calyx control failures; never admit a lens.

The Ross KSG point estimator and MaxSim representation are imported unchanged.
Cache distances across label permutations (PC-07). Random geometry is ONLY a
negative control. A single-reading random control diagnoses chunk-count effects;
it is never the representation of a real document. All original failures remain.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.segment import segment, ROLE_SENTENCE
from senses.trajectory_population import publish
from text.calyx_bits import unit, maxsim_profiles, chebyshev_matrix, raw_bits, K, plain
from text.chunk_anchor import chunks_of
from text.chunk_spans import chunk_spans, CONTRACT
from text.atomise_pilot import MIN_CHARS, MAX_CHARS, WORDS

SEEDS = (7, 17, 27, 37, 47)
PERMUTATIONS = 199
FLOOR = 0.05


def require(ok, code, message):
    if not ok:
        raise ValueError(f"{code}: {message}")


def read_bound(path, receipts):
    path = Path(path)
    raw = path.read_bytes()
    receipts[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return raw


def point(D, y):
    require(np.isfinite(D).all() and (D >= 0).all(), "MNLM_NULL_DISTANCE", "finite distances required")
    require(min(np.unique(y, return_counts=True)[1]) > K, "MNLM_NULL_SUPPORT", "insufficient class support")
    for i in range(len(y)):
        require(((D[i] == 0) & (y == y[i])).sum() - 1 < K,
                "MNLM_CALYX_DEGENERATE", "zero kth same-class radius")
    b = float(raw_bits(D, y))
    require(np.isfinite(b), "MNLM_NULL_NONFINITE", "invalid reference estimate")
    return {"raw": b, "bits": max(0., b)}


def distribution(values):
    return {"n": len(values), "min": min(values), "max": max(values),
            "median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
            "at_or_above_floor": sum(v >= FLOOR for v in values), "values": values}


def coverage(rows, by_row, texts):
    stats = {k: 0 for k in ("entities", "source_chars", "source_nonspace", "legacy_chars",
                           "legacy_nonspace", "legacy_missing_nonspace", "affected_entities",
                           "short_spans", "long_spans", "low_word_spans", "fallback_entities",
                           "lossless_chunks", "lossless_forced_cuts")}
    examples = []
    for r, idx in sorted(by_row.items()):
        text = rows[r]["text"]
        old = [texts[j] for j in idx]
        require(old == chunks_of(text), "MNLM_NULL_CHUNK_LINEAGE", f"legacy chunks differ at row {r}")
        accepted = []
        for s, e, role in segment(text):
            if role != ROLE_SENTENCE:
                continue
            t = text[s:e].strip()
            stats["short_spans"] += len(t) < MIN_CHARS
            stats["long_spans"] += len(t) > MAX_CHARS
            stats["low_word_spans"] += len(WORDS.findall(t)) < 4
            if MIN_CHARS <= len(t) <= MAX_CHARS and len(WORDS.findall(t)) >= 4:
                accepted.append(t)
        stats["fallback_entities"] += not accepted
        fresh = chunk_spans(text)
        require("".join(s["text"] for s in fresh) == text,
                "MNLM_NULL_COVERAGE", f"new chunker lost source characters at {r}")
        # Old chunks must be ordered source substrings before loss can be counted.
        end = 0
        for t in old:
            start = text.find(t, end)
            require(start >= end, "MNLM_NULL_CHUNK_LINEAGE", f"non-source chunk at {r}")
            end = start + len(t)
        source_nonspace = sum(not c.isspace() for c in text)
        old_nonspace = sum(not c.isspace() for t in old for c in t)
        missing = source_nonspace - old_nonspace
        stats["entities"] += 1
        stats["source_chars"] += len(text)
        stats["source_nonspace"] += source_nonspace
        stats["legacy_chars"] += sum(map(len, old))
        stats["legacy_nonspace"] += old_nonspace
        stats["legacy_missing_nonspace"] += missing
        stats["affected_entities"] += missing > 0
        stats["lossless_chunks"] += len(fresh)
        stats["lossless_forced_cuts"] += sum(s["forced_cut"] for s in fresh)
        if missing and len(examples) < 10:
            examples.append({"row": r, "entity": rows[r]["entity"], "missing_nonspace": missing,
                             "source_sha256": hashlib.sha256(text.encode()).hexdigest()})
    return {**stats, "examples": examples, "new_contract": CONTRACT,
            "new_missing_characters": 0, "measurement_reuse_allowed": False}


def diagnose(chunked, slots, source, report_path, output):
    began = time.perf_counter()
    require(not output.exists(), "MNLM_NULL_OUTPUT_EXISTS", "use a new immutable evidence generation")
    receipts = {}
    report = json.loads(read_bound(report_path, receipts))
    best = max(report["mean_bits"], key=report["mean_bits"].get)
    anchors_raw = read_bound(chunked / "anchors.json", receipts)
    anchors = json.loads(anchors_raw)
    summary = json.loads(read_bound(chunked / "summary.json", receipts))
    rows_raw = read_bound(source / "rows.jsonl", receipts)
    require(hashlib.sha256(rows_raw).hexdigest() == summary["parent_rows_sha256"],
            "MNLM_NULL_PARENT_DIGEST", "source rows differ from chunk publication")
    require(hashlib.sha256(anchors_raw).hexdigest() == summary["anchors_sha256"],
            "MNLM_NULL_ANCHOR_DIGEST", "anchor publication changed")
    rows = [json.loads(l) for l in rows_raw.decode("utf-8").split("\n") if l]
    table = pq.read_table(io.BytesIO(read_bound(chunked / "chunks" / "atoms.parquet", receipts))).to_pydict()
    require(all(len(v) == len(table["row"]) for v in table.values()), "MNLM_NULL_ROWS", "chunk columns differ")
    by_row = {}
    for i, r in enumerate(table["row"]):
        require(type(r) is int and 0 <= r < len(rows), "MNLM_NULL_ROWS", "chunk source row outside source")
        require(table["entity"][i] == rows[r]["entity"], "MNLM_NULL_ENTITY", "chunk identity differs")
        idx = by_row.setdefault(r, [])
        require(table["chunk"][i] == len(idx), "MNLM_NULL_CHUNK_ORDER", "unordered chunk index")
        idx.append(i)
    require(len(by_row) == summary["entities"] and len(table["row"]) == summary["chunks"],
            "MNLM_NULL_POPULATION", "population extent differs from publication")
    C = np.load(io.BytesIO(read_bound(slots / f"{best}.npy", receipts)), allow_pickle=False)
    require(C.ndim == 2 and len(C) == len(table["row"]), "MNLM_NULL_SLOT_ROWS", "slot rows differ")
    C = unit(C.astype(np.float64))
    require(set(anchors) == set(report["anchors"]), "MNLM_NULL_ANCHORS", "report anchor roster differs")
    loaded = time.perf_counter()
    cov = coverage(rows, by_row, table["text"])
    audited = time.perf_counter()
    results = {}
    original_rng = np.random.default_rng(20261019)
    original_random = unit(np.random.default_rng(7).normal(size=(len(C), 768)))
    profile_seconds = permutation_seconds = 0.
    for prop, an in anchors.items():
        ids, y = an["rows"], np.asarray(an["labels"])
        require(len(ids) == len(set(ids)) == len(y) == report["anchors"][prop]["n"],
                "MNLM_NULL_UNITS", "independent entity rows must be unique and aligned")
        require(all(r in by_row for r in ids), "MNLM_NULL_MISSING_ROW", "anchor lacks chunk readings")
        t = time.perf_counter()
        prof = maxsim_profiles([C[by_row[r]] for r in ids], ids)
        D = chebyshev_matrix(prof)
        real = point(D, y)
        shuffled = point(D, original_rng.permutation(y))
        old_random = point(chebyshev_matrix(maxsim_profiles([original_random[by_row[r]] for r in ids], ids)), y)
        replay = {"real": abs(real["bits"] - report["per_lens"][best][prop]["chunk_multi"]["bits"]),
                  "labels_permuted": abs(shuffled["bits"] - report["controls"]["labels_permuted"]["per_anchor"][prop]),
                  "random_lens": abs(old_random["bits"] - report["controls"]["random_lens"]["per_anchor"][prop])}
        require(max(replay.values()) < 1e-10, "MNLM_NULL_REPLAY", f"{prop}: persisted assay did not replay: {replay}")
        random_counts, random_single = [], []
        for seed in SEEDS:
            V = unit(np.random.default_rng(seed).normal(size=(len(C), 768)))
            random_counts.append(point(chebyshev_matrix(maxsim_profiles([V[by_row[r]] for r in ids], ids)), y)["bits"])
            single = [V[by_row[r][:1]] for r in ids]
            random_single.append(point(chebyshev_matrix(maxsim_profiles(single, ids)), y)["bits"])
        profile_seconds += time.perf_counter() - t
        t = time.perf_counter()
        # Label permutations preserve the exact class counts and reuse geometry.
        rng = np.random.default_rng(2026101901)
        perm = [point(D, rng.permutation(y))["bits"] for _ in range(PERMUTATIONS)]
        permutation_seconds += time.perf_counter() - t
        results[prop] = {"real": real, "original_random": old_random, "original_shuffle": shuffled,
                         "replay_absolute_errors": replay, "n": len(y),
                         "chunks_per_entity": [len(by_row[r]) for r in ids],
                         "permuted_labels": distribution(perm),
                         "variable_count_random": distribution(random_counts),
                         "one_reading_random": distribution(random_single),
                         "calibration_minimum_met": len(y) >= 500}
        print(json.dumps({"anchor": prop, "replayed": True, "permutation_floor_crossings": sum(v >= FLOOR for v in perm)}), flush=True)
    finished = time.perf_counter()
    out = {"schema": "chunked-null-diagnosis-v1", "acceptance": "author-checked",
           "inputs": receipts, "best_lens": best, "anchor_family": "A2", "domain": summary["family"],
           "seed_roster": list(SEEDS), "permutations": PERMUTATIONS, "floor_unchanged": FLOOR,
           "coverage": cov, "anchors": results, "served": False, "scientifically_admitted_lenses": [],
           "conclusion_scope": "diagnosis only; no subtraction of nulls or change to admission gates",
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "reference_code_sha256": hashlib.sha256(Path(__file__).parents[1].joinpath("text/calyx_bits.py").read_bytes()).hexdigest(),
           "timing": {"load_hash_seconds": loaded-began, "coverage_seconds": audited-loaded,
                      "profiles_seconds": profile_seconds, "permutations_seconds": permutation_seconds,
                      "total_before_publication_seconds": finished-began}}
    output.mkdir(parents=True, exist_ok=False)
    publish(output / "report.json", plain(out))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("chunked", "slots", "source", "report", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    diagnose(a.chunked, a.slots, a.source, a.report, a.output)


if __name__ == "__main__":
    main()
