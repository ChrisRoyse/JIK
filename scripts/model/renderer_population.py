"""Pinned, bounded historical-pilot population for renderer conditioning research.

CPU preparation only. This exposed development population is not a final test,
admitted panel, or lossless replacement for the historical pilot slot store.
Cost: O(N + source bytes); one typed lane resident at a time, no pair matrix.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.constellation_lm import bucket
from senses.trajectory_population import publish

SCHEMA = "renderer-population-v1"
HISTORY, CAPACITY = 8, 512


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(ok, code):
    if not ok:
        raise ValueError("MNLM_RP_" + code)


def text_key(text):
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def inventory(table):
    """One possible target per document, selected independently of slot values."""
    n = len(table["doc"])
    flags = np.zeros(n, np.uint8)
    candidates = {}
    first = 0
    for i, (doc, pos, text) in enumerate(zip(table["doc"], table["pos"], table["text"])):
        if i == 0 or doc != table["doc"][i-1]:
            require(doc not in candidates, "NONCONTIGUOUS_DOCUMENT")
            first = i
            candidates[doc] = []
        require(pos == i-first, "SOURCE_POSITION")
        flags[i] = ((1 if i-first < HISTORY else 0) | (2 if not text.strip() else 0)
                    | (4 if len(text.encode("utf-8")) > CAPACITY else 0))
        if flags[i] == 0:
            candidates[doc].append(i)
    return flags, candidates


def select(table, candidates, counts):
    pools = {s: [] for s in counts}
    for doc, choices in candidates.items():
        b = bucket(doc)
        if b >= 95 or not choices:
            continue  # no historical test target enters this experiment
        split = "train" if b < 90 else ("calibration" if b < 93 else "evaluation")
        key = hashlib.sha256(("renderer-v1:" + doc).encode()).hexdigest()
        target = choices[int(key[16:32], 16) % len(choices)]
        pools[split].append((key, target))
    selected = []
    for split, count in counts.items():
        require(len(pools[split]) >= count, "UNDERPOWERED_" + split.upper())
        selected.extend((split, target) for _, target in sorted(pools[split])[:count])
    return selected, {s: len(p) for s, p in pools.items()}


def prepare(a):
    require(not a.output.exists(), "OUTPUT_EXISTS")
    started = time.perf_counter(); phases = {}
    torch.set_num_threads(32)
    a.output.mkdir(parents=True)
    manifest = json.loads(a.manifest.read_bytes())
    prep = json.loads((a.predictor/"report.json").read_bytes())
    crep = json.loads((a.codebook/"report.json").read_bytes())
    lenses = prep["lenses"]
    require(prep["schema"] == "recognising-predictor-v1", "PREDICTOR_ROUTE")
    require(lenses == crep["lenses"] and set(lenses) == set(manifest["lenses"]), "LENSES")
    sources = {"atoms": a.atoms/"atoms.parquet", "manifest": a.manifest,
               "predictor-report": a.predictor/"report.json", "predictor": a.predictor/"predictor.pt",
               "codebook-report": a.codebook/"report.json", "codebook": a.codebook/"codebook.npz",
               "codes": a.codebook/"codes.npy"}
    pins = {k: digest(p) for k, p in sources.items()}
    require(pins["atoms"] == manifest["atoms_sha256"], "ATOMS_PIN")
    require(pins["predictor"] == prep["predictor_sha256"], "PREDICTOR_PIN")
    require(pins["codebook"] == prep["codebook_sha256"] == crep["codebook_sha256"], "CODEBOOK_PIN")
    require(pins["codes"] == crep["codes_sha256"], "CODES_PIN")
    table = pq.read_table(sources["atoms"]).to_pydict()
    require(len(table["doc"]) == manifest["atoms"] == crep["atoms"], "SOURCE_ROWS")
    flags, candidates = inventory(table)
    counts = {"train": a.train, "calibration": a.calibration, "evaluation": a.evaluation}
    selected, available = select(table, candidates, counts)
    targets = np.asarray([i for _, i in selected], np.int64)
    indices = targets[:, None] + np.arange(-HISTORY, 1)[None, :]
    train_mask = np.asarray([bucket(d) < 90 for d in table["doc"]])
    train_hashes = {text_key(t) for t, yes in zip(table["text"], train_mask) if yes}
    rows = []
    for row, ((split, target), ix) in enumerate(zip(selected, indices)):
        rows.append(dict(row=row, split=split, target=int(target), doc=table["doc"][target],
                         domain=table["domain"][target], indices=ix.tolist(),
                         atoms=[{k: table[k][i] for k in table} for i in ix],
                         seen_in_predictor_training=text_key(table["text"][target]) in train_hashes))
    (a.output/"rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    np.save(a.output/"eligibility.npy", flags); np.save(a.output/"indices.npy", indices)
    all_codes = np.load(sources["codes"], mmap_mode="r", allow_pickle=False)
    require(all_codes.shape == (len(table["doc"]), len(prep["codes"])), "CODE_ROWS")
    np.save(a.output/"codes.npy", all_codes[indices])
    for key, name in (("predictor", "predictor.pt"), ("predictor-report", "predictor-report.json"),
                      ("codebook", "codebook.npz"), ("codebook-report", "codebook-report.json"),
                      ("manifest", "slot-manifest.json")):
        shutil.copyfile(sources[key], a.output/name)
    phases["metadata_and_selection"] = time.perf_counter()-started
    centres = {}; lane_times = {}; raw_pins = {}
    for k, name in enumerate(lenses):
        tick = time.perf_counter(); path = a.slots/(name+".npy")
        pin = digest(path)
        require(pin == manifest["lenses"][name]["slot_sha256"], "SLOT_PIN_"+name)
        raw = np.load(path, allow_pickle=False)
        require(raw.shape == (len(flags), prep["dims"][k]) and raw.dtype == np.float16, "SLOT_SHAPE")
        # Exact historical Data recipe, independently per lane, no GPU lens forward.
        x = torch.from_numpy(raw)
        centre = x[torch.from_numpy(train_mask)].float().mean(0)
        picked = x[torch.from_numpy(indices)].float()
        require(bool(torch.isfinite(centre).all()) and bool(torch.isfinite(picked).all()), "NONFINITE")
        delta = picked-centre
        require(bool((delta.norm(dim=-1) > 0).all()), "ZERO_SLOT")
        np.save(a.output/(name+".npy"), F.normalize(delta, dim=-1).half().numpy())
        centres[name] = centre.numpy()
        del x, raw, picked, delta
        require(digest(path) == pin, "SLOT_DRIFT")
        raw_pins[name] = pin; lane_times[name] = time.perf_counter()-tick
        print(json.dumps(dict(lane=name, seconds=lane_times[name])), flush=True)
    phases["centering_extract_and_source_hashes"] = sum(lane_times.values())
    tick = time.perf_counter()
    np.savez(a.output/"centres.npz", **centres)
    require({k: digest(p) for k, p in sources.items()} == pins, "SOURCE_DRIFT")
    files = {p.name: dict(sha256=digest(p), bytes=p.stat().st_size)
             for p in sorted(a.output.iterdir()) if p.is_file()}
    phases["final_source_and_output_hashes"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    report = dict(schema=SCHEMA, source_atoms=len(flags), selected=len(rows), counts=counts,
                  history=HISTORY, capacity_bytes=CAPACITY, eligible_documents=available,
                  eligibility_flags={str(k): int(v) for k, v in Counter(flags.tolist()).items()},
                  flag_bits={"1": "fewer_than_eight_preceding_atoms", "2": "empty_target", "4": "target_over_512_bytes"},
                  sources={k: dict(path=str(p), sha256=pins[k]) for k, p in sources.items()},
                  source_slot_sha256=raw_pins, source_slots=str(a.slots), lenses=lenses, dims=prep["dims"],
                  files=files, phases_seconds=phases, lane_seconds=lane_times, wall_seconds=wall,
                  phase_coverage=sum(phases.values())/wall, torch=str(torch.__version__), mean_threads=32,
                  source_code_sha256=digest(Path(__file__)), scientific_admission=False, serving_ready=False,
                  final_test=False, composition_success=False,
                  limitations=["historical pilot instrument truncation remains", "dev already inspected",
                               "near-duplicate and pretraining exposure unresolved", "512-byte bounded diagnostic",
                               "historical training did not persist centering; replayed from pinned raw slots"])
    publish(a.output/"report.json", report)
    print(json.dumps(dict(selected=len(rows), report_sha256=digest(a.output/"report.json"), wall_seconds=wall)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("atoms", "slots", "manifest", "predictor", "codebook", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--train", type=int, default=4096)
    p.add_argument("--calibration", type=int, default=256)
    p.add_argument("--evaluation", type=int, default=256)
    a = p.parse_args()
    require(all(getattr(a, n) > 0 for n in ("train", "calibration", "evaluation")), "COUNTS")
    prepare(a)


if __name__ == "__main__":
    main()
