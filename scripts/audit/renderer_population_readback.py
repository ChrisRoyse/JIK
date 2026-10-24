"""Cold structural and optional full-source replay of renderer-population-v1."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(ok, code):
    if not ok:
        raise ValueError("MNLM_RP_READ_"+code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def bucket(doc):
    return int(hashlib.sha256(doc.encode()).hexdigest()[:8], 16) % 100


def read(root, pin, source=False):
    started = time.perf_counter()
    check(sha(root/"report.json") == pin, "REPORT_PIN")
    r = json.loads((root/"report.json").read_bytes())
    check(r["schema"] == "renderer-population-v1" and r["history"] == 8 and r["capacity_bytes"] == 512, "CONTRACT")
    check(all(r.get(k) is False for k in ("scientific_admission", "serving_ready", "final_test", "composition_success")), "FALSE_CLAIM")
    names = {"rows.jsonl", "eligibility.npy", "indices.npy", "codes.npy", "centres.npz", "predictor.pt",
             "predictor-report.json", "codebook.npz", "codebook-report.json", "slot-manifest.json"}
    names |= {n+".npy" for n in r["lenses"]}
    check(set(r["files"]) == names and len(r["lenses"]) == len(set(r["lenses"])) == len(r["dims"]), "FILE_SET")
    for name, meta in r["files"].items():
        check(sha(root/name) == meta["sha256"] and (root/name).stat().st_size == meta["bytes"], "FILE_"+name)
    pred = json.loads((root/"predictor-report.json").read_bytes())
    book = json.loads((root/"codebook-report.json").read_bytes())
    manifest = json.loads((root/"slot-manifest.json").read_bytes())
    check(pred["lenses"] == book["lenses"] == r["lenses"] and pred["dims"] == r["dims"], "LENSES")
    check(sha(root/"predictor.pt") == pred["predictor_sha256"] == r["sources"]["predictor"]["sha256"], "PREDICTOR")
    check(sha(root/"codebook.npz") == pred["codebook_sha256"] == book["codebook_sha256"], "CODEBOOK")
    check(manifest["atoms_sha256"] == r["sources"]["atoms"]["sha256"], "SOURCE_ATOMS")
    check({n: manifest["lenses"][n]["slot_sha256"] for n in r["lenses"]} == r["source_slot_sha256"], "SOURCE_SLOTS")
    rows = [json.loads(line) for line in (root/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    idx = np.load(root/"indices.npy", allow_pickle=False)
    flags = np.load(root/"eligibility.npy", allow_pickle=False)
    codes = np.load(root/"codes.npy", allow_pickle=False)
    centres = np.load(root/"centres.npz", allow_pickle=False)
    check(len(rows) == r["selected"] == sum(r["counts"].values()) and idx.shape == (len(rows), 9), "ROWS")
    check(idx.dtype == np.int64 and np.all(np.diff(idx, axis=1) == 1) and idx.min() >= 0 and idx.max() < r["source_atoms"], "INDICES")
    check(flags.dtype == np.uint8 and flags.shape == (r["source_atoms"],) and np.all(flags <= 7), "ELIGIBILITY")
    check(dict(Counter(x["split"] for x in rows)) == r["counts"], "SPLITS")
    check(len({x["doc"] for x in rows}) == len(rows), "DOCUMENT_LEAKAGE")
    check(codes.shape == (len(rows), 9, len(pred["codes"])) and np.issubdtype(codes.dtype, np.integer), "CODES")
    check(all(np.all((codes[..., k] >= 0) & (codes[..., k] < c)) for k, c in enumerate(pred["codes"])), "CODE_RANGE")
    check({str(k): v for k, v in Counter(flags.tolist()).items()} == r["eligibility_flags"], "FLAG_COUNTS")
    for i, row in enumerate(rows):
        check(row["row"] == i and row["indices"] == idx[i].tolist() and row["target"] == int(idx[i, -1]), "ROW_ORDER")
        b = bucket(row["doc"])
        split = "train" if b < 90 else "calibration" if b < 93 else "evaluation" if b < 95 else "test"
        check(row["split"] == split and split != "test", "SPLIT_HASH")
        atoms = row["atoms"]
        check(len(atoms) == 9 and all(x["doc"] == row["doc"] and x["domain"] == row["domain"] for x in atoms), "HISTORY_DOCUMENT")
        check([x["pos"] for x in atoms] == list(range(atoms[0]["pos"], atoms[0]["pos"]+9)), "HISTORY_POSITIONS")
        target = atoms[-1]["text"]
        check(bool(target.strip()) and len(target.encode("utf-8")) <= 512 and flags[row["target"]] == 0, "COMPLETE_TARGET")
    for name, d in zip(r["lenses"], r["dims"]):
        x = np.load(root/(name+".npy"), mmap_mode="r", allow_pickle=False)
        check(x.shape == (len(rows), 9, d) and x.dtype == np.float16, "SLOT_SHAPE")
        check(np.isfinite(x).all() and np.all(abs(np.linalg.norm(x.astype(np.float32), axis=-1)-1) < .001), "SLOT_NORM")
        check(centres[name].shape == (d,) and centres[name].dtype == np.float32 and np.isfinite(centres[name]).all(), "CENTRE")
    check(.9 <= r["phase_coverage"] <= 1.001 and abs(sum(r["phases_seconds"].values())/r["wall_seconds"]-r["phase_coverage"]) < 1e-10, "TIMING")
    source_read = None
    if source:
        source_read = replay_source(root, r, rows, idx, flags, codes, centres)
    check(sha(root/"report.json") == pin, "REPORT_DRIFT")
    return dict(report_sha256=pin, selected=len(rows), split_counts=r["counts"],
                files=len(names), source_replay=source_read, scientific_admission=False,
                acceptance="author-checked", wall_seconds=time.perf_counter()-started)


def replay_source(root, r, rows, idx, flags, codes, centres):
    import pyarrow.parquet as pq
    import torch
    for record in r["sources"].values():
        check(sha(record["path"]) == record["sha256"], "SOURCE_PIN")
    table = pq.read_table(r["sources"]["atoms"]["path"]).to_pydict()
    train = np.asarray([bucket(d) < 90 for d in table["doc"]])
    train_hashes = {hashlib.sha256(t.strip().lower().encode()).hexdigest()
                    for t, yes in zip(table["text"], train) if yes}
    for row, ix in zip(rows, idx):
        check(row["atoms"] == [{k: table[k][j] for k in table} for j in ix], "SOURCE_TEXT")
        seen = hashlib.sha256(row["atoms"][-1]["text"].strip().lower().encode()).hexdigest() in train_hashes
        check(row["seen_in_predictor_training"] == seen, "SEEN_FLAG")
    actual_flags = np.zeros(len(flags), np.uint8)
    for i, (pos, text) in enumerate(zip(table["pos"], table["text"])):
        actual_flags[i] = int(pos < 8) | (2 if not text.strip() else 0) | (4 if len(text.encode()) > 512 else 0)
    check(np.array_equal(actual_flags, flags), "SOURCE_ELIGIBILITY")
    pools = {k: [] for k in r["counts"]}
    by_doc = {}
    for i, (doc, flag) in enumerate(zip(table["doc"], flags)):
        if flag == 0 and bucket(doc) < 95:
            by_doc.setdefault(doc, []).append(i)
    for doc, choices in by_doc.items():
        b = bucket(doc); split = "train" if b < 90 else "calibration" if b < 93 else "evaluation"
        key = hashlib.sha256(("renderer-v1:"+doc).encode()).hexdigest()
        pools[split].append((key, choices[int(key[16:32], 16) % len(choices)]))
    # JSON key ordering is not row ordering; validate each named split separately.
    for split in r["counts"]:
        check([row["target"] for row in rows if row["split"] == split] ==
              [t for _, t in sorted(pools[split])[:r["counts"][split]]], "SOURCE_SELECTION")
    check({k: len(v) for k, v in pools.items()} == r["eligible_documents"], "SOURCE_POOL_COUNTS")
    original_codes = np.load(r["sources"]["codes"]["path"], mmap_mode="r", allow_pickle=False)
    check(np.array_equal(codes, original_codes[idx]), "SOURCE_CODES")
    torch.set_num_threads(32)
    for name in r["lenses"]:
        path = Path(r["source_slots"])/(name+".npy")
        check(sha(path) == r["source_slot_sha256"][name], "SOURCE_LANE")
        raw = np.load(path, allow_pickle=False)
        centre = torch.from_numpy(raw[train]).float().mean(0).numpy()
        check(np.array_equal(centre, centres[name]), "SOURCE_CENTRE")
        v = torch.from_numpy(raw[idx]).float()-torch.from_numpy(centre)
        expected = (v/v.norm(dim=-1, keepdim=True).clamp_min(1e-12)).half().numpy()
        check(np.array_equal(expected, np.load(root/(name+".npy"), allow_pickle=False)), "SOURCE_SLOT_VALUES")
        del raw, v, expected
        print(json.dumps(dict(source_lane_checked=name)), flush=True)
    return dict(all_source_rows_and_codes_equal=True, all_selected_slots_equal=True, centres_equal=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", type=Path, required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--source-replay", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); check(not a.output.exists(), "OUTPUT_EXISTS")
    result = read(a.population, a.report_sha256, a.source_replay)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
