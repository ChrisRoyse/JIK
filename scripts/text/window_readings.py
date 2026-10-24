"""Dense frozen senses with every tokenizer window retained separately (#12).

Uses the existing learned-lens tokenizer/provenance and native window forward.
This is a new CUDA runtime identity. No head truncation or window averaging.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import pyarrow.parquet as pq
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"lens"))
from lens.catalogue import by_name
from lens.learned import LearnedLens, weights_sha256, preprocessing_sha256, TOKEN_BUDGET, MAX_WINDOWS
from lens.lenscommon import take_lease, run_cli, LensError
from senses.trajectory_population import publish
import text_windows

RUNTIME, BATCH = "dense-window-readings-cuda-v1", 32


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, code):
    if not ok:
        raise LensError(code, "full-window measurement contract refused")


def load_chunks(root, report_hash):
    raw = (root/"report.json").read_bytes()
    require(sha(raw) == report_hash, "MNLM_WINDOW_INPUT_REPORT")
    report = json.loads(raw)
    require(report["complete"] and not report["errors"] and report["calibration_only"], "MNLM_WINDOW_INPUT_INCOMPLETE")
    records, sources = [], {}
    for family, item in sorted(report["families"].items()):
        sr = (root/family/"summary.json").read_bytes()
        require(sha(sr) == item["summary_sha256"], "MNLM_WINDOW_INPUT_SUMMARY")
        summary = json.loads(sr)
        if not item["ready"]:
            require(summary["chunks"] == 0, "MNLM_WINDOW_UNAVAILABLE")
            continue
        body = (root/family/"chunks/atoms.parquet").read_bytes()
        require(sha(body) == summary["chunks_sha256"], "MNLM_WINDOW_INPUT_CHUNKS")
        rows = pq.read_table(io.BytesIO(body)).to_pylist()
        require(len(rows) == summary["chunks"], "MNLM_WINDOW_INPUT_COUNT")
        records.extend({**r, "family": family} for r in rows)
        sources[family] = {"summary_sha256": sha(sr), "chunks_sha256": sha(body), "chunks": len(rows)}
    require(bool(records), "MNLM_WINDOW_NO_INPUT")
    return records, sources


@torch.inference_mode()
def forward(lens, texts):
    encoded, mapping = lens._tokenize(texts)
    width = encoded["input_ids"].shape[1]
    window_batch = min(BATCH, max(1, TOKEN_BUDGET//width))
    values = []
    for start in range(0, len(mapping), window_batch):
        batch = {k: v[start:start+window_batch].cuda() for k,v in encoded.items()}
        values.append(lens._windows(batch))
    vectors = np.concatenate(values).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    require(np.isfinite(vectors).all() and (norms > 0).all(), "MNLM_WINDOW_NONFINITE")
    vectors /= norms
    require(mapping == sorted(mapping), "MNLM_WINDOW_ORDER")
    counts = np.bincount(mapping, minlength=len(texts))
    require((counts > 0).all() and (counts <= MAX_WINDOWS).all(), "MNLM_WINDOW_COVERAGE")
    return vectors, np.r_[0, np.cumsum(counts)], lens.segments


def measure(name, records, root):
    started = time.perf_counter()
    spec = by_name(name)
    require(spec["kind"] == "dense", "MNLM_WINDOW_DENSE_ONLY")
    lens = LearnedLens(spec, "cuda")
    require(not lens.model.training and all(not p.requires_grad for p in lens.model.parameters()), "MNLM_WINDOW_UNFROZEN")
    original_contract = lens.contract()
    contract = {**original_contract, "runtime": RUNTIME, "kind": "dense-window-multi", "shape": f"MultiWindows({lens.dim})",
                "window_reduction": "none:ordered-window-vectors", "execution_unit": "fixed-source-batch-32-v1",
                "batch": BATCH, "torch": torch.__version__, "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(0), "platform": sys.platform,
                "backend_code_sha256": sha(Path(sys.modules[LearnedLens.__module__].__file__).read_bytes()),
                "runtime_code_sha256": sha(Path(__file__).read_bytes())}
    lens_id = sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    loaded = time.perf_counter()
    dest = root/name
    dest.mkdir()
    texts = [r["text"] for r in records]
    first = texts[:BATCH]
    x, off, seg = forward(lens, first)
    x2, off2, seg2 = forward(lens, first)
    require(np.array_equal(x, x2) and np.array_equal(off, off2) and seg == seg2, "MNLM_WINDOW_NONDETERMINISTIC")
    # A real reassembled document probes overflow without a synthetic repetition.
    documents = defaultdict(list)
    for r in records:
        documents[(r["family"], r["row"])].append(r["text"])
    candidates = [(k, "".join(v)) for k,v in documents.items() if 1800 <= sum(map(len,v)) <= 3000]
    require(bool(candidates), "MNLM_WINDOW_REAL_PROBE_MISSING")
    probe_key, probe_text = max(candidates, key=lambda item: len(item[1]))
    px, po, ps = forward(lens, [probe_text])
    px2, po2, ps2 = forward(lens, [probe_text])
    require(np.array_equal(px, px2) and ps == ps2 and np.array_equal(po, po2), "MNLM_WINDOW_NONDETERMINISTIC")
    np.savez(dest/"probe.npz", vectors=px, offsets=po)
    publish(dest/"probe.json", {"family": probe_key[0], "row": probe_key[1], "text_sha256": sha(probe_text.encode()),
                               "trace": ps[0], "purpose": "real full-document overflow/determinism probe only"})
    probed = time.perf_counter()
    vectors, offsets, cursor, count_hist = [], [0], 0, Counter()
    forward_seconds = trace_seconds = 0.0
    with gzip.open(dest/"traces.jsonl.gz", "wt", encoding="utf-8", newline="\n") as trace:
        for start in range(0, len(texts), BATCH):
            at = time.perf_counter()
            if start == 0:
                batch_vectors, batch_offsets, traces = x, off, seg
            else:
                batch_vectors, batch_offsets, traces = forward(lens, texts[start:start+BATCH])
            forward_seconds += time.perf_counter()-at
            at = time.perf_counter()
            vectors.append(batch_vectors)
            for j, tr in enumerate(traces):
                index = start+j
                row = records[index]
                n = int(batch_offsets[j+1]-batch_offsets[j])
                record = {"input": index, "family": row["family"], "row": row["row"], "chunk": row["chunk"],
                          "entity": row["entity"], "vector_span": [cursor, cursor+n], "trace": tr}
                trace.write(json.dumps(record, separators=(",", ":"))+"\n")
                cursor += n; offsets.append(cursor); count_hist[n] += 1
            trace_seconds += time.perf_counter()-at
    values = np.concatenate(vectors)
    np.savez(dest/"slots.npz", vectors=values, offsets=np.asarray(offsets, dtype=np.int64))
    require(weights_sha256(lens.snap)[0] == lens.sha256 and preprocessing_sha256(lens.snap) == lens.preprocessing,
            "MNLM_WINDOW_INSTRUMENT_DRIFT")
    files = {n: {"sha256": sha((dest/n).read_bytes()), "bytes": (dest/n).stat().st_size}
             for n in ("slots.npz", "traces.jsonl.gz", "probe.npz", "probe.json")}
    elapsed = time.perf_counter()-started
    report = {"name": name, "lens_id": lens_id, "contract": contract, "inputs": len(records), "windows": len(values),
              "window_histogram": dict(count_hist), "probe_windows": len(px), "determinism_probe_equal": True,
              "weights_and_preprocessing_unchanged": True, "files": files, "served": False, "scientific_admission": False,
              "timing": {"load_seconds": loaded-started, "probe_seconds": probed-loaded, "forward_seconds": forward_seconds,
                         "trace_seconds": trace_seconds, "write_rehash_seconds": elapsed-(probed-started)-forward_seconds-trace_seconds,
                         "wall_seconds": elapsed}}
    publish(dest/"report.json", report)
    print(json.dumps({k: report[k] for k in ("name", "inputs", "windows", "probe_windows", "timing")}), flush=True)
    del lens
    torch.cuda.empty_cache()
    return {"report_sha256": sha((dest/"report.json").read_bytes()), "lens_id": lens_id, "windows": len(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--calibration-report-sha256", required=True)
    p.add_argument("--lenses", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "MNLM_WINDOW_EXISTS")
    names = a.lenses.split(",")
    require(bool(names) and len(names) == len(set(names)), "MNLM_WINDOW_ROSTER")
    require(torch.cuda.is_available(), "MNLM_WINDOW_NO_GPU")
    residents = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    take_lease("12", "calibration-full-window-senses")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, sources = load_chunks(a.calibration, a.calibration_report_sha256)
    a.output.mkdir(parents=True)
    (a.output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    published, errors = {}, []
    for name in names:
        try:
            published[name] = measure(name, records, a.output)
        except (LensError, OSError, ValueError, RuntimeError) as exc:
            errors.append({"lens": name, "error": str(exc)})
            print(json.dumps(errors[-1]), flush=True)
            torch.cuda.empty_cache()
    publish(a.output/"report.json", {"schema": RUNTIME, "calibration_report_sha256": a.calibration_report_sha256,
            "sources": sources, "inputs": len(records), "requested": names, "lenses": published, "errors": errors,
            "complete": not errors, "scientific_admission": False, "served": False})
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(run_cli(main))
