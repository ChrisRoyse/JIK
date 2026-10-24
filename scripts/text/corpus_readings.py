"""Streaming full-window frozen text readings of a lossless constellation corpus.

PC-07/28/35: one instrument load per pass, bounded source batches, one columnar
file per instrument. Tokenizer windows and their native dimensions remain typed.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lens.catalogue import by_name
from lens.learned import LearnedLens, weights_sha256, preprocessing_sha256
from lenscommon import take_lease, release_lease
from senses.trajectory_population import publish
from text.constellation_corpus import canonical, digest, file_sha, need
from text.constellation_input import Corpus
from text.window_readings import BATCH, forward, transformers

SCHEMA = "text-constellation-window-bank-v1"
ROSTER = ("all-mpnet-base-v2", "codebert-mean", "e5-base-v2", "finbert-mean",
          "pubmedbert-embeddings", "specter2-base")


def measure(name, corpus, output):
    start = time.perf_counter()
    destination = output/name
    destination.mkdir()
    spec = by_name(name)
    need(spec["kind"] == "dense", "LENS_KIND")
    lens = LearnedLens(spec, "cuda")
    need(not lens.model.training and all(not p.requires_grad for p in lens.model.parameters()), "LENS_FROZEN")
    contract = dict(**lens.contract())
    contract.update(runtime=SCHEMA, window_runtime="dense-window-readings-cuda-v1",
        reduction="none; ordered complete tokenizer windows", execution_unit="source-order-batch-32",
        batch=BATCH, dimensions=lens.dim, torch=torch.__version__, transformers=transformers.__version__,
        cuda=torch.version.cuda, device=torch.cuda.get_device_name(), platform=sys.platform,
        code={"corpus_readings": file_sha(Path(__file__)),
              "window_readings": file_sha(Path(sys.modules[forward.__module__].__file__)),
              "learned": file_sha(Path(sys.modules[LearnedLens.__module__].__file__))})
    lens_id = digest(canonical(contract))
    phases = dict(load_seconds=time.perf_counter()-start, forward_seconds=0., persistence_seconds=0.,
                  source_seconds=0., probe_seconds=0.)
    schema = pa.schema([("index", pa.int64()), ("atom_id", pa.string()),
        ("text_sha256", pa.string()), ("vectors", pa.list_(pa.list_(pa.float32(), lens.dim))),
        ("trace", pa.string())])
    offset, windows, multiple, longest, probe = 0, 0, 0, None, None
    iterator = iter(corpus.batches(BATCH))
    with pq.ParquetWriter(destination/"readings.parquet", schema, compression="zstd") as writer:
        while True:
            tick = time.perf_counter()
            rows = next(iterator, None)
            phases["source_seconds"] += time.perf_counter()-tick
            if rows is None:
                break
            need(all(r["kind"] == "text" for r in rows), "LAYOUT_REQUIRES_EXPLICIT_ABSENCE")
            texts = [r["text"] for r in rows]
            tick = time.perf_counter()
            values, bounds, traces = forward(lens, texts)
            phases["forward_seconds"] += time.perf_counter()-tick
            if offset == 0:
                tick = time.perf_counter()
                replay, b2, t2 = forward(lens, texts)
                need(np.array_equal(values, replay) and np.array_equal(bounds, b2) and traces == t2,
                     "LENS_NONDETERMINISTIC")
                phases["probe_seconds"] += time.perf_counter()-tick
                probe = dict(first_batch=len(rows), exact=True)
            tick = time.perf_counter()
            saved = []
            for i, row in enumerate(rows):
                lo, hi = map(int, bounds[i:i+2])
                native = values[lo:hi]
                need(len(traces[i]["windows"]) == len(native), "TRACE_COVERAGE")
                saved.append(dict(index=offset+i, atom_id=row["atom_id"], text_sha256=row["text_sha256"],
                                  vectors=native.tolist(), trace=canonical(traces[i]).decode()))
                if longest is None or len(row["text"]) > len(longest["text"]):
                    longest = row
                windows += len(native)
                multiple += len(native) > 1
            writer.write_table(pa.Table.from_pylist(saved, schema=schema))
            offset += len(rows)
            phases["persistence_seconds"] += time.perf_counter()-tick
    tick = time.perf_counter()
    values, bounds, traces = forward(lens, [longest["text"]])
    replay, b2, t2 = forward(lens, [longest["text"]])
    need(np.array_equal(values, replay) and np.array_equal(bounds, b2) and traces == t2
         and len(values) > 1, "REAL_OVERFLOW_PROBE")
    np.savez(destination/"overflow.npz", vectors=values, offsets=bounds)
    publish(destination/"overflow.json", dict(atom=longest["atom_id"], text_sha256=longest["text_sha256"],
        windows=len(values), trace=traces[0], exact=True))
    probe["overflow_windows"] = len(values)
    phases["probe_seconds"] += time.perf_counter()-tick
    tick = time.perf_counter()
    need(weights_sha256(lens.snap)[0] == lens.sha256 and preprocessing_sha256(lens.snap) == lens.preprocessing,
         "LENS_DRIFT")
    need(offset == corpus.report["counts"]["atoms"], "READING_COVERAGE")
    files = {n: dict(bytes=(destination/n).stat().st_size, sha256=file_sha(destination/n))
             for n in ("readings.parquet", "overflow.npz", "overflow.json")}
    corpus.fresh()
    phases["pin_seconds"] = time.perf_counter()-tick
    wall = time.perf_counter()-start
    report = dict(schema=SCHEMA, corpus=corpus.pin, name=name, contract=contract, lens_id=lens_id,
        inputs=offset, windows=windows, multiwindow_atoms=multiple, dimensions=lens.dim, files=files,
        determinism=probe, frozen_unchanged=True, scientific_admission=False, served=False,
        phases_seconds=phases, wall_seconds=wall, phase_coverage=sum(phases.values())/wall)
    publish(destination/"report.json", report)
    print(json.dumps(dict(name=name, atoms=offset, windows=windows, multiwindow_atoms=multiple, seconds=wall)), flush=True)
    del lens
    torch.cuda.empty_cache()
    return dict(report_sha256=file_sha(destination/"report.json"), lens_id=lens_id, windows=windows)


def main(a):
    start = time.perf_counter()
    corpus = Corpus(a.corpus, a.pin)
    need(torch.cuda.is_available(), "GPU_REQUIRED")
    need(not a.output.exists(), "BANK_EXISTS")
    resident = subprocess.check_output(["nvidia-smi"], text=True)
    need(not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                     "--format=csv,noheader"], text=True).strip(), "GPU_BUSY")
    take_lease("28", "text-constellation-readings")
    try:
        a.output.mkdir(parents=True)
        (a.output/"gpu-before.txt").write_text(resident)
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        outputs, errors = {}, []
        for name in ROSTER:
            try:
                outputs[name] = measure(name, corpus, a.output)
            except Exception as exc:
                errors.append(dict(name=name, error=type(exc).__name__+":"+str(exc)))
                print(json.dumps(errors[-1]), flush=True)
                torch.cuda.empty_cache()
        publish(a.output/"bank.json", dict(schema=SCHEMA, corpus=a.pin, roster=list(ROSTER), instruments=outputs,
            errors=errors, complete=not errors, scientific_admission=False, served=False,
            code_sha256=file_sha(Path(__file__)), seconds=time.perf_counter()-start))
        need(not errors, "READINGS_INCOMPLETE")
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("corpus", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--pin", required=True)
    main(p.parse_args())
