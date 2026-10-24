"""Real learned GPU instruments through UCR, with physical slot readback.

Uses retained integration atoms and cached checkpoints only. No new dataset,
anchor fit or admission. A separate index version pins this host's runtime.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import canonical
from senses.pipeline import ingest_file, file_hash
from senses.store import Store
from catalogue import by_name
from learned_ucr import measure_learned_ucr
from lenscommon import LensError, take_lease, run_cli
from measure import nvidia_smi
from slotio import slot_digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lenses", default="ucr-all-minilm-l6-v2,ucr-answerai-colbert-small,ucr-opensearch-sparse-doc-v2,ucr-potion-base-8m")
    args = ap.parse_args()
    src, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    before = nvidia_smi()
    (out / "resident-before.txt").write_text(before)
    take_lease("15", "UCR learned-input boundary FSV, small cached models, projected <2 GiB model memory")
    import torch
    torch.use_deterministic_algorithms(True)
    store = Store(out / "channels.sqlite", create=True)
    for source in ("wikidata5m", "wikipedia-en-20231101"):
        path = src / (source + ".ucr.jsonl")
        ingest_file(store, path, json.loads(path.with_suffix(".manifest.json").read_text()), store.head)
    generation = store.head
    store.close()
    cal = out / "lenses" / "gpu" / "cal"
    cal.mkdir(parents=True)
    fams = {}
    for path in (src / "lenses" / "integration" / "cal").glob("*.jsonl"):
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()][:16]
        fams[path.stem] = rows
        (cal / path.name).write_bytes(b"".join(canonical(r) + b"\n" for r in rows))
    policies = out / "policies.json"
    policies.write_bytes((src / "policies.json").read_bytes())
    entries, checks = {}, []
    slots = cal.parent / "slots"
    for name in args.lenses.split(","):
        entry = measure_learned_ucr(by_name(name), fams, slots, {}, out / "channels.sqlite", policies, 8)
        entries[name] = entry
        checks.append({"name": name + "_actual_gpu_probe", "pass": entry["probe"]["byte_identical"]
                       and entry["probe"]["present"] > 0 and entry["contract"]["runtime"]["device"].startswith("NVIDIA")})
        for fam, meta in entry["families"].items():
            path = Path(meta["slot"])
            got = slot_digest(path)
            with np.load(path, allow_pickle=False) as z:
                present = ~z["absent"]
                absent_preserved = not present.any() if fam.startswith("a2-") else present.all()
                kind = by_name(name)["kind"]
                layout = ("tokens" in z and "offsets" in z) if kind == "multi" else ("indptr" in z and "data" in z) if kind == "sparse" else "X" in z
            checks.append({"name": name + "_" + fam + "_cold_readback", "pass": got["digest"] == meta["digest"]
                           and got["n"] == len(fams[fam]) and bool(absent_preserved) and layout})
    # Failed measurement must not mutate the authoritative observation store.
    bad = copy.deepcopy(fams)
    next(iter(bad.values()))[0]["text"] = "unbound answer injected into calibration"
    refused = None
    try:
        measure_learned_ucr(by_name(args.lenses.split(",")[0]), bad, slots, {}, out / "channels.sqlite", policies, 8)
    except LensError as exc:
        refused = exc.code
    cold = Store(out / "channels.sqlite")
    checks.append({"name": "unbound_surface_refused_without_mutation", "pass": refused == "MNLM_UCR_SURFACE_BINDING" and cold.head == generation})
    census = cold.audit()
    cold.close()
    # Actual tokenizer, real source text, repeated only as a labelled overflow
    # boundary control. Overflow must refuse before any model window is dropped.
    from learned import LearnedLens
    lens = LearnedLens(by_name(args.lenses.split(",")[0]), "cuda")
    real = next(r["text"] for rs in fams.values() for r in rs if r["text"].strip())
    refused = None
    try:
        lens._tokenize([real * 1000])
    except LensError as exc:
        refused = exc.code
    checks.append({"name": "whole_atom_overflow_refused", "pass": refused == "MNLM_LENS_WINDOW_BUDGET"})
    del lens
    torch.cuda.empty_cache()
    report = {"acceptance": "author-checked", "metric_kind": "GPU integration; not information admission", "generation": generation,
              "checks": checks, "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks),
              "store_readback": census, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "resident_before": before}
    (out / "registry.json").write_bytes(canonical(entries))
    (out / "report.json").write_bytes(canonical(report) + b"\n")
    (out / "SHA256.json").write_bytes(canonical({str(p.relative_to(out)): file_hash(p) for p in out.rglob("*") if p.is_file()
                                               and not p.name.endswith(("-shm", "-wal"))}))
    print(json.dumps({"passed": report["passed"], "failed": report["failed"], "gpu_peak_allocated_bytes": report["gpu_peak_allocated_bytes"]}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(run_cli(main))
