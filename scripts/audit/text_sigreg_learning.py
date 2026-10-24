"""Matched atom-association learning with a declared per-slot anti-collapse term."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"lens")]
import torch
from audit.text_association_learning import train, cold
from claims.codata_source import check
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.text_content_scale import build_content
from model.text_sigreg import SigregData
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(a.corpus, a.corpus_pin)
    data = SigregData(corpus, Bank(a.bank, a.bank_pin, corpus), ModelConfig(), a.strength)
    phases = dict(source_and_association_targets=time.perf_counter()-started)
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    check(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "CONTENT_GPU_BUSY")
    take_lease("28", "text-association-sigreg")
    try:
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tick = time.perf_counter()
        model = build_content(data, a)
        phases["model_and_input_scale"] = time.perf_counter()-tick
        print(json.dumps(dict(input_scale=data.input_scale)), flush=True)
        tick = time.perf_counter()
        report = cold(a, model, data) if a.cold else train(a, model, data)
        phases["cold_replay" if a.cold else "training_and_physical_checks"] = time.perf_counter()-tick
        report.update(phases_seconds=phases, main_function_wall_seconds=time.perf_counter()-started,
            process_startup_included=False, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), acceptance="author-checked only")
        report["phase_coverage"] = sum(phases.values())/report["main_function_wall_seconds"]
        check(.90 <= report["phase_coverage"] <= 1.001, "CONTENT_PHASE_COVERAGE")
        publish(a.output/"report.json", report)
        print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"),
            seconds=report["main_function_wall_seconds"], cold_exact=report.get("exact"))), flush=True)
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("corpus", "bank", "identity-population", "identity-native", "identity-source", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--strength", type=float, choices=(0., .02), required=True)
    p.add_argument("--cold", type=Path)
    p.add_argument("--cold-pin")
    run(p.parse_args())
