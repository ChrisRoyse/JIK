"""Cold streaming replay and independent functional dense attention readback."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROCESS_STARTED = time.perf_counter()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from claims.native_evidence import NativeEvidence
from claims.native_memory import NativeMemory
from lenscommon import take_lease, run_cli
from model.attention import Attention, Linear
from model.contracts import ModelConfig
from model.native_stream_input import from_memory, descriptor
from model.stream_attention import observe, blocks
from senses.perception_corpus import pinned, save, sha


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_STREAM_READBACK_"+code)


def linear(x, weight, bias):
    y = F.linear(x.contiguous().reshape(-1, weight.shape[1]), weight, None)+bias
    return y.reshape(*x.shape[:-1], weight.shape[0])


def dense(module, query, native, pad, heads):
    """No calls to Attention.forward, Linear.forward or streaming arithmetic."""
    p = dict(module.named_parameters())
    project = lambda key, x: linear(x, p[key+".weight"], p[key+".bias"])
    b, length, width = query.shape
    memory = project("projector", native)
    q = project("attention.q", query).reshape(b, length, heads, width//heads).transpose(1, 2)
    k = project("attention.k", memory).reshape(b, memory.shape[1], heads, width//heads).transpose(1, 2)
    v = project("attention.v", memory).reshape_as(k.transpose(1, 2)).transpose(1, 2)
    scores = q.matmul(k.transpose(-2, -1))/math.sqrt(width//heads)
    probabilities = torch.softmax(scores.masked_fill(pad[:, None, None], -torch.inf), dim=-1)
    value = probabilities.matmul(v).transpose(1, 2).contiguous().reshape(b, length, width)
    return project("attention.out", value)


def gradients(module, query, forward):
    module.zero_grad(set_to_none=True)
    q = query.detach().clone().cuda().requires_grad_(True)
    output = forward(q)
    output.square().sum().backward()
    return dict(output=output.detach().cpu(), query_gradient=q.grad.detach().cpu(),
                gradients={n: p.grad.detach().cpu() for n, p in module.named_parameters()})


def tensor_pairs(a, b):
    need(set(a) == set(b) == {"output", "query_gradient", "gradients"}
         and a["gradients"].keys() == b["gradients"].keys(), "OUTPUT_ROSTER")
    return [(a["output"], b["output"]), (a["query_gradient"], b["query_gradient"])] + [
        (a["gradients"][n], b["gradients"][n]) for n in a["gradients"]]


def main(a):
    started = PROCESS_STARTED
    report = pinned(a.root/"report.json", a.report_pin)
    need(report["equivalence_passed"] and report["atol"] == 1e-5 and report["rtol"] == 1e-4,
         "PRECOMMITTED_GATE")
    for name, expected in report["code"].items():
        need(sha(Path(__file__).resolve().parents[1]/name) == expected, "EXECUTION_CODE")
    need(sha(a.root/"weights.pt") == report["weights_pin"]
         and sha(a.root/"memory-at-read.sqlite") == report["memory_pin"], "RETAINED_BYTES")
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader"], text=True)
    need(not resident.strip(), "GPU_BUSY")
    take_lease("28", "native-stream-cold-readback")
    need(torch.cuda.is_available(), "NO_GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(32)
    checker = NativeEvidence(report["bank"], report["report_pin"], report["inputs"], report["corpus"])
    memory = NativeMemory(a.root/"memory-at-read.sqlite", checker, create=False, retained=report["retained"])
    streams = tuple(s for m, r in zip(report["moments"], report["requests"])
                    for s in from_memory(memory, m["id"], **r))
    config = ModelConfig(**report["config"])
    modules = nn.ModuleDict()
    for stream in streams:
        if stream.instrument not in modules:
            modules[stream.instrument] = nn.ModuleDict(dict(
                projector=Linear(stream.values.shape[1], config.width), attention=Attention(config)))
    modules.load_state_dict(torch.load(a.root/"weights.pt", map_location="cpu", weights_only=True), strict=True)
    modules.cuda()
    phases = dict(open_sources_and_checkpoint=time.perf_counter()-started)
    tick = time.perf_counter()
    need(len(streams) == len(report["rows"]) == 135, "READING_ROSTER")
    metrics = []
    for stream, row in zip(streams, report["rows"]):
        need(descriptor(stream) == {k: row[k] for k in descriptor(stream)}, "TYPED_PROVENANCE")
        position = 0
        for lo, hi in row["coverage"]:
            need(lo == position and 0 < hi-lo <= 256, "BLOCK_COVERAGE")
            position = hi
        need(position == stream.values.shape[0] and row["native_sha256"] ==
             row["provenance"]["entry"]["sha256"], "COMPLETE_COVERAGE")
        need(sha(a.root/row["proof"]) == row["proof_pin"], "PROOF_BYTES")
        proof = torch.load(a.root/row["proof"], map_location="cpu", weights_only=True)
        module = modules[stream.instrument]
        replay = gradients(module, proof["query"], lambda q: observe(
            module["attention"], module["projector"], q, stream.blocks(config),
            counts=stream.counts, config=config))
        need(all(torch.equal(x, y) for x, y in tensor_pairs(replay, proof["streamed"])), "EXACT_RESTART")
        values = torch.from_numpy(stream.values.copy())[None].cuda()
        pad = torch.zeros(values.shape[:2], device="cuda", dtype=torch.bool)
        independent = gradients(module, proof["query"], lambda q: dense(module, q, values, pad, config.heads))
        pairs = tensor_pairs(independent, replay)
        need(all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in pairs), "DENSE_GRADIENTS")
        metrics.append(dict(reference=stream.reference, exact_restart=True,
                            maximum_absolute_error=max(float((x-y).abs().max()) for x, y in pairs)))
    phases["cold_outputs_and_gradients"] = time.perf_counter()-tick
    tick = time.perf_counter()
    # A real unequal-length batch exercises rows whose later blocks are entirely padding.
    candidates = [s for s in streams if s.instrument == "image_dinov2"]
    long, short = max(candidates, key=lambda s: s.counts), min(candidates, key=lambda s: s.counts)
    need(long.counts[0] > short.counts[0]+256, "UNEQUAL_REAL_ROWS")
    values = torch.zeros((2, long.counts[0], long.values.shape[1]))
    for i, s in enumerate((long, short)):
        values[i, :s.counts[0]] = torch.from_numpy(s.values.copy())
    counts = (long.counts[0], short.counts[0])
    pad = torch.arange(values.shape[1])[None] >= torch.tensor(counts)[:, None]
    module = modules[long.instrument]
    with torch.no_grad():
        query = module["projector"](values[:, :1].cuda()).cpu()
    streamed = gradients(module, query, lambda q: observe(module["attention"], module["projector"],
        q, blocks(values, pad, config), counts=counts, config=config))
    independent = gradients(module, query, lambda q: dense(module, q, values.cuda(), pad.cuda(), config.heads))
    need(all(torch.allclose(x, y, atol=1e-5, rtol=1e-4)
             for x, y in tensor_pairs(streamed, independent)), "UNEQUAL_MASK_GRADIENTS")
    torch.save(dict(query=query, counts=counts, references=[long.reference, short.reference],
                    dense=independent, streamed=streamed), a.root/"unequal-rows.pt")
    phases["unequal_real_rows"] = time.perf_counter()-tick
    tick = time.perf_counter()
    controls = []
    for name, test in (
        ("changed-gradient", lambda: tensor_pairs(proof["streamed"], {**proof["streamed"],
            "query_gradient": proof["streamed"]["query_gradient"]+1})),
        ("changed-output", lambda: tensor_pairs(proof["streamed"], {**proof["streamed"],
            "output": proof["streamed"]["output"]+1})),
    ):
        need(not all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in test()),
             "CONTROL_ACCEPTED")
        controls.append(name)
    need(sha(a.root/"weights.pt") == report["weights_pin"]
         and sha(a.root/"memory-at-read.sqlite") == report["memory_pin"], "RETAINED_MUTATION")
    memory.close()
    phases["controls_and_close"] = time.perf_counter()-tick
    result = dict(author_checked=True, report_pin=a.report_pin, readings=len(metrics),
        exact_restart=True, maximum_absolute_error=max(m["maximum_absolute_error"] for m in metrics),
        metrics=metrics, controls=controls, unequal_counts=counts,
        unequal_proof_pin=sha(a.root/"unequal-rows.pt"), config=asdict(config),
        scientific_admission=False, learned_capability=False, training_ready=False,
        phases=phases, seconds=time.perf_counter()-started, code_pin=sha(Path(__file__)))
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    pin = save(a.output, result)
    print(json.dumps(dict(pin=pin, readings=len(metrics), exact_restart=True,
        maximum_error=result["maximum_absolute_error"], unequal_counts=counts,
        seconds=result["seconds"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--report-pin", required=True)
    p.add_argument("--output", required=True, type=Path)
    raise SystemExit(run_cli(lambda: main(p.parse_args())))
