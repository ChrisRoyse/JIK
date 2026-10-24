"""Native episodic memory -> bounded differentiable attention -> retained proof."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
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

from claims.native_evidence import NativeEvidence, pair, rational
from claims.native_memory import NativeMemory
from lenscommon import take_lease, run_cli
from model.attention import Attention, Linear
from model.contracts import ModelConfig
from model.native_stream_input import from_memory, descriptor
from model.stream_attention import observe, blocks, NativeBlock
from senses.contract import SenseError
from senses.perception_corpus import save, sha

ATOL, RTOL = 1e-5, 1e-4


def need(ok, code):
    if not ok:
        raise ValueError("MNLM_STREAM_FSV_"+code)


def branch(config, dims):
    return nn.ModuleDict(dict(projector=Linear(dims, config.width), attention=Attention(config)))


def execution(module, query, stream, config, mode):
    module.zero_grad(set_to_none=True)
    q = query.detach().clone().cuda().requires_grad_(True)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if mode == "dense":
        native = torch.from_numpy(stream.values.copy())[None].cuda()
        result = module["attention"](q, module["projector"](native))
    else:
        result = observe(module["attention"], module["projector"], q,
            stream.blocks(config), counts=stream.counts, config=config)
    loss = result.square().sum()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()-baseline
    saved = dict(output=result.detach().cpu(), query_gradient=q.grad.detach().cpu(),
                 gradients={n: p.grad.detach().cpu() for n, p in module.named_parameters()})
    need(all(torch.isfinite(v).all() for v in saved["gradients"].values()), "FINITE_GRADIENT")
    return saved, peak


def compare(left, right):
    pairs = [("output", left["output"], right["output"]),
             ("query_gradient", left["query_gradient"], right["query_gradient"])]
    need(left["gradients"].keys() == right["gradients"].keys(), "GRADIENT_ROSTER")
    pairs.extend((n, left["gradients"][n], right["gradients"][n]) for n in left["gradients"])
    return dict(maximum_absolute_error=max(float((a-b).abs().max()) for _, a, b in pairs),
        failed=[dict(name=n, maximum_absolute_error=float((a-b).abs().max()),
                     maximum_tolerance_ratio=float(((a-b).abs()/(ATOL+RTOL*b.abs())).max()))
                for n, a, b in pairs if not torch.allclose(a, b, atol=ATOL, rtol=RTOL)])


def main(a):
    started = PROCESS_STARTED
    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader"], text=True)
    need(not resident.strip(), "GPU_BUSY")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    take_lease("28", "native-stream-arithmetic")
    need(torch.cuda.is_available(), "NO_GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(32)
    torch.manual_seed(20261020)
    config = ModelConfig()
    checker = NativeEvidence(a.bank, a.report_pin, a.inputs, a.corpus)
    memory = NativeMemory(a.output/"memory.sqlite", checker, create=True)
    moments, requests, sources = [], [], []
    for clip in ("video_5930", "video_6074"):
        image, audio = (checker.measurement(clip, m) for m in ("image", "audio"))
        span = image["event_time"]["support"]
        moment = checker.moment(clip, pair(sum(map(rational, span))/2))
        for ep in (image, audio, moment):
            for op in ("intake", "check_source", "promote"):
                command = dict(op=op, episode=ep if op == "intake" else ep["id"])
                if op == "promote":
                    command["receipt"] = memory.memory.state[ep["id"]]["receipt"]["id"]
                memory.apply(command, expected_head=memory.retained()["head"])
        source = moment["source"]
        request = dict(clock_id=source["clock_id"], split_group=source["split_group"],
                       source_pin=source["source"]["media"], through=[2, 1])
        moments.append(moment)
        requests.append(request)
        sources.append(audio["id"])
    streams = tuple(s for m, r in zip(moments, requests) for s in from_memory(memory, m["id"], **r))
    need(len(streams) == 135 and sum(s.values.nbytes for s in streams) == 260935680, "CENSUS")
    retained = memory.retained()
    with sqlite3.connect(a.output/"memory-at-read.sqlite") as backup:
        memory.journal.db.backup(backup)
    modules = nn.ModuleDict()
    for stream in streams:
        if stream.instrument not in modules:
            modules[stream.instrument] = branch(config, stream.values.shape[1])
    torch.save(modules.state_dict(), a.output/"weights.pt")
    modules.cuda()
    phases = dict(open_and_prepare=time.perf_counter()-started)
    tick = time.perf_counter()
    rows, controls = [], []
    (a.output/"proofs").mkdir()
    for i, stream in enumerate(streams):
        module = modules[stream.instrument]
        # Initial query derived from this real reading; fixed bytes retained.
        with torch.no_grad():
            query = module["projector"](torch.from_numpy(stream.values[:1].copy())[None].cuda()).cpu()
        dense, dense_peak = execution(module, query, stream, config, "dense")
        streamed, stream_peak = execution(module, query, stream, config, "stream")
        file = f"proofs/{i:03d}.pt"
        torch.save(dict(query=query, dense=dense, streamed=streamed), a.output/file)
        comparison = compare(dense, streamed)
        coverage = []
        h = hashlib.sha256()
        for block in stream.blocks(config):
            h.update(block.values.numpy().tobytes())
            coverage.append([block.offset, block.offset+block.values.shape[1]])
        need(h.hexdigest() == json.loads(stream.metadata)["entry"]["sha256"], "COMPLETE_NATIVE_BYTES")
        rows.append(dict(**descriptor(stream), proof=file, proof_pin=sha(a.output/file),
            coverage=coverage, native_sha256=h.hexdigest(), **comparison,
            dense_extra_peak_bytes=dense_peak, streamed_extra_peak_bytes=stream_peak))
    phases["native_dense_stream_and_gradients"] = time.perf_counter()-tick
    tick = time.perf_counter()
    chosen = max(streams, key=lambda s: s.values.shape[0])
    module = modules[chosen.instrument]
    query = torch.zeros((1, 1, config.width), device="cuda")
    original = list(chosen.blocks(config))

    def refusal(name, call, code):
        try:
            call()
        except (ValueError, SenseError) as exc:
            need(str(exc).startswith(code), "CONTROL_REASON:"+name+":"+str(exc))
            controls.append(dict(name=name, code=str(exc)))
        else:
            need(False, "CONTROL_ACCEPTED:"+name)

    def invoke(items, counts=None, cfg=config):
        return observe(module["attention"], module["projector"], query, iter(items),
                       counts=chosen.counts if counts is None else counts, config=cfg)
    with torch.no_grad():
        refusal("missing-tail", lambda: invoke(original[:-1]), "MNLM_STREAM_COVERAGE")
        refusal("reordered-blocks", lambda: invoke(original[::-1]), "MNLM_STREAM_ORDER")
        refusal("repeated-block", lambda: invoke(original+[original[-1]]), "MNLM_STREAM_ORDER")
        refusal("false-count", lambda: invoke(original, (chosen.counts[0]+1,)), "MNLM_STREAM_COVERAGE")
        refusal("bool-count", lambda: invoke(original, (True,)), "MNLM_STREAM_COUNTS")
        refusal("fixed-budget", lambda: invoke(original, cfg=replace(config, max_native_elements=1)),
                "MNLM_STREAM_QUERY_BUDGET")
        refusal("fixed-attention-budget", lambda: invoke(original, cfg=replace(config, max_attention_elements=1)),
                "MNLM_STREAM_BLOCK_BUDGET")
        bad = original[0].values.clone(); bad[0, 0, 0] = float("nan")
        refusal("nonfinite-native", lambda: invoke([NativeBlock(0, bad, original[0].padding)]),
                "MNLM_STREAM_VALUES")
        bad = original[0].values[:, :, :-1]
        refusal("wrong-native-shape", lambda: invoke([NativeBlock(0, bad, original[0].padding)]),
                "MNLM_STREAM_NATIVE")
        refusal("false-training", lambda: list(chosen.blocks(config, purpose="training")),
                "MNLM_NATIVE_STREAM_ADMISSION")
    memory.apply(dict(op="invalidate", episode=sources[0], reason="withdraw source support for boundary probe"),
                 expected_head=memory.retained()["head"])
    refusal("stale-retained-head", lambda: list(chosen.blocks(config)), "MNLM_NATIVE_STREAM_STALE")
    refusal("withdrawn-moment", lambda: from_memory(memory, moments[0]["id"], **requests[0]),
            "MNLM_NATIVE_MEMORY_SUPPORT_WITHDRAWN_OR_UNCHECKED")
    memory.close()
    phases["negative_controls"] = time.perf_counter()-tick
    result = dict(schema="native-stream-witness-v1", author_checked=True, config=asdict(config),
        report_pin=a.report_pin, bank=str(a.bank), inputs=str(a.inputs), corpus=str(a.corpus),
        moments=moments, requests=requests, retained=retained,
        memory_pin=sha(a.output/"memory-at-read.sqlite"), weights_pin=sha(a.output/"weights.pt"),
        rows=rows, controls=controls, atol=ATOL, rtol=RTOL, phases=phases,
        seconds=time.perf_counter()-started, scientific_admission=False, training_ready=False,
        behavioural_compliance_established=False, learned_capability=False,
        code={n: sha(Path(__file__).resolve().parents[1]/n) for n in (
            "model/stream_attention.py", "model/native_stream_input.py",
            "model/identity_loop.py", "model/loop_checkpoint.py", "audit/native_stream_witness.py")})
    result["phase_coverage"] = sum(phases.values())/result["seconds"]
    need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
    result["equivalence_passed"] = all(not r["failed"] for r in rows)
    pin = save(a.output/"report.json", result)
    print(json.dumps(dict(pin=pin, readings=len(rows), controls=len(controls),
        maximum_error=max(r["maximum_absolute_error"] for r in rows),
        seconds=result["seconds"], equivalence_passed=result["equivalence_passed"])), flush=True)
    need(result["equivalence_passed"], "DENSE_EQUIVALENCE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bank", "inputs", "corpus", "output"):
        p.add_argument("--"+key, required=True, type=Path)
    p.add_argument("--report-pin", required=True)
    raise SystemExit(run_cli(lambda: main(p.parse_args())))
