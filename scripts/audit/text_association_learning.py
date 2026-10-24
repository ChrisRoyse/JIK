"""Connected native association learning with source tapes and exact restart."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/"lens")]
import numpy as np
import torch

from audit.text_dynamics_checks import checkpoint_model
from audit.text_learning_support import controls as source_controls, tape
from claims.codata_source import check
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.loop_training_checkpoint import fingerprint
from model.native_world_checkpoint import save as save_model
from model.text_associations import AssociationData, RECIPE, update
from model.text_context_regions import PrefixSession
from model.text_dynamics import frozen_pin, optimizer_for
from model.text_dynamics_checkpoint import save, load
from model.text_semantic_data import build_semantic
from model.text_sequence_data import FLAGS, PURPOSE
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def continuation(root, pin, model, data, evidence, examples):
    optimizer, manifest = load(root, model, data, expected=pin, evidence=evidence, purpose=PURPOSE)
    value = update(model, optimizer, data, examples, ["general", "legal", "science"], manifest["frozen"])
    return dict(update=value, fingerprint=fingerprint(model, optimizer))


def train(a, model, data):
    np.save(a.output/"teacher.npy", data.teacher, allow_pickle=False)
    initial = save_model(a.output/"initial-model", model,
        tuple(model.start(g["context"]) for g in data.groups), data.bind)
    representation = data.representation_controls(model)
    negative = source_controls(a.output, model, data, a.output/"initial-model", initial)
    association_controls = data.controls(model)
    baseline, examples = tape(a.output/"baseline.db", model, data, a.output/"initial-model", initial,
        session_class=PrefixSession)
    data.bind_examples(examples)
    by_left = {e["left"]: e for e in examples}
    evidence = dict(retained=baseline["retained"], examples=baseline["examples"])
    optimizer = optimizer_for(model)
    frozen, initial_fingerprint = frozen_pin(model), fingerprint(model, optimizer)
    pins = {"initial-training": save(a.output/"initial-training", model, optimizer, data,
        step=0, parent=None, evidence=evidence, frozen=frozen)}
    metrics, tapes, updates = {"baseline": data.evaluate(model)}, {"baseline": baseline}, {}
    step = 0
    for name, domains in (("cycle-1", ["general", "legal"]), ("cycle-2", ["general", "legal", "science"])):
        updates[name] = []
        for epoch in range(RECIPE["epochs"]):
            result = update(model, optimizer, data, by_left, domains, frozen)
            updates[name].append(result)
            step += 1
            print(json.dumps(dict(stage=name, epoch=epoch, **result)), flush=True)
        pins[name] = save(a.output/name, model, optimizer, data, step=step,
            parent=pins["initial-training" if name == "cycle-1" else "cycle-1"], evidence=evidence, frozen=frozen)
        tapes[name], _ = tape(a.output/(name+".db"), model, data, a.output/name/"model",
            checkpoint_model(a.output/name), session_class=PrefixSession)
        metrics[name] = data.evaluate(model)
    continued = continuation(a.output/"cycle-1", pins["cycle-1"], model, data, evidence, by_left)
    repeated = continuation(a.output/"cycle-1", pins["cycle-1"], model, data, evidence, by_left)
    check(continued == repeated, "ASSOCIATION_OPTIMIZER_RESTART")
    optimizer, _ = load(a.output/"initial-training", model, data, expected=pins["initial-training"],
        evidence=evidence, purpose=PURPOSE)
    check(fingerprint(model, optimizer) == initial_fingerprint, "ASSOCIATION_ROLLBACK")
    return dict(schema="text-association-learning-witness-v1", data=data.bind, recipe=RECIPE,
        initial_model=initial, checkpoints=pins, evidence=evidence, initial_fingerprint=initial_fingerprint,
        association_metrics=metrics, association_controls=association_controls, source_controls=negative,
        representation_controls=representation, tapes=tapes, updates=updates, continuation=continued,
        optimizer_steps=step, training_node_exposures=sum(r["nodes"] for rows in updates.values() for r in rows),
        auxiliary_restart_probe_updates=2, auxiliary_restart_probe_node_exposures=208,
        rollback=True, flags=FLAGS, successful_acquisition_demonstrated=False,
        interpretation="learning_native_association_geometry; not grounded facts, admitted motifs, or D2 bits")


def cold(a, model, data):
    check(file_sha(a.cold/"report.json") == a.cold_pin, "ASSOCIATION_COLD_PIN")
    expected = json.loads((a.cold/"report.json").read_bytes())
    check(expected["data"] == data.bind and expected["recipe"] == RECIPE, "ASSOCIATION_COLD_BINDING")
    check(np.array_equal(np.load(a.cold/"teacher.npy", allow_pickle=False), data.teacher), "ASSOCIATION_COLD_TEACHER")
    metrics, tapes, examples = {}, {}, None
    for name in ("baseline", "cycle-1", "cycle-2"):
        cp = a.cold/"initial-model" if name == "baseline" else a.cold/name/"model"
        pin = expected["initial_model"] if name == "baseline" else checkpoint_model(a.cold/name)
        tapes[name], examples = tape(a.cold/(name+".db"), model, data, cp, pin,
            retained=expected["tapes"][name]["retained"], session_class=PrefixSession)
        check(tapes[name] == expected["tapes"][name], "ASSOCIATION_COLD_TAPE")
        metrics[name] = data.evaluate(model)
        check(metrics[name] == expected["association_metrics"][name], "ASSOCIATION_COLD_METRICS")
        if name == "baseline":
            data.bind_examples(examples)
            baseline_examples = {e["left"]: e for e in examples}
    continued = continuation(a.cold/"cycle-1", expected["checkpoints"]["cycle-1"], model, data,
        expected["evidence"], baseline_examples)
    check(continued == expected["continuation"], "ASSOCIATION_COLD_CONTINUATION")
    optimizer, _ = load(a.cold/"initial-training", model, data, expected=expected["checkpoints"]["initial-training"],
        evidence=expected["evidence"], purpose=PURPOSE)
    check(fingerprint(model, optimizer) == expected["initial_fingerprint"], "ASSOCIATION_COLD_ROLLBACK")
    return dict(schema="text-association-cold-readback-v1", source=a.cold_pin, exact=True,
        tapes=tapes, association_metrics=metrics, continuation=continued, rollback=True, flags=FLAGS)


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(a.corpus, a.corpus_pin)
    data = AssociationData(corpus, Bank(a.bank, a.bank_pin, corpus), ModelConfig())
    phases = dict(source_and_association_targets=time.perf_counter()-started)
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    check(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "ASSOCIATION_GPU_BUSY")
    take_lease("28", "text-native-association-learning")
    try:
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tick = time.perf_counter()
        model = build_semantic(data, a)
        phases["model"] = time.perf_counter()-tick
        tick = time.perf_counter()
        report = cold(a, model, data) if a.cold else train(a, model, data)
        phases["cold_replay" if a.cold else "training_and_physical_checks"] = time.perf_counter()-tick
        report.update(phases_seconds=phases, main_function_wall_seconds=time.perf_counter()-started,
            process_startup_included=False, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), acceptance="author-checked only")
        report["phase_coverage"] = sum(phases.values())/report["main_function_wall_seconds"]
        check(.90 <= report["phase_coverage"] <= 1.001, "ASSOCIATION_PHASE_COVERAGE")
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
    p.add_argument("--cold", type=Path)
    p.add_argument("--cold-pin")
    run(p.parse_args())
