"""Two reversible scratch-model sequence-learning cycles with native text slots."""
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

from audit.text_dynamics_checks import checkpoint_model, controls, frozen_control, next_update
from audit.text_learning_support import build, controls as source_controls, decisions, tape
from claims.codata_source import check
from lenscommon import take_lease, release_lease
from model.constellation_learning import trainable
from model.contracts import ModelConfig
from model.loop_training_checkpoint import fingerprint
from model.native_world_checkpoint import save as model_save
from model.text_context_regions import PrefixSession
from model.text_dynamics import DynamicsData, PLAN, frozen_pin, optimizer_for, train_cycle
from model.text_dynamics_checkpoint import save, load
from model.text_sequence_data import FLAGS, PURPOSE
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def replay(a, model, data):
    check(file_sha(a.cold/"report.json") == a.cold_pin, "DYNAMICS_COLD_PIN")
    expected = json.loads((a.cold/"report.json").read_bytes())
    check(expected["data"] == data.bind, "DYNAMICS_COLD_CONTRACT")
    reports, examples = {}, None
    for name in ("baseline", "cycle-1", "cycle-2"):
        cp = a.cold/("initial-model" if name == "baseline" else name)
        cp = cp if name == "baseline" else cp/"model"
        pin = expected["initial_model"] if name == "baseline" else checkpoint_model(a.cold/name)
        report, rows = tape(a.cold/(name+".db"), model, data, cp, pin,
                           retained=expected["tapes"][name]["retained"], session_class=PrefixSession)
        check(report == expected["tapes"][name], "DYNAMICS_COLD_NEURAL_REPLAY")
        reports[name] = report["retained"]
        if examples is None:
            examples = rows
    data.bind_examples(examples)
    evidence = expected["evidence"]
    continued = next_update(a.cold/"cycle-1", expected["checkpoints"]["cycle-1"], model, data, evidence, examples)
    check(continued == expected["continuation"], "DYNAMICS_COLD_OPTIMIZER_REPLAY")
    optimizer, _ = load(a.cold/"initial-training", model, data,
        expected=expected["checkpoints"]["initial-training"], evidence=evidence, purpose=PURPOSE)
    check(fingerprint(model, optimizer) == expected["initial_fingerprint"], "DYNAMICS_COLD_ROLLBACK")
    return dict(exact=True, tapes=reports, continuation=continued, rollback=True, flags=FLAGS)


def learn(a, model, data, phases):
    tick = time.perf_counter()
    initial_model = model_save(a.output/"initial-model", model,
        tuple(model.start(g["context"]) for g in data.groups), data.bind)
    representation = data.representation_controls(model) if hasattr(data, "representation_controls") else None
    negative = source_controls(a.output, model, data, a.output/"initial-model", initial_model)
    phases["source_controls"] = time.perf_counter()-tick
    tick = time.perf_counter()
    baseline, examples = tape(a.output/"baseline.db", model, data,
        a.output/"initial-model", initial_model, session_class=PrefixSession)
    check(len(examples) == 197 and baseline["events"] == 394, "DYNAMICS_TAPE_COVERAGE")
    data.bind_examples(examples)
    phases["baseline_commit_reveal"] = time.perf_counter()-tick
    optimizer = optimizer_for(model)
    frozen, initial_fingerprint = frozen_pin(model), fingerprint(model, optimizer)
    evidence = dict(retained=baseline["retained"], examples=baseline["examples"])
    tick = time.perf_counter()
    pins = {"initial-training": save(a.output/"initial-training", model, optimizer, data,
        step=0, parent=None, evidence=evidence, frozen=frozen)}
    controls_result = controls(a.output/"initial-training", pins["initial-training"],
        model, optimizer, data, evidence, examples)
    standard_control = frozen_control(model)
    phases["initial_checkpoint_and_controls"] = time.perf_counter()-tick
    tapes, cycles, changes, step = {"baseline": baseline}, {}, {}, 0
    for name, domains in (("cycle-1", PLAN["cycle_1"]), ("cycle-2", PLAN["cycle_2"])):
        tick = time.perf_counter()
        cycles[name] = train_cycle(model, optimizer, data, examples, domains, frozen)
        step += cycles[name]["steps"]
        phases[name+"_update"] = time.perf_counter()-tick
        tick = time.perf_counter()
        parent = pins["initial-training" if name == "cycle-1" else "cycle-1"]
        pins[name] = save(a.output/name, model, optimizer, data, step=step,
                          parent=parent, evidence=evidence, frozen=frozen)
        phases[name+"_checkpoint"] = time.perf_counter()-tick
        tick = time.perf_counter()
        tapes[name], later = tape(a.output/(name+".db"), model, data, a.output/name/"model",
                                  checkpoint_model(a.output/name), session_class=PrefixSession)
        changes[name] = sum(a["identity"] != b["identity"] for a, b in zip(examples, later))
        phases[name+"_commit_reveal"] = time.perf_counter()-tick
        if name == "cycle-1":
            tick = time.perf_counter()
            continuation = next_update(a.output/name, pins[name], model, data, evidence, examples)
            repeated = next_update(a.output/name, pins[name], model, data, evidence, examples)
            check(continuation == repeated, "DYNAMICS_OPTIMIZER_RESTART")
            optimizer, _ = load(a.output/name, model, data, expected=pins[name], evidence=evidence, purpose=PURPOSE)
            phases["continuation_and_restore"] = time.perf_counter()-tick
    tick = time.perf_counter()
    check(step == 40 and frozen_pin(model) == frozen, "DYNAMICS_FINAL_CURSOR")
    optimizer, _ = load(a.output/"initial-training", model, data,
        expected=pins["initial-training"], evidence=evidence, purpose=PURPOSE)
    check(fingerprint(model, optimizer) == initial_fingerprint, "DYNAMICS_ROLLBACK")
    phases["rollback"] = time.perf_counter()-tick
    return dict(schema="text-dynamics-learning-witness-v1", data=data.bind, initial_model=initial_model,
        checkpoints=pins, evidence=evidence, initial_fingerprint=initial_fingerprint,
        representation_controls=representation,
        tapes=tapes, cycles=cycles, continuation=continuation, source_controls=negative,
        learning_controls=controls_result, standard_control=standard_control,
        rollback=True, optimizer_steps=step, supervised_transitions=1520, flags=FLAGS,
        frozen_identity_module_and_references=True, coupled_prediction_fingerprints_changed=changes,
        identity_compliance_established=False,
        decisions={"cycle-1": decisions(baseline, tapes["cycle-1"]),
                   "cycle-2": decisions(tapes["cycle-1"], tapes["cycle-2"])},
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for _, p in trainable(model)))


def run(a, *, data_type=DynamicsData, model_factory=build):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    phases = {}
    corpus = Corpus(a.corpus, a.corpus_pin)
    data = data_type(corpus, Bank(a.bank, a.bank_pin, corpus), ModelConfig())
    phases["source"] = time.perf_counter()-started
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    check(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "DYNAMICS_GPU_BUSY")
    take_lease("28", "text-constellation-dynamics-learning")
    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = model_factory(data, a)
        phases["model"] = time.perf_counter()-tick
        if a.cold:
            tick = time.perf_counter()
            report = replay(a, model, data)
            phases["cold_replay"] = time.perf_counter()-tick
        else:
            report = learn(a, model, data, phases)
        report.update(phases_seconds=phases, main_function_wall_seconds=time.perf_counter()-started,
            process_startup_included=False, acceptance="author-checked only",
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            driver_sha256=file_sha(Path(__file__)), checks_sha256=file_sha(Path(__file__).with_name("text_dynamics_checks.py")))
        report["phase_coverage"] = sum(phases.values())/report["main_function_wall_seconds"]
        check(.90 <= report["phase_coverage"] <= 1.001, "DYNAMICS_PHASE_COVERAGE")
        publish(a.output/"report.json", report)
        print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"), seconds=report["main_function_wall_seconds"],
                              steps=report.get("optimizer_steps"), cold_exact=report.get("exact"))), flush=True)
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
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--causal-regions", action="store_true")
    mode.add_argument("--semantic-regions", action="store_true")
    mode.add_argument("--semantic-transformer", action="store_true")
    args = p.parse_args()
    if args.semantic_regions or args.semantic_transformer:
        from functools import partial
        from model.text_semantic_data import SemanticData, build_semantic
        run(args, data_type=partial(SemanticData, transformer=args.semantic_transformer), model_factory=build_semantic)
    elif args.causal_regions:
        from model.text_causal_regions import CausalRegionData
        run(args, data_type=CausalRegionData)
    else:
        run(args)
