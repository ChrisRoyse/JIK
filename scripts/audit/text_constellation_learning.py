"""Run two real, reversible text-constellation diagnostic learning cycles on CUDA."""
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
import torch

from audit.text_learning_support import build, controls, decisions, tape, train_cycle, update
from claims.codata_source import check
from lenscommon import take_lease, release_lease
from model.constellation_learning import frozen_pin
from model.constellation_training_checkpoint import optimizer_for
from model.contracts import ModelConfig
from model.loop_training_checkpoint import fingerprint
from model.native_world_checkpoint import save as model_save
from model.text_sequence_checkpoint import save, load
from model.text_sequence_data import FLAGS, PLAN, PURPOSE, SequenceData
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def checkpoint_model(root):
    return json.loads((root/"manifest.json").read_bytes())["model"]


def next_update(root, pin, model, data, evidence, example):
    optimizer, _ = load(root, model, data, expected=pin, evidence=evidence, purpose=PURPOSE)
    loss = update(model, optimizer, data, example)
    return dict(loss=loss, fingerprint=fingerprint(model, optimizer))


def cold(a, model, data):
    check(file_sha(a.cold/"report.json") == a.cold_pin, "TEXT_COLD_PIN")
    expected = json.loads((a.cold/"report.json").read_bytes())
    check(expected["data"] == data.bind, "TEXT_COLD_CONTRACT")
    reports, examples = {}, None
    for name in ("baseline", "cycle-1", "cycle-2"):
        record = expected["tapes"][name]
        cp = a.cold/("initial-model" if name == "baseline" else name)/("model" if name != "baseline" else "")
        pin = expected["initial_model"] if name == "baseline" else checkpoint_model(a.cold/name)
        report, rows = tape(a.cold/(name+".db"), model, data, cp, pin, retained=record["retained"])
        check(report == record, "TEXT_COLD_NEURAL_REPLAY")
        reports[name] = report["retained"]
        if examples is None:
            examples = rows
    example = next(e for e in examples if data.by_index[e["left"]]["split"] == "train")
    replay = next_update(a.cold/"cycle-1", expected["checkpoints"]["cycle-1"], model, data,
                         expected["tapes"]["baseline"]["retained"], example)
    check(replay == expected["continuation"], "TEXT_COLD_OPTIMIZER_REPLAY")
    optimizer, _ = load(a.cold/"initial-training", model, data,
        expected=expected["checkpoints"]["initial-training"], evidence=expected["tapes"]["baseline"]["retained"],
        purpose=PURPOSE)
    check(fingerprint(model, optimizer) == expected["initial_fingerprint"], "TEXT_COLD_ROLLBACK")
    return dict(exact=True, tapes=reports, continuation=replay, rollback=True, flags=FLAGS)


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    phases = {}
    corpus = Corpus(a.corpus, a.corpus_pin)
    bank = Bank(a.bank, a.bank_pin, corpus)
    data = SequenceData(corpus, bank, ModelConfig())
    phases["source"] = time.perf_counter()-started
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    check(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "TEXT_GPU_BUSY_OR_MISSING")
    take_lease("28", "text-constellation-diagnostic-learning")
    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = build(data, a)
        phases["model"] = time.perf_counter()-tick
        if a.cold:
            tick = time.perf_counter()
            report = cold(a, model, data)
            phases["cold_replay"] = time.perf_counter()-tick
        else:
            tick = time.perf_counter()
            initial_model = model_save(a.output/"initial-model", model,
                tuple(model.start(g["context"]) for g in data.groups), data.bind)
            negative = controls(a.output, model, data, a.output/"initial-model", initial_model)
            phases["controls"] = time.perf_counter()-tick
            tick = time.perf_counter()
            baseline, examples = tape(a.output/"baseline.db", model, data, a.output/"initial-model", initial_model)
            check(len(examples) == 197 and baseline["events"] == 394, "TEXT_TAPE_COVERAGE")
            phases["baseline_commit_reveal"] = time.perf_counter()-tick
            optimizer = optimizer_for(model, PLAN)
            frozen, initial_fingerprint = frozen_pin(model), fingerprint(model, optimizer)
            tick = time.perf_counter()
            pins = {"initial-training": save(a.output/"initial-training", model, optimizer, data,
                step=0, parent=None, evidence=baseline["retained"], frozen=frozen)}
            phases["initial_checkpoint"] = time.perf_counter()-tick
            tapes, cycles, step, continuation = {"baseline": baseline}, {}, 0, None
            for name, domains in (("cycle-1", PLAN["cycle_1"]), ("cycle-2", PLAN["cycle_2"])):
                tick = time.perf_counter()
                cycles[name] = train_cycle(model, optimizer, data, examples, domains, frozen)
                step += cycles[name]["steps"]
                phases[name+"_update"] = time.perf_counter()-tick
                tick = time.perf_counter()
                parent = pins["initial-training" if name == "cycle-1" else "cycle-1"]
                pins[name] = save(a.output/name, model, optimizer, data, step=step, parent=parent,
                                  evidence=baseline["retained"], frozen=frozen)
                phases[name+"_checkpoint"] = time.perf_counter()-tick
                tick = time.perf_counter()
                tapes[name], later_examples = tape(a.output/(name+".db"), model, data,
                    a.output/name/"model", checkpoint_model(a.output/name))
                check([e["identity"] for e in later_examples] == [e["identity"] for e in examples], "TEXT_IDENTITY_REGRESSION")
                phases[name+"_commit_reveal"] = time.perf_counter()-tick
                if name == "cycle-1":
                    tick = time.perf_counter()
                    example = next(e for e in examples if data.by_index[e["left"]]["split"] == "train")
                    continuation = next_update(a.output/name, pins[name], model, data, baseline["retained"], example)
                    repeated = next_update(a.output/name, pins[name], model, data, baseline["retained"], example)
                    check(continuation == repeated, "TEXT_OPTIMIZER_RESTART")
                    optimizer, _ = load(a.output/name, model, data, expected=pins[name], evidence=baseline["retained"], purpose=PURPOSE)
                    phases["continuation_and_restore"] = time.perf_counter()-tick
            tick = time.perf_counter()
            check(step == 1520 and frozen_pin(model) == frozen, "TEXT_FINAL_CURSOR")
            optimizer, _ = load(a.output/"initial-training", model, data,
                expected=pins["initial-training"], evidence=baseline["retained"], purpose=PURPOSE)
            check(fingerprint(model, optimizer) == initial_fingerprint, "TEXT_ROLLBACK")
            phases["rollback"] = time.perf_counter()-tick
            report = dict(schema="text-constellation-learning-witness-v1", data=data.bind, initial_model=initial_model,
                checkpoints=pins, initial_fingerprint=initial_fingerprint, tapes=tapes, cycles=cycles,
                continuation=continuation, controls=negative, rollback=True, optimizer_steps=step,
                frozen_core_retained=True, identity_outputs_retained=True, flags=FLAGS,
                decisions={"cycle-1": decisions(baseline, tapes["cycle-1"]),
                           "cycle-2": decisions(tapes["cycle-1"], tapes["cycle-2"])},
                parameters=sum(p.numel() for p in model.parameters()))
        report.update(phases_seconds=phases, main_function_wall_seconds=time.perf_counter()-started,
                      process_startup_included=False, acceptance="author-checked only",
                      driver_sha256=file_sha(Path(__file__)), support_sha256=file_sha(Path(__file__).with_name("text_learning_support.py")))
        report["phase_coverage"] = sum(phases.values())/report["main_function_wall_seconds"]
        check(.90 <= report["phase_coverage"] <= 1.001, "TEXT_PHASE_COVERAGE")
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
    run(p.parse_args())
