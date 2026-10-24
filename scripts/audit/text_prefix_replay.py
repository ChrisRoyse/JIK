"""Change region availability on retained learned checkpoints; no repeated optimization."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import torch

from audit.text_learning_support import build, tape, decisions, update
from claims.codata_source import check
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model import native_world_checkpoint
from model.loop_training_checkpoint import fingerprint
from model.text_context_regions import ContextSequenceData, PrefixSession
from model.text_sequence_checkpoint import load as training_load
from model.text_sequence_data import FLAGS, PURPOSE
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def controls(data):
    left = next(i for i, g in data.by_index.items() if g["split"] == "development")
    prefix = data.by_index[left]["indices"][:1]
    results = []
    for name, bad in (("future", prefix+[data.successor[left]]),
                      ("other_context", prefix+[data.groups[0]["indices"][0]])):
        try:
            data.region_indices(left, supplied=bad)
        except ValueError as exc:
            check(str(exc) == "MNLM_CODATA_TEXT_CONTEXT_FUTURE_OR_OTHER", "TEXT_PREFIX_CONTROL_CODE")
            results.append(name)
        else:
            check(False, "TEXT_PREFIX_CONTROL_ACCEPTED")
    return results


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    check(file_sha(a.run/"report.json") == a.pin, "TEXT_PREFIX_PARENT_PIN")
    parent = json.loads((a.run/"report.json").read_bytes())
    corpus = Corpus(a.corpus, a.corpus_pin)
    data = ContextSequenceData(corpus, Bank(a.bank, a.bank_pin, corpus), ModelConfig())
    check(data.base_bind == parent["data"], "TEXT_PREFIX_PARENT_DATA")
    phases = dict(source=time.perf_counter()-started)
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    check(torch.cuda.is_available() and not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "TEXT_PREFIX_GPU_BUSY")
    take_lease("28", "text-observed-prefix-regions")
    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = build(data, a)
        expected = None
        if a.cold:
            check(file_sha(a.cold/"report.json") == a.cold_pin, "TEXT_PREFIX_COLD_PIN")
            expected = json.loads((a.cold/"report.json").read_bytes())
            check(expected["data"] == data.bind and expected["parent"] == a.pin, "TEXT_PREFIX_COLD_BINDING")
        phases["model"] = time.perf_counter()-tick
        records, snapshots, comparison, baseline_examples = {}, {}, {}, None
        for name in ("baseline", "cycle-1", "cycle-2"):
            tick = time.perf_counter()
            path = a.run/"initial-model" if name == "baseline" else a.run/name/"model"
            pin = parent["initial_model"] if name == "baseline" else json.loads((a.run/name/"manifest.json").read_bytes())["model"]
            native_world_checkpoint.load(path, model, expected=pin, binding=data.base_bind)
            if expected is None:
                snapshot = a.output/(name+"-model")
                snapshots[name] = native_world_checkpoint.save(snapshot, model,
                    tuple(model.start(g["context"]) for g in data.groups), data.bind)
                record, examples = tape(a.output/(name+".db"), model, data, snapshot, snapshots[name], session_class=PrefixSession)
            else:
                snapshot = a.cold/(name+"-model")
                snapshots[name] = expected["snapshots"][name]
                record, examples = tape(a.cold/(name+".db"), model, data, snapshot, snapshots[name],
                    retained=expected["tapes"][name]["retained"], session_class=PrefixSession)
                check(record == expected["tapes"][name], "TEXT_PREFIX_COLD_REPLAY")
            if baseline_examples is None:
                baseline_examples = examples
            check([e["identity"] for e in examples] == [e["identity"] for e in baseline_examples], "TEXT_PREFIX_IDENTITY_REGRESSION")
            records[name] = record
            comparison[name] = {domain: {lane: dict(fixed=old["prediction"],
                prefix=record["summary"]["development"][domain]["instruments"][lane]["prediction"],
                persistence=old["persistence"])
                for lane, old in values["instruments"].items()}
                for domain, values in parent["tapes"][name]["summary"]["development"].items()}
            phases[name+"_replay"] = time.perf_counter()-tick
        tick = time.perf_counter()
        # The shared training references already contain every training prefix.
        # Reusing their optimizer work is therefore checked, not merely assumed.
        first = next(e for e in baseline_examples if data.by_index[e["left"]]["split"] == "train")
        proxy = SimpleNamespace(inputs=data.inputs, bind=data.base_bind)
        optimizer, _ = training_load(a.run/"cycle-1", model, proxy,
            expected=parent["checkpoints"]["cycle-1"], evidence=parent["tapes"]["baseline"]["retained"], purpose=PURPOSE)
        loss = update(model, optimizer, data, first)
        continuation = dict(loss=loss, fingerprint=fingerprint(model, optimizer))
        check(continuation == parent["continuation"], "TEXT_PREFIX_GRADIENT_REUSE")
        optimizer, _ = training_load(a.run/"initial-training", model, proxy,
            expected=parent["checkpoints"]["initial-training"], evidence=parent["tapes"]["baseline"]["retained"], purpose=PURPOSE)
        check(fingerprint(model, optimizer) == parent["initial_fingerprint"], "TEXT_PREFIX_ROLLBACK")
        negatives = controls(data)
        phases["gradient_reuse_rollback_controls"] = time.perf_counter()-tick
        wall = time.perf_counter()-started
        check(.9 <= sum(phases.values())/wall <= 1.001, "TEXT_PREFIX_PHASE_COVERAGE")
        result = dict(schema="text-prefix-region-replay-v1", parent=a.pin, data=data.bind,
            snapshots=snapshots, tapes=records, comparisons=comparison, flags=FLAGS,
            additional_optimizer_training_steps=0, inherited_training_steps=parent["optimizer_steps"],
            diagnostic_next_step_exact=continuation, rollback=True, identity_outputs_retained=True,
            negative_controls=negatives, cold_exact=expected is not None,
            decisions={"cycle-1": decisions(records["baseline"], records["cycle-1"]),
                       "cycle-2": decisions(records["cycle-1"], records["cycle-2"])},
            phases_seconds=phases, main_function_wall_seconds=wall, phase_coverage=sum(phases.values())/wall,
            acceptance="author-checked only", driver_sha256=file_sha(Path(__file__)))
        publish(a.output/"report.json", result)
        print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"), seconds=wall,
                             cold_exact=result["cold_exact"], gradient_reuse_exact=True)))
    finally:
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("run", "corpus", "bank", "identity-population", "identity-native", "identity-source", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("pin", "corpus-pin", "bank-pin"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--cold", type=Path)
    p.add_argument("--cold-pin")
    run(p.parse_args())
