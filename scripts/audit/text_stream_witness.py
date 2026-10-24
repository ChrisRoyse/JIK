"""Real text constellations through the intrinsic model; observations are not Echoes."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import torch

from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.episode_input import EpisodeBatch
from model.identity_loop import IdentityLoopPredictor
from model.loop_input import LoopInputs
from model.native_identity_world import NativeIdentityWorld, SensoryIdentityCore
from model.native_world_checkpoint import save, load
from model.text_episode_input import TextInputs
from senses.contract import canonical, digest
from senses.trajectory_population import publish
from text.constellation_corpus import file_sha, need
from text.constellation_input import Corpus
from text.corpus_bank import Bank


def tensor_pin(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_pin(s):
    return dict(context=s.context, sequence=s.sequence, event=s.last_event,
                world=tensor_pin(s.hidden), identity=tensor_pin(s.identity_hidden))


def output_pin(o):
    return digest(dict(states=[state_pin(s) for s in o["states"]],
        region=o["predicted_constellation"].regions, phase=o["phase"],
        slots={n: tensor_pin(x) for n, x in o["slots"].items()}))


def select(corpus):
    chosen, groups, longest, offset = [], [], None, 0
    for i, metadata in enumerate(corpus.documents):
        rows = corpus.document(i)
        if metadata["domain"] == "general":
            groups.append(list(range(len(chosen), len(chosen)+len(rows))))
            chosen.extend((row, offset+j) for j, row in enumerate(rows))
        for j, row in enumerate(rows):
            if longest is None or len(row["text"]) > len(longest[0]["text"]):
                longest = row, offset+j
        offset += len(rows)
    need(len(groups) == 2 and all(len(g) > 3 for g in groups), "STREAM_PROOF_POPULATION")
    probe = next((i for i, (r, _) in enumerate(chosen) if r["atom_id"] == longest[0]["atom_id"]), len(chosen))
    if probe == len(chosen):
        chosen.append(longest)
    operations = []
    for step in range(max(map(len, groups))):
        for context, group in zip(("text-A", "text-B"), groups):
            if step < len(group):
                operations.append(dict(context=context, index=group[step]))
    return chosen, groups[0], probe, operations


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(a.corpus, a.corpus_pin)
    bank = Bank(a.bank, a.bank_pin, corpus)
    chosen, refs, probe, operations = select(corpus)
    cfg = ModelConfig()
    inputs = TextInputs(bank, chosen, cfg)
    identity = LoopInputs(a.identity_population, a.identity_native, a.identity_source)
    binding = dict(inputs=inputs.binding, standard_source=identity.source_id,
        operations=digest(operations), probe=chosen[probe][0]["atom_id"], refs=refs,
        mode="source_observation_without_action_phase_advance", scientific_admission=False)
    phase = dict(source_seconds=time.perf_counter()-started)
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    need(torch.cuda.is_available() and not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
         "--format=csv,noheader"], text=True).strip(), "STREAM_GPU_BUSY_OR_MISSING")
    take_lease("28", "text-intrinsic-consumer")
    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20261020)
        core = IdentityLoopPredictor(cfg, identity.specs,
            identity.batch(identity.standard_units, cfg, "cpu"), source_id=identity.source_id)
        model = NativeIdentityWorld(SensoryIdentityCore(core, inputs.specs, inputs.temporal_catalogue),
            inputs.references(refs, "cpu"), world_source_id=inputs.binding).cuda().eval()
        allowed = torch.ones(len(refs), dtype=torch.bool, device="cuda")
        states = {n: model.bind_state(model.core.start(n)) for n in ("text-A", "text-B")}
        parameters = {n: tensor_pin(p) for n, p in model.named_parameters()}
        phase["model_seconds"] = time.perf_counter()-tick

        def observe(context, index, **extra):
            row = chosen[index][0]
            x = inputs.batch(index, through=row["char_end"])
            event = digest(dict(context=context, row=row["atom_id"], input=inputs.binding))
            with torch.no_grad():
                return model(x, (states[context],), allowed=allowed, phase=states[context].sequence % 5,
                    event_ids=[event], evidence_origin=["source_observation"], advance_phase=False, **extra)

        tick = time.perf_counter()
        if a.cold:
            expected = json.loads((a.cold/"report.json").read_bytes())
            need(file_sha(a.cold/"report.json") == a.cold_pin and expected["binding"] == binding, "STREAM_COLD_PIN")
            recovered = load(a.cold/"initial", model, expected=expected["initial"], binding=binding)
            states = {s.context: s for s in recovered}
        else:
            initial = save(a.output/"initial", model, tuple(states.values()), binding)
        calls = []
        for operation in operations:
            context, index = operation["context"], operation["index"]
            other = "text-B" if context == "text-A" else "text-A"
            before, untouched = state_pin(states[context]), state_pin(states[other])
            output = observe(context, index)
            new = output["states"][0]
            need(new.sequence == states[context].sequence == 0 and output["phase"] == "Observation",
                 "STREAM_PHASE_ADVANCED")
            need(tensor_pin(new.hidden) != before["world"] and tensor_pin(new.identity_hidden) != before["identity"],
                 "STREAM_DEAD_COUPLING")
            states[context] = new
            need(state_pin(states[other]) == untouched, "STREAM_CONTEXT_LEAK")
            calls.append(dict(**operation, source=chosen[index][0]["atom_id"],
                through=chosen[index][0]["char_end"], before=before, after=state_pin(new), fingerprint=output_pin(output)))
        final_states = [state_pin(s) for s in states.values()]
        continuation = observe("text-A", probe)
        continuation_pin = output_pin(continuation)
        phase["observations_seconds"] = time.perf_counter()-tick
        tick = time.perf_counter()
        if a.cold:
            need(calls == json.loads((a.cold/"calls.json").read_bytes())
                 and final_states == expected["final_states"] and continuation_pin == expected["continuation"],
                 "STREAM_FRESH_PROCESS_REPLAY")
            recovered = load(a.cold/"final", model, expected=expected["checkpoint"], binding=binding)
            states = {s.context: s for s in recovered}
            need(output_pin(observe("text-A", probe)) == continuation_pin, "STREAM_CHECKPOINT_REPLAY")
            phase["checkpoint_seconds"] = time.perf_counter()-tick
            wall = time.perf_counter()-started
            publish(a.output/"report.json", dict(cold=a.cold_pin, exact=True, operations=len(calls),
                final_states=final_states, continuation=continuation_pin, phases_seconds=phase,
                wall_seconds=wall, phase_coverage=sum(phase.values())/wall, scientific_admission=False,
                served=False, acceptance="author-checked only"))
            print(json.dumps(dict(cold_exact=True, operations=len(calls), seconds=wall)))
            return
        checkpoint = save(a.output/"final", model, tuple(states.values()), binding)
        recovered = load(a.output/"final", model, expected=checkpoint, binding=binding)
        states = {s.context: s for s in recovered}
        need(output_pin(observe("text-A", probe)) == continuation_pin, "STREAM_CHECKPOINT_REPLAY")
        controls = []
        def deny(name, fn, code):
            try:
                fn()
            except Exception as exc:
                actual = getattr(exc, "code", str(exc))
                need(actual == code, "STREAM_CONTROL_CODE:"+name+":"+actual)
                controls.append(dict(name=name, code=actual))
            else:
                raise ValueError("MNLM_TEXT_CORPUS_CONTROL_ACCEPTED:"+name)
        row = chosen[probe][0]
        deny("future_source", lambda: inputs.batch(probe, through=row["char_end"]-1), "MNLM_TEXT_STREAM_FUTURE")
        deny("unadmitted_training", lambda: inputs.batch(probe, through=row["char_end"], purpose="training"),
             "MNLM_TEXT_STREAM_ADMISSION_REQUIRED")
        bad = EpisodeBatch(inputs, probe)
        bad.pop(inputs.specs[0].name)
        deny("missing_native_lane", lambda: bad.validate(inputs.specs, cfg), "MNLM_EPISODE_ROSTER")
        x = inputs.batch(probe, through=row["char_end"])
        args = dict(allowed=allowed, phase=0, event_ids=[digest(dict(control="phase", input=inputs.binding))])
        deny("invented_action_receipt", lambda: model(x, (states["text-A"],), **args,
             evidence_origin=["execution_receipt"], advance_phase=False), "MNLM_LOOP_OBSERVATION_MODE")
        with torch.no_grad():
            ordinary = model(x, (states["text-A"],), **args, evidence_origin=["source_observation"])
            explicit = model(x, (states["text-A"],), **args, evidence_origin=["source_observation"], advance_phase=True)
        need(output_pin(ordinary) == output_pin(explicit) and ordinary["states"][0].sequence == 1,
             "STREAM_DEFAULT_PHASE_REGRESSION")
        need([state_pin(s) for s in states.values()] == final_states, "STREAM_CONTROL_MUTATION")
        need({n: tensor_pin(p) for n, p in model.named_parameters()} == parameters, "STREAM_WEIGHT_DRIFT")
        phase["checkpoint_controls_seconds"] = time.perf_counter()-tick
        wall = time.perf_counter()-started
        publish(a.output/"calls.json", calls)
        report = dict(schema="text-intrinsic-observation-witness-v1", binding=binding, initial=initial,
            checkpoint=checkpoint, final_states=final_states, continuation=continuation_pin, controls=controls,
            source_atoms=len(chosen), source_observations=len(calls), native_instruments=len(inputs.specs),
            native_window_counts={s.name: sum(x[s.name].count for x in inputs.lanes) for s in inputs.specs},
            parameters=sum(p.numel() for p in model.parameters()), optimizer_steps=0, standard=model.standard_id,
            phase_advances=0, fabricated_echoes=0, scientific_admission=False, served=False,
            useful_learning_established=False, default_phase_transition_unchanged=True,
            code_sha256=file_sha(Path(__file__)), phases_seconds=phase, wall_seconds=wall,
            phase_coverage=sum(phase.values())/wall, acceptance="author-checked only")
        publish(a.output/"report.json", report)
        print(json.dumps(dict(report_sha256=file_sha(a.output/"report.json"), source_atoms=len(chosen),
            operations=len(calls), controls=len(controls), parameters=report["parameters"], seconds=wall)), flush=True)
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
