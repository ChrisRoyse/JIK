"""Full recorded utterances through the real typed predictor; engineering only."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

STARTED = time.perf_counter()
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
import torch
from torch.nn import functional as F
from audit.timed_world_witness import bundle, fingerprint
from audit.timed_speech_witness import request
from claims.timed_speech_memory import SpeechMemory
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.episode_input import EpisodeWorldInputs
from model.identity_loop import IdentityLoopPredictor
from model.loop_input import LoopInputs
from model.native_identity_world import NativeIdentityWorld, SensoryIdentityCore
from model.native_world_checkpoint import code, load, save as snapshot
from model.stream_attention import observe
from model.timed_world_input import TimedWorldInputs
from model.window_maxsim import maxsim
from senses.contract import digest
from senses.perception_corpus import pinned, sha
from senses.timed_readings import Bank, Inputs, need
from senses.timed_speech import FLAGS, save


def main(a):
    a.output.mkdir(exist_ok=False, parents=True)
    source = Inputs(a.inputs, a.input_pin, a.source)
    manifest = pinned(a.bank/"bank.json", a.bank_pin)
    bank = Bank(a.bank, a.bank_pin, source, {n: digest(v) for n, v in manifest["instruments"].items()})
    retained = pinned(a.inputs/"rollback-retained.json", a.retained_pin)
    need(sha(a.inputs/"rollback.sqlite") == a.memory_pin, "MEMORY_PIN")
    shutil.copy2(a.inputs/"rollback.sqlite", a.output/"memory.sqlite")
    memory = SpeechMemory(a.output/"memory.sqlite", source.evidence, create=False, retained=retained)
    cfg = ModelConfig()
    selected = TimedWorldInputs(bank, memory, cfg)
    traversals = []
    episodes = EpisodeWorldInputs(selected, trace=traversals.append)
    identity = LoopInputs(a.identity_population, a.identity_native, a.identity_source)
    previous = pinned(a.previous/"report.json", a.previous_pin) if a.previous else None
    if previous:
        need(previous["binding"] == episodes.binding and previous["code"] == code(), "COLD_CONTRACT")
    save(a.output/"episode-records.json", episodes.records)
    census = {m["id"]: {n: dict(rows=x.count, readings=len(x.pieces), dims=x.dims)
        for n, x in lanes.items()} for m, lanes in zip(episodes.moments, episodes.lanes, strict=True)}
    save(a.output/"census.json", census)
    need(sum(x["rows"]*x["dims"]*4 for lanes in census.values() for x in lanes.values())
         == 2960381952, "COMPLETE_EPISODE_CENSUS")
    phases = dict(source_and_memory=time.perf_counter()-STARTED)
    controls, outputs = [], []

    def deny(name, fn, expected):
        try:
            fn()
        except Exception as exc:
            actual = getattr(exc, "code", str(exc))
            need(actual == expected, "CONTROL:"+name+":"+actual)
            controls.append(dict(name=name, refused=actual))
        else:
            raise ValueError("MNLM_EPISODE_CONTROL_ACCEPTED:"+name)

    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader"], text=True)
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    need(not resident.strip(), "GPU_BUSY")
    take_lease("28", "complete-recorded-episode-model")
    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20261020)
        core = IdentityLoopPredictor(cfg, identity.specs,
            identity.batch(identity.standard_units, cfg, "cpu"), source_id=identity.source_id)
        model = NativeIdentityWorld(SensoryIdentityCore(core, episodes.specs, episodes.temporal_catalogue),
            selected.batch([0, 2], "cpu"), world_source_id=episodes.binding).cuda().eval()
        need(model.standard_id == "b71ea5d1ca3ce454aec6c71a67723f32a9770e5a1f130566265a04e2a5906016",
             "STANDARD_CHANGED")
        states = (model.start("AMI:A"), model.start("AMI:B"))
        initial = snapshot(a.output/"initial", model, states, episodes.binding)
        if previous:
            states = load(a.previous/"initial", model, expected=previous["initial"], binding=episodes.binding)
        allowed = torch.ones(2, dtype=torch.bool, device="cuda")
        (a.output/"outputs").mkdir()
        phases["model_publication"] = time.perf_counter()-tick

        def invoke(index, phase, state, native=None):
            event = digest(dict(binding=episodes.binding, row=episodes.moments[index]["id"], phase=phase,
                                scope="engineering_replay_of_recorded_utterance"))
            return model(episodes.batch(index) if native is None else native, (state,), allowed=allowed,
                phase=phase, event_ids=[event], evidence_origin=["source_observation"])

        tick = time.perf_counter()
        for phase in range(2):
            for context in range(2):
                index = 2*context+phase
                with torch.no_grad():
                    output = invoke(index, phase, states[context])
                actual = bundle(output)
                name = f"{phase}-{context}.pt"
                if previous:
                    expected = torch.load(a.previous/"outputs"/name, weights_only=True)
                    need(set(actual) == set(expected) and all(torch.equal(v, expected[n])
                         for n, v in actual.items()), "COLD_FORWARD")
                torch.save(actual, a.output/"outputs"/name)
                outputs.append(dict(file=name, sha256=sha(a.output/"outputs"/name),
                    fingerprint=fingerprint(actual), index=index, phase=phase, context=context))
                states = tuple(output["states"][0] if i == context else s for i, s in enumerate(states))
                print(json.dumps(dict(episode=index, phase=phase, full_native_consumed=True)), flush=True)
            if phase == 0:
                pending_states = states
                pending = snapshot(a.output/"pending", model, states, episodes.binding)
                states = load(a.output/"pending", model, expected=pending, binding=episodes.binding)
        need(len(traversals) == 128, "STREAM_TRAVERSALS")
        save(a.output/"episode-traversals.json", traversals)
        episodes.trace = None
        phases["complete_episodes_and_readback"] = time.perf_counter()-tick
        tick = time.perf_counter()
        # Full shortest episode backward, including recomputed region recognition.
        output = invoke(3, 1, pending_states[1])
        loss = output["constellation_regions"].sum()+sum(v[..., 0].mean() for v in output["slots"].values())
        loss.backward()
        gradients = {}
        for prefix in [kind+s.name for kind in ("core.projectors.", "core.temporal_projectors.")
                       for s in episodes.specs]+["core.to_identity", "core.from_identity"]:
            norm = sum(float(p.grad.detach().norm()) for n, p in model.named_parameters()
                       if n.startswith(prefix+".") and p.grad is not None)
            need(norm > 0, "GRADIENT:"+prefix)
            gradients[prefix] = norm
        model.zero_grad(set_to_none=True)
        phases["short_complete_episode_backward"] = time.perf_counter()-tick
        tick = time.perf_counter()
        # A complete real 20-frame ConvNeXt episode fits the original dense cap.
        lane = episodes.batch(3)["image_convnext"]
        blocks = list(lane.blocks(cfg))
        values = torch.cat([b.values for b in blocks], 1).cuda()
        positions = torch.cat([b.positions for b in blocks], 1).cuda()
        need(values.numel() <= cfg.max_native_elements and values.shape[1] <= cfg.max_native_tokens,
             "DENSE_ORACLE_BUDGET")
        projector, positioner = model.core.projectors[lane.name], model.core.temporal_projectors[lane.name]
        attention = model.core.observe[lane.name]
        results = []
        for streamed in (False, True):
            model.zero_grad(set_to_none=True)
            query = model.core.identity.phase_embedding.weight[0].detach()[None, None].clone().requires_grad_()
            value = (observe(attention, projector, query, iter(blocks), counts=(lane.count,), config=cfg,
                             positioner=positioner) if streamed
                     else attention(query, projector(values)+positioner(positions)))
            value.square().sum().backward()
            results.append([value.detach().clone(), query.grad.detach().clone()] +
                [p.grad.detach().clone() for module in (projector, positioner, attention)
                 for p in module.parameters()])
        need(all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in zip(*results)), "DENSE_GRADIENT")
        comparison = dict(lane=lane.name, frames=len(lane.pieces), native_rows=lane.count,
                          max_abs=max(float((x-y).abs().max()) for x, y in zip(*results)))
        # Independently compare the new streamed region reduction and its gradients.
        spec = next(s for s in episodes.specs if s.name == lane.name)
        slot = episodes.specs.index(spec)
        query_layer = model.query[lane.name]
        book = getattr(model, f"world_values_{slot}")
        mask = getattr(model, f"world_padding_{slot}")
        results = []
        for streamed in (False, True):
            model.zero_grad(set_to_none=True)
            context = F.normalize(model.core.identity.phase_embedding.weight[0].detach(),
                                  dim=-1)[None, None].clone().requires_grad_()
            keys = F.normalize(projector(book), dim=-1)
            if streamed:
                value = model._stream_regions(spec, lane, context, keys, mask)
            else:
                query = F.normalize(query_layer(context+F.normalize(projector(values), dim=-1)), dim=-1)
                value = maxsim(query, keys, mask, cfg).mean(1)
            value.square().sum().backward()
            results.append([value.detach().clone(), context.grad.detach().clone()] +
                [p.grad.detach().clone() for module in (projector, query_layer)
                 for p in module.parameters()])
        need(all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in zip(*results)),
             "DENSE_REGION_GRADIENT")
        comparison["recognition_max_abs"] = max(float((x-y).abs().max()) for x, y in zip(*results))
        model.zero_grad(set_to_none=True)
        phases["dense_gradient_comparison"] = time.perf_counter()-tick
        tick = time.perf_counter()
        native = episodes.batch(3)
        first = "image_sam"
        deny("unadmitted_training", lambda: episodes.batch(3, purpose="training"), "MNLM_EPISODE_ADMISSION")
        removed = episodes.batch(3)
        del removed[first]
        deny("missing_lane", lambda: invoke(3, 1, pending_states[1], removed), "MNLM_EPISODE_ROSTER")
        for name, changed in (
            ("dropped_reading", replace(native[first], pieces=native[first].pieces[:-1])),
            ("reordered_readings", replace(native[first], pieces=native[first].pieces[::-1])),
            ("missing_time", replace(native[first], temporal=())),
            ("foreign_event", replace(native[first], rows=("0"*64,)))):
            altered = episodes.batch(3)
            altered[first] = changed
            deny(name, lambda: invoke(3, 1, pending_states[1], altered), "MNLM_EPISODE_BINDING")
        moment = episodes.moments[3]
        deny("future_transcript", lambda: memory.consume(moment["id"],
             **{**request(moment), "through": moment["content"]["at"]}), "MNLM_TIMED_SPEECH_FUTURE_ANNOTATION")
        deny("checkpoint_binding", lambda: load(a.output/"pending", model,
             expected=pending, binding="0"*64), "MNLM_NATIVE_WORLD_CONTRACT")
        before = memory.snapshot()
        audio = next(memory.memory.episodes[k] for k in moment["evidence"]
                     if memory.memory.episodes[k]["content"].get("medium") == "audio")
        memory.apply(dict(op="invalidate", episode=audio["id"], reason="complete episode withdrawal proof"),
                     expected_head=memory.retained()["head"])
        after = memory.snapshot()
        deny("withdrawn_existing_batch", lambda: invoke(3, 1, pending_states[1], native), "MNLM_TIMED_MODEL_STALE")
        deny("withdrawn_stream", lambda: next(native[first].blocks(cfg)), "MNLM_TIMED_MODEL_STALE")
        memory.close()
        memory = SpeechMemory(a.output/"memory.sqlite", source.evidence, create=False, retained=after["retained"])
        need(memory.snapshot() == after, "WITHDRAWAL_RESTART")
        deny("withdrawn_restart", lambda: memory.consume(moment["id"], **request(moment)),
             "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
        memory.close()
        shutil.copy2(a.inputs/"rollback.sqlite", a.output/"rollback.sqlite")
        memory = SpeechMemory(a.output/"rollback.sqlite", source.evidence, create=False, retained=retained)
        need(memory.snapshot() == before, "MEMORY_ROLLBACK")
        selected = TimedWorldInputs(bank, memory, cfg)
        episodes = EpisodeWorldInputs(selected)
        restored = load(a.output/"pending", model, expected=pending, binding=episodes.binding)
        with torch.no_grad():
            rolled = bundle(invoke(3, 1, restored[1]))
        expected = torch.load(a.output/"outputs"/"1-1.pt", weights_only=True)
        need(all(torch.equal(v, expected[n]) for n, v in rolled.items()), "MODEL_ROLLBACK")
        phases["controls_withdrawal_restart_rollback"] = time.perf_counter()-tick
        result = dict(acceptance="author-checked", code=code(), witness_code=sha(Path(__file__)),
            input_pin=a.input_pin, bank_pin=a.bank_pin, binding=episodes.binding, temporal_id=model.core.temporal_id,
            initial=initial, pending=pending, standard=model.standard_id, world=model.world_id,
            outputs=outputs, controls=controls, gradients=gradients, dense_comparison=comparison,
            records=sha(a.output/"episode-records.json"), traversals=sha(a.output/"episode-traversals.json"),
            census=sha(a.output/"census.json"), flags=FLAGS, cold_exact=bool(previous), rollback_exact=True,
            complete_episodes=4, parameters=sum(p.numel() for p in model.parameters()),
            learned_update=False, useful_capability_proven=False, serving=False,
            rehearsal="recorded utterances in engineering phases; no action consequence or future forecast",
            before=before["retained"], after=after["retained"], peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            phases=phases, seconds=time.perf_counter()-STARTED)
        result["phase_coverage"] = sum(phases.values())/result["seconds"]
        need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
        save(a.output/"report.json", result)
        print(json.dumps(dict(report=sha(a.output/"report.json"), seconds=result["seconds"],
                              controls=len(controls), complete_episodes=4)), flush=True)
    finally:
        memory.close()
        release_lease()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("inputs", "source", "bank", "output", "identity-population", "identity-native", "identity-source"):
        p.add_argument("--"+n, type=Path, required=True)
    for n in ("input-pin", "bank-pin", "memory-pin", "retained-pin"):
        p.add_argument("--"+n, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--previous-pin")
    main(p.parse_args())
