"""Real timed readings through complete world prediction and intrinsic identity.

Five phases rehearse software on fixed recordings. They are not audience echoes
of model actions, future-prediction evidence, instrument admission, or training.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from dataclasses import replace
import hashlib
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
from audit.timed_speech_witness import request
from claims.timed_speech_memory import SpeechMemory
from lenscommon import take_lease, release_lease
from model.contracts import ModelConfig
from model.identity_loop import IdentityLoopPredictor
from model.loop_input import LoopInputs
from model.native_identity_world import NativeIdentityWorld, SensoryIdentityCore
from model.native_world_checkpoint import save as checkpoint_save, load as checkpoint_load, code
from model.timed_world_input import TimedWorldInputs
from model.event_temporal_input import features
from model.stream_attention import blocks, observe
from model.window_maxsim import maxsim, tile
from senses.contract import canonical, digest
from senses.perception_corpus import pinned, sha
from senses.timed_readings import Bank, Inputs, need
from senses.timed_speech import FLAGS, save


def bundle(out):
    result = {kind+":"+n: v.detach().cpu() for kind in
              ("slots", "regions", "identity_regions", "identity_slots") for n, v in out[kind].items()}
    result["common"] = out["constellation_regions"].detach().cpu()
    for i, state in enumerate(out["states"]):
        result[f"world:{i}"] = state.hidden.detach().cpu()
        result[f"identity:{i}"] = state.identity_hidden.detach().cpu()
    return result


def fingerprint(values):
    return digest({n: dict(shape=list(t.shape), dtype=str(t.dtype),
        sha256=hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()) for n, t in values.items()})


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    source = Inputs(a.inputs, a.input_pin, a.source)
    raw = pinned(a.bank/"bank.json", a.bank_pin)
    bank = Bank(a.bank, a.bank_pin, source, {n: digest(v) for n, v in raw["instruments"].items()})
    retained = pinned(a.inputs/"rollback-retained.json", a.retained_pin)
    need(sha(a.inputs/"rollback.sqlite") == a.memory_pin, "MEMORY_PIN")
    shutil.copy2(a.inputs/"rollback.sqlite", a.output/"memory.sqlite")
    memory = SpeechMemory(a.output/"memory.sqlite", source.evidence, create=False, retained=retained)
    identity = LoopInputs(a.identity_population, a.identity_native, a.identity_source)
    cfg = ModelConfig()
    inputs = TimedWorldInputs(bank, memory, cfg)
    need(len(inputs.moments) == 4 and len(inputs.specs) == 16, "PROOF_CORPUS")
    # Entire first source moment from each provider speaker, no per-slot truncation.
    refs = [0, 2]
    need(inputs.moments[0]["content"]["who"] != inputs.moments[2]["content"]["who"], "REGION_SPEAKERS")
    world_refs = inputs.batch(refs, "cpu")
    temporal_records = {m["id"]: {s.instrument: json.loads(raw)
        for s, raw in zip(streams, times, strict=True)}
        for m, streams, times in zip(inputs.moments, inputs.streams, inputs.temporal, strict=True)}
    save(a.output/"temporal-records.json", temporal_records)
    temporal_positions = {m["id"]: {s.instrument: features((raw,), (len(s.values),),
        len(s.values), "cpu") for s, raw in zip(streams, times, strict=True)}
        for m, streams, times in zip(inputs.moments, inputs.streams, inputs.temporal, strict=True)}
    torch.save(temporal_positions, a.output/"temporal-positions.pt")
    previous = pinned(a.previous/"report.json", a.previous_pin) if a.previous else None
    if previous:
        need(previous["binding"] == inputs.binding and previous["code"] == code(), "COLD_CONTRACT")
    phases = {"source_and_memory": time.perf_counter()-STARTED}
    controls, outputs, gradients = [], [], {}

    def deny(name, fn, expected):
        try:
            fn()
        except Exception as exc:
            actual = getattr(exc, "code", str(exc))
            need(actual == expected, "CONTROL:"+name+":"+actual)
            controls.append(dict(name=name, refused=actual))
        else:
            raise ValueError("MNLM_TIMED_WORLD_CONTROL_ACCEPTED:"+name)

    resident = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader"], text=True)
    (a.output/"gpu-before.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    need(not resident.strip(), "GPU_BUSY")
    take_lease("28", "native-constellation-identity-world")
    try:
        tick = time.perf_counter()
        torch.set_num_threads(32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20261020)
        core = IdentityLoopPredictor(cfg, identity.specs,
            identity.batch(identity.standard_units, cfg, "cpu"), source_id=identity.source_id)
        need(core.standard_id == "b71ea5d1ca3ce454aec6c71a67723f32a9770e5a1f130566265a04e2a5906016",
             "FIXED_STANDARD")
        model = NativeIdentityWorld(SensoryIdentityCore(core, inputs.specs, inputs.temporal_catalogue),
                                   world_refs, world_source_id=inputs.binding).cuda().eval()
        states = (model.start("AMI:A"), model.start("AMI:B"))
        initial = checkpoint_save(a.output/"initial", model, states, inputs.binding)
        if previous:
            states = checkpoint_load(a.previous/"initial", model,
                                     expected=previous["initial"], binding=inputs.binding)
        allowed = torch.ones(len(refs), dtype=torch.bool, device="cuda")
        (a.output/"outputs").mkdir()
        phases["model_and_publication"] = time.perf_counter()-tick
        tick = time.perf_counter()

        def invoke(context, phase, *, current=None, native=None, origin="source_observation"):
            inputs.fresh()
            row = context*2 + phase%2
            observed = inputs.batch([row], "cuda") if native is None else native
            event = digest(dict(binding=inputs.binding, context=context, phase=phase,
                                source=inputs.moments[row]["id"], kind="engineering_phase_rehearsal"))
            out = model(observed, (states[context] if current is None else current,),
                allowed=allowed, phase=phase, event_ids=[event], evidence_origin=[origin])
            inputs.fresh()
            return out

        pending = None
        for phase in range(5):
            for context in range(2):
                with torch.no_grad():
                    out = invoke(context, phase)
                actual = bundle(out)
                region = out["predicted_constellation"].regions[0]
                for lane, slot in out["predicted_constellation"].slots.items():
                    index = model.reference_rows[lane].index(region)
                    reference = model.references()[lane]
                    need(slot.values.shape[1] == int((~reference.padding[index]).sum()),
                         "PREDICTION_WINDOW_LOSS")
                filename = f"{phase}-{context}.pt"
                if previous:
                    expected = torch.load(a.previous/"outputs"/filename, weights_only=True)
                    need(set(actual) == set(expected) and all(torch.equal(v, expected[n])
                         for n, v in actual.items()), "COLD_FORWARD")
                torch.save(actual, a.output/"outputs"/filename)
                outputs.append(dict(phase=phase, context=context, fingerprint=fingerprint(actual),
                    file=filename, file_sha256=sha(a.output/"outputs"/filename),
                    region=out["predicted_constellation"].regions[0],
                    native_shapes={n: list(s.values.shape) for n, s in out["predicted_constellation"].slots.items()}))
                states = tuple(out["states"][0] if i == context else s for i, s in enumerate(states))
            if phase == 2:
                pending_states = states
                pending = checkpoint_save(a.output/"pending", model, states, inputs.binding)
                states = checkpoint_load(a.output/"pending", model, expected=pending, binding=inputs.binding)
        phases["complete_model_phases_and_readback"] = time.perf_counter()-tick
        tick = time.perf_counter()
        with torch.no_grad():
            original = invoke(0, 3, current=pending_states[0])
            changed = invoke(0, 3, current=replace(pending_states[0],
                             identity_hidden=pending_states[0].identity_hidden+.125))
        identity_effect = any(not torch.equal(original["slots"][n], changed["slots"][n])
                              for n in original["slots"])
        need(identity_effect, "IDENTITY_NOT_INTRINSIC")
        # Independent contexts have no shared mutable hidden state.
        with torch.no_grad():
            other = bundle(invoke(1, 3, current=pending_states[1]))
        expected_other = torch.load(a.output/"outputs"/"3-1.pt", weights_only=True)
        need(all(torch.equal(v, expected_other[n]) for n, v in other.items()), "CONTEXT_LEAK")
        model.zero_grad(set_to_none=True)
        out = invoke(0, 3, current=pending_states[0])
        loss = out["constellation_regions"].sum()
        loss = loss + sum(x[..., 0].mean() for x in out["slots"].values())
        loss.backward()
        groups = ["core.to_identity", "core.from_identity"] + [
            "core.projectors."+s.name for s in inputs.specs] + [
            "core.identity.anchor."+s.name for s in identity.specs] + [
            "core.temporal_projectors."+s.name for s in inputs.specs]
        for prefix in groups:
            value = sum(float(p.grad.detach().norm()) for n, p in model.named_parameters()
                        if n.startswith(prefix+".") and p.grad is not None)
            need(value > 0, "GRADIENT:"+prefix)
            gradients[prefix] = value
        model.zero_grad(set_to_none=True)
        phases["intrinsic_coupling_gradients"] = time.perf_counter()-tick
        tick = time.perf_counter()
        # Dense comparison fits the ORIGINAL cap. Lower the blocked path's cap;
        # no raised budget to make a dense oracle fit.
        spec = next(s for s in inputs.specs if s.name == "image_sam")
        with torch.no_grad():
            observed = inputs.batch([0], "cuda")[spec.name].values
            query = F.normalize(model.core.projectors[spec.name](observed[:, :512]), dim=-1)
            keys = F.normalize(model.core.projectors[spec.name](model.references()[spec.name].values), dim=-1)
        mask = model.references()[spec.name].padding
        comparisons = []
        for tied in (False, True):
            selected_keys = keys if not tied else keys[:, :1].expand(-1, 4096, -1).contiguous()
            results = []
            for blocked in (False, True):
                q, k = query.detach().requires_grad_(), selected_keys.detach().requires_grad_()
                value = (maxsim(q, k, mask, replace(cfg, max_attention_elements=1_048_576))
                         if blocked else tile(q, k, mask))
                value.sum().backward()
                results.append((value.detach(), q.grad.detach(), k.grad.detach()))
            need(all(torch.allclose(x, y, atol=1e-5, rtol=1e-4)
                     for x, y in zip(*results)), "MAXSIM_DENSE_GRADIENT")
            comparisons.append(dict(tied=tied, max_abs=max(float((x-y).abs().max())
                for x, y in zip(*results)), queries=512, reference_windows=4096))
        phases["maxsim_dense_gradient_comparison"] = time.perf_counter()-tick
        tick = time.perf_counter()
        # Numerical interventions are quarantined diagnostics, never source
        # observations, learned updates, or evidence of physical time accuracy.
        temporal_effect = {}
        observed = inputs.batch([0], "cuda")
        with torch.no_grad():
            for name in ("image_sam", "audio_wavlm", "text_e5-base-v2"):
                x = observed[name]
                record = json.loads(x.temporal[0])
                # Public consumption must reject a 1ms window shift. Private
                # arithmetic establishes that time affects attention.
                from fractions import Fraction
                for window in record["windows"]:
                    for span in [window["interval"], *window["components"]]:
                        for j, value in enumerate(span):
                            shifted = Fraction(*value)+Fraction(1, 1000)
                            span[j] = [shifted.numerator, shifted.denominator]
                altered = replace(x, temporal=(canonical(record),))
                deny("shifted_reading_"+name, lambda: invoke(0, 0,
                    current=model.start("AMI:A"), native={**observed, name: altered}),
                    "MNLM_TEMPORAL_BINDING")
                lane_index = next(i for i, s in enumerate(inputs.specs) if s.name == name)
                query = (model.core.initial[lane_index]
                         +model.core.identity.phase_embedding.weight[0])[None, None]
                values = []
                for positions in (x.positions(), altered.positions()):
                    values.append(observe(model.core.observe[name], model.core.projectors[name], query,
                        blocks(x.values, x.padding, cfg, positions=positions),
                        counts=tuple((~x.padding).sum(1).tolist()), config=cfg,
                        positioner=model.core.temporal_projectors[name]))
                delta = float((values[0]-values[1]).abs().max())
                need(delta > 0, "TEMPORAL_INERT:"+name)
                temporal_effect[name] = dict(shift_seconds=[1, 1000], max_abs=delta,
                    observed=values[0].cpu(), counterfactual=values[1].cpu())
            baseline = bundle(invoke(0, 0, current=model.start("AMI:A"), native=observed))
            kept = {n: p.detach().clone() for n, p in model.core.temporal_projectors.named_parameters()}
            try:
                for p in model.core.temporal_projectors.parameters():
                    p.zero_()
                ablated = bundle(invoke(0, 0, current=model.start("AMI:A"), native=observed))
            finally:
                for n, p in model.core.temporal_projectors.named_parameters():
                    p.copy_(kept[n])
            need(not torch.equal(baseline["world:0"], ablated["world:0"])
                 and not torch.equal(baseline["common"], ablated["common"]), "TEMPORAL_MODEL_INERT")
            temporal_ablation = {n: dict(baseline=baseline[n], zero_time_projection=ablated[n])
                                 for n in ("world:0", "identity:0", "common")}
        torch.save(dict(reading_shift=temporal_effect, model_ablation=temporal_ablation),
                   a.output/"temporal-sensitivity.pt")
        phases["temporal_sensitivity_and_public_refusals"] = time.perf_counter()-tick
        tick = time.perf_counter()
        # Real mixed-length rows exercise padding; the SAM branch spans blocks.
        comparison_rows = inputs.batch([0, 1], "cuda")
        temporal_attention = []
        for name in ("image_sam", "audio_wavlm"):
            x = comparison_rows[name]
            positioner = model.core.temporal_projectors[name]
            projector, attention = model.core.projectors[name], model.core.observe[name]
            positions = x.positions()
            query = model.core.identity.phase_embedding.weight[0].detach()[None, None].expand(2, 1, -1)
            results = []
            for blocked in (False, True):
                model.zero_grad(set_to_none=True)
                q = query.clone().requires_grad_()
                if blocked:
                    value = observe(attention, projector, q,
                        blocks(x.values, x.padding, cfg, positions=positions),
                        counts=tuple((~x.padding).sum(1).tolist()), config=cfg, positioner=positioner)
                else:
                    value = attention(q, projector(x.values)+positioner(positions), padding=x.padding)
                value.square().sum().backward()
                results.append([value.detach().clone(), q.grad.detach().clone()] +
                    [p.grad.detach().clone() for module in (projector, attention, positioner)
                     for p in module.parameters()])
            need(all(torch.allclose(a, b, atol=1e-5, rtol=1e-4) for a, b in zip(*results)),
                 "TEMPORAL_ATTENTION_DENSE:"+name)
            temporal_attention.append(dict(lane=name, counts=(~x.padding).sum(1).tolist(),
                max_abs=max(float((a-b).abs().max()) for a, b in zip(*results))))
        model.zero_grad(set_to_none=True)
        phases["temporal_dense_gradient_comparison"] = time.perf_counter()-tick
        tick = time.perf_counter()
        deny("phase_order", lambda: invoke(0, 4, current=pending_states[0]), "MNLM_LOOP_PHASE_ORDER")
        deny("generated_echo", lambda: invoke(0, 3, current=pending_states[0], origin="generated_proposal"),
             "MNLM_LOOP_CAPTURED_ECHO")
        deny("foreign_standard", lambda: invoke(0, 3,
             current=replace(pending_states[0], standard_id="0"*64)), "MNLM_LOOP_STANDARD_ID")
        native = inputs.batch([1], "cuda")
        first = next(iter(native))
        deny("missing_temporal_support", lambda: invoke(0, 3, current=pending_states[0],
             native={**native, first: replace(native[first], temporal=())}), "MNLM_TEMPORAL_MISSING")
        for field, value in (("at", [0, 1]), ("available_after", [0, 1]), ("through", [0, 1])):
            record = json.loads(native[first].temporal[0])
            record[field] = value
            changed = replace(native[first], temporal=(canonical(record),))
            deny("tampered_"+field, lambda: invoke(0, 3, current=pending_states[0],
                 native={**native, first: changed}), "MNLM_TEMPORAL_BINDING")
        record = json.loads(native[first].temporal[0])
        record["source"]["clock_id"] = "0"*64
        deny("foreign_clock", lambda: invoke(0, 3, current=pending_states[0],
             native={**native, first: replace(native[first], temporal=(canonical(record),))}),
             "MNLM_TEMPORAL_BINDING")
        deny("missing_lane", lambda: invoke(0, 3, current=pending_states[0],
             native={n: x for n, x in native.items() if n != first}), "MNLM_CODATA_CONSTELLATION_ROSTER")
        deny("instrument_mismatch", lambda: invoke(0, 3, current=pending_states[0],
             native={**native, first: replace(native[first], identity="0"*64)}), "MNLM_MODEL_IDENTITY")
        deny("training_unadmitted", lambda: inputs.batch([0], "cuda", purpose="training"),
             "MNLM_TIMED_MODEL_ADMISSION")
        deny("checkpoint_pin", lambda: checkpoint_load(a.output/"pending", model,
             expected="0"*64, binding=inputs.binding), "MNLM_NATIVE_WORLD_PIN")
        deny("checkpoint_binding", lambda: checkpoint_load(a.output/"pending", model,
             expected=pending, binding="0"*64), "MNLM_NATIVE_WORLD_CONTRACT")
        deny("immutable_snapshot", lambda: checkpoint_save(a.output/"pending", model, states, inputs.binding),
             "MNLM_NATIVE_WORLD_EXISTS")
        shutil.copytree(a.output/"pending", a.output/"corrupt")
        with (a.output/"corrupt"/"state.pt").open("r+b") as damaged:
            damaged.seek(128)
            byte = damaged.read(1)
            damaged.seek(128)
            damaged.write(bytes([byte[0] ^ 1]))
        deny("checkpoint_payload_corruption", lambda: checkpoint_load(a.output/"corrupt", model,
             expected=pending, binding=inputs.binding), "MNLM_NATIVE_WORLD_BYTES")
        with torch.no_grad():
            frozen = model.core.identity.ref_values_0.clone()
            model.core.identity.ref_values_0[0, 0, 0] += 1
        deny("standard_mutation", model.check_reference, "MNLM_LOOP_STANDARD_DRIFT")
        with torch.no_grad():
            model.core.identity.ref_values_0.copy_(frozen)
        model.check_reference()
        phases["negative_controls"] = time.perf_counter()-tick
        tick = time.perf_counter()
        before = memory.snapshot()
        first_moment = inputs.moments[0]
        # Select by source type, not incidental evidence list order.
        audio = next(memory.memory.episodes[k] for k in first_moment["evidence"]
                     if memory.memory.episodes[k]["claim_kind"] == "measurement"
                     and memory.memory.episodes[k]["content"].get("medium") == "audio")
        memory.apply(dict(op="invalidate", episode=audio["id"], reason="complete-model withdrawal probe"),
                     expected_head=memory.retained()["head"])
        after = memory.snapshot()
        deny("stale_model_support", lambda: invoke(0, 0), "MNLM_TIMED_MODEL_STALE")
        deny("withdrawn_source", lambda: bank.streams(memory, first_moment["id"], **request(first_moment)),
             "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
        memory.close()
        memory = SpeechMemory(a.output/"memory.sqlite", source.evidence, create=False, retained=after["retained"])
        need(memory.snapshot() == after, "MEMORY_RESTART")
        deny("restart_withdrawn", lambda: bank.streams(memory, first_moment["id"], **request(first_moment)),
             "MNLM_TIMED_SPEECH_SUPPORT_WITHDRAWN_OR_UNCHECKED")
        memory.close()
        shutil.copy2(a.inputs/"rollback.sqlite", a.output/"rollback.sqlite")
        memory = SpeechMemory(a.output/"rollback.sqlite", source.evidence, create=False, retained=retained)
        need(memory.snapshot() == before, "MEMORY_ROLLBACK")
        inputs = TimedWorldInputs(bank, memory, cfg)
        states = checkpoint_load(a.output/"initial", model, expected=initial, binding=inputs.binding)
        with torch.no_grad():
            rolled = bundle(invoke(0, 0))
        expected = torch.load(a.output/"outputs"/"0-0.pt", weights_only=True)
        need(all(torch.equal(v, expected[n]) for n, v in rolled.items()), "MODEL_ROLLBACK")
        need(sha(a.identity_source/"profile.md") == source.report["identity_pin"], "IDENTITY_CHANGED")
        phases["withdraw_restart_and_complete_model_rollback"] = time.perf_counter()-tick
        result = dict(acceptance="author-checked", flags=FLAGS, binding=inputs.binding, code=code(),
            temporal_id=model.core.temporal_id,
            temporal_records=sha(a.output/"temporal-records.json"),
            temporal_positions=sha(a.output/"temporal-positions.pt"),
            temporal_sensitivity=sha(a.output/"temporal-sensitivity.pt"),
            temporal_context_connected=True, physical_time_accuracy_proven=False,
            temporal_attention_comparison=temporal_attention,
            witness_code=sha(Path(__file__)),
            input_pin=a.input_pin, bank_pin=a.bank_pin, standard=model.standard_id, world=model.world_id,
            initial=initial, pending=pending, outputs=outputs, controls=controls, gradients=gradients,
            maxsim_comparison=comparisons, identity_effect=identity_effect, context_isolation=True,
            cold_exact=bool(previous), rollback_exact=True, identity_pin=sha(a.identity_source/"profile.md"),
            complete_native_predictor_connected=True, learned_update=False, useful_capability_proven=False,
            rehearsal="fixed-source observations in software phases; no audience response or future forecast",
            parameters=sum(p.numel() for p in model.parameters()),
            reference_elements=sum(x.values.numel() for x in world_refs.values()),
            before=before["retained"], after=after["retained"],
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), phases=phases, seconds=time.perf_counter()-STARTED)
        result["phase_coverage"] = sum(phases.values())/result["seconds"]
        need(result["phase_coverage"] >= .9, "PHASE_COVERAGE")
        save(a.output/"report.json", result)
        print(json.dumps(dict(report=sha(a.output/"report.json"), seconds=result["seconds"],
            controls=len(controls), complete_native_predictor_connected=True)))
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
