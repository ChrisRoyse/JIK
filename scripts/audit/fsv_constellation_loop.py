"""Full native constellations through the real prediction/echo/correction lifecycle."""
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import fsv_predictive_loop as protocol
from audit.constellation_readback import independent, decode, cosine, edit_distance
from claims.codata_source import check, sha
from claims.episodic_memory import digest, canonical
from model.constellation import Constellation, unpack, ordered_distance
from model.constellation_loop import ConstellationLoop
from model.constellation_world import ConstellationWorld
from model.loop_session import output_digest
from model.native import NativeInput
from senses.contract import SenseError


def boundary(session, root):
    import numpy as np
    import torch
    session._ensure_model()
    base = session.model
    original = digest(session.snapshot())
    core = base.core
    # Real multiwindow references already present in the retained checkpoint.
    # This is a representation test, not a new identity/world evidence source.
    with torch.random.fork_rng(devices=[]):
        model = ConstellationWorld(core, core.references(), world_source_id=core.source_id).to(session.device)
    for name in base.query:
        model.query[name].load_state_dict(base.query[name].state_dict())
        model.place[name].load_state_dict(base.place[name].state_dict())
    model.eval()
    lengths = {s.name: (~getattr(model, f"world_padding_{i}")).sum(1).tolist()
               for i, s in enumerate(model.specs)}
    selected = max(range(10), key=lambda j: sum(v[j] for v in lengths.values()))
    native = {}
    for i, spec in enumerate(model.specs):
        ref = core.references()[spec.name]
        native[spec.name] = NativeInput(spec.identity, (ref.rows[selected],), ref.values[selected:selected+1],
                                       ref.padding[selected:selected+1], ref.absent[selected:selected+1])
    allowed = torch.zeros(10, device=session.device, dtype=torch.bool)
    allowed[selected] = True
    args = dict(allowed=allowed, phase=0, event_ids=[digest(["real-constellation-boundary", selected])],
                evidence_origin=["source_observation"])
    state = (model.start("multiwindow-boundary"),)
    def forward():
        with torch.no_grad():
            return model(native, state, **args)
    output = forward()
    prediction = output["predicted_constellation"]
    packed = prediction.pack(model.specs, model.config)
    check(all(len(decode(packed[s.name])) == lengths[s.name][selected] for s in model.specs),
          "BOUNDARY_WINDOW_COUNT")
    check(len(set(prediction.regions)) == 1 and all(int(o[0]) == selected for o in output["world_owner"].values()),
          "BOUNDARY_REGION_JOIN")
    controls = []
    def refuse(name, code, fn):
        try:
            fn()
        except (ValueError, SenseError) as exc:
            check(getattr(exc, "code", str(exc)) == code, "BOUNDARY_WRONG_REFUSAL_"+name)
        else:
            raise AssertionError(name+" did not refuse")
        controls.append(dict(name=name, code=code))
    first, second = model.specs[:2]
    missing = dict(native)
    missing.pop(first.name)
    refuse("missing_input_instrument", "MNLM_CODATA_CONSTELLATION_ROSTER",
           lambda: Constellation(missing, "observed").validate(model.specs, model.config))
    wrong = dict(native)
    wrong[first.name] = replace(native[first.name], identity=second.identity)
    refuse("swapped_input_instrument", "MNLM_MODEL_IDENTITY",
           lambda: Constellation(wrong, "observed").validate(model.specs, model.config))
    mismatched = dict(native)
    mismatched[first.name] = replace(native[first.name], rows=("another-observation",))
    refuse("mixed_observation_constellation", "MNLM_CODATA_CONSTELLATION_ALIGNMENT",
           lambda: Constellation(mismatched, "observed").validate(model.specs, model.config))
    flat = dict(packed[first.name], shape=[first.dims*lengths[first.name][selected]])
    refuse("flattened_prediction", "MNLM_CODATA_CONSTELLATION_SHAPE",
           lambda: unpack(flat, first, max_windows=model.config.max_native_tokens))
    refuse("swapped_prediction_instrument", "MNLM_CODATA_CONSTELLATION_SLOT_IDENTITY",
           lambda: unpack(packed[first.name], second, max_windows=model.config.max_native_tokens))
    bad_bytes = dict(packed[first.name], sha256="0"*64)
    refuse("changed_prediction_bytes", "MNLM_CODATA_CONSTELLATION_BYTES",
           lambda: unpack(bad_bytes, first, max_windows=model.config.max_native_tokens))
    malformed = dict(packed[first.name], base64="!"*len(packed[first.name]["base64"]))
    refuse("malformed_prediction_payload", "MNLM_CODATA_CONSTELLATION_BYTES",
           lambda: unpack(malformed, first, max_windows=model.config.max_native_tokens))
    refuse("empty_region_mask", "MNLM_CODATA_CONSTELLATION_ALLOWED",
           lambda: model(native, state, **dict(args, allowed=torch.zeros_like(allowed))))
    # Disallowed native rows never enter projections. Poisoning them cannot alter output.
    saved = {}
    for i, spec in enumerate(model.specs):
        value = getattr(model, f"world_values_{i}")
        saved[i] = value.clone()
        value[~allowed] = float("nan")
    invariant = forward()
    check(output_digest(invariant) == output_digest(output)
          and invariant["predicted_constellation"].pack(model.specs, model.config) == packed,
          "BOUNDARY_FORBIDDEN_REGION_LEAK")
    refuse("reference_tamper_detected", "MNLM_WORLD_REFERENCE_DRIFT", model.check_reference)
    for i, values in saved.items():
        getattr(model, f"world_values_{i}").copy_(values)
    model.check_reference()
    controls.append(dict(name="forbidden_regions_do_not_enter_forecast", unchanged=True))
    # Independently recompute sequence distances over actual, distinct native windows.
    window_audit = {}
    for i, spec in enumerate(model.specs):
        actual = decode(packed[spec.name])
        ref = getattr(model, f"world_values_{i}")[selected, ~getattr(model, f"world_padding_{i}")[selected]]
        vectors = ref.cpu().tolist()
        matrix = [[cosine(x, y) for y in actual] for x in vectors]
        check(abs(edit_distance(matrix)-ordered_distance(np.array(matrix))) < 1e-12, "BOUNDARY_SCORE")
        check(all(cosine(x, y) >= np.sqrt(.99)-1e-6 for x, y in zip(vectors, actual)), "BOUNDARY_PLACEMENT")
        exact = [[cosine(x, y) for y in vectors] for x in vectors]
        dropped = [[cosine(x, y) for y in vectors[:1]] for x in vectors]
        reversed_windows = [[cosine(x, y) for y in reversed(vectors)] for x in vectors]
        check(len(vectors) > 1 and edit_distance(dropped) > edit_distance(exact)
              and edit_distance(reversed_windows) > edit_distance(exact), "BOUNDARY_ORDER_OR_DROP")
        window_audit[spec.name] = dict(instrument=spec.identity, windows=len(vectors), input=vectors,
            prediction=packed[spec.name], cosines=matrix, ordered_loss=edit_distance(matrix),
            exact_loss=edit_distance(exact), dropped_loss=edit_distance(dropped),
            reversed_loss=edit_distance(reversed_windows))
    # Trainability boundary: each typed branch gets finite gradients, frozen reference buffers do not.
    model.zero_grad(set_to_none=True)
    differentiable = model(native, state, **args)
    terms = []
    for i, spec in enumerate(model.specs):
        pred = differentiable["predicted_constellation"].slots[spec.name]
        target = getattr(model, f"world_values_{i}")[selected:selected+1, :pred.values.shape[1]]
        terms.append((1-(pred.values*target).sum(-1))[~pred.padding].mean())
    torch.stack(terms).mean().backward()
    check(all(model.place[s.name].weight.grad is not None
              and bool(torch.isfinite(model.place[s.name].weight.grad).all())
              and bool(model.place[s.name].weight.grad.abs().sum() > 0) for s in model.specs), "BOUNDARY_GRADIENT")
    check(all(not v.requires_grad and v.grad is None for n, v in model.named_buffers()), "BOUNDARY_REFERENCE_GRADIENT")
    model.zero_grad(set_to_none=True)
    core.zero_grad(set_to_none=True)
    base.check_reference()
    check(digest(session.snapshot()) == original, "BOUNDARY_LIVE_STATE_MUTATED")
    result = dict(schema="constellation-boundary-v1", region=prediction.regions[0], selected=selected,
        standard=base.standard_id, world=base.world_id, actual_multiwindow_slots=window_audit,
        controls=controls, differentiable_per_instrument=True, reference_gradients=False,
        live_snapshot_unchanged=True, representation_only=True)
    protocol.write(root/"constellation-boundary.json", result)
    return dict(file="constellation-boundary.json", sha256=sha(canonical(result)), controls=len(controls),
                full_native_windows={n: r["windows"] for n, r in window_audit.items()})


if __name__ == "__main__":
    sys.exit(protocol.run_cli(lambda: protocol.main(loop_type=ConstellationLoop, readback=independent, boundary=boundary)))
