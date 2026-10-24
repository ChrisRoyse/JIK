"""Read original native checkpoint tensors back from materialized CPU model state.

This inspects bindings only: no CPU model forward, admission or training.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1]/"lens")]
from senses.contract import SenseError
from senses.perception_corpus import save, sha


def need(ok, reason):
    if not ok:
        raise ValueError("MNLM_LAYOUT_READBACK_"+reason)


def target_key(instrument, original):
    """Independent source-name accounting, from the pinned upstream conversion rules."""
    prefixes = dict(siglip2="vision_model.", mae="vit.", sam="vision_encoder.",
                    convnext="convnext.", ast="audio_spectrogram_transformer.", clap="audio_model.")
    if instrument in prefixes:
        prefix = prefixes[instrument]
        if not original.startswith(prefix):
            return None
        key = original if instrument == "sam" else original[len(prefix):]
    else:
        key = original.replace("pos_conv_embed.conv.weight_g",
            "pos_conv_embed.conv.parametrizations.weight.original0").replace(
            "pos_conv_embed.conv.weight_v", "pos_conv_embed.conv.parametrizations.weight.original1")
    if instrument in ("mae", "ast"):
        key = key.replace("encoder.layer.", "layers.")
        for old, new in (("attention.query", "q_proj"), ("attention.key", "k_proj"),
                         ("attention.value", "v_proj"), ("attention.output.dense", "attention.o_proj"),
                         ("intermediate.dense", "mlp.fc1"), ("output.dense", "mlp.fc2")):
            key = key.replace(old, new)
    return key


def controls(model, source):
    import torch
    from native_assets import bind, state_sha
    original = state_sha(model)
    key = min(source, key=lambda k: source[k].numel())
    cases = []
    altered = dict(source); del altered[key]
    cases.append(("missing", altered, lambda k:k, "MNLM_NATIVE_WEIGHT_LAYOUT"))
    altered = dict(source); altered["unaccounted"] = source[key]
    cases.append(("extra", altered, lambda k:k, "MNLM_NATIVE_WEIGHT_LAYOUT"))
    altered = dict(source); altered[key] = source[key].reshape(-1)[:0]
    cases.append(("shape", altered, lambda k:k, "MNLM_NATIVE_WEIGHT_SHAPE"))
    altered = dict(source); altered[key] = source[key].double()
    cases.append(("dtype", altered, lambda k:k, "MNLM_NATIVE_WEIGHT_SHAPE"))
    altered = dict(source); altered["duplicate."+key] = source[key]
    cases.append(("collision", altered, lambda k:k.removeprefix("duplicate."),
                  "MNLM_NATIVE_WEIGHT_COLLISION"))
    results = []
    for label, values, select, expected in cases:
        try:
            bind(model, values, select, lambda k:False)
        except SenseError as exc:
            need(exc.code == expected, "CONTROL_ERROR:"+label)
        else:
            raise ValueError("MNLM_LAYOUT_READBACK_ACCEPTED:"+label)
        need(state_sha(model) == original, "MUTATED_BEFORE_REJECTION:"+label)
        results.append(dict(control=label, code=expected, state_unchanged=True))
    loader = model.load_state_dict
    def corrupt(*args, **kwargs):
        result = loader(*args, **kwargs)
        with torch.no_grad():
            model.state_dict()[key].reshape(-1).view(torch.uint8)[0] ^= 1
        return result
    model.load_state_dict = corrupt
    try:
        bind(model, source, lambda k:k, lambda k:False)
    except SenseError as exc:
        need(exc.code == "MNLM_NATIVE_WEIGHT_READBACK", "POST_LOAD_CONTROL")
        results.append(dict(control="changed-loaded-byte", code=exc.code))
    else:
        raise ValueError("MNLM_LAYOUT_READBACK_ACCEPTED:changed-loaded-byte")
    finally:
        model.load_state_dict = loader
        bind(model, source, lambda k:k, lambda k:False)
    need(state_sha(model) == original, "CONTROL_RESTORATION")
    return results


def main(a):
    started = time.perf_counter()
    import torch
    import transformers
    import native_vision
    import native_audio
    from native_assets import read_assets
    torch.set_num_threads(32)
    a.output.mkdir(parents=True, exist_ok=False)
    phases = dict(imports=time.perf_counter()-started)
    reports, faults = {}, []
    for medium, module in (("image",native_vision), ("audio",native_audio)):
        for name, spec in module.SPECS.items():
            start = time.perf_counter()
            row, config, processor, tensors = read_assets(a.assets, a.asset_plan, spec["repo"])
            if medium == "image":
                model, cfg, layout = module.construct(name, config, tensors)
            else:
                model, extractor, cfg, layout = module.construct(name, config, processor, tensors)
            state, mapping, selected, records = model.state_dict(), {}, {}, []
            for original, tensor in tensors.items():
                target = target_key(name, original)
                if target is None:
                    need(original in layout["excluded_tensors"], "UNACCOUNTED_SOURCE:"+original)
                    continue
                need(target in state and target not in selected, "TARGET:"+original)
                actual = state[target]
                expected_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
                actual_bytes = actual.detach().cpu().contiguous().numpy().tobytes()
                need(tensor.shape == actual.shape and tensor.dtype == actual.dtype
                     and expected_bytes == actual_bytes, "ORIGINAL_BYTES:"+original)
                mapping[original], selected[target] = target, tensor
                records.append(dict(original=original, target=target, shape=list(tensor.shape),
                    dtype=str(tensor.dtype), bytes=len(actual_bytes),
                    sha256=hashlib.sha256(actual_bytes).hexdigest()))
            need(set(selected) == set(state) and len(records) == layout["selected_tensors"],
                 "COMPLETE_MODEL")
            need(hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
                 == layout["mapping_sha256"], "MAPPING_HASH")
            proof = dict(repo=row["repo"], revision=row["revision"], layout=layout, tensors=records,
                model_source_pin=sha(Path(inspect.getfile(type(model)))), model_device="cpu",
                forward_executed=False)
            reports[name] = dict(report_pin=save(a.output/(name+".json"),proof),
                                 tensors=len(records), bytes=sum(r["bytes"] for r in records),
                                 schema=layout["schema"])
            phases[name] = time.perf_counter()-start
            if name == "siglip2":
                start = time.perf_counter()
                faults = controls(model, selected)
                phases["negative_controls"] = time.perf_counter()-start
            del state, model, selected, tensors, actual, tensor, actual_bytes, expected_bytes
            gc.collect()
            print(json.dumps(dict(instrument=name, **reports[name])),flush=True)
    need(not torch.cuda.is_initialized(), "UNEXPECTED_CUDA")
    report = dict(author_checked=True, models=reports, negative_controls=faults, phases=phases,
                  code_pin=sha(Path(__file__)), transformers=transformers.__version__,
                  seconds=time.perf_counter()-started, forward_executed=False,
                  scientific_admission=False)
    report["phase_coverage"] = sum(phases.values())/report["seconds"]
    need(report["phase_coverage"] >= 0.9, "PHASE_COVERAGE")
    pin = save(a.output/"report.json",report)
    print(json.dumps(dict(report_pin=pin, seconds=report["seconds"],
                         phase_coverage=report["phase_coverage"])),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets","asset-plan","output"):
        p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
