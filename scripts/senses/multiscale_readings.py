"""Ordered frozen readings at every source scale; per-instrument edge geometry."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.commissioned_windows import (instrument, tokenize, infer, Encoder, FILES, BATCH,
                                         check_residents, pinned, sha, file_records)
from lens.lenscommon import take_lease, run_cli
from senses.trajectory_population import publish
from audit.multiscale_reading_readback import population, plan, check
import numpy as np
import torch
import tokenizers


def prepare(pop_root, pop_pin, models, names, output):
    check(not output.exists(), "OUTPUT_EXISTS"); started = time.perf_counter()
    _, d = population(pop_root, pop_pin)
    check(len(names) == len(set(names)) and all(Path(n).name == n for n in names), "INSTRUMENT_NAMES")
    output.mkdir(parents=True); entries = []
    tokenize_seconds = 0.0; model_load_seconds = 0.0
    for name in names:
        at = time.perf_counter(); root = models/name
        rep, pins, tok = instrument(root, pinned(root/"report.json")); model_load_seconds += time.perf_counter()-at
        at = time.perf_counter(); ids, offsets, traces = tokenize(tok, [u["text"] for u in d["units"]], 128)
        tokenize_seconds += time.perf_counter()-at
        dest = output/name; dest.mkdir()
        np.save(dest/"input_ids.npy", ids); np.save(dest/"offsets.npy", offsets); publish(dest/"traces.json", traces)
        check({n: pinned(root/n) for n in FILES} == pins, "MODEL_DRIFT")
        entries.append(dict(name=name, model_files=pins, config=rep["config"], windows=len(ids),
                            tokens=int(np.count_nonzero(ids)),
                            files=file_records(dest, ("input_ids.npy", "offsets.npy", "traces.json"))))
    publish(output/"instruments.json", entries); check(pinned(pop_root/"report.json") == pop_pin, "POPULATION_DRIFT")
    files = file_records(output, ("instruments.json",)); elapsed = time.perf_counter()-started
    report = dict(schema="multiscale-native-plan-v1", population_sha256=pop_pin, files=files,
                  instruments=len(entries), complete=True, scientific_admission=False, served=False, meaning_asserted=False,
                  code_sha256=pinned(Path(__file__)), tokenization_code_sha256=pinned(Path(sys.modules[tokenize.__module__].__file__)),
                  timing=dict(tokenize_seconds=tokenize_seconds, model_load_seconds=model_load_seconds,
                              other_seconds=elapsed-tokenize_seconds-model_load_seconds, wall_seconds=elapsed))
    publish(output/"report.json", report)
    return dict(instruments=[dict(name=e["name"], windows=e["windows"], tokens=e["tokens"]) for e in entries], timing=report["timing"])


def node_indices(node, offsets):
    return np.asarray([j for i in node["unit_indices"] if i is not None
                       for j in range(int(offsets[i]), int(offsets[i+1]))], dtype=np.int64)


def geometry(nodes, edges, offsets, vectors, lens_id):
    by_id = {n["id"]:node_indices(n, offsets) for n in nodes}
    output = []; pairs = 0
    for e in edges:
        ix, iy = by_id[e["from"]], by_id[e["to"]]; pairs += len(ix)*len(iy)
        if len(ix) and len(iy):
            matrix = vectors[ix] @ vectors[iy].T
            directed = [float(matrix.max(1).mean()), float(matrix.max(0).mean())]
            score, status = min(directed), "measured_geometry_only"
        else:
            directed, score, status = None, None, "no_lexical_reading"
        output.append(dict(lens_id=lens_id, **{"from":e["from"], "to":e["to"]},
                           overlap_group=e["overlap_group"], visible_after=e["visible_after"],
                           directed_maxsim=directed, maxsim=score, status=status))
    return output, pairs


def visible(data, document, field, cutoff, group):
    source = next((s for s in data["sources"] if s["id"] == document), None)
    check(source is not None and data["groups"][document] == group, "CAUSAL_GROUP")
    f = next((f for f in source["fields"] if f["name"] == field), None)
    check(f is not None and type(cutoff) is int and 0 <= cutoff <= len(f["text"]), "CAUSAL_CUTOFF")
    nodes = [n for n in data["nodes"] if n["document_id"] == document and n["field_name"] == field]
    check(all(n["visible_after"] == max(b for _,b in n["source_spans"]) and n["overlap_group"] == group
              for n in nodes), "CAUSAL_SOURCE")
    ids = {n["id"] for n in nodes if n["visible_after"] <= cutoff}
    edges = [i for i,e in enumerate(data["edges"]) if e["from"] in ids and e["to"] in ids]
    return dict(node_ids=sorted(ids), edge_indices=edges)


def measure(pop_root, plan_root, plan_pin, models, output, desktop):
    check(not output.exists(), "OUTPUT_EXISTS"); started = time.perf_counter()
    pr, entries, d = plan(plan_root, plan_pin, pop_root, models); checked = time.perf_counter()
    check(torch.cuda.is_available(), "NO_GPU")
    residents = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    allowed = json.loads(desktop.read_bytes()) if desktop else None
    check_residents(residents, allowed); take_lease("27", "multiscale-native-readings")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True); (output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    publish(output/"desktop-residents.json", allowed)
    phases = dict(input_check_seconds=checked-started, model_load_seconds=0.0, forward_seconds=0.0,
                  association_seconds=0.0, model_rehash_seconds=0.0)
    lanes = {}; total_pairs = 0
    for e in entries:
        at = time.perf_counter(); root = models/e["name"]; cfg = e["config"]
        raw = (root/"model.pt").read_bytes(); check(sha(raw) == e["model_files"]["model.pt"], "MODEL_PIN")
        model = Encoder(cfg["vocab"], cfg["width"], cfg["layers"], cfg["heads"], 128, 0.0)
        model.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
        model.requires_grad_(False); model.eval(); model.cuda()
        check(not model.training and all(not p.requires_grad for p in model.parameters()), "UNFROZEN")
        contract = dict(name=e["name"], model_files=e["model_files"], device="cuda", dtype="float32",
                        runtime="commissioned-multiscale-native-v1", batch=BATCH, max_len=128, window_reduction="none",
                        torch=torch.__version__, cuda=torch.version.cuda, tokenizers=tokenizers.__version__, platform=sys.platform,
                        device_name=torch.cuda.get_device_name(0), runtime_code_sha256=pinned(Path(__file__)),
                        encoder_code_sha256=pinned(Path(sys.modules[Encoder.__module__].__file__)),
                        forward_code_sha256=pinned(Path(sys.modules[infer.__module__].__file__)))
        lens_id = sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
        ids = np.load(plan_root/e["name"]/"input_ids.npy", allow_pickle=False)
        offsets = np.load(plan_root/e["name"]/"offsets.npy", allow_pickle=False)
        phases["model_load_seconds"] += time.perf_counter()-at
        at = time.perf_counter(); values = []
        for start in range(0, len(ids), BATCH):
            part = ids[start:start+BATCH]; v = infer(model, part)
            check(np.array_equal(v, infer(model, part)), "NONDETERMINISTIC")
            values.append(v)
        vectors = np.vstack(values); phases["forward_seconds"] += time.perf_counter()-at
        at = time.perf_counter(); geo, pairs = geometry(d["nodes"], d["edges"], offsets, vectors, lens_id)
        phases["association_seconds"] += time.perf_counter()-at; total_pairs += pairs
        dest = output/e["name"]; dest.mkdir(); np.save(dest/"vectors.npy", vectors); publish(dest/"edges.json", geo)
        at = time.perf_counter(); check({n:pinned(root/n) for n in FILES} == e["model_files"], "MODEL_DRIFT")
        phases["model_rehash_seconds"] += time.perf_counter()-at
        lanes[e["name"]] = dict(lens_id=lens_id, contract=contract, window_pairs=pairs, frozen_unchanged=True,
                                repeated_windows_equal=len(vectors), files=file_records(dest, ("vectors.npy", "edges.json")))
        del model, raw; torch.cuda.empty_cache()
    check(pinned(plan_root/"report.json") == plan_pin and pinned(pop_root/"report.json") == pr["population_sha256"], "INPUT_DRIFT")
    publish(output/"instruments.json", lanes)
    files = file_records(output, ("instruments.json", "gpu-before.txt", "desktop-residents.json"))
    elapsed = time.perf_counter()-started; phases["publication_other_seconds"] = elapsed-sum(phases.values())
    report = dict(schema="multiscale-native-readings-v1", plan_sha256=plan_pin, population_sha256=pr["population_sha256"],
                  instruments=len(lanes), window_pairs=total_pairs, files=files, complete=True, scientific_admission=False,
                  served=False, meaning_asserted=False, timing=dict(**phases, wall_seconds=elapsed))
    publish(output/"report.json", report); return report


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare"); run = sub.add_parser("measure")
    for s in (prep, run):
        for n in ("population", "models", "output"):
            s.add_argument("--"+n, type=Path, required=True)
    prep.add_argument("--population-sha256", required=True); prep.add_argument("--names", nargs="+", required=True)
    run.add_argument("--plan", type=Path, required=True); run.add_argument("--plan-sha256", required=True)
    run.add_argument("--reviewed-desktop-residents", type=Path)
    a = p.parse_args()
    if a.mode == "prepare":
        result = prepare(a.population, a.population_sha256, a.models, a.names, a.output)
    else:
        result = measure(a.population, a.plan, a.plan_sha256, a.models, a.output, a.reviewed_desktop_residents)
    print(json.dumps(result))


if __name__ == "__main__":
    sys.exit(run_cli(main))
