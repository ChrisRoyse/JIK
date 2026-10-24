"""Lossless window preparation and frozen commissioned CUDA readings (#12).

Preparation is CPU tokenization only, never a lens forward. A separately pinned
plan is consumed by one leased CUDA job. Every non-overlapping token window has
its own fp32 vector; no pooling across windows or instruments.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from tokenizers import Tokenizer
import tokenizers
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"lens"))
from lens.lenscommon import LensError, run_cli, take_lease
from senses.train_embedder import Encoder, PAD
from senses.trajectory_population import publish
from text.window_readings import load_chunks
import text_windows

PLAN = "commissioned-window-plan-v1"
RUNTIME = "commissioned-window-readings-cuda-v1"
BATCH = 32
FILES = ("model.pt", "tokenizer.json", "report.json", "config.json")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, suffix):
    if not ok:
        raise LensError("MNLM_CW_"+suffix, "commissioned full-window contract refused")


def pinned(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def instrument(root, report_pin):
    raw = (root/"report.json").read_bytes()
    require(sha(raw) == report_pin, "REPORT_PIN")
    report = json.loads(raw)
    hashes = {n: pinned(root/n) for n in FILES}
    require(hashes["model.pt"] == report["weights_sha256"], "WEIGHTS_PIN")
    require(hashes["tokenizer.json"] == report["tokenizer_sha256"], "TOKENIZER_PIN")
    require(report["schema"] == "commissioned-sense-v1" and report["device"] == "cuda", "TRAINING_RUNTIME")
    cfg = report["config"]
    require(json.loads((root/"config.json").read_bytes()) == cfg and cfg["max_len"] == 128
            and cfg["pool"] == "mean" and cfg["norm"] == "unit", "CONFIG")
    require(Path(report["name"]).name == report["name"] and report["name"] not in ("", ".", ".."), "NAME")
    tok = Tokenizer.from_file(str(root/"tokenizer.json")); tok.no_padding(); tok.no_truncation()
    require(tok.token_to_id("<pad>") == PAD and tok.get_vocab_size() <= cfg["vocab"], "VOCAB")
    return report, hashes, tok


def tokenize(tok, texts, width):
    encodings = tok.encode_batch(texts, add_special_tokens=False)
    ids, mapping, offsets, lengths = [], [], [], []
    for row, enc in enumerate(encodings):
        require(bool(enc.ids) and PAD not in enc.ids and not enc.overflowing, "EMPTY_OR_PAD_SOURCE")
        require(len(enc.ids) == len(enc.offsets), "TOKEN_ALIGNMENT")
        for start in range(0, len(enc.ids), width):
            part, spans = enc.ids[start:start+width], enc.offsets[start:start+width]
            lengths.append(len(part)); mapping.append(row)
            ids.append(part+[PAD]*(width-len(part)))
            offsets.append([list(s) for s in spans]+[[0, 0]]*(width-len(part)))
    tensor = np.asarray(ids, dtype=np.int64)
    mask = [[1]*n+[0]*(width-n) for n in lengths]
    traces = text_windows.build(texts, "", dict(input_ids=ids, attention_mask=mask,
                                offset_mapping=offsets, special_tokens_mask=[[0]*width for _ in ids]), mapping, "dense")
    counts = np.bincount(mapping, minlength=len(texts))
    return tensor, np.r_[0, np.cumsum(counts)].astype(np.int64), traces


def file_records(root, names):
    return {n: dict(sha256=pinned(root/n), bytes=(root/n).stat().st_size) for n in names}


def prepare(calibration, calibration_pin, model_root, model_pin, output):
    require(not output.exists(), "EXISTS"); started = time.perf_counter()
    rep, pins, tok = instrument(model_root, model_pin)
    records, sources = load_chunks(calibration, calibration_pin)
    loaded = time.perf_counter(); output.mkdir(parents=True)
    # Invariant source and tokenizer are resolved once per plan, not per row.
    inputs, indices, traces = tokenize(tok, [r["text"] for r in records], rep["config"]["max_len"])
    documents = defaultdict(list)
    for row in records:
        documents[(row["family"], row["row"])].append(row["text"])
    candidates = [(key, "".join(parts)) for key, parts in documents.items()
                  if 1800 <= sum(map(len, parts)) <= 3000]
    require(bool(candidates), "REAL_PROBE_MISSING")
    key, text = max(candidates, key=lambda item: len(item[1]))
    probe, probe_indices, probe_trace = tokenize(tok, [text], rep["config"]["max_len"])
    require(len(probe) > 1, "REAL_PROBE_NOT_OVERFLOW")
    tokenized = time.perf_counter()
    np.save(output/"input_ids.npy", inputs); np.save(output/"offsets.npy", indices)
    np.save(output/"probe_ids.npy", probe)
    with gzip.open(output/"traces.jsonl.gz", "wt", encoding="utf-8", newline="\n") as f:
        for i, (r, tr) in enumerate(zip(records, traces)):
            f.write(json.dumps(dict(input=i, **{k: r[k] for k in ("family", "row", "entity", "chunk")},
                                   vector_span=indices[i:i+2].tolist(), trace=tr), separators=(",", ":"))+"\n")
    publish(output/"probe.json", dict(family=key[0], row=key[1], text_sha256=sha(text.encode()),
                                      trace=probe_trace[0], vector_span=probe_indices.tolist()))
    publish(output/"instrument-report.json", rep)
    files = file_records(output, ("input_ids.npy", "offsets.npy", "traces.jsonl.gz", "probe_ids.npy",
                                  "probe.json", "instrument-report.json"))
    require({n: pinned(model_root/n) for n in FILES} == pins, "INSTRUMENT_DRIFT")
    elapsed = time.perf_counter()-started
    report = dict(schema=PLAN, calibration_report_sha256=calibration_pin, sources=sources,
                  instrument_report_sha256=model_pin, instrument_files=pins, name=rep["name"], files=files,
                  inputs=len(records), windows=len(inputs), tokens=int(np.count_nonzero(inputs)),
                  window_histogram=dict(Counter(map(int, np.diff(indices)))), probe_windows=len(probe),
                  preparation_code_sha256=pinned(Path(__file__)), tokenizers=tokenizers.__version__,
                  special_tokens=False, max_len=rep["config"]["max_len"], window_overlap=0,
                  window_reduction="none:ordered-window-vectors", complete=True, measured=False,
                  scientific_admission=False, served=False, timing=dict(input_seconds=loaded-started,
                  tokenize_seconds=tokenized-loaded, write_rehash_seconds=elapsed-(tokenized-started), wall_seconds=elapsed))
    publish(output/"report.json", report)
    return report


def load_plan(root, pin):
    raw = (root/"report.json").read_bytes(); require(sha(raw) == pin, "PLAN_PIN")
    rep = json.loads(raw)
    require(rep["schema"] == PLAN and rep["complete"] is True and rep["measured"] is False
            and rep["scientific_admission"] is False and rep["served"] is False, "PLAN_CLAIM")
    for name, entry in rep["files"].items():
        require(Path(name).name == name and pinned(root/name) == entry["sha256"]
                and (root/name).stat().st_size == entry["bytes"], "PLAN_FILE")
    return rep


def check_residents(raw, allowlist=None):
    observed = {(int(r[0].strip()), r[1].strip()) for r in list(csv.reader(io.StringIO(raw)))[1:] if r}
    allowed = set()
    if allowlist is not None:
        allowed = {(r["pid"], r["process_name"]) for r in allowlist["desktop_residents"]}
        require(allowlist["schema"] == "reviewed-windows-desktop-residents-v1" and sys.platform == "win32", "RESIDENT_SCOPE")
    require(observed <= allowed, "GPU_RESIDENT")


@torch.inference_mode()
def infer(model, ids):
    require(ids.dtype == np.int64 and ids.ndim == 2 and ids.shape[1] == 128
            and len(ids) <= BATCH and np.all(np.any(ids != PAD, axis=1)), "FORWARD_IDS")
    result = model(torch.from_numpy(np.array(ids, copy=True)).cuda()).float().cpu().numpy()
    require(np.isfinite(result).all() and np.allclose(np.linalg.norm(result, axis=1), 1, rtol=0, atol=3e-6), "VECTORS")
    return result


def measure(plan_root, plan_pin, calibration, model_root, output, desktop=None):
    require(not output.exists(), "EXISTS"); started = time.perf_counter()
    plan = load_plan(plan_root, plan_pin)
    rep, pins, _ = instrument(model_root, plan["instrument_report_sha256"])
    require(pins == plan["instrument_files"], "PLAN_INSTRUMENT")
    # Cold source/tokenization check before any CUDA allocation.
    from audit.commissioned_window_readback import read_plan
    cold = read_plan(plan_root, calibration, model_root, plan_pin)
    require(torch.cuda.is_available(), "NO_GPU")
    residents = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True)
    desktop_raw = desktop.read_bytes() if desktop else None
    check_residents(residents, json.loads(desktop_raw) if desktop_raw else None)
    take_lease("12", "commissioned-full-windows-"+rep["name"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cfg = rep["config"]
    model = Encoder(cfg["vocab"], cfg["width"], cfg["layers"], cfg["heads"], cfg["max_len"], 0.0)
    # Load the already-pinned bytes, never a path re-open after checking a hash.
    raw = (model_root/"model.pt").read_bytes(); require(sha(raw) == pins["model.pt"], "WEIGHTS_PIN")
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
    model.requires_grad_(False); model.eval(); model.cuda()
    require(not model.training and all(not p.requires_grad for p in model.parameters()), "UNFROZEN")
    contract = dict(runtime=RUNTIME, name=rep["name"], model_files=pins, instrument_report_sha256=plan["instrument_report_sha256"],
                    shape=f"MultiWindows({cfg['width']})", max_len=128, batch=BATCH, pad=PAD, dtype="float32",
                    execution_unit="fixed-window-batches-32-width-128-source-order-v1", window_overlap=0,
                    window_reduction="none:ordered-window-vectors", special_tokens=False, device="cuda",
                    torch=torch.__version__, cuda=torch.version.cuda, tokenizers=tokenizers.__version__,
                    device_name=torch.cuda.get_device_name(0), platform=sys.platform,
                    encoder_code_sha256=pinned(Path(sys.modules[Encoder.__module__].__file__)),
                    runtime_code_sha256=pinned(Path(__file__)))
    lens_id = sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    ids = np.load(plan_root/"input_ids.npy", mmap_mode="r", allow_pickle=False)
    probe_ids = np.load(plan_root/"probe_ids.npy", allow_pickle=False)
    loaded = time.perf_counter(); output.mkdir(parents=True)
    (output/"gpu-before.txt").write_text(residents, encoding="utf-8")
    publish(output/"desktop-residents.json", json.loads(desktop_raw) if desktop_raw else {"desktop_residents": []})
    publish(output/"plan-readback.json", cold)
    values = np.lib.format.open_memmap(output/"vectors.npy", mode="w+", dtype=np.float32, shape=(len(ids), cfg["width"]))
    repeated = 0; forward_seconds = 0.0
    for start in range(0, len(ids), BATCH):
        at = time.perf_counter(); x = infer(model, ids[start:start+BATCH])
        values[start:start+len(x)] = x
        if start == 0 or start+BATCH >= len(ids):
            require(np.array_equal(x, infer(model, ids[start:start+BATCH])), "NONDETERMINISTIC")
            repeated += len(x)
        forward_seconds += time.perf_counter()-at
    values.flush(); del values
    at = time.perf_counter(); probes = []
    for start in range(0, len(probe_ids), BATCH):
        part = probe_ids[start:start+BATCH]; x = infer(model, part)
        require(np.array_equal(x, infer(model, part)), "NONDETERMINISTIC")
        probes.append(x)
    np.save(output/"probe-vectors.npy", np.vstack(probes)); forward_seconds += time.perf_counter()-at
    require({n: pinned(model_root/n) for n in FILES} == pins, "INSTRUMENT_DRIFT")
    require(pinned(plan_root/"report.json") == plan_pin, "PLAN_DRIFT")
    files = file_records(output, ("vectors.npy", "probe-vectors.npy", "plan-readback.json", "gpu-before.txt", "desktop-residents.json"))
    elapsed = time.perf_counter()-started
    report = dict(schema=RUNTIME, plan_report_sha256=plan_pin, contract=contract, lens_id=lens_id, files=files,
                  inputs=plan["inputs"], windows=plan["windows"], tokens=plan["tokens"], probe_windows=plan["probe_windows"],
                  repeated_windows_equal=repeated+len(probe_ids), weights_tokenizer_report_unchanged=True,
                  complete=True, scientific_admission=False, served=False, timing=dict(load_seconds=loaded-started,
                  forward_seconds=forward_seconds, write_rehash_seconds=elapsed-(loaded-started)-forward_seconds, wall_seconds=elapsed))
    publish(output/"report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare"); run = sub.add_parser("measure")
    for parser in (prep, run):
        for arg in ("calibration", "instrument", "output"):
            parser.add_argument("--"+arg, type=Path, required=True)
    prep.add_argument("--calibration-report-sha256", required=True); prep.add_argument("--instrument-report-sha256", required=True)
    run.add_argument("--plan", type=Path, required=True); run.add_argument("--plan-report-sha256", required=True)
    run.add_argument("--reviewed-desktop-residents", type=Path, help="Windows only: exact PIDs/paths inspected before the job")
    a = p.parse_args()
    if a.mode == "prepare":
        r = prepare(a.calibration, a.calibration_report_sha256, a.instrument, a.instrument_report_sha256, a.output)
    else:
        r = measure(a.plan, a.plan_report_sha256, a.calibration, a.instrument, a.output, a.reviewed_desktop_residents)
    print(json.dumps({k: r[k] for k in ("schema", "inputs", "windows", "tokens", "timing")}))


if __name__ == "__main__":
    sys.exit(run_cli(main))
