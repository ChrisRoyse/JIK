"""Cold source/tokenizer/window/native replay for commissioned instruments.

No model forward on CPU. Source and tokenizer reconstruction do not call the
producer's tokenize/prepare path. GPU replay is an explicit separate mode.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.window_readback import check, sha, validate_trace, validate_vectors

PLAN_FILES = {"input_ids.npy", "offsets.npy", "traces.jsonl.gz", "probe_ids.npy", "probe.json", "instrument-report.json"}
MODEL_FILES = {"model.pt", "tokenizer.json", "report.json", "config.json"}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_files(root, record, names):
    check(set(record) == names, "MNLM_COLD_CW_FILES")
    for name in names:
        pin = record[name]
        check((root/name).stat().st_size == pin["bytes"] and digest(root/name) == pin["sha256"], "MNLM_COLD_CW_FILE_HASH")


def read_sources(root, plan):
    raw = (root/"report.json").read_bytes(); source = json.loads(raw)
    check(sha(raw) == plan["calibration_report_sha256"] and source["complete"] and not source["errors"]
          and source["calibration_only"], "MNLM_COLD_CW_SOURCE")
    check(set(plan["sources"]) == {f for f, v in source["families"].items() if v["ready"]}, "MNLM_COLD_CW_FAMILIES")
    records = []; documents = defaultdict(list)
    for family, pin in sorted(plan["sources"].items()):
        raw = (root/family/"summary.json").read_bytes(); summary = json.loads(raw)
        body = (root/family/"chunks/atoms.parquet").read_bytes()
        check(sha(raw) == pin["summary_sha256"] == source["families"][family]["summary_sha256"]
              and sha(body) == summary["chunks_sha256"] == pin["chunks_sha256"], "MNLM_COLD_CW_SOURCE_HASH")
        rows = pq.read_table(io.BytesIO(body)).to_pylist()
        check(len(rows) == summary["chunks"] == pin["chunks"], "MNLM_COLD_CW_SOURCE_COUNT")
        for row in rows:
            records.append(dict(row, family=family)); documents[(family, row["row"])].append(row["text"])
    return records, documents


def validate_native_ids(ids, width, vocab):
    check(ids.dtype == np.int64 and ids.ndim == 2 and ids.shape[1] == width and len(ids) > 0
          and np.all(ids >= 0) and np.all(ids < vocab) and np.all(np.any(ids > 0, axis=1)), "MNLM_COLD_CW_IDS")
    for row in ids:
        zeros = np.flatnonzero(row == 0)
        check(not len(zeros) or np.all(row[zeros[0]:] == 0), "MNLM_COLD_CW_PADDING")


def validate_prepared(trace, text, reference, native, width):
    check(trace["prefix"] == "" and trace["offset_unit"] == "unicode-codepoint", "MNLM_COLD_CW_PREFIX")
    validate_trace(trace, text, dict(input_ids=reference.ids, offset_mapping=reference.offsets), width)
    expected = [reference.ids[i:i+width]+[0]*max(0, width-len(reference.ids[i:i+width]))
                for i in range(0, len(reference.ids), width)]
    check(expected and np.array_equal(native, np.asarray(expected, np.int64)), "MNLM_COLD_CW_FULL_IDS")
    check(len(trace["windows"]) == len(expected), "MNLM_COLD_CW_WINDOW_COUNT")
    for w, row in zip(trace["windows"], expected):
        check(w["input_ids"] == row and w["attention_mask"] == [int(i != 0) for i in row]
              and w["special_tokens_mask"] == [0]*width, "MNLM_COLD_CW_FORWARD_TRACE")
    return len(reference.ids)


def read_plan(root, calibration, model_root, plan_pin):
    started = time.perf_counter(); raw = (root/"report.json").read_bytes(); plan = json.loads(raw)
    check(sha(raw) == plan_pin and plan["schema"] == "commissioned-window-plan-v1", "MNLM_COLD_CW_PLAN")
    check(plan["complete"] is True and all(plan[k] is False for k in ("measured", "scientific_admission", "served")), "MNLM_COLD_CW_CLAIM")
    check(plan["max_len"] == 128 and plan["special_tokens"] is False and plan["window_overlap"] == 0
          and plan["window_reduction"] == "none:ordered-window-vectors", "MNLM_COLD_CW_RULES")
    read_files(root, plan["files"], PLAN_FILES)
    check(set(plan["instrument_files"]) == MODEL_FILES, "MNLM_COLD_CW_INSTRUMENT")
    for name, pin in plan["instrument_files"].items():
        check(digest(model_root/name) == pin, "MNLM_COLD_CW_MODEL_HASH")
    raw = (model_root/"report.json").read_bytes(); report = json.loads(raw)
    check(sha(raw) == plan["instrument_report_sha256"] and report == json.loads((root/"instrument-report.json").read_bytes())
          and report["name"] == plan["name"] and report["schema"] == "commissioned-sense-v1" and report["device"] == "cuda"
          and report["weights_sha256"] == plan["instrument_files"]["model.pt"]
          and report["tokenizer_sha256"] == plan["instrument_files"]["tokenizer.json"], "MNLM_COLD_CW_INSTRUMENT")
    cfg = json.loads((model_root/"config.json").read_bytes())
    check(cfg == report["config"] and cfg["max_len"] == plan["max_len"], "MNLM_COLD_CW_CONFIG")
    tok = Tokenizer.from_file(str(model_root/"tokenizer.json")); tok.no_padding(); tok.no_truncation()
    check(tok.token_to_id("<pad>") == 0, "MNLM_COLD_CW_PAD")
    records, documents = read_sources(calibration, plan)
    ids = np.load(root/"input_ids.npy", allow_pickle=False); offsets = np.load(root/"offsets.npy", allow_pickle=False)
    validate_native_ids(ids, 128, cfg["vocab"])
    check(offsets.dtype == np.int64 and offsets.shape == (len(records)+1,) and offsets[0] == 0
          and offsets[-1] == len(ids) and np.all(np.diff(offsets) > 0), "MNLM_COLD_CW_OFFSETS")
    with gzip.open(root/"traces.jsonl.gz", "rt", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f]
    check(len(traces) == len(records) == plan["inputs"] and len(ids) == plan["windows"], "MNLM_COLD_CW_COUNTS")
    tokens = gaps = 0
    for index, (row, trace) in enumerate(zip(records, traces)):
        check(trace["input"] == index and all(trace[k] == row[k] for k in ("family", "row", "entity", "chunk"))
              and trace["vector_span"] == offsets[index:index+2].tolist(), "MNLM_COLD_CW_ROW_BINDING")
        ref = tok.encode(row["text"], add_special_tokens=False)
        tokens += validate_prepared(trace["trace"], row["text"], ref, ids[offsets[index]:offsets[index+1]], 128)
        gaps += sum(b-a for a, b in trace["trace"]["unmapped_source_spans"])
    check(tokens == plan["tokens"] == np.count_nonzero(ids)
          and {str(k): v for k, v in Counter(map(int, np.diff(offsets))).items()} == plan["window_histogram"], "MNLM_COLD_CW_TOTALS")
    probe = json.loads((root/"probe.json").read_bytes()); text = "".join(documents[(probe["family"], probe["row"])])
    check(sha(text.encode()) == probe["text_sha256"], "MNLM_COLD_CW_PROBE_SOURCE")
    probe_ids = np.load(root/"probe_ids.npy", allow_pickle=False); validate_native_ids(probe_ids, 128, cfg["vocab"])
    check(len(probe_ids) == plan["probe_windows"] > 1 and probe["vector_span"] == [0, len(probe_ids)], "MNLM_COLD_CW_PROBE")
    validate_prepared(probe["trace"], text, tok.encode(text, add_special_tokens=False), probe_ids, 128)
    return dict(acceptance="author-checked", plan_report_sha256=plan_pin, inputs=len(records), windows=len(ids), tokens=tokens,
                probe_windows=len(probe_ids), normalization_gap_characters=gaps, all_token_ids_offsets_equal=True,
                instrument_files=plan["instrument_files"], gpu_forward_performed=False, scientific_admission=False,
                wall_seconds=time.perf_counter()-started)


def read_measurement(root, plan_root, calibration, model_root):
    raw = (root/"report.json").read_bytes(); report = json.loads(raw)
    check(report["schema"] == "commissioned-window-readings-cuda-v1" and report["complete"] is True
          and report["scientific_admission"] is False and report["served"] is False, "MNLM_COLD_CW_MEASUREMENT_CLAIM")
    cold = read_plan(plan_root, calibration, model_root, report["plan_report_sha256"])
    contract = report["contract"]; cfg = json.loads((model_root/"config.json").read_bytes())
    check(sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()) == report["lens_id"]
          and contract["model_files"] == cold["instrument_files"] and contract["device"] == "cuda"
          and contract["window_reduction"] == "none:ordered-window-vectors" and contract["batch"] == 32
          and contract["max_len"] == 128 and contract["dtype"] == "float32" and contract["pad"] == 0
          and contract["shape"] == f"MultiWindows({cfg['width']})", "MNLM_COLD_CW_CONTRACT")
    read_files(root, report["files"], {"vectors.npy", "probe-vectors.npy", "plan-readback.json", "gpu-before.txt", "desktop-residents.json"})
    v = np.load(root/"vectors.npy", allow_pickle=False); offsets = np.load(plan_root/"offsets.npy", allow_pickle=False)
    validate_vectors(v, offsets, cold["inputs"], cold["windows"], cfg["width"])
    p = np.load(root/"probe-vectors.npy", allow_pickle=False)
    validate_vectors(p, np.asarray([0, len(p)], dtype=np.int64), 1, cold["probe_windows"], cfg["width"])
    check(all(report[k] == cold[k] for k in ("inputs", "windows", "tokens", "probe_windows"))
          and report["weights_tokenizer_report_unchanged"] is True and report["repeated_windows_equal"] > len(p), "MNLM_COLD_CW_NATIVE_COUNTS")
    return dict(acceptance="author-checked", report_sha256=sha(raw), lens_id=report["lens_id"], plan=cold,
                all_native_shapes_norms_hashes_equal=True, gpu_replay_performed=False, scientific_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "calibration", "instrument", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--plan-report-sha256", required=True)
    p.add_argument("--publication", type=Path)
    a = p.parse_args()
    r = (read_measurement(a.publication, a.plan, a.calibration, a.instrument) if a.publication else
         read_plan(a.plan, a.calibration, a.instrument, a.plan_report_sha256))
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(r, f, sort_keys=True)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
