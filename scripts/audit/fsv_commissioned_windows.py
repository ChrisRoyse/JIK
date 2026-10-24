"""Persisted source/model, coverage and publication controls for commissioned windows."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.commissioned_window_readback import (read_plan, read_measurement, validate_prepared,
                                              validate_native_ids, validate_vectors, digest)
from senses.commissioned_windows import instrument, prepare, check_residents


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "calibration", "instrument", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--publication", type=Path)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False); checks = []

    def put(name, value):
        with (a.output/(name+".json")).open("x", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, allow_nan=False)

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def refuses(name, code, fn):
        observed = None
        try:
            fn()
        except (ValueError, RuntimeError) as exc:
            observed = str(exc)
        check(name, observed is not None and code in observed, observed=observed, expected=code)

    plan = json.loads((a.plan/"report.json").read_bytes()); plan_pin = digest(a.plan/"report.json")
    paths = [a.plan/"report.json", a.calibration/"report.json"]
    paths += [a.plan/n for n in plan["files"]]+[a.instrument/n for n in plan["instrument_files"]]
    if a.publication:
        publication = json.loads((a.publication/"report.json").read_bytes())
        paths += [a.publication/"report.json"]+[a.publication/n for n in publication["files"]]
    before = {str(path): digest(path) for path in paths}; put("before", before)
    cold = read_plan(a.plan, a.calibration, a.instrument, plan_pin); put("cold-plan", cold)
    check("B1 source/model/tokenizer files pinned", cold["instrument_files"] == plan["instrument_files"])
    check("B2 every retained token and offset cold-reconstructs", cold["all_token_ids_offsets_equal"])
    check("B2 actual source document exercises overflow", cold["probe_windows"] > 1)
    probe = json.loads((a.plan/"probe.json").read_bytes())
    rows = pq.read_table(a.calibration/probe["family"]/"chunks/atoms.parquet").to_pylist()
    text = "".join(r["text"] for r in rows if r["row"] == probe["row"])
    tok = Tokenizer.from_file(str(a.instrument/"tokenizer.json")); tok.no_padding(); tok.no_truncation()
    ref = tok.encode(text, add_special_tokens=False); ids = np.load(a.plan/"probe_ids.npy", allow_pickle=False)

    def trace_control(name, changed, native, code):
        put(name, changed); np.save(a.output/(name+".npy"), native)
        refuses("B2 "+name, code, lambda: validate_prepared(json.loads((a.output/(name+".json")).read_bytes()),
                 text, ref, np.load(a.output/(name+".npy"), allow_pickle=False), 128))

    bad = copy.deepcopy(probe["trace"]); bad["windows"].pop()
    trace_control("dropped-tail", bad, ids[:-1], "MNLM_WINDOW_READBACK_TOKENS")
    bad = copy.deepcopy(probe["trace"]); changed = ids.copy(); changed[0, 0] += 1; bad["windows"][0]["input_ids"][0] += 1
    trace_control("coherently-changed-token", bad, changed, "MNLM_WINDOW_READBACK_TOKENS")
    bad = copy.deepcopy(probe["trace"]); bad["surface_sha256"] = "0"*64
    trace_control("substituted-source", bad, ids, "MNLM_WINDOW_READBACK_TEXT")
    bad = copy.deepcopy(probe["trace"]); bad["windows"].reverse()
    trace_control("reordered-windows", bad, ids[::-1], "MNLM_WINDOW_READBACK_ALIGNMENT")
    changed = ids.copy(); changed[0, 1] = 0; np.save(a.output/"internal-padding.npy", changed)
    refuses("B2 unexplained internal padding refuses", "MNLM_COLD_CW_PADDING", lambda: validate_native_ids(changed, 128, 32000))
    for field, value, code in (("complete", False, "MNLM_COLD_CW_CLAIM"),
                               ("measured", True, "MNLM_COLD_CW_CLAIM"),
                               ("scientific_admission", True, "MNLM_COLD_CW_CLAIM")):
        root = a.output/field; root.mkdir(); bad = {**plan, field: value}
        (root/"report.json").write_text(json.dumps(bad), encoding="utf-8")
        refuses("B3 false "+field+" refuses", code, lambda: read_plan(root, a.calibration, a.instrument, digest(root/"report.json")))
    stale = a.output/"stale-tokenizer"; stale.mkdir()
    for name in ("model.pt", "config.json", "report.json"):
        os.link(a.instrument/name, stale/name)  # Never write hard-linked immutable originals.
    (stale/"tokenizer.json").write_bytes((a.instrument/"tokenizer.json").read_bytes()+b"\n")
    refuses("B1 stale tokenizer refuses before CUDA load", "MNLM_CW_TOKENIZER_PIN",
            lambda: instrument(stale, plan["instrument_report_sha256"]))
    refuses("B3 overwrite refuses before tokenization/GPU", "MNLM_CW_EXISTS",
            lambda: prepare(a.calibration, "0"*64, a.instrument, "0"*64, a.plan))
    unexpected = "pid, process_name, used_gpu_memory [MiB]\n999999, python.exe, 1024 MiB\n"
    (a.output/"unexpected-resident.txt").write_text(unexpected, encoding="utf-8")
    refuses("B3 unexpected GPU process refuses", "MNLM_CW_GPU_RESIDENT", lambda: check_residents(unexpected))
    if a.publication:
        measured = read_measurement(a.publication, a.plan, a.calibration, a.instrument); put("cold-measurement", measured)
        check("B3 native vectors/norms/pins cold-read", measured["all_native_shapes_norms_hashes_equal"])
        vectors = np.load(a.publication/"probe-vectors.npy", allow_pickle=False)
        offsets = np.asarray([0, len(vectors)], dtype=np.int64)
        for name in ("nonfinite", "zero", "missing-window"):
            bad = vectors.copy()
            if name == "nonfinite": bad[0, 0] = np.nan
            elif name == "zero": bad[0] = 0
            else: bad = bad[:-1]
            path = a.output/(name+".npy"); np.save(path, bad)
            refuses("B3 "+name+" refuses", "MNLM_WINDOW_READBACK_VECTORS",
                    lambda: validate_vectors(np.load(path, allow_pickle=False), offsets, 1, len(vectors), vectors.shape[1]))
    after = {str(path): digest(path) for path in paths}; put("after", after)
    check("B1/B3 source and positive publication unchanged", before == after)
    result = dict(checks=checks, passed=sum(c["passed"] for c in checks), total=len(checks),
                  acceptance="author-checked", gpu_measurement_present=a.publication is not None,
                  scientific_admission=False, cold_plan=cold)
    put("fsv", result); print(json.dumps({k: result[k] for k in ("passed", "total", "gpu_measurement_present")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
