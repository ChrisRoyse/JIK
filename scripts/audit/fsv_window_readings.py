"""Cold full-window FSV plus persisted corrupt-copy probes on real readings."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.window_readback import readback, validate_trace, validate_vectors, sha, snapshot_dir, canonical_assets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("publication", "calibration", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--overflow-reference", type=Path)
    p.add_argument("--overflow-calibration", type=Path)
    p.add_argument("--expected-lenses", help="Explicit frozen subset; default is the original six-instrument roster")
    a = p.parse_args()
    roster = {'e5-base-v2', 'all-mpnet-base-v2', 'pubmedbert-embeddings', 'finbert-mean', 'specter2-base', 'codebert-mean'}
    expected = a.expected_lenses.split(',') if a.expected_lenses else sorted(roster)
    if not expected or len(set(expected)) != len(expected) or not set(expected) <= roster:
        raise ValueError('MNLM_WINDOW_FSV_EXPECTED_ROSTER')
    if bool(a.overflow_reference) != bool(a.overflow_calibration):
        raise ValueError("MNLM_WINDOW_FSV_REFERENCE_PAIR")
    started = time.perf_counter(); phases = {}
    a.output.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, ok):
        checks.append({"name": name, "pass": bool(ok)})
    def refuses(name, code, fn):
        try:
            fn(); ok = False
        except ValueError as exc:
            ok = code in str(exc)
        check(name, ok)
    report_hash = sha((a.publication/"report.json").read_bytes())
    cold = readback(a.publication, a.calibration)
    if set(cold['lenses']) != set(expected):
        raise ValueError('MNLM_WINDOW_FSV_TARGET_ROSTER')
    overflow_root = a.overflow_reference or a.publication
    overflow_calibration = a.overflow_calibration or a.calibration
    overflow = readback(overflow_root, overflow_calibration) if a.overflow_reference else cold
    if a.overflow_reference:
        if ({n:v["lens_id"] for n,v in cold["lenses"].items()}
                != {n:v["lens_id"] for n,v in overflow["lenses"].items() if n in expected}):
            raise ValueError("MNLM_WINDOW_FSV_REFERENCE_INSTRUMENTS")
        check("B1 overflow reference has identical frozen instrument identities", True)
    phases["target_and_reference_cold_readback"] = time.perf_counter()-started
    tick = time.perf_counter()
    check("B1 all explicitly expected instruments and sources pinned", set(cold['lenses']) == set(expected))
    check("B2 all saved token IDs/offsets equal full untruncated input", all(r["all_token_ids_offsets_equal"] for r in cold["lenses"].values()))
    check("B2 real long documents exercise multiple windows", all(r["probe_windows"] > 1 for r in overflow["lenses"].values()))
    name = "e5-base-v2"
    if name not in overflow['lenses']:
        raise ValueError('MNLM_WINDOW_FSV_E5_OVERFLOW_REFERENCE_REQUIRED')
    root = overflow_root/name
    lens = json.loads((root/"report.json").read_bytes())
    assets = lens["contract"]["preprocessing_sha256"]
    native_names = {k.replace("/", "\\"): v for k,v in assets.items()}
    path = a.output/"asset-separators.json"; path.write_text(json.dumps(native_names), encoding="utf-8")
    check("B1 host asset separators preserve exact file identity", canonical_assets(json.loads(path.read_bytes())) == canonical_assets(assets))
    bad = dict(assets); bad[next(iter(bad))] = "0"*64
    path = a.output/"changed-asset-hash.json"; path.write_text(json.dumps(bad), encoding="utf-8")
    check("B1 changed asset contents still refuse", canonical_assets(json.loads(path.read_bytes())) != canonical_assets(assets))
    bad = dict(assets); bad["extra.json"] = "0"*64
    path = a.output/"extra-asset.json"; path.write_text(json.dumps(bad), encoding="utf-8")
    check("B1 extra asset still refuses", canonical_assets(json.loads(path.read_bytes())) != canonical_assets(assets))
    nested = next(k for k in assets if "/" in k or "\\" in k)
    bad = dict(assets); bad[nested.replace("\\", "/") if "\\" in nested else nested.replace("/", "\\")] = assets[nested]
    path = a.output/"ambiguous-asset.json"; path.write_text(json.dumps(bad), encoding="utf-8")
    refuses("B1 ambiguous normalized asset names refuse", "MNLM_WINDOW_READBACK_ASSET_PATH", lambda: canonical_assets(json.loads(path.read_bytes())))
    probe = json.loads((root/"probe.json").read_bytes())
    rows = pq.read_table(overflow_calibration/probe["family"]/"chunks/atoms.parquet").to_pylist()
    text = "".join(r["text"] for r in rows if r["row"] == probe["row"])
    tok = AutoTokenizer.from_pretrained(str(snapshot_dir(lens["contract"]["repo"])), trust_remote_code=False)
    reference = tok(lens["contract"]["prefix"]+text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    def trace_control(name, changed, code):
        path = a.output/(name+".json")
        path.write_text(json.dumps(changed), encoding="utf-8")
        refuses(name, code, lambda: validate_trace(json.loads(path.read_bytes()), text, reference, lens["contract"]["max_len"]))
    bad = copy.deepcopy(probe["trace"]); bad["windows"].pop()
    trace_control("B2 dropped tail window refuses", bad, "MNLM_WINDOW_READBACK_TOKENS")
    bad = copy.deepcopy(probe["trace"])
    w = bad["windows"][0]; i = next(i for i,(m,s) in enumerate(zip(w["attention_mask"], w["special_tokens_mask"])) if m and not s)
    w["input_ids"][i] += 1
    trace_control("B2 changed forward token refuses", bad, "MNLM_WINDOW_READBACK_TOKENS")
    bad = copy.deepcopy(probe["trace"]); bad["surface_sha256"] = "0"*64
    trace_control("B1 reading bound to another source refuses", bad, "MNLM_WINDOW_READBACK_TEXT")
    bad = copy.deepcopy(probe["trace"]); bad["windows"].reverse()
    trace_control("B2 reordered windows refuse", bad, "MNLM_WINDOW_READBACK_ALIGNMENT")
    bad = copy.deepcopy(probe["trace"]); bad["unmapped_source_spans"] = []
    trace_control("B2 hidden normalization gaps refuse", bad, "MNLM_WINDOW_READBACK_NORMALIZATION")
    with np.load(root/"probe.npz") as z:
        values, indices = z["vectors"].copy(), z["offsets"].copy()
    def vector_control(name, v, idx, code):
        path = a.output/(name+".npz")
        np.savez(path, vectors=v, offsets=idx)
        def cold_values():
            with np.load(path) as z:
                validate_vectors(z["vectors"], z["offsets"], 1, len(values), values.shape[1])
        refuses(name, code, cold_values)
    bad = values.copy(); bad[0,0] = np.nan
    vector_control("B3 nonfinite vector refuses", bad, indices, "MNLM_WINDOW_READBACK_VECTORS")
    bad = values.copy(); bad[0] = 0
    vector_control("B3 collapsed zero vector refuses", bad, indices, "MNLM_WINDOW_READBACK_VECTORS")
    vector_control("B3 missing window vector refuses", values[:-1], indices, "MNLM_WINDOW_READBACK_VECTORS")
    vector_control("B3 wrong vector dimension refuses", values[:,:-1], indices, "MNLM_WINDOW_READBACK_VECTORS")
    bad = indices.copy(); bad[-1] -= 1
    vector_control("B3 broken source-to-vector range refuses", values, bad, "MNLM_WINDOW_READBACK_INDICES")
    partial = a.output/"partial"; partial.mkdir()
    report = json.loads((a.publication/"report.json").read_bytes()); report["complete"] = False
    (partial/"report.json").write_text(json.dumps(report), encoding="utf-8")
    refuses("B3 partial publication refuses", "MNLM_WINDOW_READBACK_INCOMPLETE", lambda: readback(partial, a.calibration))
    cp = subprocess.run([sys.executable, "-I", str(Path(__file__).parents[1]/"text/window_readings.py"),
                         "--calibration", str(a.calibration), "--calibration-report-sha256", "0"*64,
                         "--lenses", name, "--output", str(a.publication)], capture_output=True, text=True, encoding="utf-8")
    (a.output/"overwrite.log").write_text(cp.stdout+cp.stderr, encoding="utf-8")
    check("B3 immutable output refuses before GPU acquisition", cp.returncode == 2 and "MNLM_WINDOW_EXISTS" in cp.stderr)
    check("B3 original publication report unchanged", sha((a.publication/"report.json").read_bytes()) == report_hash)
    if sha((overflow_root/"report.json").read_bytes()) != overflow["report_sha256"]:
        raise ValueError("MNLM_WINDOW_FSV_REFERENCE_DRIFT")
    phases["persisted_corruption_controls_and_immutability"] = time.perf_counter()-tick
    elapsed = time.perf_counter()-started
    result = {"checks": checks, "passed": sum(c["pass"] for c in checks), "total": len(checks),
              "cold_readback": cold, "overflow_reference_readback": overflow,
              "overflow_reference_separate": bool(a.overflow_reference), "phases_seconds": phases,
              "expected_lenses": expected,
              "wall_seconds": elapsed, "phase_coverage": sum(phases.values())/elapsed,
              "acceptance": "author-checked", "scientific_admission": False}
    (a.output/"fsv.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("passed", "total")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
