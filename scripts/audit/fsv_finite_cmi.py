"""Bounded #12 commissioning, real retained negative + known-truth boundaries.

Run -I --input <developer-audit input> --output <new evidence directory>.
No downloads, learned model, GPU work, configuration or live ledger mutation.
The frozen comparison roster is written before any evaluation is generated.
Retain all artifacts under #12; no automatic deletion. No admission certificate.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finite_cmi as cmi
from finite_cmi_native import file_sha, native_states
from finite_cmi_readback import coverage_grid, replay, check_native
from lenscommon import LensError
from production.gate import require_admitted

NAMES = ("noise", "one-bit", "xor", "duplicate", "osc80", "osc512", "no-repeat", "full-conditioning", "absent")
REAL_SHA = {"record-vector.npz": "ce0976085308c49592802f46929c043d833c10e8c5d46cad6feb76d82634c245",
            "a2-wikidata-p54.jsonl": "f922824446c9a2311fdf261297272a5bad425e2b11294b7abf05ad247754b41d"}


def write(path, value):
    with Path(path).open("xb") as f:
        f.write(cmi.canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def frozen_plan():
    comparisons = {f"{name}-{n}": {"candidate_event": "native candidate coordinate 0 > 0; absent -> false",
                                   "outcome_event": "binary recorded outcome equals 1",
                                   "evaluation_units": n,
                                   "panel_ids": [] if name in ("noise", "one-bit", "absent") else ["complete-native-panel"]}
                   for n in (600, 2000) for name in NAMES}
    comparisons["real-duplicate"] = {"candidate_event": "record-vector coordinate 0 > 0",
                                     "outcome_event": "retained anchored label equals 0", "evaluation_units": 1864,
                                     "panel_ids": ["complete-native-panel"]}
    return {"schema": cmi.SCHEMA, "alpha": cmi.ALPHA, "seeds": list(cmi.SEEDS),
            "design_units": cmi.DESIGN_UNITS, "comparisons": comparisons,
            "assumptions": {"sampling": "iid-independent-units",
                            "justification": "Artificial independent Bernoulli/uniform controls only. Real duplicate is an algebraic redundancy control; its IID sampling is NOT established.",
                            "witnesses": "fixed-before-evaluation", "conditioning": "complete-native-state"}}


def arrays_for(name, n, seed):
    rng = np.random.default_rng(seed)
    count = n + cmi.DESIGN_UNITS
    x, y, z = [rng.integers(0, 2, count).astype(np.float32) for _ in range(3)]
    absent, truth = np.zeros(count, bool), 0
    if name in ("one-bit", "no-repeat"):
        x, truth = y.copy(), 1
    if name == "xor":
        y, truth = (x.astype(int) ^ z.astype(int)).astype(np.float32), 1
    if name == "duplicate":
        x, y = z.copy(), z.copy()
    if name.startswith("osc"):
        z = rng.uniform(-1, 1, count).astype(np.float32)
        x = (np.sin(int(name[3:]) * z.astype(np.float64)) > 0).astype(np.float32)
        y = x.copy()
    if name == "no-repeat":
        z = rng.uniform(-1, 1, count).astype(np.float32)
    if name == "full-conditioning":
        x, z = y.copy(), np.column_stack((z, y))
    if name == "absent":
        x, absent = np.zeros(count, np.float32), np.ones(count, bool)
    panel = name not in ("noise", "one-bit", "absent")
    return x[:, None], y, z.reshape(count, -1), absent, truth, panel


def experiment(root, plan, comparison, seed, x, y, z, absent, truth, panel, real=None):
    directory = root / f"{comparison}-{seed}"
    directory.mkdir()
    n = len(y)
    candidate = directory / "candidate.npz"
    with candidate.open("xb") as f:
        np.savez(f, X=x, absent=absent)
    descriptors = []
    if panel:
        p = directory / "panel.npz"
        if real:
            shutil.copyfile(real, p)  # Preserve full native windows and layout.
        else:
            with p.open("xb") as f:
                np.savez(f, X=z, absent=np.zeros(n, bool))
        descriptors = [{"id": "complete-native-panel", "path": str(p), "sha256": file_sha(p)}]
    ids = plan["comparisons"][comparison]["panel_ids"]
    states = native_states(descriptors, ids, n)
    units = [f"{comparison}/{seed}/unit/{i}" for i in range(n)]
    if real:
        labels = [json.loads(line) for line in (root / "real-labels.jsonl").read_text(encoding="utf-8").splitlines()]
        units = [r["cluster"] for r in labels]
        assert all(r["source"] == "wikidata5m" for r in labels)
    u, v = (x[:, 0] > 0).tolist(), y.astype(bool).tolist()
    outcome = directory / "outcome.npy"
    with outcome.open("xb") as f:
        np.save(f, y.astype(bool), allow_pickle=False)
    payload = {"units": units, "states": [base64.b64encode(s).decode("ascii") for s in states],
               "u": u, "v": v, "candidate": {"path": candidate.name, "sha256": file_sha(candidate)},
               "outcome": {"path": outcome.name, "sha256": file_sha(outcome)},
               "panel": [{**d, "path": Path(d["path"]).name} for d in descriptors], "panel_ids": ids,
               "true_cmi_bits": truth, "artificial_boundary_only": not bool(real),
               "iid_assumptions_substantiated": not bool(real)}
    write(directory / "inputs.json", payload)
    # Execute from persisted inputs, not transient generation output.
    payload = json.loads((directory / "inputs.json").read_bytes())
    states = [base64.b64decode(s) for s in payload["states"]]
    d = cmi.DESIGN_UNITS
    design = list(zip(payload["units"][:d], states[:d]))
    evaluation = list(zip(payload["units"][d:], states[d:], payload["u"][d:], payload["v"][d:]))
    result = cmi.estimate(plan, comparison, seed, design, evaluation)
    write(directory / "result.json", result)
    checked = replay(directory, plan)
    assert result["bits_lcb"] <= truth + 1e-10, (comparison, seed, result["bits_lcb"])
    return {**checked, "true_cmi_bits": truth, "code": result["code"],
            "artificial_boundary_only": not bool(real), "production_admission": False}


def boundaries(root, plan):
    checks = []
    def reject(name, code, action):
        before = cmi.digest(plan)
        try:
            action()
        except (cmi.BoundError, LensError, FileExistsError) as exc:
            actual = getattr(exc, "code", "EXISTS")
            assert actual == code, (name, code, actual)
            after = cmi.digest(plan)
            assert before == after
            checks.append({"boundary": name, "code": actual, "plan_before": before, "plan_after": after})
        else:
            raise AssertionError(f"{name}: invalid boundary accepted")
    design = [(f"d{i}", b"[]") for i in range(128)]
    rows = [(f"e{i}", b"[]", bool(i % 2), bool(i % 3)) for i in range(600)]
    def call(p=plan, d=design, e=rows, comparison="noise-600", seed=42):
        return cmi.estimate(p, comparison, seed, d, e)
    reject("repeated evaluation unit", "MNLM_CMI_UNIT_REUSE", lambda: call(e=[rows[1]] + rows[1:]))
    reject("design/evaluation overlap", "MNLM_CMI_UNIT_REUSE", lambda: call(d=[("e0", b"[]")] + design[1:]))
    reject("nonboolean witness", "MNLM_CMI_WITNESS", lambda: call(e=[("e0", b"[]", 1, True)] + rows[1:]))
    reject("underpowered units", "MNLM_CMI_SAMPLE", lambda: call(e=rows[:49]))
    reject("short design split", "MNLM_CMI_SAMPLE", lambda: call(d=design[:127]))
    reject("changed evaluation extent", "MNLM_CMI_SAMPLE_PLAN", lambda: call(e=rows[:500]))
    reject("unbudgeted comparison", "MNLM_CMI_COMPARISON", lambda: call(comparison="post-hoc"))
    reject("unbudgeted seed", "MNLM_CMI_COMPARISON", lambda: call(seed=47))
    p = copy.deepcopy(plan)
    p["assumptions"]["sampling"] = "dependent-time-steps"
    reject("dependent units", "MNLM_CMI_ASSUMPTIONS", lambda: call(p=p))
    p = copy.deepcopy(plan)
    p["alpha"] = 0.1
    reject("raised error budget", "MNLM_CMI_PLAN", lambda: call(p=p))
    reject("inconsistent table", "MNLM_CMI_COUNT", lambda: cmi.context_bound(100, 600, 40, 40, 60, 0.01))
    minority = cmi.context_bound(2000, 2000, 49, 49, 49, 0.01)
    assert minority["supported"] is False and minority["weighted_bits_lcb"] == 0
    checks.append({"boundary": "minority floor unchanged at 50", "code": "PASS"})
    reject("missing panel", "MNLM_CMI_PANEL_INCOMPLETE", lambda: native_states([], ["required"], 2))
    native = root / "native-boundaries"
    native.mkdir()
    slots = {
        "dense": {"X": np.asarray([[1, 2], [2, 1]], np.float32), "absent": np.zeros(2, bool)},
        "multi": {"tokens": np.asarray([[1, 2], [2, 1], [2, 1], [1, 2]], np.float32),
                  "offsets": np.asarray([0, 2, 4], np.int64), "absent": np.zeros(2, bool)},
        "sparse": {"data": np.ones(4, np.float32), "indices": np.asarray([0, 2, 1, 2], np.int32),
                   "indptr": np.asarray([0, 2, 4], np.int64), "shape": np.asarray([2, 4], np.int64),
                   "absent": np.zeros(2, bool)},
        "absence": {"X": np.zeros((2, 2), np.float32), "absent": np.asarray([False, True])}}
    for name, arrays in slots.items():
        path = native / f"{name}.npz"
        with path.open("xb") as f:
            np.savez(f, **arrays)
        desc = {"id": name, "path": str(path), "sha256": file_sha(path)}
        states = native_states([desc], [name], 2)
        assert states[0] != states[1], f"{name}: native distinction collapsed"
        payload = {"panel": [{**desc, "path": path.name}], "panel_ids": [name],
                   "states": [base64.b64encode(s).decode("ascii") for s in states]}
        write(native / f"{name}.json", payload)
        check_native(native, json.loads((native / f"{name}.json").read_bytes()))
        checks.append({"boundary": f"{name} retains native distinction", "code": "PASS"})
        reject(f"{name} changed hash", "MNLM_CMI_NATIVE_CHANGED",
               lambda: native_states([{**desc, "sha256": "0" * 64}], [name], 2))
    path = native / "dense.npz"
    before = file_sha(path)
    with np.load(path) as q:
        altered = q["X"].reshape(2, 1, 2)
    change = native / "shape-changed.npz"
    with change.open("xb") as f:
        np.savez(f, X=altered, absent=np.zeros(2, bool))
    reject("same values changed shape", "MNLM_CMI_NATIVE_CHANGED",
           lambda: native_states([{"id": "dense", "path": str(change), "sha256": before}], ["dense"], 2))
    change = native / "value-changed.npz"
    with change.open("xb") as f:
        np.savez(f, X=np.zeros((2, 2), np.float32), absent=np.zeros(2, bool))
    reject("same layout changed values", "MNLM_CMI_NATIVE_CHANGED",
           lambda: native_states([{"id": "dense", "path": str(change), "sha256": before}], ["dense"], 2))
    desc = {"id": "dense", "path": str(native / "dense.npz"), "sha256": before}
    reject("repeated conditioning slot", "MNLM_CMI_PANEL_INCOMPLETE",
           lambda: native_states([desc, desc], ["dense", "dense"], 2))
    reject("wrong native row count", "MNLM_CMI_NATIVE_ROWS", lambda: native_states([desc], ["dense"], 3))
    path = root / "noise-600-42" / "result.json"
    before = file_sha(path)
    reject("published result overwrite", "EXISTS", lambda: write(path, {"production_admission": True}))
    assert file_sha(path) == before
    checks.append({"boundary": "publication before/after", "before": before, "after": file_sha(path)})
    admission = root / "not-an-admission"
    admission.mkdir()
    positive = json.loads((root / "one-bit-2000-42" / "result.json").read_bytes())
    assert positive["bits_lcb"] >= cmi.BIT_FLOOR
    write(admission / "panel.json", positive)
    before = file_sha(admission / "panel.json")
    reject("positive bound is not Stage 3 admission", "MNLM_STAGE4_PANEL_UNADMITTED",
           lambda: require_admitted(admission, admission / "absent-ledger.jsonl"))
    assert not (admission / "absent-ledger.jsonl").exists() and file_sha(admission / "panel.json") == before
    checks.append({"boundary": "Stage 4 refusal leaves publication unchanged", "before": before,
                   "after": file_sha(admission / "panel.json"), "ledger_created": False})
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    start, phases = time.monotonic(), {}
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output
    inputs_before = {name: file_sha(args.input / name) for name in REAL_SHA}
    assert inputs_before == REAL_SHA
    plan = frozen_plan()
    cmi.validate_plan(plan)
    write(root / "plan.json", plan)
    write(root / "runtime.json", {"python": sys.version, "platform": platform.platform(),
                                  "numpy": np.__version__, "scipy": scipy.__version__})
    code = root / "code"
    code.mkdir()
    for p in (Path(__file__), Path(__file__).with_name("finite_cmi_readback.py"),
              Path(cmi.__file__), Path(cmi.__file__).with_name("finite_cmi_native.py"),
              Path(cmi.__file__).with_name("slot_layout.py"), Path(require_admitted.__code__.co_filename)):
        shutil.copyfile(p, code / p.name)
    phases["setup"] = time.monotonic() - start
    phase = time.monotonic()
    results = []
    for n in (600, 2000):
        for seed in cmi.SEEDS:
            for name in NAMES:
                x, y, z, absent, truth, panel = arrays_for(name, n, seed)
                results.append(experiment(root, plan, f"{name}-{n}", seed, x, y, z, absent, truth, panel))
        print(f"cold-replayed {len(results)} controls", flush=True)
    phases["known_truth_controls_and_readback"] = time.monotonic() - phase
    phase = time.monotonic()
    shutil.copyfile(args.input / "a2-wikidata-p54.jsonl", root / "real-labels.jsonl")
    labels = [json.loads(s) for s in (root / "real-labels.jsonl").read_text(encoding="utf-8").splitlines()]
    with np.load(args.input / "record-vector.npz", allow_pickle=False) as q:
        x, absent = q["X"].copy(), q["absent"].copy()
    assert len(x) == len(labels) and len({r["cluster"] for r in labels}) == len(labels)
    assert not absent.any(), "real duplicate control requires complete candidate rows"
    y = np.asarray([r["y"] == 0 for r in labels])
    results.append(experiment(root, plan, "real-duplicate", 42, x, y, x, absent, 0, True,
                              args.input / "record-vector.npz"))
    phases["real_input_and_readback"] = time.monotonic() - phase
    phase = time.monotonic()
    checks = boundaries(root, plan)
    coverage = coverage_grid(cmi.interval)
    phases["negative_boundaries_and_interval_coverage"] = time.monotonic() - phase
    inputs_after = {name: file_sha(args.input / name) for name in REAL_SHA}
    assert inputs_before == inputs_after
    summary = {"schema": cmi.SCHEMA, "acceptance": "author-checked only", "production_admission": False,
               "cases": results, "boundaries": checks, "interval_grid": coverage,
               "inputs_before": inputs_before, "inputs_after": inputs_after,
               "phases_seconds_measured": phases, "wall_seconds_measured": time.monotonic() - start,
               "limitations": ["IID units and fixed evaluation-independent witnesses are assumptions, not certified by IDs.",
                               "Real retained data is an algebraic negative control only; IID not established.",
                               "No-repeat native conditioning is underpowered, never rounded to create repeated contexts.",
                               "Floating numerical intervals are checked, not formally rounded interval arithmetic.",
                               "Method has not been independently accepted; Stage 3 remains uncommissioned."]}
    write(root / "report.json", summary)
    # Explicit bounded experiment directory only; no scanning production/data roots.
    files = sorted(p for p in root.rglob("*") if p.is_file())
    write(root / "manifest.json", {"files": {p.relative_to(root).as_posix(): file_sha(p) for p in files},
                                   "retention_owner": "ChrisRoyse/MNLM#12", "production_admission": False})
    cold = json.loads((root / "manifest.json").read_bytes())
    assert all(file_sha(root / p) == h for p, h in cold["files"].items())
    print(json.dumps({"cases": len(results), "boundaries": len(checks), "files": len(files),
                      "report_sha256": file_sha(root / "report.json"), "manifest_sha256": file_sha(root / "manifest.json"),
                      "wall_seconds": summary["wall_seconds_measured"], "production_admission": False}))


if __name__ == "__main__":
    main()
