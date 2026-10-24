"""A measured lane may be unavailable: retain typed refusals and null scores."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.guard_score_readback import read, sha, lines
from senses.contract import canonical


def main():
    ap = argparse.ArgumentParser()
    for n in ("measurements", "plan", "renders", "population", "comparison", "conditioning", "out"):
        ap.add_argument("--" + n, required=True, type=Path)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    original = sha(a.measurements / "report.json")
    shadow = a.out / "measurements"; shutil.copytree(a.measurements, shadow)
    rep = json.loads((shadow / "report.json").read_bytes()); name = rep["lenses"][0]
    pool = lines(a.renders / "texts.jsonl"); index = {r["id"]: i for i, r in enumerate(pool)}
    cases, targets = lines(a.renders / "candidates.jsonl"), lines(a.renders / "targets.jsonl")
    replayed = {i for b in rep["readings"][name]["replays"] for i in b["rows"]}
    calibration = {c["text_id"] for c in cases if c["split"] == "calibration"}
    source = next(index[t["text_id"]] for t in targets if index[t["text_id"]] not in replayed and t["text_id"] not in calibration)
    output = next(index[c["text_id"]] for c in cases if c["split"] == "evaluation" and c["text_id"] is not None
                  and index[c["text_id"]] not in replayed and c["text_id"] not in calibration)
    selected = sorted({source, output})
    vectors = np.load(shadow / (name + ".npy"), allow_pickle=False)
    valid = np.load(shadow / (name + ".valid.npy"), allow_pickle=False)
    assert valid[selected].all()
    vectors[selected] = 0; valid[selected] = False
    np.save(shadow / (name + ".npy"), vectors, allow_pickle=False)
    np.save(shadow / (name + ".valid.npy"), valid, allow_pickle=False)
    for i in selected:
        rep["readings"][name]["statuses"][i] = "MNLM_GUARD_NONFINITE_OR_ZERO"
    rep["readings"][name]["measured"] -= len(selected)
    for file in (name + ".npy", name + ".valid.npy"):
        rep["files"][file] = {"sha256": sha(shadow / file), "bytes": (shadow / file).stat().st_size}
    (shadow / "report.json").write_bytes(canonical(rep))
    command = [sys.executable, "-I", str(Path(__file__).resolve().parents[1] / "model/guard_score.py"),
               "--measurements", str(shadow), "--measurements-sha256", sha(shadow / "report.json"), "--output", str(a.out / "scores")]
    for n in ("plan", "renders", "population", "comparison", "conditioning"):
        command += ["--" + n, str(getattr(a, n))]
    with (a.out / "score.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    args = copy.copy(a); args.root = a.out / "scores"; args.measurements = shadow; args.report_sha256 = sha(args.root / "report.json")
    cold = read(args)
    (a.out / "cold-readback.json").write_bytes(canonical(cold))
    traces, controls = lines(args.root / "traces.jsonl"), lines(args.root / "controls.jsonl")
    affected = [c for c in traces if c["text_id"] in {pool[i]["id"] for i in selected}]
    source_controls = [c for c in controls if c["text_id"] == pool[source]["id"]]
    checks = {"unavailable_output_refused": bool(affected) and all(c["slots"][name]["cosine"] is None
              and c["slots"][name]["status"] == "MNLM_GUARD_SLOT_UNAVAILABLE" and not c["geometric_pass"] for c in affected),
              "unavailable_source_control_retained": bool(source_controls) and all(c["slots"][name]["cosine"] is None for c in source_controls),
              "all_assertions_refused": all(c["asserted"] is False for c in traces),
              "original_unchanged": sha(a.measurements / "report.json") == original}
    report = {"author_checked": True, "boundary_probe_only": True, "passed": all(checks.values()), "checks": checks,
              "mutated_lens": name, "mutated_rows": selected, "affected_render_cases": len(affected),
              "original_report_sha256": original, "mutant_report_sha256": sha(shadow / "report.json"), "scientific_admission": False}
    (a.out / "fsv.json").write_bytes(canonical(report)); print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
