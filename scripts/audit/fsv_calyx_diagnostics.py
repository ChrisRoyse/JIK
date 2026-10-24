"""Persisted corrupt-copy controls around the real native-window diagnostic."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_diagnostic_readback import load_inputs, readback, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("diagnostic", "slots", "calibration", "fsv", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    inputs = load_inputs(a.slots, a.calibration, a.fsv)
    before = {p.name: sha(p.read_bytes()) for p in a.diagnostic.iterdir() if p.is_file()}
    cold = readback(a.diagnostic, inputs)
    checks = [{"boundary": "B1 source/native/cache", "name": "all real profiles, distances and pairs rebuilt", "pass": True},
              {"boundary": "B2 geometry/numerics", "name": "all 3582 label nulls, 15 random scores and point claims replayed", "pass": True},
              {"boundary": "B3 publication/claim", "name": "complete diagnostic explicitly grants no admission", "pass": True}]
    key = "biomedical-P1057"; sense = "e5-base-v2"
    def put(path, value):
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    def control(name, boundary, expected, edit):
        dest = a.output/name; shutil.copytree(a.diagnostic, dest)
        root = json.loads((dest/"report.json").read_bytes())
        edit(dest, root)
        put(dest/(key+".json"), root["findings"][key]); put(dest/"report.json", root)
        try:
            readback(dest, inputs); error = "no refusal"
        except ValueError as exc:
            error = str(exc)
        checks.append({"boundary": boundary, "name": name, "expected": expected, "observed": error,
                       "pass": error == "MNLM_DIAGNOSTIC_READBACK_"+expected,
                       "mutated_report_sha256": sha((dest/"report.json").read_bytes())})
    def artifact(edit):
        def mutate(dest, root):
            path = dest/(key+".npz")
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            edit(arrays); np.savez(path, **arrays)
            root["findings"][key]["artifact_sha256"] = sha(path.read_bytes())
        return mutate
    control("wrong-source", "B1", "BINDING", lambda d,r: r.update(source_report_sha256="0"*64))
    control("changed-label-rebound", "B1", "LABELS", artifact(lambda z: z["labels"].__setitem__(0,99)))
    control("missing-pair-rebound", "B1", "PAIRS", artifact(lambda z: z.update(pairs=z["pairs"][:-1])))
    control("changed-distance-rebound", "B1", "GEOMETRY", artifact(lambda z: z[sense+"__distance"].__setitem__((0,1),.9)))
    control("changed-pair-reading-rebound", "B1", "PAIR_GEOMETRY", artifact(lambda z: z[sense+"__pair_readings"].__setitem__(0,.9)))
    control("false-point-score", "B2", "POINT", lambda d,r: r["findings"][key]["standalone_raw_bits"].update({sense:9.0}))
    control("changed-null-score-rebound", "B2", "NULL_VALUES", artifact(lambda z: z[sense+"__null_raw_bits"].__setitem__(0,9.0)))
    control("hidden-null-crossings", "B2", "NULL_CLAIMS", lambda d,r: r["findings"][key]["label_nulls"][sense].update(at_or_above_point_floor=0))
    control("incomplete-publication", "B3", "PUBLICATION", lambda d,r: r.update(complete=False))
    control("false-admission", "B3", "CLAIMS", lambda d,r: r.update(scientific_admission=True))
    control("relaxed-floor", "B3", "RULES", lambda d,r: r["rules"].update(point_floor=.01))
    checks.append({"boundary": "B3", "name": "original diagnostic unchanged", "pass": before ==
                   {p.name: sha(p.read_bytes()) for p in a.diagnostic.iterdir() if p.is_file()}})
    result = {"passed": sum(c["pass"] for c in checks), "total": len(checks), "checks": checks,
              "cold_readback": cold, "acceptance": "author-checked"}
    put(a.output/"fsv.json", result)
    print(json.dumps({k:result[k] for k in ("passed", "total")}))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
