"""Independent SQL recount/native readback and fault controls for the support census.

The enumerated binary states below are known-truth boundary controls only;
they never enter a source, instrument training, information assay or admission.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / "lens")]
from conditional_support import load, recount, state_bytes
from senses.contract import CHANNELS, canonical, require
from senses.pipeline import file_hash
from senses.trajectory_population import publish


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, condition, **detail):
        checks.append(dict(name=name, passed=bool(condition), **detail))
    def refuses(name, fn, code):
        error = None
        try:
            fn()
        except Exception as exc:
            error = str(exc)
        check(name, error is not None and code in error, error=error)
    before = {n: file_hash(args.census / n) for n in ("intent.json", "report.json", "states.json", "COMMITTED")}
    report, rows = load(args.census, before["report.json"])
    index = {r["atom"]: r for r in rows}
    check("one_observation_per_retained_unit", len(index) == len({r["unit"] for r in rows}) == report["rows"])
    # Independent implementation: SQL aggregation, no call to the production counter.
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE states (" + ",".join(c + " TEXT NOT NULL" for c in CHANNELS) + ")")
    db.executemany("INSERT INTO states VALUES(" + ",".join("?" for _ in CHANNELS) + ")",
                   [tuple(r["slots"][c]["sha256"] for c in CHANNELS) for r in rows])
    mismatches = []
    for case in report["comparisons"]:
        columns = ",".join(case["conditioning"])
        counts = [r[0] for r in db.execute("SELECT count(*) FROM states GROUP BY " + columns)]
        complete = sum(all(r["slots"][c]["kind"] != "Absent" for c in [case["candidate"], *case["conditioning"]]) for r in rows)
        if (len(counts), max(counts), sum(n for n in counts if n >= 100), complete) != (
                case["context_buckets"], case["largest_bucket"], case["potentially_supported_rows_upper"], case["complete_cases"]):
            mismatches.append(case["candidate"] + ":" + columns)
    db.close()
    check("every_conditioning_count_SQL_recount", not mismatches, comparisons=len(report["comparisons"]), mismatches=mismatches)
    # The previously retained six-observation transport proof supplies physical native values.
    # Re-serialize their complete states independently of state_bytes().
    reference_rows = json.loads((args.reference / "rows.json").read_bytes())
    reference_manifest = json.loads((args.reference / "manifest.json").read_bytes())
    matches = 0
    for c in CHANNELS:
        p = args.reference / (c + ".npz")
        assert file_hash(p) == reference_manifest["slots"][c]["sha256"]
        with np.load(p, allow_pickle=False) as z:
            for i, row in enumerate(reference_rows):
                lo, hi = z["offsets"][i:i+2]
                values = z["tokens"][lo:hi]
                header = dict(shape=list(values.shape), dtype="<f4", kind="Absent" if bool(z["absent"][i]) else "Multi",
                              windows=int(z["windows"][i]))
                raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n" + values.tobytes()
                actual = {**header, "sha256": hashlib.sha256(raw).hexdigest()}
                assert actual == index[row["atom"]]["slots"][c], (c, i)
                matches += 1
    check("prior_physical_native_slots_match_census", matches == len(reference_rows) * len(CHANNELS), readings=matches)
    # Faults never mutate the authoritative census or its source.
    bad = copy.deepcopy(rows[:2]); bad[1]["unit"] = bad[0]["unit"]
    refuses("duplicate_unit", lambda: recount(bad, report["exact_constants"]), "MNLM_SUPPORT_UNITS")
    bad = copy.deepcopy(rows[:2]); bad[0]["slots"].pop("history")
    refuses("missing_conditioning_channel", lambda: recount(bad, report["exact_constants"]), "MNLM_SUPPORT_CHANNELS")
    bad = copy.deepcopy(rows[:2]); bad[0]["slots"]["history"]["shape"] = [1, 127]
    refuses("changed_native_shape", lambda: recount(bad, report["exact_constants"]), "MNLM_SUPPORT_NATIVE")
    control = args.output / "missing-commit"; control.mkdir()
    for n in ("report.json", "states.json", "intent.json"):
        (control / n).write_bytes((args.census / n).read_bytes())
    # The ordinary reader's missing marker is a hard failure, never inferred success.
    refuses("unpublished_census", lambda: load(control, before["report.json"]), "MNLM_SUPPORT_UNCOMMITTED")
    (control / "COMMITTED").write_bytes((args.census / "COMMITTED").read_bytes())
    (control / "states.json").write_bytes((control / "states.json").read_bytes() + b" ")
    refuses("changed_state_bytes", lambda: load(control, before["report.json"]), "MNLM_UCR_INPUT_DIGEST")
    (control / "states.json").write_bytes((args.census / "states.json").read_bytes())
    bad_report = copy.deepcopy(report); bad_report["comparisons"][0]["largest_bucket"] += 1
    (control / "report.json").write_bytes(canonical(bad_report))
    changed_sha = file_hash(control / "report.json")
    marker = json.loads((control / "COMMITTED").read_bytes()); marker["report_sha256"] = changed_sha
    (control / "COMMITTED").write_bytes(canonical(marker))
    refuses("resealed_wrong_count", lambda: load(control, changed_sha), "MNLM_SUPPORT_RECOUNT")
    values = np.zeros((1,128), np.float32)
    a, _ = state_bytes({"tokens": values, "kind": "Multi"}, 1)
    b, _ = state_bytes({"tokens": values, "kind": "Multi"}, 2)
    check("window_count_is_conditioning", a != b)
    values[0,0] = float("nan")
    refuses("nonfinite_native", lambda: state_bytes({"tokens": values, "kind": "Multi"}, 1), "MNLM_SUPPORT_NATIVE")

    # A genuine repeated binary context can pass the necessary-support screen.
    varied = list(CHANNELS[:5])
    synthetic = []
    for i in range(1600):
        s = {}
        for c in CHANNELS:
            bit = ((i // 50) >> varied.index(c)) & 1 if c in varied else 0
            s[c] = dict(sha256=hashlib.sha256(bytes([bit])).hexdigest(), shape=[1,128], dtype="<f4", kind="Multi", windows=1)
        synthetic.append(dict(atom=str(i), unit=str(i), slots=s))
    constants = {c: c not in varied for c in CHANNELS}
    positive = recount(synthetic, constants)
    check("known_truth_repeated_context_positive", positive["possible_fifth_steps_upper"] == 5
          and positive["stage3_admitted"] is False)
    unique = copy.deepcopy(synthetic)
    for i,r in enumerate(unique):
        for c in varied:
            r["slots"][c]["sha256"] = hashlib.sha256((c+str(i)).encode()).hexdigest()
    negative = recount(unique, constants)
    check("unique_context_refuses_support", negative["possible_fifth_steps_upper"] == 0)
    collision = copy.deepcopy(unique)
    for r in collision:
        for c in varied:
            r["slots"][c]["sha256"] = "0"*64
    over = recount(collision, constants)
    check("hash_collision_only_overstates_support", over["possible_fifth_steps_upper"] == 5 and over["stage3_admitted"] is False)
    check("source_census_unchanged", before == {n: file_hash(args.census / n) for n in before})
    result = dict(acceptance="author-checked only", checks=checks, passed=sum(c["passed"] for c in checks),
                  failed=sum(not c["passed"] for c in checks), census_report_sha256=before["report.json"],
                  source_files=before, controls_are_training_data=False, stage3_admitted=False)
    publish(args.output / "report.json", result)
    print(json.dumps({k: result[k] for k in ("passed", "failed")}))
    return bool(result["failed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("census", "reference", "output"):
        ap.add_argument("--"+n, type=Path, required=True)
    sys.exit(run(ap.parse_args()))
