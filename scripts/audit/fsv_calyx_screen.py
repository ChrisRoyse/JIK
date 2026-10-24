"""Physical before/action/after proof for Calyx screening using retained reports.

Controls mutate copies of real measurements; they are never scientific results.
The CLI is executed in a separate process, then JSON and input hashes are read
independently. No GPU or new measurement population is involved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish
from text.calyx_screen import ScreeningError, screen


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cold_check(path):
    """Independent reader: no calls to screen(), replay() or their validators."""
    value = json.loads(path.read_bytes())
    src = Path(value["source"]["path"])
    if sha(src) != value["source"]["sha256"]:
        raise ValueError("FSV_SOURCE_DIGEST")
    source = json.loads(src.read_bytes())
    result = value["screening"]
    flags = (result["meaning_rule_met"], result["served"], result["shannon_conditional_certified"])
    assert flags == (False, False, False) and result["scientifically_admitted_lenses"] == []
    expected = {(name, anchor) for name, c in source["controls"].items()
                for anchor, bits in c["per_anchor"].items() if bits >= 0.05}
    actual = {(f["control"], f["anchor"]) for f in result["failed_controls"]}
    assert expected == actual and result["controls_pass"] == (not expected)
    for q, row in result["per_anchor"].items():
        bad = any(anchor == q for _, anchor in expected)
        assert row["controls_pass"] == (not bad)
        assert not bad or not row["candidate_roster"]
        for i, lens in enumerate(row["candidate_roster"]):
            assert source["per_lens"][lens][q]["chunk_multi"]["ci_low"] >= 0.05
            for previous in row["candidate_roster"][:i]:
                pairs = source["pair_nmi"]
                corr = pairs.get(f"{lens}|{previous}", pairs.get(f"{previous}|{lens}"))
                assert corr <= 0.6
    return {"source": str(src), "source_sha256": sha(src), "output_sha256": sha(path),
            "before": value["source_flags"],
            "after": {"controls_pass": result["controls_pass"], "meaning_rule_met": False},
            "failed_controls": result["failed_controls"],
            "candidate_counts": {q: len(r["candidate_roster"]) for q, r in result["per_anchor"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); phases = {}
    files = [args.reports / f"{family}-v1.json" for family in
             ("biomedical", "code", "finance", "legal", "science")]
    original_hashes = {str(p): sha(p) for p in files}
    cli = Path(__file__).resolve().parents[1] / "text" / "calyx_screen.py"
    command = [sys.executable, "-I", str(cli)]
    for path in files:
        command.extend(["--report", str(path)])
    destination = args.output / "replay"
    command.extend(["--output", str(destination)])
    proc = subprocess.run(command, capture_output=True, text=True)
    (args.output / "cli.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    outputs = [destination / (original_hashes[str(p)] + ".json") for p in files]
    readbacks = [cold_check(p) for p in outputs]
    phases["real_cli_and_five_cold_reads"] = time.perf_counter() - start
    tick = time.perf_counter()
    checks = []

    def check(name, condition, detail):
        checks.append({"name": name, "pass": bool(condition), "detail": detail})
        assert condition, name

    for item in readbacks:
        check("cold readback " + Path(item["source"]).name, True, item)
    for family in ("code", "finance"):
        digest = original_hashes[str(args.reports / (family + "-v1.json"))]
        item = next(r for r in readbacks if r["source_sha256"] == digest)
        # The legacy input cached a misleading mean-based pass; newer producers
        # correctly cache false. The per-anchor refusal must hold for both.
        check("per-anchor controls refuse " + family,
              not item["after"]["controls_pass"] and bool(item["failed_controls"]), item)
        misleading = copy.deepcopy(json.loads((args.reports / (family + "-v1.json")).read_bytes()))
        misleading["controls_pass"] = True
        for control in misleading["controls"].values():
            control["mean_bits"] = 0.0
        control_path = args.output / (family + "-misleading-summary.json")
        control_path.write_text(json.dumps(misleading), encoding="utf-8")
        corrected = screen(json.loads(control_path.read_bytes()))
        check("cached mean and pass cannot mask " + family,
              not corrected["controls_pass"] and corrected["failed_controls"] == item["failed_controls"],
              {"control_path": str(control_path), "failed_controls": corrected["failed_controls"]})

    real = json.loads((args.reports / "legal-v1.json").read_bytes())
    anchor = next(iter(real["anchors"]))
    baseline = screen(real)
    original = baseline["per_anchor"][anchor]["candidate_roster"][0]
    mutations = args.output / "boundary-inputs"
    mutations.mkdir()

    def persist(name, value):
        p = mutations / (name + ".json")
        p.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
        return p

    def refuse(name, value):
        p = persist(name, value)
        try:
            screen(json.loads(p.read_bytes()))
        except ScreeningError as error:
            check(name, str(error).startswith("MNLM_CALYX_SCREEN_INPUT:"), str(error))
        else:
            check(name, False, "unexpected acceptance")

    low = copy.deepcopy(real)
    low["per_lens"][original][anchor]["chunk_multi"]["ci_low"] = 0.049
    result = screen(json.loads(persist("low-ci", low).read_bytes()))
    check("strong point with weak interval",
          original not in result["per_anchor"][anchor]["candidate_roster"],
          {"point": low["per_lens"][original][anchor]["chunk_multi"]["bits"], "ci_low": 0.049})

    duplicate = copy.deepcopy(real)
    alias = "control-duplicate-lens"
    duplicate["per_lens"][alias] = copy.deepcopy(real["per_lens"][original])
    names = set(real["per_lens"]) - set(real["refused_lenses"])
    for lens in names:
        pair = real["pair_nmi"]
        duplicate["pair_nmi"][f"{alias}|{lens}"] = (1.0 if lens == original else
            pair.get(f"{original}|{lens}", pair.get(f"{lens}|{original}")))
    result = screen(json.loads(persist("duplicate-lens", duplicate).read_bytes()))
    selected = result["per_anchor"][anchor]["candidate_roster"]
    check("duplicate cannot become another perspective", len({alias, original} & set(selected)) == 1, selected)

    failed = copy.deepcopy(real)
    failed["controls"]["random_lens"]["per_anchor"][anchor] = 0.05
    failed["controls"]["random_lens"]["mean_bits"] = 0.0
    result = screen(json.loads(persist("control-at-floor", failed).read_bytes()))
    check("control at floor overrides cached mean", not result["controls_pass"]
          and result["per_anchor"][anchor]["candidate_roster"] == [], result["failed_controls"])

    missing = copy.deepcopy(real)
    del missing["controls"]["random_lens"]["per_anchor"][anchor]
    refuse("missing-control", missing)
    invalid = copy.deepcopy(real)
    invalid["controls"]["random_lens"]["per_anchor"][anchor] = float("nan")
    refuse("nonfinite-control", invalid)
    invalid = copy.deepcopy(real)
    invalid["per_lens"][original][anchor]["chunk_multi"]["ci_low"] = float("inf")
    refuse("nonfinite-interval", invalid)
    invalid = copy.deepcopy(real)
    invalid["per_lens"][original][anchor]["chunk_multi"]["n"] -= 1
    refuse("mismatched-population", invalid)
    invalid = copy.deepcopy(real)
    invalid["pair_nmi"].pop(next(iter(invalid["pair_nmi"])))
    refuse("missing-pair-evidence", invalid)
    invalid = copy.deepcopy(real)
    invalid["per_lens"][original][anchor]["chunk_multi"]["ci_low"] = 100.0
    refuse("reversed-interval", invalid)
    check("screen never certifies D2", len(baseline["per_anchor"][anchor]["candidate_roster"]) >= 5
          and not baseline["meaning_rule_met"] and not baseline["shannon_conditional_certified"],
          baseline["admission_blockers"])

    # All-or-nothing evidence publication and source binding are physical tests.
    hashes = {str(p): sha(p) for p in outputs}
    retry = subprocess.run(command, capture_output=True, text=True)
    (args.output / "retry.log").write_text(retry.stdout + retry.stderr, encoding="utf-8")
    check("immutable replay refuses overwrite", retry.returncode != 0
          and "MNLM_UCR_OUTPUT_EXISTS" in retry.stderr
          and hashes == {str(p): sha(p) for p in outputs}, {"returncode": retry.returncode})
    mixed_out = args.output / "mixed-replay"
    mixed = subprocess.run([sys.executable, "-I", str(cli), "--report", str(mutations / "missing-control.json"),
                            "--report", str(files[0]), "--output", str(mixed_out)], capture_output=True, text=True)
    (args.output / "mixed.log").write_text(mixed.stdout + mixed.stderr, encoding="utf-8")
    check("bad input does not hide valid sibling", mixed.returncode != 0
          and "MNLM_CALYX_SCREEN_INPUT" in mixed.stderr
          and (mixed_out / (original_hashes[str(files[0])] + ".json")).is_file(), mixed.stderr)

    tampered_source = mutations / "tampered-source.json"
    tampered_source.write_bytes(files[0].read_bytes())
    bound = json.loads(outputs[0].read_bytes())
    bound["source"]["path"] = str(tampered_source)
    tampered_output = persist("bound-before-tamper", bound)
    cold_check(tampered_output)
    tampered_source.write_bytes(tampered_source.read_bytes() + b" ")
    try:
        cold_check(tampered_output)
    except ValueError as error:
        check("source mutation caught on cold read", str(error) == "FSV_SOURCE_DIGEST", str(error))
    else:
        check("source mutation caught on cold read", False, "missed mutation")
    check("original source bytes unchanged", original_hashes == {str(p): sha(p) for p in files}, original_hashes)
    phases["physical_controls_and_final_source_readback"] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    summary = {"schema": "fsv-calyx-screen-v1", "acceptance": "author-checked only",
               "checks": checks, "passed": len(checks), "boundaries": 3,
               "wall_seconds": wall, "phases_seconds": phases, "phase_coverage": sum(phases.values())/wall,
               "inputs": original_hashes,
               "code_sha256": sha(Path(__file__)), "screen_sha256": sha(cli)}
    publish(args.output / "fsv.json", summary)
    print(json.dumps({"passed": len(checks), "report": str(args.output / "fsv.json")}))


if __name__ == "__main__":
    main()
