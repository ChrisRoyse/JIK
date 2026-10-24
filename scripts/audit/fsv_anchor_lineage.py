"""Original-byte, canonical-record and value-set consumer boundary evidence."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.anchor_lineage_readback import read, parents, witness, sha, lines
from text.anchor_lineage import judge
from senses.contract import canonical
from senses.trajectory_population import publish


def main():
    ap = argparse.ArgumentParser()
    for n in ("root", "raw", "out"):
        ap.add_argument("--" + n, type=Path, required=True)
    a = ap.parse_args(); start = time.perf_counter(); phases = {}
    a.out.mkdir(parents=True, exist_ok=False)
    pin = sha(a.root / "report.json"); rep = json.loads((a.root / "report.json").read_bytes())
    selection = parents(a.root, rep)
    original = witness(a.raw, rep["original_archive_sha256"], {(c["entity"], c["property"]) for c in selection})
    phases["full_original_replay"] = time.perf_counter() - start
    publish(a.out / "original-replay.json", original)
    tick = time.perf_counter(); cold = read(a.root, pin, original)
    publish(a.out / "cold-readback.json", cold)
    checks = [{"name": "B1 all fixed labels traced to complete original/canonical value sets", "passed": True}]
    phases["positive_cold_read"] = time.perf_counter() - tick
    tick = time.perf_counter()

    def fault(name, fn, code, file=None):
        shadow = a.out / name; shutil.copytree(a.root, shadow)
        changed = copy.deepcopy(rep)
        if file:
            objects = lines(shadow / file); fn(objects)
            (shadow / file).write_bytes(b"".join(canonical(o) + b"\n" for o in objects))
            changed["files"][file] = {"sha256": sha(shadow / file), "bytes": (shadow / file).stat().st_size}
        else:
            fn(changed)
        (shadow / "report.json").write_bytes(canonical(changed))
        try:
            read(shadow, sha(shadow / "report.json"), original)
            observed = "accepted"
        except ValueError as exc:
            observed = str(exc)
        checks.append({"name": name, "expected": "MNLM_RP_READ_" + code, "observed": observed,
                       "passed": observed == "MNLM_RP_READ_" + code, "mutant_report_sha256": sha(shadow / "report.json")})

    fault("B1_changed_original_source_pin", lambda r: r.update(original_archive_sha256="0" * 64), "AL_SOURCE_BINDING")
    fault("B1_changed_label_binding", lambda r: r[0].update(legacy_value="Q99999999999999"), "AL_SELECTION", "selected.jsonl")
    fault("B1_changed_raw_byte", lambda r: r[0].update(line_hex=r[0]["line_hex"] + "20"), "AL_SOURCE_COMPLETENESS", "raw-selected.jsonl")
    fault("B2_omitted_source_occurrence", lambda r: r.pop(), "AL_SOURCE_COMPLETENESS", "raw-selected.jsonl")
    fault("B2_changed_canonical_value", lambda r: r[0]["record"]["f"].update(o="Q99999999999999"), "AL_CANON_PAYLOAD", "canonical-selected.jsonl")
    fault("B2_changed_canonical_row", lambda r: r[0]["record"].update(row=-1), "AL_CANON_POINTER", "canonical-selected.jsonl")
    fault("B3_invented_value", lambda r: r[0]["values"].append("Q99999999999999"), "AL_CASE_VALUES", "cases.jsonl")
    fault("B3_invented_currentness", lambda r: r[0].update(currentness_certified=True), "AL_CASE_VALUES", "cases.jsonl")

    def false_unique(rows):
        case = next(c for c in rows if c["status"] == "multiple")
        case.update(values=[case["legacy_value"]], status="singleton")

    fault("B3_multiple_values_cannot_be_silently_singular", false_unique, "AL_CASE_VALUES", "cases.jsonl")
    fault("B3_false_admission", lambda r: r.update(scientific_admission=True), "AL_CLAIM")
    fault("B3_inflated_summary", lambda r: r["summaries"]["biomedical/P1057"].update(cases=505), "AL_SUMMARY")
    phases["persisted_fault_controls"] = time.perf_counter() - tick
    tick = time.perf_counter(); cases = lines(a.root / "cases.jsonl"); verdicts = []
    for c in cases:
        for v in c["values"]:
            verdicts.append(judge(c, v))
        verdicts.append(judge(c, c["legacy_value"], unique=True))
        verdicts.append(judge(c, "Q99999999999999"))
    check_supported = all(judge(c, v)["status"] == "source_supported" for c in cases for v in c["values"])
    check_unique = all(judge(c, c["legacy_value"], unique=True)["status"] ==
                       ("source_unique" if c["status"] == "singleton" else "MNLM_A2_AMBIGUOUS_VALUE_SET") for c in cases)
    checks += [{"name": "B3 every actual value including alternatives is source-supported", "passed": check_supported},
               {"name": "B3 all multivalued records refuse single-answer certification", "passed": check_unique},
               {"name": "B3 unobserved value is unavailable, never labelled false reality", "passed":
                all(judge(c, "Q99999999999999")["status"] == "MNLM_A2_VALUE_UNOBSERVED" for c in cases)},
               {"name": "B3 no value-set verdict asserts real-world truth", "passed":
                all(v["asserted"] is False and v["real_world_correctness_certified"] is False for v in verdicts)}]
    (a.out / "consumer-verdicts.jsonl").write_bytes(b"".join(canonical(v) + b"\n" for v in verdicts))
    checks.append({"name": "B3 original publication unchanged and final cold read survives", "passed":
                   sha(a.root / "report.json") == pin and read(a.root, pin, original) == cold})
    phases["consumer_and_final_readback"] = time.perf_counter() - tick
    wall = time.perf_counter() - start
    result = {"author_checked": True, "passed": all(c["passed"] for c in checks), "checks": checks,
              "source_report_sha256": pin, "consumer_verdicts": len(verdicts), "scientific_admission": False,
              "phases_seconds": phases, "wall_seconds": wall, "phase_coverage": sum(phases.values()) / wall}
    publish(a.out / "fsv.json", result)
    print(json.dumps({"passed": result["passed"], "checks": len(checks), "failed": [c for c in checks if not c["passed"]],
                      "wall_seconds": wall, "phase_coverage": result["phase_coverage"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
