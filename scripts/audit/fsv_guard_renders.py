"""Persisted render/calibration/input-pool boundary controls; no fake observations."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.guard_render_readback import read, lines, sha
from senses.contract import canonical


def main():
    ap = argparse.ArgumentParser()
    for name in ("root", "population", "comparison", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    before = sha(a.root / "report.json")
    cold = read(a.root, before, a.population, a.comparison)
    (a.out / "cold-readback.json").write_bytes(canonical(cold))
    original = json.loads((a.root / "report.json").read_bytes())
    checks = [{"name": "B1 pinned population, checkpoint and all output bytes read back", "passed": True}]

    def fault(name, mutate, code, file=None):
        shadow = a.out / name
        shutil.copytree(a.root, shadow)
        rep = copy.deepcopy(original)
        if file:
            objects = lines(shadow / file)
            mutate(objects)
            (shadow / file).write_bytes(b"".join(canonical(o) + b"\n" for o in objects))
            rep["files"][file] = {"bytes": (shadow / file).stat().st_size, "sha256": sha(shadow / file)}
        else:
            mutate(rep)
        (shadow / "report.json").write_bytes(canonical(rep))
        try:
            read(shadow, sha(shadow / "report.json"), a.population, a.comparison)
            observed = "accepted"
        except ValueError as exc:
            observed = str(exc)
        checks.append({"name": name, "expected": "MNLM_RP_READ_" + code, "observed": observed,
                       "passed": observed == "MNLM_RP_READ_" + code})

    fault("B1_wrong_population_pin", lambda r: r.update(population_sha256="0" * 64), "GUARD_RENDER_SOURCE")
    fault("B1_wrong_checkpoint", lambda r: r["checkpoint_sha256"].update(true="0" * 64), "GUARD_RENDER_CHECKPOINT")
    fault("B2_evaluation_as_calibration", lambda r: r["calibration_policy"].update(split="evaluation"), "GUARD_CALIBRATION_POLICY")
    fault("B2_threshold_tuned_after_results", lambda r: r["calibration_policy"].update(quantile=.01), "GUARD_CALIBRATION_POLICY")
    fault("B2_missing_case", lambda r: r.pop(), "GUARD_RENDER_ORDER", "candidates.jsonl")
    fault("B2_reordered_cases", lambda r: r.reverse(), "GUARD_RENDER_ORDER", "candidates.jsonl")
    fault("B2_invented_anchor", lambda r: r[0].update(independent_anchor_status="correct"), "GUARD_RENDER_BYTES_OR_CLAIM", "candidates.jsonl")
    fault("B2_premature_assertion", lambda r: r[0].update(asserted=True), "GUARD_RENDER_BYTES_OR_CLAIM", "candidates.jsonl")

    def replace_refusal(rows):
        row = next(c for c in rows if c["status"] != "complete_unasserted")
        row.update(text=".", status="complete_unasserted")

    fault("B3_placeholder_cannot_replace_refusal", replace_refusal, "GUARD_RENDER_BYTES_OR_CLAIM", "candidates.jsonl")
    fault("B3_changed_pool_bytes", lambda r: r[0].update(text=r[0]["text"] + "x"), "GUARD_RENDER_TEXT", "texts.jsonl")
    fault("B3_false_grounding_certificate", lambda r: r.update(independent_anchors_evaluated=True), "GUARD_RENDER_CLAIM")
    after = sha(a.root / "report.json")
    checks.append({"name": "B3 originals unchanged and full cold replay survives", "passed": before == after
                   and read(a.root, before, a.population, a.comparison) == cold})
    report = {"author_checked": True, "passed": all(c["passed"] for c in checks), "checks": checks,
              "before_report_sha256": before, "after_report_sha256": after,
              "scientific_admission": False, "guard_evaluated": False}
    (a.out / "fsv.json").write_bytes(canonical(report))
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "failed": [c for c in checks if not c["passed"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
