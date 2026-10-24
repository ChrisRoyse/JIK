"""Operator-exclusion boundaries, including refusal before retiring output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "canon"), str(Path(__file__).resolve().parents[1] / "lens")]
from common import Paths
from dedup import load_population
from lenscommon import read_family


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sandbox = out / "control-only"
    target = sandbox / "out" / "atoms" / "scope-control"
    target.mkdir(parents=True)
    (target / "manifest.jsonl").write_text('{"sentinel":"known-truth output-retirement control"}\n')
    repo = Path(__file__).resolve().parents[2]
    common = ["--root", str(sandbox / "raw"), "--out", str(sandbox / "out"), "--version", "scope-control"]
    checks = []

    def state():
        return {str(p.relative_to(sandbox)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sandbox.rglob("*") if p.is_file()}

    for script, options in (("canon/atomise.py", ["--sources", "cuyahoga-eighth-district", "--retire"]),
                            ("canon/dedup.py", ["--sources", "cuyahoga-eighth-district", "--step", "exact"]),
                            ("canon/canonicalise.py", ["--source", "cuyahoga-eighth-district"]),
                            ("lens/measure.py", ["--families", "a6-cuyahoga-cited-cohort"])):
        before = state()
        proc = subprocess.run([sys.executable, "-I", str(repo / "scripts" / script), *common, *options],
                              capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        log = proc.stdout + proc.stderr
        (out / (Path(script).stem + ".log")).write_text(log)
        checks.append({"boundary": script, "pass": proc.returncode != 0 and "MNLM_SOURCE_EXCLUDED" in log and before == state(),
                       "returncode": proc.returncode, "before": before, "after": state()})
    paths = Paths(sandbox / "raw", sandbox / "out", "scope-control")
    paths.dedup.mkdir(parents=True)
    (paths.dedup / "population.json").write_text(json.dumps({"sources": ["cuyahoga-eighth-district"]}))
    cal = sandbox / "cal"
    cal.mkdir()
    (cal / "different-name.jsonl").write_text(json.dumps({"source": "cuyahoga-eighth-district"}) + "\n")
    for name, call in (("old_population_reuse", lambda: load_population(paths)), ("renamed_calibration_reuse", lambda: read_family(cal, "different-name"))):
        before, code = state(), None
        try:
            call()
        except RuntimeError as exc:
            code = str(exc)
        checks.append({"boundary": name, "pass": code is not None and code.startswith("MNLM_SOURCE_EXCLUDED") and before == state(), "error": code})
    report = {"acceptance": "author-checked", "scope": "known-truth refusal controls, no training data", "checks": checks,
              "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
