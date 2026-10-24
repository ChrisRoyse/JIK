"""Real >=500 calibration path plus corrupt-copy boundary controls (#12)."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calibration_readback import readback, sha

PRODUCER = Path(__file__).parents[1]/"text/calibration_population.py"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "contract", "publication", "output"):
        p.add_argument("--"+arg, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    def check(name, ok):
        results.append({"name": name, "pass": bool(ok)})
    def put(path, value):
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    def rejects(name, code, fn):
        try:
            fn(); ok = False
        except ValueError as exc:
            ok = code in str(exc)
        check(name, ok)
    before = {str(p.relative_to(a.publication)): sha(p.read_bytes()) for p in a.publication.rglob("*") if p.is_file()}
    cold = readback(a.source, a.publication, a.contract)
    check("B1 real samples >=500 independently reselected with unchanged classes", cold["ready_questions"] == 3)
    check("B2 every real chunk reconstructs full source", cold["all_source_spans_equal"])
    check("B3 all unavailable questions retained", sum(f["unavailable"] for f in cold["families"].values()) == 12)
    def run(name, config=a.contract, dest=None):
        cp = subprocess.run([sys.executable, "-I", str(PRODUCER), "--source", str(a.source), "--contract", str(config),
                             "--output", str(dest or a.output/name)], capture_output=True, text=True, encoding="utf-8")
        (a.output/(name+".log")).write_text(cp.stdout+cp.stderr, encoding="utf-8")
        return cp
    cp = run("repeat")
    check("B3 actual CLI replay succeeds", cp.returncode == 0)
    repeat = readback(a.source, a.output/"repeat", a.contract)
    check("B3 replay source selections and physical chunks identical", all(
          (a.publication/family/filename).read_bytes() == (a.output/"repeat"/family/filename).read_bytes()
          for family in cold["families"] for filename in (["anchors.json", "chunks/atoms.parquet"]
          if cold["families"][family]["entities"] else ["anchors.json"])))
    cp = run("overwrite", dest=a.publication)
    check("B3 existing generation refuses", cp.returncode != 0 and "MNLM_CALPOP_EXISTS" in cp.stderr)
    config = json.loads(a.contract.read_bytes())
    bad = copy.deepcopy(config); bad["source_report_sha256"] = "0"*64
    path = a.output/"bad-source.json"; put(path, bad)
    cp = run("bad-source", path)
    check("B1 wrong source report refuses", cp.returncode != 0 and "MNLM_CALPOP_SOURCE_REPORT" in cp.stderr)
    bad = copy.deepcopy(config); bad["families"]["biomedical"]["rows_sha256"] = "0"*64
    path = a.output/"bad-rows.json"; put(path, bad)
    cp = run("bad-rows", path)
    report = json.loads((a.output/"bad-rows/report.json").read_bytes())
    check("B1 corrupt family refuses while four siblings continue", cp.returncode == 2 and not report["complete"]
          and len(report["families"]) == 4 and "MNLM_CALPOP_INPUT_HASH" in report["errors"][0]["error"])
    bad = copy.deepcopy(config); bad["families"]["biomedical"]["questions"]["P1057"].reverse()
    path = a.output/"bad-classes.json"; put(path, bad)
    cp = run("bad-classes", path)
    report = json.loads((a.output/"bad-classes/report.json").read_bytes())
    check("B1 reordered class ontology refuses", cp.returncode == 2 and "MNLM_CALPOP_QUESTION_CHANGED" in report["errors"][0]["error"])
    def clone(name):
        dest = a.output/name
        shutil.copytree(a.publication, dest)
        return dest
    def rebind(dest, family, summary):
        put(dest/family/"summary.json", summary)
        rep = json.loads((dest/"report.json").read_bytes())
        rep["families"][family]["summary_sha256"] = sha((dest/family/"summary.json").read_bytes())
        put(dest/"report.json", rep)
    dest = clone("changed-label")
    path = dest/"biomedical/anchors.json"; v = json.loads(path.read_bytes()); v["P1057"]["labels"][0] = (v["P1057"]["labels"][0]+1)%6
    put(path, v)
    sm = json.loads((dest/"biomedical/summary.json").read_bytes()); sm["anchors_sha256"] = sha(path.read_bytes())
    rebind(dest, "biomedical", sm)
    rejects("B3 changed label with recomputed hashes refuses", "MNLM_CALPOP_READBACK_SAMPLE", lambda: readback(a.source, dest, a.contract))
    dest = clone("missing-tail")
    path = dest/"biomedical/chunks/atoms.parquet"; t = pq.read_table(path); pq.write_table(t.slice(0, len(t)-1), path)
    sm = json.loads((dest/"biomedical/summary.json").read_bytes()); sm["chunks_sha256"] = sha(path.read_bytes())
    rebind(dest, "biomedical", sm)
    rejects("B2 deleted tail despite recomputed hashes refuses", "MNLM_CALPOP_READBACK_COVERAGE", lambda: readback(a.source, dest, a.contract))
    dest = clone("false-power")
    sm = json.loads((dest/"code/summary.json").read_bytes()); sm["power"]["P178"]["available"] = True
    rebind(dest, "code", sm)
    rejects("B3 unavailable question promoted without source support refuses", "MNLM_CALPOP_READBACK_POWER", lambda: readback(a.source, dest, a.contract))
    check("B3 original publication bytes unchanged", all(sha((a.publication/name).read_bytes()) == h for name,h in before.items()))
    out = {"checks": results, "passed": sum(x["pass"] for x in results), "total": len(results),
           "cold_readback": cold, "acceptance": "author-checked"}
    put(a.output/"fsv.json", out)
    print(json.dumps({k:out[k] for k in ("passed", "total")}))
    if out["passed"] != out["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
