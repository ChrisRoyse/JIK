"""Panels for the whole-model impact test (#12/#25), from the whole-panel assays (rules predeclared on #12).

A = the 12 pilot senses the predictor was built on. B = A plus every sense in the final panel of any family
whose nulls passed (panel-assay.json). Prints B-minus-A split by kind as two comma lists (dense | sparse+rater)
and writes impact-panels.json beside the assays.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lens"))
from lens.catalogue import by_name
from senses.trajectory_population import publish

A = ("e5-base-v2,all-mpnet-base-v2,gte-modernbert-base,qwen3-embedding-0.6b,pubmedbert-embeddings,medembed-base,"
     "legal-bert-mean,bge-financial-matryoshka,finbert-mean,specter2-base,allenai-specter,unixcoder-mean").split(",")


def main():
    root = Path(sys.argv[1])
    fams = {}
    for f in sorted(root.glob("*/panel-assay.json")):
        r = json.loads(f.read_bytes())
        fams[f.parent.name] = {"final_panel": r["final_panel"], "nulls_pass": r["nulls_pass"]}
    if len(fams) != 5:
        raise SystemExit(f"MNLM_IMPACT_FAMILIES: expected 5 panel assays under {root}, found {sorted(fams)}")
    extra = sorted({n for v in fams.values() if v["nulls_pass"] for n in v["final_panel"]} - set(A))
    dense = [n for n in extra if by_name(n)["kind"] == "dense"]
    other = [n for n in extra if by_name(n)["kind"] in ("sparse", "rater")]
    B = A + dense + other
    rec = {"schema": "impact-panels-v1", "families": fams, "A": A, "B": B, "extra_dense": dense, "extra_catalogue": other}
    out = root / "impact-panels.json"
    if out.exists():  # idempotent re-run: the persisted panels must equal the recomputed ones, else fail closed
        old = json.loads(out.read_bytes())
        if {k: old.get(k) for k in rec} != rec:
            raise SystemExit(f"MNLM_IMPACT_PANELS_DIFFER: {out} disagrees with the assays it was built from")
    else:
        publish(out, rec)
    print(",".join(dense) or "-", ",".join(other) or "-", ",".join(B))


if __name__ == "__main__":
    main()
