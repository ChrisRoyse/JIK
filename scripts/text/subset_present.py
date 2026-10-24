"""Graph-present subset of a split-property anchor (a2-domain-split-v4-present, #12/#15).

Rule (predeclared before any v4 result): keep the rows whose graph-sense reading is not the typed-absence
vector (absence indicator, the last dimension, is 0). Every slot file is cut to those rows in the same order, so
D1 is recomputed on this population and record_assay runs unchanged. The all-rows v4 assay is run as well;
both are reported.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish


def main():
    src, dst, slots_src, slots_dst = map(Path, sys.argv[1:5])
    kg = sys.argv[5]
    summ = json.loads((src / "summary.json").read_bytes())
    rows = [json.loads(l) for l in (src / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    G = np.load(slots_src / f"{kg}.npy")
    if len(G) != len(rows):
        raise SystemExit(f"MNLM_ANCHOR_ROWS: {kg} has {len(G)} rows, anchor {len(rows)}")
    keep = np.flatnonzero(G[:, -1] < 0.5)
    dst.mkdir(parents=True, exist_ok=False); slots_dst.mkdir(parents=True, exist_ok=False)
    with (dst / "rows.jsonl").open("x", encoding="utf-8") as fh:
        for i in keep:
            fh.write(json.dumps(rows[i], sort_keys=True) + "\n")
    pq.write_table(pq.read_table(src / "atoms.parquet").take(keep), dst / "atoms.parquet")
    for f in sorted(slots_src.glob("*.npy")):
        np.save(slots_dst / f.name, np.load(f)[keep])
    publish(dst / "summary.json", {**summ, "anchor_version": summ["anchor_version"] + "-present", "parent_rows_sha256": summ["rows_sha256"],
        "subset_rule": f"{kg} absence indicator == 0", "rows": int(len(keep)), "parent_rows": len(rows),
        "rows_sha256": hashlib.sha256((dst / "rows.jsonl").read_bytes()).hexdigest(),
        "subset_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"family": summ.get("family"), "rows": int(len(keep)), "parent_rows": len(rows)}))


if __name__ == "__main__":
    main()
