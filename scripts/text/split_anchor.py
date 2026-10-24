"""Split-property A2 anchor (a2-domain-split-v4, #12/#15): one half of an entity's record is the anchor, the
other half is input that a graph sense may read (rules predeclared before any reading).

From an a2-domain-record-v3 family: the anchor keeps the properties at ODD frequency ranks (1st, 3rd, ...,
19th) of that family's 20 anchor properties; the even-ranked properties (and every rarer property) are not
labels and may be read by the graph sense. Rows, texts and order are unchanged, so the text embedders'
slots are the same files (hard-linked, not recomputed; their hashes are re-checked).
"""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.trajectory_population import publish


def main():
    src, dst, slots_src, slots_dst = (Path(x) for x in sys.argv[1:5])
    summ = json.loads((src / "summary.json").read_bytes())
    props = summ["properties"]
    anchor, inputs = props[0::2], props[1::2]
    dst.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(l) for l in (src / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    with (dst / "rows.jsonl").open("x", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "y": {p: r["y"][p] for p in anchor}}, sort_keys=True) + "\n")
    os.link(src / "atoms.parquet", dst / "atoms.parquet")
    publish(dst / "summary.json", {**{k: v for k, v in summ.items() if k not in ("properties", "rows_sha256", "values", "present_per_property")},
        "anchor_version": "a2-domain-split-v4", "parent_rows_sha256": summ["rows_sha256"], "properties": anchor,
        "graph_input_properties_ranked": inputs, "values": {p: summ["values"][p] for p in anchor},
        "present_per_property": {p: summ["present_per_property"][p] for p in anchor},
        "rows_sha256": hashlib.sha256((dst / "rows.jsonl").read_bytes()).hexdigest(),
        "split_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    slots_dst.mkdir(parents=True, exist_ok=False)
    for f in sorted(slots_src.glob("*.npy")):
        if f.name.startswith("kg-"):
            continue  # graph senses depend on the masked set; recomputed for this anchor
        os.link(f, slots_dst / f.name)
    print(json.dumps({"family": summ.get("family"), "anchor": anchor, "inputs": inputs, "rows": len(rows)}))


if __name__ == "__main__":
    main()
