"""Read back real scripture coordinate joins against independent canonical fields.

This proves only exact coordinate agreement in two captured Genesis prefixes.
It is not a semantic-equivalence or general versification-resolution claim.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from senses.adapters import atom, prepare
from senses.contract import canonical, digest
from senses.pipeline import file_hash, read_bound
from senses.referents import ReferentMap
from senses.store import Store


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {n: file_hash(root / n) for n in ("verse_index.npz", "verse_keys.jsonl", "scripture-population.json")}
    metadata = json.loads((root / "scripture-population.json").read_text())
    keys = {r["key_id"]: r for r in map(json.loads, (root / "verse_keys.jsonl").read_text().splitlines())}
    capture = [r for r in json.loads((root / "capture.json").read_text()) if r["source"] in ("unfoldingword-uhb", "unfoldingword-ult")]
    selected = {(r["source"], r["shard"]): r for r in capture}
    shard_ids = {i: (s["source"], s["shard"]) for i, s in enumerate(metadata["shards"]) if (s["source"], s["shard"]) in selected}
    mapping_rows = []
    with np.load(root / "verse_index.npz", allow_pickle=False) as z:
        mask = np.isin(z["shard"], list(shard_ids)) & (z["row"] < 64)
        for k, s, r in zip(z["key_id"][mask], z["shard"][mask], z["row"][mask]):
            source, shard = shard_ids[int(s)]
            key = keys[int(k)]
            mapping_rows.append({"source": source, "shard": shard, "row": int(r), "status": "resolved",
                                 "key": f"scripture-coordinate:{key['book']}.{key['chapter']}.{key['verse']}",
                                 "evidence": {"index_sha256": hashes["verse_index.npz"], "key_id": int(k)}})
    artifact = {"schema": "ucr-referents-v1", "rows": mapping_rows}
    resolver = ReferentMap(artifact, digest(artifact))
    gold, records = [], []
    for cap in capture:
        for e in read_bound(root / (cap["source"] + ".jsonl"), cap["sha256"]):
            # Gold is independently read from canonical fields, never the index.
            f = e["record"]["f"]
            gold.append({"source": cap["source"], "shard": cap["shard"], "row": e["record"]["row"],
                         "key": f"scripture-coordinate:{f['book']}.{f['chapter']}.{f['verse']}"})
            records.append(atom(resolver.apply(prepare(e))))
    score = resolver.evaluate(gold)
    (out / "mapping.json").write_bytes(canonical(artifact))
    (out / "gold.json").write_bytes(canonical(gold))
    (out / "records.jsonl").write_bytes(b"".join(canonical(r) + b"\n" for r in records))
    s = Store(out / "referents.sqlite", create=True)
    split = [r for r in records if r["locator"]["source"] == "unfoldingword-uhb"]
    second = [r for r in records if r["locator"]["source"] == "unfoldingword-ult"]
    base = s.ingest(split, {"count": len(split), "input_sha256": digest(split)}, None)["generation"]
    empty_before = all(s.context(r, "co_reference", base) == [] for r in split)
    final = s.ingest(second, {"count": len(second), "input_sha256": digest(second)}, base)["generation"]
    s.close()
    cold = Store(out / "referents.sqlite")
    links = [cold.context(r, "co_reference", final) for r in split]
    expected = {(r["locator"]["row"]): r["atom_id"] for r in second}
    resolved = {(r["source"], r["shard"], r["row"]) for r in mapping_rows}
    def expected_links(r):
        row = r["locator"]["row"]
        return [expected[row]] if all((s, "01-GEN", row) in resolved for s in ("unfoldingword-uhb", "unfoldingword-ult")) else []
    exact = all([x["atom_id"] for x in xs] == expected_links(r) for r, xs in zip(split, links))
    blocked = all(cold.context(r, "co_reference", final, [xs[0]["unit"]]) == [] for r, xs in zip(split, links) if xs)
    history = all(cold.context(r, "co_reference", base) == [] for r in split)
    corrupt = copy.deepcopy(artifact)
    corrupt["rows"][0]["key"] = "scripture-coordinate:WRONG.0.0"
    wrong = ReferentMap(corrupt, digest(corrupt)).evaluate(gold)
    checks = {"resolved_canonical_gold_agreement": score["correct"] > 0 and score["wrong"] == 0,
              "before_no_other_edition": empty_before, "after_exact_cross_source_links": exact,
              "old_generation_unchanged": history, "label_unit_exclusion": blocked, "wrong_key_detected": wrong["wrong"] == 1}
    report = {"acceptance": "author-checked", "scope": __doc__, "inputs_sha256": hashes, "coordinate_gold": score,
              "unresolved_gold": [r for r in gold if (r["source"], r["shard"], r["row"]) not in resolved],
              "coverage_complete": score["unresolved"] == 0,
              "checks": checks, "passed": sum(checks.values()), "failed": sum(not c for c in checks.values()), "readback": cold.audit()}
    cold.close()
    (out / "report.json").write_bytes(canonical(report) + b"\n")
    (out / "SHA256.json").write_bytes(canonical({p.name: file_hash(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({"passed": report["passed"], "failed": report["failed"], "coordinate_gold": score}))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
