"""Cold byte/shape/provenance readback for the learned visual commissioning run.

Only four-row boundary artifacts and the declared 416-row feature selection are
read. No outcome metric is calculated and no instrument earns scientific admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
import numpy as np
from senses.contract import canonical, require, digest
from senses.pipeline import file_hash
from production.publication import Publication
from production.native_rows import mapped_rows, one, value_hash
from vision_encoder import SPEC


def inventory(root):
    seal = json.loads((root / "SHA256.json").read_bytes())
    for name, expected in seal.items():
        p = (root / name).resolve()
        require(p.is_relative_to(root.resolve()) and file_hash(p) == expected,
                "MNLM_VISION_READBACK", "sealed boundary file changed")
    return len(seal)


def publication(root):
    pub = Publication(root)
    counts, values, visual_bytes, visual_rows, rows = {}, {}, 0, 0, set()
    try:
        state = pub.verify()
        for key in sorted(pub.plan["shards"]):
            shard, _ = pub.read(key)  # physical blobs, native layout, contract, bindings and chain
            source_rows = [json.loads(x) for x in pub.blob(shard["rows_sha256"]).read_bytes().splitlines()]
            rows.update(x["atom"] for x in source_rows)
            for name, slot in shard["slots"].items():
                counts[name] = counts.get(name, 0) + len(source_rows)
                with mapped_rows(pub.blob(slot["sha256"])) as arrays:
                    h = value_hash(arrays)
                    values[key + "/" + name] = h
                    if name == SPEC["name"]:
                        require(slot["contract"]["channel"] == "perception" and slot["seal"]["dims"] == 384,
                                "MNLM_VISION_READBACK", "visual instrument misdeclared")
                        for i, source in enumerate(source_rows):
                            row = one(arrays, i)
                            require(row["tokens"].shape == (362, 384) and not row["absent"]
                                    and np.isfinite(row["tokens"]).all(), "MNLM_VISION_READBACK", "native patches lost")
                            visual_bytes += row["tokens"].nbytes
                            visual_rows += 1
        require(state["shards"] == len(pub.plan["shards"]), "MNLM_VISION_READBACK", "incomplete publication")
        return {"state": state, "rows": len(rows), "counts": counts, "native_digests": values,
                "visual_value_bytes": visual_bytes, "visual_rows": visual_rows,
                "source_ids_sha256": digest(sorted(rows))}
    finally:
        pub.close()


def run(boundary, native, out):
    before = inventory(boundary)
    report = json.loads((boundary / "report.json").read_bytes())
    require(report["passed"] == len(report["checks"]) and all(c["pass"] for c in report["checks"])
            and not report["stage3_admitted"] and not report["stage4_accepted"],
            "MNLM_VISION_READBACK", "incorrect boundary verdict")
    first = publication(boundary / "publication/publication")
    cached = publication(boundary / "cached-publication/publication")
    require(first["native_digests"] == cached["native_digests"] and first["visual_rows"] == 4
            and len(first["counts"]) == 7, "MNLM_VISION_READBACK", "cached native values differ")
    witness = np.load(boundary / "visual-witness.npy", allow_pickle=False)
    sha = hashlib.sha256(witness.tobytes()).hexdigest()
    declared = next(c for c in report["checks"] if c["name"] == "native_visual_determinism")
    require(sha == declared["digest"] and list(witness.shape) == declared["shape"],
            "MNLM_VISION_READBACK", "witness differs from declared measurement")
    measured = publication(native / "native/publication")
    require(measured["visual_rows"] == measured["rows"] == 416 and len(measured["counts"]) == 5
            and set(measured["counts"].values()) == {416} and measured["visual_value_bytes"] == 231309312,
            "MNLM_VISION_READBACK", "feature selection is incomplete")
    consumer = json.loads((native / "consumer.json").read_bytes())
    require(consumer["visual_features_only"] and not consumer["new_outcome_evaluation"]
            and not consumer["stage3_admitted"] and not consumer["stage4_accepted"],
            "MNLM_VISION_READBACK", "feature run was incorrectly promoted")
    require(before == inventory(boundary), "MNLM_VISION_READBACK", "readback mutated sealed evidence")
    result = {"status": "author-checked", "boundary_sealed_files": before, "boundary_checks": report["passed"],
              "witness_sha256": sha, "boundary": first, "measurement": measured,
              "stream": json.loads((native / "stream.json").read_bytes()), "consumer": consumer,
              "semantic_understanding_demonstrated": False, "stage3_admitted": False, "stage4_accepted": False}
    require(not out.exists(), "MNLM_VISION_READBACK", "do not overwrite existing readback evidence")
    out.write_bytes(canonical(result))
    print(json.dumps({"out": str(out), "sha256": file_hash(out), "boundary_checks": report["passed"],
                      "rows": measured["rows"], "visual_bytes": measured["visual_value_bytes"]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for n in ("boundary", "native", "out"):
        ap.add_argument("--" + n, required=True, type=Path)
    a = ap.parse_args()
    run(a.boundary, a.native, a.out)
