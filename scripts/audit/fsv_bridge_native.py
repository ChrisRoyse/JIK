"""Run the complete retained physical-observation pilot through Stages 1/2 APIs.

Linear typed-byte passes and indexed SQLite writes; no pixel decoding, embeddings,
or new acquisitions. Canonical/atom builders perform full cold closure readbacks.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "canon"),
               str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import canonical, require, digest
from senses.pipeline import file_hash
from native_population import build as canonicalize
from native_atoms import build as atomize, open_atoms
from training_scope import POLICY_SHA256


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    path = a.export / "observations.ucr.jsonl"
    manifest_path = a.export / "observations.ucr.manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest["count"] == 23702 and manifest["input"]["adapter"] == "bridge-observations-v2",
            "MNLM_BRIDGE_NATIVE_SCOPE", "complete v2 export required")
    plan = {"schema": "native-canon-plan-v1", "training_scope_sha256": POLICY_SHA256,
            "source_rows": {"bridge-v2-rlds": manifest["count"]}, "exports": [{"name": "bridge-physical-v2",
            "path": str(path.resolve()), "manifest": str(manifest_path.resolve()), "manifest_sha256": file_hash(manifest_path),
            "sha256": manifest["sha256"], "bytes": path.stat().st_size, "count": manifest["count"]}]}
    (a.out / "plan.json").write_bytes(canonical(plan))
    first = canonicalize(plan, a.out / "population")
    print(json.dumps({"canonicalization": first}), flush=True)
    second = atomize(a.out / "population", first["manifest_sha256"], a.out / "atoms")
    print(json.dumps({"atomization": second}), flush=True)
    # Cold physical census from independent connections, after builder handles close.
    with sqlite3.connect((a.out / "population/native.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        counts = {"observations": db.execute("SELECT count(*) FROM observations").fetchone()[0],
                  "units": db.execute("SELECT count(DISTINCT unit) FROM observations").fetchone()[0],
                  "perception_payloads": db.execute("SELECT count(*) FROM payloads WHERE channel='perception'").fetchone()[0]}
    with sqlite3.connect((a.out / "atoms/lineage.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        counts["atoms"] = db.execute("SELECT count(*) FROM lineage").fetchone()[0]
        counts["coordinate_units"] = db.execute("SELECT DISTINCT span_unit FROM lineage").fetchall()
        counts["order_edges"] = db.execute("SELECT count(*) FROM edges").fetchone()[0]
    require(counts["observations"] == counts["atoms"] == counts["perception_payloads"] == 23702
            and counts["units"] == 416 and counts["coordinate_units"] == [("step",)]
            and counts["order_edges"] == 23702 - 622, "MNLM_BRIDGE_NATIVE_READBACK", "typed observation/step conservation failed")
    atoms = open_atoms(a.out / "atoms", second["manifest_sha256"])
    require(file_hash(path) == manifest["sha256"], "MNLM_BRIDGE_NATIVE_SOURCE", "source changed")
    report = {"acceptance": "author-checked only", "counts": counts,
              "source_sha256": manifest["sha256"], "population_manifest_sha256": first["manifest_sha256"],
              "atoms_manifest_sha256": second["manifest_sha256"], "phase_coverage": [first["phase_coverage"], second["phase_coverage"]],
              "stage2_scope_complete": atoms["stage2_scope_complete"], "full_stage2_complete": False,
              "stage3_admitted": False, "stage4_accepted": False}
    (a.out / "report.json").write_bytes(canonical(report))
    print(json.dumps(report), flush=True)
