"""Exercise real Stage 2 atomisation -> UCR export -> Stage 3 slot consumer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import blake3
import numpy as np
import zstandard

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens"),
                str(Path(__file__).resolve().parents[1] / "canon")]
from atomise import atomise_shard
from senses.contract import canonical, digest
from senses.pipeline import export_canonical, file_hash, ingest_file, read_bound, write_records
from senses.store import Store
from senses.instruments import contracts
from universal import measure_universal, coverage_report
from slotio import slot_digest
from lenscommon import Paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    capture = json.loads((Path(args.input) / "capture.json").read_text())
    checks, total = [], 0
    store = Store(root / "integration.sqlite", create=True)
    cal = root / "lenses" / "integration" / "cal"
    cal.mkdir(parents=True)
    policies, fams = {}, {}
    for source, family, count in (("wikidata5m", "a2-wikidata-p31", 64), ("wikipedia-en-20231101", "integration-prose", 1)):
        item = next(x for x in capture if x["source"] == source)
        envelopes = list(read_bound(Path(args.input) / (source + ".jsonl"), item["sha256"]))[:count]
        shard = item["shard"]
        canon_dir, atoms_dir = root / "canon" / source, root / "atoms" / source
        canon_dir.mkdir(parents=True)
        path = canon_dir / (shard + ".jsonl.zst")
        with path.open("wb") as f, zstandard.ZstdCompressor().stream_writer(f) as zw:
            for e in envelopes:
                zw.write(canonical(e["record"]) + b"\n")
        manifest = {**envelopes[0]["source_manifest"], "out_blake3": blake3.blake3(path.read_bytes()).hexdigest(),
                    "docs": count, "out_bytes": path.stat().st_size, "scope": "captured prefix for integration proof", "parent_capture_sha256": item["sha256"]}
        am = atomise_shard({"source": source, "shard": shard, "canon_dir": str(canon_dir), "out_dir": str(atoms_dir),
                           "unique_rows": [e["record"]["row"] for e in envelopes], "cuts": {}})
        atoms_file = atoms_dir / (shard + ".atoms.npz")
        plan = {"shards": [{"source": source, "shard": shard, "canon": str(path), "atoms": str(atoms_file),
                            "atoms_sha256": file_hash(atoms_file), "manifest": manifest, "atom_manifest": am}]}
        export = root / (source + ".ucr.jsonl")
        em = write_records(export, export_canonical(plan), {"plan": plan})
        ing = ingest_file(store, export, em, store.head)
        checks.append({"name": "stage2_atom_count_" + source, "pass": ing["added"] == am["atoms"], "atoms": am["atoms"]})
        total += am["atoms"]
        records = list(read_bound(export, em["sha256"]))
        # Small explicit proof scope for the Stage 3 interface, not a powered assay.
        selected = records[:64]
        rows = [{"i": i, "source": source, "shard": shard, "row": r["locator"]["row"], "span": r["locator"]["span"],
                 "text": r["channels"]["surface"]["items"][0]["text"] if r["channels"]["surface"]["items"] else "",
                 "family": family, "y": 0, "y_name": "boundary-only", "cluster": r["unit"]} for i, r in enumerate(selected)]
        (cal / (family + ".jsonl")).write_bytes(b"".join(canonical(r) + b"\n" for r in rows))
        fams[family] = rows
        policies[family] = {"schema": "ucr-view-v1", "question": family, "allow_channels": ["surface", "fields", "structure", "provenance"],
                            "deny_fields": [], "deny_relations": [], "blocked_units": [r["unit"] for r in records],
                            "label_aliases": [], "rationale": "interface boundary proof only, no information claim"}
    policy_path = root / "policies.json"
    policy_path.write_bytes(canonical(policies))
    store.close()
    registry = {"lenses": {}}
    for spec in contracts():
        ent = measure_universal(spec, fams, root / "lenses" / "integration" / "slots", {}, root / "integration.sqlite", policy_path)
        ent["spec"] = spec
        registry["lenses"][spec["name"]] = ent
        for fam, result in ent["families"].items():
            cold = slot_digest(Path(result["slot"]))
            checks.append({"name": "stage3_readback_" + spec["name"] + "_" + fam, "pass": cold["digest"] == result["digest"] and cold["n"] == len(fams[fam])})
    cov = coverage_report(registry, {q: {"n": len(r)} for q, r in fams.items()})
    checks.append({"name": "coverage_does_not_certify", "pass": not cov["served"] and all(q["atoms_ge5_admitted"] == 0 for q in cov["questions"].values())})
    # A second registered text instrument is a second available instrument; only the contract admits it as a sense.
    first = registry["lenses"]["ucr-surface-v1"]
    registry["lenses"]["second-text-embedding-control"] = first
    check_cov = coverage_report(registry, {q: {"n": len(r)} for q, r in fams.items()})
    checks.append({"name": "second_text_instrument_available_not_admitted", "pass": not check_cov["admitted"] and not check_cov["served"]
                   and all(q["atoms_ge5_admitted"] == 0 for q in check_cov["questions"].values())})
    checks.append({"name": "empty_channel_not_registered", "pass": registry["lenses"]["ucr-perception-v1"].get("error") == "MNLM_UCR_CHANNEL_UNOBSERVED"})
    from learned_ucr import expand_rows, specs as surface_specs
    from catalogue import LEARNED
    # These are known-layout boundary controls, not learned-model measurements.
    native = np.asarray([[1, 2], [3, 4], [5, 6]], np.float32)
    expanded = expand_rows((native, np.asarray([0, 1, 3])), np.asarray([0, 2]), 4, "multi", 2)
    checks.append({"name": "learned_multi_preserves_absence", "pass": expanded["offsets"].tolist() == [0, 1, 1, 3, 3]
                   and expanded["absent"].tolist() == [False, True, False, True] and np.array_equal(expanded["tokens"], native)})
    expanded = expand_rows(native[:2], np.asarray([0, 2]), 4, "dense", 2)
    checks.append({"name": "learned_dense_preserves_absence", "pass": expanded["absent"].tolist() == [False, True, False, True]
                   and np.array_equal(expanded["X"][[0, 2]], native[:2])})
    checks.append({"name": "every_learned_model_reads_surface_channel", "pass": {s["perspective"] for s in surface_specs(LEARNED)} == {"surface"}})
    report = {"acceptance": "author-checked", "metric_kind": "Stage 2/3 integration boundary, not admission", "atoms": total,
              "checks": checks, "passed": sum(c["pass"] for c in checks), "failed": sum(not c["pass"] for c in checks), "coverage": cov}
    (root / "report.json").write_bytes(canonical(report) + b"\n")
    (root / "registry.json").write_bytes(canonical(registry) + b"\n")
    (root / "SHA256.json").write_bytes(canonical({str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file() and not p.name.endswith(("-wal", "-shm"))}))
    print(json.dumps(report))
    return int(report["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
