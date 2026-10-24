"""Cold-check native guard readings and all frozen identity/source/batch pins."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import check, sha
from audit.guard_render_readback import lines
from senses.contract import digest
from senses.trajectory_population import publish


def read(root, pin, plan, renders):
    check(sha(root / "report.json") == pin, "GUARD_MEASURE_PIN")
    rep = json.loads((root / "report.json").read_bytes())
    check(rep["schema"] == "strict-guard-measurements-v1" and all(rep[k] is False for k in
          ("scientific_admission", "guard_evaluated", "historical_identity_equivalence_certified",
           "independent_anchors_evaluated")), "GUARD_MEASURE_CLAIM")
    check(sha(plan / "report.json") == rep["plan_sha256"], "GUARD_MEASURE_PLAN_PIN")
    prepared = json.loads((plan / "report.json").read_bytes())
    check(rep["renders_sha256"] == prepared["renders_sha256"] == sha(renders / "report.json")
          and rep["texts_sha256"] == prepared["texts_sha256"] == sha(renders / "texts.jsonl"), "GUARD_MEASURE_SOURCE")
    names = rep["lenses"]
    check(names == prepared["lenses"] and len(names) == len(set(names)) == 12
          and set(rep["identities"]) == set(rep["readings"]) == set(names)
          and rep["runtime_fork_declared"] == prepared.get("runtime_fork_declared", False), "GUARD_MEASURE_LANES")
    files = {"gpu-before.txt"} | {name + suffix for name in names for suffix in (".npy", ".valid.npy")}
    check(set(rep["files"]) == files, "GUARD_MEASURE_FILES")
    for name, meta in rep["files"].items():
        check(sha(root / name) == meta["sha256"] and (root / name).stat().st_size == meta["bytes"], "GUARD_MEASURE_FILE_PIN")
    texts = lines(renders / "texts.jsonl")
    result = {}
    for name in names:
        item, ident, record = prepared["instruments"][name], rep["identities"][name], rep["readings"][name]
        check(sha(plan / (name + ".json")) == item["token_plan_sha256"], "GUARD_MEASURE_TOKEN_PIN")
        tokens = json.loads((plan / (name + ".json")).read_bytes())
        check([r["text_id"] for r in tokens] == [r["id"] for r in texts], "GUARD_MEASURE_TOKEN_ORDER")
        check(all(ident[k] == item[k] for k in ("recipe", "revision", "weights_sha256", "preprocessing_sha256"))
              and ident["name"] == name and ident["runtime"] == prepared["runtime"]
              and ident["execution_code_sha256"] == rep["source_code_sha256"]
              and ident["float32_repeated_forward"] is True and ident["deterministic_algorithms"] is True
              and ident["tf32"] is False and ident["forward_dtype"] == "float32" and ident["stored_dtype"] == "float16"
              and ident["historical_identity_equivalence_certified"] is False
              and record["identity_sha256"] == digest(ident), "GUARD_MEASURE_IDENTITY")
        slots = np.load(root / (name + ".npy"), allow_pickle=False)
        valid = np.load(root / (name + ".valid.npy"), allow_pickle=False)
        check(slots.shape == (len(texts), item["recipe"]["dims"]) and slots.dtype == np.float16
              and valid.shape == (len(texts),) and valid.dtype == np.bool_, "GUARD_MEASURE_SHAPE")
        check(np.isfinite(slots).all() and np.all(slots[~valid] == 0)
              and np.all(abs(np.linalg.norm(slots[valid].astype(np.float64), axis=1) - 1) < .001), "GUARD_MEASURE_VALUES")
        check(record["rows"] == len(texts) and record["measured"] == int(valid.sum())
              and len(record["statuses"]) == len(texts), "GUARD_MEASURE_COUNTS")
        for t, ok, status in zip(tokens, valid, record["statuses"]):
            expected = "measured" if ok else ("MNLM_GUARD_NONFINITE_OR_ZERO" if t["status"] == "ready" else t["status"])
            check(status == expected and (not ok or t["status"] == "ready"), "GUARD_MEASURE_STATUS")
        batches = record["batches"]
        flattened = [i for batch in batches for i in batch]
        ready = [i for i, t in enumerate(tokens) if t["status"] == "ready"]
        check(sorted(flattened) == ready and digest(batches) == ident["batch_geometry_sha256"], "GUARD_MEASURE_COVERAGE")
        check(all(0 < len(batch) <= 256 and max(tokens[i]["tokens"] for i in batch) * len(batch)
                  <= item["recipe"]["batch_token_budget"] for batch in batches), "GUARD_MEASURE_BUDGET")
        probes = sorted({0, len(batches) // 2, len(batches) - 1})
        check(len(record["replays"]) == len(probes), "GUARD_MEASURE_REPLAY")
        for b, replay in zip(probes, record["replays"]):
            indices = batches[b]
            good = np.asarray(indices)[valid[indices]]
            check(replay["rows"] == indices and replay["byte_identical"] is True
                  and replay["stored_sha256"] == hashlib.sha256(slots[good].tobytes()).hexdigest()
                  and len(replay["fp32_sha256"]) == 64, "GUARD_MEASURE_REPLAY")
        result[name] = {"rows": len(texts), "measured": int(valid.sum()), "replay_batches": len(probes),
                        "identity_sha256": record["identity_sha256"]}
    check(.9 <= rep["phase_coverage"] <= 1.001 and abs(sum(rep["phases_seconds"].values()) / rep["wall_seconds"]
          - rep["phase_coverage"]) < 1e-10, "GUARD_MEASURE_TIMING")
    return {"author_checked": True, "report_sha256": pin, "lenses": result,
            "scientific_admission": False, "historical_identity_equivalence_certified": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("root", "plan", "renders", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--report-sha256", required=True)
    a = ap.parse_args()
    result = read(a.root, a.report_sha256, a.plan, a.renders)
    publish(a.output, result)
    print(json.dumps(result))
