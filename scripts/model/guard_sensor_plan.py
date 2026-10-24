"""CPU-only full-token and frozen-asset plan for historical-pilot guard readback.

Native instrument slots remain separate. An over-window input is unavailable for
that lane; no text is truncated, replaced or silently assigned another runtime.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from lens.learned import snapshot_dir, weights_sha256, preprocessing_sha256
from lens.catalogue import by_name
from audit.renderer_population_readback import sha, check
from audit.guard_render_readback import read, lines
from senses.contract import canonical
from senses.trajectory_population import publish


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("renders", "population", "comparison", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--renders-sha256", required=True)
    ap.add_argument("--declare-runtime-fork", action="store_true",
                    help="Create a diagnostic identity for this runtime; never historical equivalence")
    a = ap.parse_args()
    check(not a.output.exists(), "GUARD_PLAN_OUTPUT_EXISTS")
    start = time.perf_counter(); phases = {}
    read(a.renders, a.renders_sha256, a.population, a.comparison)
    pop = json.loads((a.population / "report.json").read_bytes())
    historical = json.loads((a.population / "slot-manifest.json").read_bytes())
    check(sha(a.population / "slot-manifest.json") == pop["files"]["slot-manifest.json"]["sha256"], "GUARD_SOURCE_INSTRUMENT_PIN")
    import torch
    import transformers
    from transformers import AutoTokenizer
    matched = str(torch.__version__) == historical["torch"]
    check(matched or a.declare_runtime_fork, "GUARD_SENSOR_RUNTIME")
    texts = lines(a.renders / "texts.jsonl")
    check(0 < len(texts) <= 2304 and len({x["id"] for x in texts}) == len(texts), "GUARD_PLAN_TEXT_COUNT")
    phases["source_and_runtime"] = time.perf_counter() - start
    a.output.mkdir(parents=True)
    instruments = {}
    for name in pop["lenses"]:
        t = time.perf_counter()
        spec, old = by_name(name), historical["lenses"][name]
        check(all(spec.get(k, "") == old[k] for k in ("repo", "pool", "prefix", "max_len")), "GUARD_SENSOR_RECIPE")
        snap = snapshot_dir(spec["repo"])
        weight_pin, weight_files = weights_sha256(snap)
        prep = preprocessing_sha256(snap)
        check(snap.name == old["revision"] and weight_pin == old["weights_sha256"], "GUARD_SENSOR_WEIGHTS")
        tokenizer = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False, local_files_only=True)
        if spec["pool"] == "last":
            tokenizer.padding_side = "left"
        phases[name + "_assets"] = time.perf_counter() - t
        t = time.perf_counter()
        encoded = tokenizer([spec.get("prefix", "") + x["text"] for x in texts], padding=False, truncation=False,
                            return_attention_mask=True, return_length=True)
        counts = encoded.pop("length")
        keys = sorted(set(encoded) & {"input_ids", "attention_mask", "token_type_ids"})
        check("input_ids" in keys and "attention_mask" in keys and len(counts) == len(texts), "GUARD_SENSOR_TOKENS")
        rows = []
        for i, item in enumerate(texts):
            n = len(encoded["input_ids"][i])
            check(n == counts[i] and n > 0 and all(len(encoded[k][i]) == n for k in keys)
                  and encoded["attention_mask"][i] == [1] * n, "GUARD_SENSOR_COMPLETE_INPUT")
            rows.append({"text_id": item["id"], "tokens": n,
                         "status": "ready" if n <= spec["max_len"] else "MNLM_GUARD_LENS_WINDOW",
                         "inputs": {k: encoded[k][i] for k in keys}})
        (a.output / (name + ".json")).write_bytes(canonical(rows))
        instruments[name] = {"recipe": {k: old[k] for k in ("repo", "pool", "prefix", "max_len", "dims", "batch_token_budget")},
                             "revision": snap.name, "weights_sha256": weight_pin, "weight_files": weight_files,
                             "preprocessing_sha256": prep, "padding_side": tokenizer.padding_side,
                             "token_plan_sha256": sha(a.output / (name + ".json")), "rows": len(rows),
                             "tokens": sum(counts), "over_window": sum(r["status"] != "ready" for r in rows)}
        phases[name + "_token_plan"] = time.perf_counter() - t
        print(json.dumps({"name": name, "rows": len(rows), "tokens": sum(counts), "over_window": instruments[name]["over_window"]}), flush=True)
    wall = time.perf_counter() - start
    report = {"schema": "strict-guard-sensor-plan-v1", "renders_sha256": a.renders_sha256,
              "texts_sha256": sha(a.renders / "texts.jsonl"), "historical_manifest_sha256": sha(a.population / "slot-manifest.json"),
              "lenses": pop["lenses"], "instruments": instruments,
              "runtime": {"torch": str(torch.__version__), "transformers": str(transformers.__version__), "platform": sys.platform,
                          "cuda": torch.version.cuda, "historical_torch_matched": matched},
              "historical_torch": historical["torch"], "runtime_fork_declared": a.declare_runtime_fork,
              "source_code_sha256": sha(Path(__file__)), "gpu_used": False, "measurements_complete": False,
              "scientific_admission": False, "historical_runtime_provenance_complete": False,
              "limitations": ["historical manifest does not pin all preprocessing/runtime dependencies",
                              "fresh state/repeated-forward identity and historical source replay must be measured before interpreting scores"],
              "phases_seconds": phases, "wall_seconds": wall, "phase_coverage": sum(phases.values()) / wall}
    publish(a.output / "report.json", report)
    print(json.dumps({"report_sha256": sha(a.output / "report.json"), "wall_seconds": wall, "phase_coverage": report["phase_coverage"]}))


if __name__ == "__main__":
    main()
