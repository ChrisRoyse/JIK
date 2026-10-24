"""Re-tokenize every guard input independently and cold-check all instrument pins."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from audit.renderer_population_readback import check, sha
from audit.guard_render_readback import lines
from lens.learned import snapshot_dir, weights_sha256, preprocessing_sha256
from senses.contract import canonical
from senses.trajectory_population import publish


def validate_row(row, text, tokenizer, recipe):
    # Independent per-text call; the producer used a single batch.
    encoded = tokenizer(recipe["prefix"] + text["text"], padding=False, truncation=False)
    expected = {k: list(v) for k, v in encoded.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
    n = len(expected["input_ids"])
    check(row == {"text_id": text["id"], "tokens": n, "inputs": expected,
                  "status": "ready" if n <= recipe["max_len"] else "MNLM_GUARD_LENS_WINDOW"}, "GUARD_TOKEN_COVERAGE")


def main():
    ap = argparse.ArgumentParser()
    for name in ("plan", "renders", "population", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True)
    a = ap.parse_args()
    start = time.perf_counter()
    check(sha(a.plan / "report.json") == a.plan_sha256, "GUARD_SENSOR_PLAN_PIN")
    rep = json.loads((a.plan / "report.json").read_bytes())
    check(rep["schema"] == "strict-guard-sensor-plan-v1" and rep["gpu_used"] is False
          and rep["measurements_complete"] is False and rep["scientific_admission"] is False
          and rep["historical_runtime_provenance_complete"] is False, "GUARD_SENSOR_PLAN_CLAIM")
    check(sha(a.renders / "report.json") == rep["renders_sha256"]
          and sha(a.renders / "texts.jsonl") == rep["texts_sha256"]
          and sha(a.population / "slot-manifest.json") == rep["historical_manifest_sha256"], "GUARD_SENSOR_PLAN_SOURCE")
    hist = json.loads((a.population / "slot-manifest.json").read_bytes())
    pop = json.loads((a.population / "report.json").read_bytes())
    check(rep["lenses"] == pop["lenses"] and set(rep["instruments"]) == set(rep["lenses"]), "GUARD_SENSOR_PLAN_LANES")
    import torch
    import transformers
    from transformers import AutoTokenizer
    matched = str(torch.__version__) == hist["torch"]
    check(rep["runtime"] == {"torch": str(torch.__version__), "transformers": str(transformers.__version__), "platform": sys.platform,
                              "cuda": torch.version.cuda, "historical_torch_matched": matched}
          and rep.get("historical_torch", hist["torch"]) == hist["torch"]
          and (matched or rep.get("runtime_fork_declared") is True), "GUARD_SENSOR_PLAN_RUNTIME")
    texts = lines(a.renders / "texts.jsonl")
    result, negatives, phases = {}, [], {}
    phases["source_and_runtime"] = time.perf_counter() - start
    for name in rep["lenses"]:
        tick = time.perf_counter()
        item, old = rep["instruments"][name], hist["lenses"][name]
        check(item["recipe"] == {k: old[k] for k in ("repo", "pool", "prefix", "max_len", "dims", "batch_token_budget")}, "GUARD_SENSOR_PLAN_RECIPE")
        snap = snapshot_dir(item["recipe"]["repo"])
        weights, files = weights_sha256(snap)
        check(weights == item["weights_sha256"] == old["weights_sha256"] and files == item["weight_files"]
              and snap.name == item["revision"] == old["revision"]
              and preprocessing_sha256(snap) == item["preprocessing_sha256"], "GUARD_SENSOR_PLAN_ASSETS")
        path = a.plan / (name + ".json")
        check(sha(path) == item["token_plan_sha256"], "GUARD_SENSOR_PLAN_TOKENS")
        rows = json.loads(path.read_bytes())
        check(len(rows) == len(texts) == item["rows"], "GUARD_SENSOR_PLAN_ROWS")
        tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=False, local_files_only=True)
        tok.padding_side = item["padding_side"]
        phases[name + "_assets"] = time.perf_counter() - tick
        tick = time.perf_counter()
        for row, text in zip(rows, texts):
            validate_row(row, text, tok, item["recipe"])
        check(sum(r["tokens"] for r in rows) == item["tokens"]
              and sum(r["status"] != "ready" for r in rows) == item["over_window"], "GUARD_SENSOR_PLAN_COUNTS")
        result[name] = {"rows": len(rows), "tokens": item["tokens"], "over_window": item["over_window"]}
        # Re-signed metadata cannot conceal a silently cut or misassigned input.
        for label in ("cut_token", "wrong_text", "false_refusal"):
            damaged = copy.deepcopy(rows[0])
            if label == "cut_token":
                damaged["inputs"]["input_ids"].pop()
            elif label == "wrong_text":
                damaged["text_id"] = "0" * 64
            else:
                damaged["status"] = "MNLM_GUARD_LENS_WINDOW" if damaged["status"] == "ready" else "ready"
            try:
                validate_row(damaged, texts[0], tok, item["recipe"])
                passed = False
            except ValueError as exc:
                passed = str(exc) == "MNLM_RP_READ_GUARD_TOKEN_COVERAGE"
            negatives.append({"lens": name, "control": label, "refused": passed})
        phases[name + "_retokenization_and_controls"] = time.perf_counter() - tick
    check(all(c["refused"] for c in negatives), "GUARD_SENSOR_PLAN_CONTROL")
    check(sha(a.plan / "report.json") == a.plan_sha256 and sha(a.renders / "texts.jsonl") == rep["texts_sha256"], "GUARD_SENSOR_PLAN_DRIFT")
    wall = time.perf_counter() - start
    report = {"author_checked": True, "plan_sha256": a.plan_sha256, "instruments": result,
              "negative_controls": negatives, "all_texts_independently_retokenized": True,
              "gpu_used": False, "scientific_admission": False, "measurements_complete": False,
              "phases_seconds": phases, "wall_seconds": wall, "phase_coverage": sum(phases.values()) / wall}
    publish(a.output, report)
    print(json.dumps({"instruments": len(result), "rows_per_instrument": len(texts), "controls": len(negatives),
                      "wall_seconds": wall, "phase_coverage": report["phase_coverage"]}))


if __name__ == "__main__":
    main()
