"""Cold closure of fixed renderer cases, strict output bytes and encoder input pool."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.renderer_population_readback import sha, check
from audit.renderer_comparison_readback import decoded
from senses.contract import digest
from senses.trajectory_population import publish


def lines(path):
    return [json.loads(s) for s in path.read_bytes().splitlines()]


def read(root, pin, population, comparison):
    check(sha(root / "report.json") == pin, "GUARD_RENDER_PIN")
    rep = json.loads((root / "report.json").read_bytes())
    check(rep["schema"] == "strict-guard-renders-v1"
          and sha(population / "report.json") == rep["population_sha256"]
          and sha(comparison / "report.json") == rep["comparison_sha256"], "GUARD_RENDER_SOURCE")
    comp = json.loads((comparison / "report.json").read_bytes())
    check(rep["conditioning_sha256"] == comp["conditioning_sha256"] and rep["runtime"] == comp["runtime"], "GUARD_RENDER_SOURCE")
    check(all(rep[k] is False for k in ("guard_evaluated", "independent_anchors_evaluated", "scientific_admission", "composition_success", "final_test")), "GUARD_RENDER_CLAIM")
    check(rep["calibration_policy"] == {"arm": "true", "route": "true", "split": "calibration", "quantile": .1,
          "method": "linear", "minimum_complete": 50, "semantic_far_certified": False}, "GUARD_CALIBRATION_POLICY")
    check(set(rep["files"]) == {"gpu-before.txt", "candidates.jsonl", "targets.jsonl", "texts.jsonl", "replays.json"}, "GUARD_RENDER_FILES")
    for name, meta in rep["files"].items():
        check(sha(root / name) == meta["sha256"] and (root / name).stat().st_size == meta["bytes"], "GUARD_RENDER_FILE")
    for arm in ("true", "gaussian", "predicted"):
        check(sha(comparison / arm / "renderer.pt") == rep["checkpoint_sha256"][arm]
              == comp["arms"][arm]["checkpoint_sha256"], "GUARD_RENDER_CHECKPOINT")
    pr = json.loads((population / "report.json").read_bytes())
    check(sha(population / "rows.jsonl") == pr["files"]["rows.jsonl"]["sha256"], "GUARD_RENDER_ROWS")
    rows = lines(population / "rows.jsonl")
    cal = [i for i, r in enumerate(rows) if r["split"] == "calibration"]
    ev = [i for i, r in enumerate(rows) if r["split"] == "evaluation"]
    check(len(cal) == len(ev) == 256 and not ({rows[i]["doc"] for i in cal} & {rows[i]["doc"] for i in ev}), "GUARD_RENDER_SPLITS")
    cases = lines(root / "candidates.jsonl")
    expected = [("true", "calibration", "true", i) for i in cal]
    expected += [(arm, "evaluation", route, i) for arm in ("true", "gaussian", "predicted") for route in ("true", "predicted") for i in ev]
    check([(c["arm"], c["split"], c["route"], c["row"]) for c in cases] == expected and rep["cases"] == len(cases) == 1792, "GUARD_RENDER_ORDER")
    pool = lines(root / "texts.jsonl")
    text_index = {t["id"]: t for t in pool}
    check(len(text_index) == len(pool) == rep["unique_complete_texts"] and pool == sorted(pool, key=lambda t: t["id"]), "GUARD_RENDER_POOL")
    for t in pool:
        raw = t["text"].encode("utf-8", "strict")
        check(t["text"].strip() and len(raw) == t["bytes"] <= 512 and hashlib.sha256(raw).hexdigest() == t["id"], "GUARD_RENDER_TEXT")
    targets = lines(root / "targets.jsonl")
    nontrain = [i for i, r in enumerate(rows) if r["split"] != "train"]
    check([t["row"] for t in targets] == nontrain and len(targets) == rep["targets"] == 512, "GUARD_RENDER_TARGETS")
    used, summary = set(), {}
    for t in targets:
        source = rows[t["row"]]
        text = source["atoms"][-1]["text"]
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        check(t == {"row": t["row"], "doc": source["doc"], "split": source["split"], "text_id": key}
              and key in text_index and text_index[key]["text"] == text, "GUARD_RENDER_TARGET_BYTES")
        used.add(key)
    for c in cases:
        status, text = decoded(c["tokens"])
        key = hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None
        check(c["status"] == status and c["text"] == text and c["text_id"] == key
              and c["case"] == digest(["strict-guard-case-v1", c["arm"], c["split"], c["route"], c["row"]])
              and c["doc"] == rows[c["row"]]["doc"] and c["asserted"] is False
              and c["independent_anchor_status"] == "MNLM_ANCHOR_UNAVAILABLE", "GUARD_RENDER_BYTES_OR_CLAIM")
        if key:
            check(key in text_index and text_index[key]["text"] == text, "GUARD_RENDER_POOL")
            used.add(key)
        group = "/".join((c["arm"], c["split"], c["route"]))
        summary.setdefault(group, Counter())[status] += 1
    check(used == set(text_index) and sum(t["bytes"] for t in pool) == rep["unique_utf8_bytes"]
          and dict(Counter(c["status"] for c in cases)) == rep["statuses"], "GUARD_RENDER_COUNTS")
    replays = json.loads((root / "replays.json").read_bytes())
    check(len(replays) == 21 and rep["replayed_rows"] == 672, "GUARD_RENDER_REPLAY")
    for i, replay in enumerate(replays):
        group, offset = divmod(i, 3)
        begin = (0, 128, 224)[offset]
        subset = cases[group * 256 + begin:group * 256 + begin + 32]
        decoded_rows = [{k: c[k] for k in ("status", "text", "tokens")} for c in subset]
        check(replay["rows"] == [c["row"] for c in subset] and replay["byte_identical"] is True
              and all(replay[k] == subset[0][k] for k in ("arm", "split", "route"))
              and replay["sha256"] == digest(decoded_rows), "GUARD_RENDER_REPLAY")
    check(.9 <= rep["phase_coverage"] <= 1.001 and abs(sum(rep["phases_seconds"].values()) / rep["wall_seconds"] - rep["phase_coverage"]) < 1e-10, "GUARD_RENDER_TIMING")
    return {"author_checked": True, "report_sha256": pin, "cases": len(cases), "targets": len(targets),
            "unique_complete_texts": len(pool), "unique_utf8_bytes": rep["unique_utf8_bytes"],
            "statuses_by_group": {k: dict(v) for k, v in summary.items()}, "all_bytes_reparsed": True,
            "guard_evaluated": False, "independent_anchors_evaluated": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("root", "population", "comparison", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--report-sha256", required=True)
    a = ap.parse_args()
    result = read(a.root, a.report_sha256, a.population, a.comparison)
    publish(a.output, result)
    print(json.dumps(result))
