"""Read routed forecasts from SQLite and score each route against source windows."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"lens"))
from audit.fsv_kernel_route import independent_score
from audit.tool_loop_readback import readback as read_journal
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.tool_temporal_source import ToolSource
from model.tool_temporal_input import ToolInputs


def readback(inputs, run, expected):
    raw = (run/"kernel"/"manifest.json").read_bytes()
    check(sha(raw) == expected == (run/"kernel"/"COMMITTED").read_text(), "KERNEL_READBACK_PIN")
    kernel = json.loads(raw)
    source_pairs = {str(ref["unit"]): dict(response=ref,
        action=inputs.population.lookup[ref["task"], ref["turn"]-1])
        for ref in inputs.population.initial_bank()}
    check(source_pairs == kernel["pairs"], "KERNEL_READBACK_PAIRS")
    snapshot = json.loads((run/"snapshot.json").read_bytes())
    original = read_journal(run/"experience.sqlite3", inputs, snapshot)
    with sqlite3.connect(f"file:{(run/'experience.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        payloads = [json.loads(raw) for raw, in db.execute("SELECT data FROM payloads")]
    records = {digest(p["forecast"]): p["forecast"] for p in payloads if "forecast" in p}
    def windows(name, unit):
        x, offsets = inputs.arrays[name]
        return x[offsets[unit]:offsets[unit+1]]
    details, scores, checked_outputs, labels = {}, 0, 0, 0
    for key, record in records.items():
        route = record["routing"]
        check(route["kernel"] == expected and route["query"] == record["request"]["reference"],
              "KERNEL_READBACK_QUERY")
        check(set(route["traces"]) == {s.name for s in inputs.specs}, "KERNEL_READBACK_ROSTER")
        ranks, votes = {}, {}
        for spec in inputs.specs:
            query = windows(spec.name, route["query"]["unit"])
            direct = {k: independent_score(query, windows(spec.name, p["action"]["unit"]))
                      for k, p in source_pairs.items()}
            exact = sorted(direct, key=lambda k: (-direct[k], k))[:kernel["plan"]["top_k"]]
            hits = route["traces"][spec.name]["hits"]
            check(all(abs(value-direct[k]) < 2e-12 for k, value in hits), "KERNEL_READBACK_SCORE")
            check(len(hits) == kernel["plan"]["top_k"], "KERNEL_READBACK_TOP_K")
            ranks[spec.name] = dict(recall=len(set(exact) & {k for k, _ in hits})/len(exact),
                                   exact=exact)
            for rank, (k, _) in enumerate(hits, 1):
                votes[k] = votes.get(k, 0.)+1/(60+rank)
            scores += len(direct)
        chosen = sorted(votes, key=lambda k: (-votes[k], k))[:kernel["plan"]["top_k"]]
        units = {source_pairs[k]["response"]["unit"] for k in chosen}
        allowed = [i for i, u in enumerate(inputs.world_units) if u in units]
        check(chosen == route["responses"] and allowed == route["allowed"]
              and int(record["region"].split(":")[0]) in units, "KERNEL_READBACK_FUSION")
        checked_outputs += 1
        details[key] = dict(context=record["request"]["context"], cutoff=record["request"]["turn"],
                            lanes=ranks, allowed=allowed)
    for payload in payloads:
        if "candidate" not in payload:
            continue
        for key, region in payload["labels"].items():
            check(region in details[key]["allowed"], "KERNEL_READBACK_LABEL")
            labels += 1
        for kind in ("baseline", "candidate"):
            for key, value in payload[kind].items():
                unit = int(value["region"].split(":")[0])
                check(inputs.world_units.index(unit) in details[key]["allowed"],
                      "KERNEL_READBACK_CANDIDATE_REGION")
                checked_outputs += 1
    return dict(complete=True, author_checked=True, kernel=expected, journal=original,
                independently_scored_action_pairs=scores, checked_prediction_regions=checked_outputs,
                checked_training_labels=labels, forecasts=details, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "run", "population-pin", "native-pin", "kernel-pin"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = ToolInputs(ToolSource(Path(args.raw)), args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    result = readback(inputs, Path(args.run), args.kernel_pin)
    result["wall_seconds"] = time.perf_counter()-started
    with (Path(args.run)/"kernel-route-readback.json").open("xb") as f:
        f.write(canonical(result))
    print(json.dumps({k: v for k, v in result.items() if k not in ("journal", "forecasts")}))


if __name__ == "__main__":
    main()
