"""Independent full-table/algebraic readback of a target-aware capacity audit."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"lens"))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical
from claims.tool_temporal_source import ToolSource
from model.tool_temporal_input import ToolInputs


def table_distance(substitutions):
    m, n = substitutions.shape
    grid = np.zeros((m+1, n+1))
    grid[:, 0], grid[0, :] = np.arange(m+1), np.arange(n+1)
    for i in range(m):
        for j in range(n):
            grid[i+1, j+1] = min(grid[i, j+1]+1, grid[i+1, j]+1,
                                  grid[i, j]+substitutions[i, j])
    return float(grid[m, n]/max(m, n))


def readback(inputs, root, expected):
    raw = (root/"report.json").read_bytes()
    check(sha(raw) == expected == (root/"COMMITTED").read_text(), "REGION_READBACK_PIN")
    report = json.loads(raw)
    check(report["uses_targets"] and not report["predictive_capability"]
          and not report["scientific_admission"], "REGION_READBACK_SCOPE")
    count, max_error, near_ties = 0, 0., []
    for value in report["result"]["results"].values():
        task, cutoff = value["task"], value["cutoff"]
        target = inputs.population.lookup[task, cutoff+1]["unit"]
        previous = inputs.population.lookup[task, cutoff-1]["unit"]
        history = sorted(set(inputs.world_units) | {
            inputs.population.lookup[task, t]["unit"] for t in range(3, cutoff, 2)})
        check(target == value["target"] and previous == value["persistence_unit"]
              and history == value["history_units"] and value["fixed_units"] == inputs.world_units,
              "REGION_READBACK_HISTORY")
        recomputed = {}
        for spec in inputs.specs:
            x, ptr = inputs.arrays[spec.name]
            def unit(k):
                a = x[ptr[k]:ptr[k+1]].astype(np.float64)
                return a/np.sqrt(np.einsum("ij,ij->i", a, a))[:, None]
            a = unit(target)
            baseline = table_distance(1-np.clip(a@unit(previous).T, -1, 1))
            recomputed[spec.name] = {}
            for key, saved in value["costs"][spec.name].items():
                b = unit(int(key))
                c = np.clip(np.einsum("ik,jk->ij", a, b), -1, 1)
                radius = report["radius"]
                # Algebraic spherical-cap maximum, independent of arccos/asin producer.
                boundary = np.sqrt(1-radius*radius)
                best = np.where(c >= boundary, 1., c*boundary+np.sqrt(1-c*c)*radius)
                measured = dict(exact=table_distance(1-c), lower=table_distance(1-best),
                                length_lower=abs(len(a)-len(b))/max(len(a), len(b)))
                delta = max(abs(measured[k]-saved[k]) for k in measured)
                check(delta < 1e-12, "REGION_READBACK_ARITHMETIC")
                count += 1
                max_error = max(max_error, delta)
                recomputed[spec.name][key] = measured
            fixed = [recomputed[spec.name][str(u)] for u in inputs.world_units]
            expected_lane = dict(persistence=baseline,
                fixed_min_exact=min(x["exact"] for x in fixed),
                fixed_min_lower=min(x["lower"] for x in fixed),
                historical_min_lower=min(x["lower"] for x in recomputed[spec.name].values()))
            check(all(abs(v-value["instruments"][spec.name][k]) < 1e-12
                      for k, v in expected_lane.items()), "REGION_READBACK_LANE")
        for pool, units in (("fixed", inputs.world_units), ("historical", history)):
            for label, cost in (("optimistically_feasible", "lower"), ("exact_better", "exact")):
                calculated = [u for u in units if all(
                    recomputed[s.name][str(u)][cost] < value["instruments"][s.name]["persistence"]
                    for s in inputs.specs)]
                published = value[pool+"_"+label]
                # An exact reuse of the persistence row can differ by rounding
                # between BLAS matmul and the independent einsum implementation.
                # Keep those strict-comparison ties explicit; do not change a gate.
                for unit_id in set(calculated) ^ set(published):
                    margins = [recomputed[s.name][str(unit_id)][cost] -
                               value["instruments"][s.name]["persistence"] for s in inputs.specs]
                    check(max(margins) <= 1e-12 and abs(max(margins)) < 1e-12,
                          "REGION_READBACK_SUMMARY")
                    near_ties.append(dict(task=task, cutoff=cutoff, pool=pool,
                                          comparison=label, unit=unit_id, margins=margins))
    return dict(complete=True, author_checked=True, recomputed_comparisons=count,
                max_absolute_error=max_error, near_zero_comparisons=near_ties,
                predictive_capability=False, report=expected)


def main():
    parser = argparse.ArgumentParser()
    for name in ("raw", "population", "native", "output", "population-pin", "native-pin", "report-pin"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = ToolInputs(ToolSource(Path(args.raw)), args.population, args.native,
                        population_pin=args.population_pin, native_pin=args.native_pin)
    result = readback(inputs, Path(args.output), args.report_pin)
    result["wall_seconds"] = time.perf_counter()-started
    with (Path(args.output)/"cold-readback.json").open("xb") as f:
        f.write(canonical(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
