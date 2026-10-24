"""Separate native channel slots from a sealed prefix reader; diagnostic only.

PC-07/13/35: one input read per observation; bounded TokenSpool writers retain
every native row. Exact tokenizer counts are preflighted before dense allocation.
No outcomes, pooled scores, random projections or scientific admission here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from senses.contract import CHANNELS, canonical, digest, require
from senses.instruments import DIM, contracts, groups, measure, slot_digest as reading_digest
from senses.pipeline import file_hash
from senses.trajectory_population import publish as put, read_pinned
from slotio import TokenSpool, slot_digest, npz_member

SCHEMA = "native-prefix-panel-v1"
STATIC = ("name", "channel", "contract", "weights_sha256", "dtype", "shape", "norm", "inputs", "kind")


def observation_counts(current, past):
    result = {}
    for c in CHANNELS:
        items = [(f, lane) for r in past for lane in ("surface", "fields", "relations")
                 for f in r["channels"][lane]["items"]] if c == "history" else (
                 [] if c == "co_reference" else [(f, c) for f in current["channels"][c]["items"]])
        result[c] = sum(1 for f, lane in items for _ in groups(f, lane))
    return result


def census_from_counts(reader, counts):
    channels = {c: dict(token_rows=0, max_token_rows=0, absent=0, projected_values_bytes=0) for c in CHANNELS}
    count = 0
    for row in counts:
        require(set(row) == set(CHANNELS) and all(type(v) is int and v >= 0 for v in row.values()),
                "MNLM_PREFIX_PANEL_PLAN", "exact per-observation native counts required")
        count += 1
        for c, n in row.items():
            p = channels[c]
            p["token_rows"] += n
            p["max_token_rows"] = max(p["max_token_rows"], n)
            p["absent"] += n == 0
            p["projected_values_bytes"] += n * DIM * 4
    require(count == len(reader.examples), "MNLM_PREFIX_PANEL_PLAN", "counted observation inventory differs")
    return {"schema": "prefix-native-count-preflight-v1", "input_manifest_sha256": reader.identity,
            "admitted": False, "dtype": "<f4", "dim": DIM, "rows": len(reader.examples),
            "contracts": contracts(), "channels": channels,
            "projected_values_bytes": sum(c["projected_values_bytes"] for c in channels.values())}


def census(reader):
    return census_from_counts(reader, (observation_counts(*reader.read(atom)[:2]) for atom in reader.examples))


def validate_plan(reader, plan):
    require(plan.get("schema") == "prefix-native-count-preflight-v1"
            and plan.get("input_manifest_sha256") == reader.identity
            and plan.get("admitted") is False and plan.get("dtype") == "<f4" and plan.get("dim") == DIM
            and type(plan.get("rows")) is int and plan["rows"] == len(reader.examples) > 0,
            "MNLM_PREFIX_PANEL_PLAN", "pin exact input count, shape and diagnostic scope")
    current = {c["channel"]: c for c in contracts()}
    declared = {c["channel"]: c for c in plan["contracts"]}
    require(set(plan["channels"]) == set(CHANNELS) == set(declared)
            and len(plan["contracts"]) == len(CHANNELS)
            and all({k: current[c][k] for k in STATIC} == {k: declared[c][k] for k in STATIC} for c in CHANNELS),
            "MNLM_PREFIX_PANEL_INSTRUMENT", "tokenizer or channel contract changed since preflight")
    for c, p in plan["channels"].items():
        require(set(p) == {"token_rows", "max_token_rows", "absent", "projected_values_bytes"}
                and all(type(v) is int and v >= 0 for v in p.values())
                and p["max_token_rows"] <= p["token_rows"] <= p["max_token_rows"] * plan["rows"]
                and p["absent"] <= plan["rows"]
                and p["projected_values_bytes"] == p["token_rows"] * DIM * 4,
                "MNLM_PREFIX_PANEL_PLAN", "invalid native row/storage census")
    require(plan["projected_values_bytes"] == sum(p["projected_values_bytes"] for p in plan["channels"].values()),
            "MNLM_PREFIX_PANEL_PLAN", "storage sum differs from channel census")


def native_state(values):
    """Measurement state only: no atom, dependency, source or lens identity."""
    return hashlib.sha256(canonical({"shape": list(values.shape), "dtype": "<f4", "absent": len(values) == 0})
                          + values.astype("<f4", copy=False).tobytes()).hexdigest()


def publish(reader, plan, output):
    started = time.perf_counter()
    validate_plan(reader, plan)
    output.mkdir(parents=True, exist_ok=False)
    roster = contracts()
    spools, rows = {}, []
    totals, maxima, missing = Counter(), Counter(), Counter()
    try:
        for c in CHANNELS:
            spools[c] = TokenSpool(output / (c + ".npz"), DIM)
        for atom in reader.examples:
            current, past, binding = reader.read(atom)
            entry = {"atom": atom, "binding": binding, "slots": {}}
            for channel in CHANNELS:
                slot = measure(current, channel, past if channel == "history" else None)
                n = len(slot["tokens"])
                totals[channel] += n
                maxima[channel] = max(maxima[channel], n)
                missing[channel] += n == 0
                p = plan["channels"][channel]
                require(n <= p["max_token_rows"] and totals[channel] <= p["token_rows"],
                        "MNLM_PREFIX_PANEL_BUDGET", "native readings exceeded fixed tokenizer census; no truncation")
                spools[channel].add(slot["tokens"], np.asarray([0, n], np.int64))
                entry["slots"][channel] = {"shape": slot["shape"], "kind": slot["kind"], "reason": slot["reason"],
                    "reading_digest": reading_digest(slot), "state_sha256": native_state(slot["tokens"]),
                    "windows": len(past) if channel == "history" else int(n > 0)}
            rows.append(entry)
        measurement_done = time.perf_counter()
        for channel in CHANNELS:
            p = plan["channels"][channel]
            require((totals[channel], maxima[channel], missing[channel]) ==
                    (p["token_rows"], p["max_token_rows"], p["absent"]),
                    "MNLM_PREFIX_PANEL_COUNT", "measurement differs from token/absence census")
            spools[channel].finish(np.asarray([r["slots"][channel]["windows"] for r in rows], np.int64))
    finally:
        for spool in spools.values():
            spool.fh.close()
            # This run exclusively owns its scratch paths; published NPZs remain.
            spool.raw.unlink(missing_ok=True)
    publication_done = time.perf_counter()
    slots, variation = {}, {}
    for channel in CHANNELS:
        path = output / (channel + ".npz")
        slots[channel] = {"seal": slot_digest(path), "sha256": file_hash(path), "bytes": path.stat().st_size,
                          "contract": next(c for c in roster if c["channel"] == channel)}
        arrays = {k: npz_member(path, k) for k in ("tokens", "offsets", "absent", "windows")}
        try:
            counter, constant, first = Counter(), True, None
            for i, row in enumerate(rows):
                lo, hi = arrays["offsets"][i:i + 2]
                values = arrays["tokens"][lo:hi]
                state = native_state(values)
                require(state == row["slots"][channel]["state_sha256"]
                        and bool(arrays["absent"][i]) == (row["slots"][channel]["kind"] == "Absent")
                        and int(arrays["windows"][i]) == row["slots"][channel]["windows"],
                        "MNLM_PREFIX_PANEL_READBACK", "cold native values or boundaries changed")
                counter[state] += 1
                if first is None:
                    first = values.copy()
                elif constant:
                    constant = np.array_equal(first, values)
            require(constant == (len(counter) == 1), "MNLM_PREFIX_PANEL_STATE", "state digest/exact equality disagree")
            variation[channel] = {"states": len(counter), "exact_constant": constant,
                                  "absent": missing[channel], "token_rows": totals[channel],
                                  "shannon_information_measured": False}
        finally:
            for array in arrays.values():
                if hasattr(array, "_mmap"):
                    array._mmap.close()
    rows_sha = put(output / "rows.json", rows)
    manifest = {"schema": SCHEMA, "purpose": "engineering_boundary", "input_manifest_sha256": reader.identity,
                "reader_code_sha256": reader.reader_code_sha256, "code_sha256": file_hash(Path(__file__)),
                "plan_sha256": digest(plan), "rows": len(rows), "rows_sha256": rows_sha,
                "slots": slots, "variation": variation, "stage3_admitted": False, "stage4_accepted": False}
    sha = put(output / "manifest.json", manifest)
    put(output / "COMMITTED", {"manifest_sha256": sha})
    result = {"manifest_sha256": sha, "rows": len(rows), "variation": variation,
              "native_bytes": sum(s["bytes"] for s in slots.values()), "stage3_admitted": False,
              "timings": {"measurement_seconds": measurement_done - started,
                          "spool_publication_seconds": publication_done - measurement_done,
                          "cold_readback_seconds": time.perf_counter() - publication_done,
                          "wall_seconds": time.perf_counter() - started}}
    put(output / "run.json", result)
    return result


def load(root, manifest_sha256):
    m = read_pinned(root / "manifest.json", manifest_sha256)
    require(m.get("schema") == SCHEMA and m.get("purpose") == "engineering_boundary"
            and m.get("stage3_admitted") is False and m.get("stage4_accepted") is False
            and set(m["slots"]) == set(CHANNELS), "MNLM_PREFIX_PANEL_MANIFEST", "diagnostic native panel required")
    require(json.loads((root / "COMMITTED").read_bytes()) == {"manifest_sha256": manifest_sha256},
            "MNLM_PREFIX_PANEL_UNCOMMITTED", "native publication is incomplete")
    rows = read_pinned(root / "rows.json", m["rows_sha256"])
    require(len(rows) == m["rows"] and len({r["atom"] for r in rows}) == len(rows),
            "MNLM_PREFIX_PANEL_COUNT", "duplicate or missing rows")
    for channel, slot in m["slots"].items():
        p = root / (channel + ".npz")
        require(p.stat().st_size == slot["bytes"] and file_hash(p) == slot["sha256"]
                and slot_digest(p) == slot["seal"] and slot["contract"]["channel"] == channel
                and slot["seal"]["n"] == len(rows), "MNLM_PREFIX_PANEL_DIGEST", "native panel bytes/layout changed")
    return m, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("trajectory", "forecast", "trajectory-structure", "forecast-structure"), required=True)
    for name in ("prefix", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prefix-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    a = parser.parse_args()
    if a.kind.startswith("trajectory"):
        from senses.trajectory_prefix import PrefixInputs
    else:
        from senses.forecast_prefix import PrefixInputs
    reader = PrefixInputs(a.prefix, a.prefix_sha256)
    if a.kind.endswith("-structure"):
        from senses.structured_prefix import PrefixInputs as StructuredInputs
        reader = StructuredInputs(reader)
    print(json.dumps(publish(reader, read_pinned(a.plan, a.plan_sha256), a.output)))
