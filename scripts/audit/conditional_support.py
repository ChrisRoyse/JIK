"""Outcome-blind necessary-support audit for exact native-context CMI (#12).

Hash buckets upper-bound exact-context counts: collisions merge, never split,
equal states. Insufficient buckets prove this estimator lacks support here;
large buckets do not prove equality, IID sampling, information or admission.
O(native bytes + choose(channels,5)*5*observations), no all-pairs distances.
"""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "lens")]
from senses.contract import CHANNELS, SenseError, canonical, digest, require
from senses.instruments import DIM, contracts, measure
from senses.pipeline import file_hash
from senses.trajectory_population import publish, read_pinned
from prefix_panel import validate_plan
from finite_cmi import MIN_UNITS

SCHEMA = "native-conditional-support-v1"
MIN_CHANNELS = 5


def state_bytes(slot, windows):
    values = slot["tokens"]
    require(isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[1] == DIM
            and values.dtype == np.dtype("float32") and bool(np.isfinite(values).all())
            and slot["kind"] == ("Absent" if len(values) == 0 else "Multi")
            and type(windows) is int and windows >= 0,
            "MNLM_SUPPORT_NATIVE", "expected complete finite native Multi/Absent values and window count")
    header = dict(shape=list(values.shape), dtype="<f4", kind=slot["kind"], windows=windows)
    return canonical(header) + b"\n" + values.astype("<f4", copy=False).tobytes(), header


def recount(rows, exact_constants):
    """No outcomes: count upper bounds using all observations, before any split."""
    require(rows and len({r["unit"] for r in rows}) == len(rows)
            and len({r["atom"] for r in rows}) == len(rows),
            "MNLM_SUPPORT_UNITS", "need unique unit and observation identities; this alone does not establish IID")
    require(set(exact_constants) == set(CHANNELS) and all(type(x) is bool for x in exact_constants.values()),
            "MNLM_SUPPORT_CONSTANTS", "exact native comparisons required; a single hash bucket is not a constant proof")
    for r in rows:
        require(set(r["slots"]) == set(CHANNELS), "MNLM_SUPPORT_CHANNELS", "missing native conditioning channel")
        for s in r["slots"].values():
            require(set(s) == {"sha256", "shape", "dtype", "kind", "windows"}
                    and isinstance(s["sha256"], str) and len(s["sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in s["sha256"])
                    and s["dtype"] == "<f4" and len(s["shape"]) == 2 and s["shape"][1] == DIM
                    and type(s["shape"][0]) is int and s["shape"][0] >= 0
                    and s["kind"] == ("Multi" if s["shape"][0] else "Absent")
                    and type(s["windows"]) is int and s["windows"] >= 0,
                    "MNLM_SUPPORT_NATIVE", "malformed native state receipt")
    channels = {c: {"hash_buckets": len({r["slots"][c]["sha256"] for r in rows}),
                    "exact_constant": exact_constants[c],
                    "present": sum(r["slots"][c]["kind"] != "Absent" for r in rows)} for c in CHANNELS}
    cases = []
    for panel in combinations(CHANNELS, MIN_CHANNELS):
        complete = sum(all(r["slots"][c]["kind"] != "Absent" for c in panel) for r in rows)
        nonconstant = all(not exact_constants[c] for c in panel)
        for candidate in panel:
            conditioned = [c for c in panel if c != candidate]
            counts = Counter(tuple(r["slots"][c]["sha256"] for c in conditioned) for r in rows)
            largest = max(counts.values())
            supported_upper = sum(n for n in counts.values() if n >= 2 * MIN_UNITS)
            cases.append(dict(candidate=candidate, conditioning=conditioned, complete_cases=complete,
                all_five_nonconstant=nonconstant, context_buckets=len(counts), largest_bucket=largest,
                potentially_supported_rows_upper=supported_upper,
                possible_fifth_step=nonconstant and complete > 0 and supported_upper > 0))
    return dict(channels=channels, comparisons=cases,
        possible_fifth_steps_upper=sum(c["possible_fifth_step"] for c in cases),
        all_nonconstant_comparisons=sum(c["all_five_nonconstant"] for c in cases),
        min_binary_class_units=MIN_UNITS, necessary_context_units=2 * MIN_UNITS,
        no_certificate_from_this_method=not any(c["possible_fifth_step"] for c in cases),
        information_measured=False, iid_established=False, stage3_admitted=False, stage4_accepted=False)


def run(reader, plan, output, source_binding, started=None):
    input_ready = time.perf_counter()
    started = input_ready if started is None else started
    validate_plan(reader, plan)
    require(not output.exists(), "MNLM_UCR_OUTPUT_EXISTS", "choose a fresh support-audit directory")
    output.mkdir(parents=True, exist_ok=False)
    # Publish intent before work; only COMMITTED makes the complete census readable.
    intent = dict(schema=SCHEMA, source=source_binding, reader_identity=reader.identity,
                  reader_code_sha256=reader.reader_code_sha256, plan_sha256=digest(plan),
                  rows=plan["rows"], contracts=contracts(), code_sha256=file_hash(Path(__file__)),
                  outcomes_visible=False, min_binary_class_units=MIN_UNITS)
    intent_sha = publish(output / "intent.json", intent)
    rows, first, constants = [], {}, {c: True for c in CHANNELS}
    totals, maxima, missing = Counter(), Counter(), Counter()
    for atom in reader.examples:
        current, past, binding = reader.read(atom)
        require(all(k not in binding or binding[k] is False for k in ("future_visible", "outcomes_visible")),
                "MNLM_SUPPORT_VISIBILITY", "future/outcome exposure invalidates this input census")
        row = dict(atom=atom, unit=current["unit"], binding_sha256=digest(binding), slots={})
        for c in CHANNELS:
            slot = measure(current, c, past if c == "history" else None)
            n = len(slot["tokens"])
            totals[c] += n
            maxima[c] = max(maxima[c], n)
            missing[c] += n == 0
            require(n <= plan["channels"][c]["max_token_rows"] and totals[c] <= plan["channels"][c]["token_rows"],
                    "MNLM_SUPPORT_COUNT", "native rows exceed the frozen tokenizer preflight")
            data, header = state_bytes(slot, len(past) if c == "history" else int(n > 0))
            if c not in first:
                first[c] = data
            elif constants[c]:
                constants[c] = first[c] == data  # full bytes, never hash-only equality
            row["slots"][c] = {**header, "sha256": hashlib.sha256(data).hexdigest()}
        rows.append(row)
    for c in CHANNELS:
        p = plan["channels"][c]
        require((totals[c], maxima[c], missing[c]) == (p["token_rows"], p["max_token_rows"], p["absent"]),
                "MNLM_SUPPORT_COUNT", "native rows differ from the complete tokenizer census")
    measured = time.perf_counter()
    report = recount(rows, constants)
    counted = time.perf_counter()
    rows_sha = publish(output / "states.json", rows)
    report.update(schema=SCHEMA, rows=len(rows), exact_constants=constants, native_token_rows=sum(totals.values()),
        native_value_bytes_processed=sum(totals.values()) * DIM * 4,
        dense_slot_files_written=0, intent_sha256=intent_sha, states_sha256=rows_sha,
        acceptance="author-checked only", sampling="task-level grouping; IID not established",
        scope="current frozen channel bank and this retained cohort; no claim about all possible instruments",
        status="MNLM_CMI_EXACT_CONTEXT_SUPPORT" if report["no_certificate_from_this_method"] else "MNLM_CMI_SUPPORT_ONLY",
        timings=dict(input_preparation_seconds=input_ready-started, native_census_seconds=measured-input_ready,
                     context_count_seconds=counted-measured))
    report_sha = publish(output / "report.json", report)
    publish(output / "COMMITTED", {"intent_sha256": intent_sha, "report_sha256": report_sha, "states_sha256": rows_sha})
    load(output, report_sha)
    publish(output / "timing.json", dict(wall_seconds=time.perf_counter()-started,
            measured_phases_seconds=counted-started, publication_and_readback_seconds=time.perf_counter()-counted))
    return report


def load(root, expected_report_sha256):
    report = read_pinned(root / "report.json", expected_report_sha256)
    try:
        marker = json.loads((root / "COMMITTED").read_bytes())
    except (OSError, ValueError) as exc:
        raise SenseError("MNLM_SUPPORT_UNCOMMITTED", "completion marker missing or malformed; rebuild in a fresh directory") from exc
    require(marker == {
        "intent_sha256": report["intent_sha256"], "report_sha256": expected_report_sha256,
        "states_sha256": report["states_sha256"]}, "MNLM_SUPPORT_UNCOMMITTED", "complete census marker required")
    intent = read_pinned(root / "intent.json", report["intent_sha256"])
    rows = read_pinned(root / "states.json", report["states_sha256"])
    require(intent["schema"] == report["schema"] == SCHEMA and intent["rows"] == report["rows"] == len(rows)
            and intent["outcomes_visible"] is False and intent["min_binary_class_units"] == MIN_UNITS,
            "MNLM_SUPPORT_SCOPE", "census scope differs from intent")
    actual = recount(rows, report["exact_constants"])
    require(all(report[k] == v for k, v in actual.items()), "MNLM_SUPPORT_RECOUNT", "retained counts or status differ")
    return report, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("prefix", "plan", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in ("prefix-sha256", "plan-sha256"):
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args()
    try:
        started = time.perf_counter()
        from senses.trajectory_prefix import PrefixInputs
        from senses.structured_prefix import PrefixInputs as Structured
        reader = Structured(PrefixInputs(a.prefix, a.prefix_sha256))
        result = run(reader, read_pinned(a.plan, a.plan_sha256), a.output,
                     {"kind": "trajectory-structure", "manifest_sha256": a.prefix_sha256}, started=started)
        print(json.dumps({k: result[k] for k in ("rows", "status", "channels", "all_nonconstant_comparisons",
                                                "possible_fifth_steps_upper", "native_token_rows")}))
    except SenseError as exc:
        print(json.dumps(exc.envelope()), file=sys.stderr)
        sys.exit(2)
