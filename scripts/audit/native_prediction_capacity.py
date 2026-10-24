"""Read measured prediction capacity without fitting or changing a model or gate."""
import argparse
import base64
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.prospective_execution_readback import canonical, journal, sha


def require(ok, code):
    if not ok:
        raise ValueError(code)


def authenticated(path, pin):
    raw = path.read_bytes()
    require(sha(raw) == pin, "CAPACITY_BYTES")
    return raw


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and len(values) and np.isfinite(values).all(),
            "CAPACITY_NATIVE_VALUES")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    require((norms > 0).all(), "CAPACITY_NATIVE_VALUES")
    return values/norms


def edit(cost):
    """Full-table recurrence, independent of the predictor's rolling-row code."""
    n, m = cost.shape
    table = np.empty((n+1, m+1), dtype=np.float64)
    table[:, 0], table[0, :] = np.arange(n+1), np.arange(m+1)
    for i in range(n):
        for j in range(m):
            table[i+1, j+1] = min(table[i, j+1]+1, table[i+1, j]+1, table[i, j]+cost[i,j])
    return float(table[-1, -1]/max(n,m))


def bounds(target, reference, radius):
    require(0 <= radius < 1 and target.shape[1] == reference.shape[1], "CAPACITY_DOMAIN")
    similarity = np.clip(target@reference.T, -1., 1.)
    # ||residual|| <= r around a unit reference implies angular displacement <= asin(r).
    best_cosine = np.cos(np.maximum(0., np.arccos(similarity)-np.arcsin(radius)))
    # Independent algebraic form of the spherical-cap support function.
    cap = np.sqrt(1-radius*radius)
    algebraic = np.where(similarity >= cap, 1.,
                        similarity*cap+radius*np.sqrt(np.maximum(0.,1-similarity**2)))
    require(np.max(np.abs(best_cosine-algebraic)) < 1e-12, "CAPACITY_CAP_READBACK")
    length = abs(len(target)-len(reference))/max(len(target),len(reference))
    lower, unchanged = edit(1-best_cosine), edit(1-similarity)
    require(length <= lower+1e-12 and lower <= unchanged+1e-12, "CAPACITY_BOUND_ORDER")
    return dict(target_windows=len(target), reference_windows=len(reference),
                length_lower_bound=length, geometric_lower_bound=lower, unchanged_loss=unchanged)


def selected(reference, prediction, roster, region):
    require(region in roster, "CAPACITY_FOREIGN_REGION")
    require(prediction.shape == reference.shape, "CAPACITY_PREDICTION_CARDINALITY")
    require(np.max(np.arccos(np.clip(np.sum(prediction*reference,axis=1),-1.,1.))) <=
            np.arcsin(.1)+1e-6, "CAPACITY_PLACEMENT_RADIUS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--seal", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    seal = json.loads(authenticated(a.evidence/"seal.json", a.seal))
    def evidence(name):
        return authenticated(a.evidence/name, seal["files"][name]["sha256"])
    summary = json.loads(evidence("summary.json"))
    code = evidence("code/model/constellation_world.py")
    require(b"residual = .1*residual / torch.linalg.vector_norm" in code,
            "CAPACITY_ARCHITECTURE_NOT_AUDITED")
    c = summary["config"]
    contract, events, payloads, head = journal(Path(c["output"])/"experience.sqlite3")
    require(head == summary["head"], "CAPACITY_HEAD")
    forecasts = {sha(canonical(p["forecast"])):p["forecast"] for p in payloads.values() if "forecast" in p}
    decisions = [p for p in payloads.values() if "candidate" in p]
    require(len(decisions) == 1, "CAPACITY_DECISIONS")
    candidates = decisions[0]["candidate"]
    kernel_path = Path(c["kernel"])
    kernel_pin = next(f["routing"]["kernel"] for f in forecasts.values())
    kernel = json.loads(authenticated(kernel_path/"manifest.json", kernel_pin))
    shared = sorted(v["response"]["unit"] for v in kernel["pairs"].values())
    base = Path(c["population"])
    population = json.loads(authenticated(base/"report.json",c["population_pin"]))
    span_pin = population["files"]["reading_spans.jsonl"]
    if isinstance(span_pin, dict):
        span_pin = span_pin["sha256"]
    spans = [json.loads(r) for r in authenticated(base/"reading_spans.jsonl",span_pin).splitlines()]
    source_refs = [json.loads(r) for r in authenticated(base/"references.jsonl",
        population["files"]["references.jsonl"]["sha256"]).splitlines()]
    messages = [json.loads(r) for r in authenticated(base/"messages.jsonl",
        population["files"]["messages.jsonl"]["sha256"]).splitlines()]
    def source_reference(index, task=None, cutoff=None):
        refs = [r for r in source_refs if r["unit"] == index and r["role"] == "recorded_response"
                and ((task is not None and r["task"] == task and r["turn"] < cutoff)
                     or (r["partition"] == "train" and r["turn"] in (3,5)))]
        require(refs, "CAPACITY_REGION_PROVENANCE")
        text = messages[index]["text"]
        require(all(sha(text.encode()) == r["text_sha256"] for r in refs), "CAPACITY_SURFACE_PIN")
        return dict(references=refs, utf8_bytes=len(text.encode()), excerpt=text[:240])
    grouped = {}
    for span in spans:
        grouped.setdefault(span["unit"],[]).append(span)
    checks, phases = [], {}
    touched = {}
    def native(root, pin):
        report_raw = authenticated(root/"report.json",pin)
        touched[str(root/"report.json")] = pin
        report = json.loads(report_raw)
        result = {}
        for name, r in report["lenses"].items():
            raw = authenticated(root/name/"report.json",r["report_sha256"])
            lane = json.loads(raw)
            blob_pin = lane["files"]["slots.npz"]["sha256"]
            authenticated(root/name/"slots.npz",blob_pin)
            touched[str(root/name/"slots.npz")] = blob_pin
            with np.load(root/name/"slots.npz",allow_pickle=False) as z:
                values, offsets = z["vectors"].copy(), z["offsets"].copy()
            require(offsets[0] == 0 and offsets[-1] == len(values) and
                    (np.diff(offsets) >= 0).all(), "CAPACITY_NATIVE_OFFSETS")
            result[name] = (values, offsets)
        return result
    source_native = native(Path(c["native"]), c["native_pin"])
    source_rows = {}
    def source_row(name, index):
        key = (name,index)
        if key not in source_rows:
            values, offsets = source_native[name]
            source_rows[key] = unit(np.concatenate([values[offsets[s["atom"]]:offsets[s["atom"]+1]]
                                                    for s in grouped[index]]))
        return source_rows[key]
    targets = {}
    for provider in c["providers"]:
        enrollment = json.loads(evidence(Path(provider["enrollment"]).relative_to(a.evidence).as_posix()))
        readings = native(Path(enrollment["native"]),enrollment["native_pin"])
        pop = Path(enrollment["population"])
        r = json.loads(authenticated(pop/"report.json",enrollment["population_pin"]))
        refs = [json.loads(v) for v in authenticated(pop/"references.jsonl",r["files"]["references.jsonl"]["sha256"]).splitlines()]
        for key,target in summary["actual_targets"].items():
            observation = target["binding"]["observation"]
            members = [r for r in refs if r["observation"] == observation and r["unit"] is not None]
            if not members:
                continue
            targets[key] = {n:unit(np.concatenate([v[o[r["unit"]]:o[r["unit"]+1]] for r in members]))
                            for n,(v,o) in readings.items()}
    require(set(targets) == set(summary["actual_targets"]) == set(candidates), "CAPACITY_ROSTER")
    phases["authenticated_inputs"] = time.perf_counter()-started
    tick = time.perf_counter()
    outcomes, pairs = {}, 0
    for key,target in targets.items():
        record = forecasts[key]
        eligible = record["region_context"]["definition"]["units"]
        request = record["request"]
        provenance = {i:source_reference(i,request["task"],request["turn"]) for i in eligible}
        rows = {}
        for n,t in target.items():
            results = {}
            for index in sorted(set(shared+eligible)):
                ref = source_row(n,index)
                results[index] = bounds(t,ref,.1)
                pairs += len(t)*len(ref)
            measured = {}
            for kind,prediction in (("forecast",record),("candidate",candidates[key])):
                slot = prediction["predictions"][n]
                raw = base64.b64decode(slot["base64"],validate=True)
                require(sha(raw) == slot["sha256"], "CAPACITY_PREDICTION_HASH")
                v = unit(np.frombuffer(raw,dtype='<f4').reshape(slot["shape"]))
                index = int(prediction["region"].split(':')[0])
                ref = source_row(n,index)
                selected(ref,v,eligible,index)
                loss = edit(1-np.clip(t@v.T,-1.,1.))
                saved = (summary["scores"][key][n]["prediction_loss"] if kind == "forecast" else
                         prediction["assessment"]["instruments"][n]["prediction_loss"])
                require(abs(loss-saved) < 1e-12 and loss >= results[index]["geometric_lower_bound"]-1e-12,
                        "CAPACITY_SCORE_REPLAY")
                measured[kind] = dict(region=index,windows=len(v),loss=loss,**results[index])
            rows[n] = dict(actual=measured,regions=results,
                eligible_length_range=[min(results[i]["reference_windows"] for i in eligible),max(results[i]["reference_windows"] for i in eligible)],
                eligible_floor=min(results[i]["geometric_lower_bound"] for i in eligible),
                shared_floor=min(results[i]["geometric_lower_bound"] for i in shared))
        common = min(eligible,key=lambda i:sum(rows[n]["regions"][i]["unchanged_loss"] for n in rows))
        shared_common = min(shared,key=lambda i:sum(rows[n]["regions"][i]["unchanged_loss"] for n in rows))
        outcomes[key] = dict(context=record["request"]["context"],eligible=eligible,slots=rows,
                            eligible_provenance=provenance,
                            best_unchanged_common_region=common,
                            best_common_per_slot={n:rows[n]["regions"][common] for n in rows},
                            best_unchanged_shared_common_region=shared_common,
                            shared_common_per_slot={n:rows[n]["regions"][shared_common] for n in rows})
    phases["capacity_bounds_and_score_replay"] = time.perf_counter()-tick
    tick = time.perf_counter()
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as e:
            require(code in str(e), "CAPACITY_CONTROL_CODE")
        else:
            raise AssertionError("CAPACITY_ACCEPTED_"+name)
        checks.append(dict(name=name,code=code))
    refuse("foreign_region","CAPACITY_FOREIGN_REGION",lambda:selected(ref,v,eligible,-1))
    refuse("changed_cardinality","CAPACITY_PREDICTION_CARDINALITY",lambda:selected(ref,np.concatenate([v,v[:1]]),eligible,index))
    refuse("altered_native_pin","CAPACITY_BYTES",lambda:authenticated(Path(c["native"])/"report.json",'0'*64))
    refuse("invalid_native_values","CAPACITY_NATIVE_VALUES",lambda:unit(v.flatten()))
    refuse("invalid_radius","CAPACITY_DOMAIN",lambda:bounds(t,ref,1.))
    response_units = {r["unit"] for r in source_refs if r["role"] == "recorded_response"}
    proposal = next(r["unit"] for r in source_refs if r["role"] == "generated_proposal"
                    and r["unit"] not in response_units)
    refuse("proposal_as_region","CAPACITY_REGION_PROVENANCE",lambda:source_reference(proposal))
    require(journal(Path(c["output"])/"experience.sqlite3")[3] == head,"CAPACITY_MUTATED_SOURCE")
    phases["controls_and_source_head"] = time.perf_counter()-tick
    wall = time.perf_counter()-started
    result = dict(complete=True,acceptance="author-checked only",source_seal=a.seal,source_head=head,
        radius=.1,outcomes=outcomes,shared_regions=len(shared),ordered_window_pairs=pairs,
        input_files=touched,controls=checks,phases=phases,wall_seconds=wall,phase_coverage=sum(phases.values())/wall,
        model_fitted=False,scientific_admission=False,selected=False,
        shared_provenance={i:source_reference(i) for i in shared},
        interpretation=dict(
            per_slot_minima_may_choose_different_regions=True,
            lower_bounds_are_optimistic_not_attainability_proofs=True,
            output_window_count_is_fixed_by_selected_region=True,
            shared_bank_comparison_does_not_authorize_expanding_route=True,
            exposed_outcomes_are_not_fresh_evaluation=True),
        scope="lower_bounds_for_existing_region_cardinality_and_bounded_placement_not_global_model_limits")
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"report.json").write_bytes(canonical(result))
    print(json.dumps(dict(complete=True,ordered_window_pairs=pairs,wall_seconds=wall,
        outcomes={k:dict(context=v["context"],common=v["best_unchanged_common_region"],
                        slots={n:dict(eligible_lengths=x["eligible_length_range"],eligible_floor=x["eligible_floor"],shared_floor=x["shared_floor"])
                               for n,x in v["slots"].items()}) for k,v in outcomes.items()})))


if __name__ == "__main__":
    main()
