"""Independent persisted-parameter, dual, and earlier-training retention readback."""
import copy

import numpy as np
import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest


def pin(extra):
    return digest({k: dict(shape=list(a.shape), dtype=a.dtype.str, sha256=sha(a.tobytes()))
                   for k, a in extra.items() if k.startswith(("query.", "place."))})


def projection(row, count):
    check(row["rows"] == count and 0 <= row["active_rows"] <= count, "UPDATE_READ_PROJECTION_ROSTER")
    n = row["active_rows"]
    gram = np.asarray(row["gram"], dtype=np.float64).reshape(n, n)
    dot = np.asarray(row["dot"], dtype=np.float64)
    check(dot.shape == (n,) and np.isfinite(gram).all() and np.isfinite(dot).all(),
          "UPDATE_READ_PROJECTION_FINITE")
    if row["status"] == "already_descent":
        check(np.all(dot <= 0) and row["scale"] == 1., "UPDATE_READ_DESCENT")
    elif row["status"] == "projected":
        dual = np.asarray(row["dual"])
        bound = -np.abs(dot)
        after = dot-gram@dual
        tolerance = 64*np.finfo(np.float64).eps*max(
            1., float(np.linalg.norm(gram, np.inf)*np.linalg.norm(dual, np.inf)+np.linalg.norm(dot, np.inf)))
        check(dual.shape == dot.shape and np.isfinite(dual).all() and np.all(dual >= 0)
              and np.allclose(gram, gram.T, rtol=0, atol=tolerance)
              and np.array_equal(bound, np.asarray(row["bound"]))
              and row["tolerance"] == tolerance
              and np.max(np.abs(after-np.asarray(row["projected_dot"]))) <= tolerance
              and np.max(after-bound) <= tolerance
              and np.max(np.abs(dual*(after-bound))) <= tolerance, "UPDATE_READ_DUAL")
        check(0 < row["scale"] <= 1 and row["projected_norm"] >= 0
              and abs(row["scale"]-(min(1., row["proposal_norm"]/row["projected_norm"])
                      if row["projected_norm"] else 1.)) < 1e-14, "UPDATE_READ_NORM")
    else:
        check(row["status"] in ("infeasible_rank", "infeasible_dual", "solver_iteration_limit"),
              "UPDATE_READ_PROJECTION_STATUS")


def read(body, manifest, proof, payload, previous, previous_payload, start, final):
    contract = manifest["binding"].get("training_retention")
    if contract is None:
        check("retained_update" not in manifest["binding"]["plan"], "UPDATE_READ_CONTRACT")
        return dict(required=False)
    check(contract["rule"] == manifest["binding"]["plan"]["retained_update"] == "retained_native_descent_v1"
          and contract["scipy"] == manifest["binding"]["plan"]["retention_scipy"]
          and contract["scope"] == "earlier_training_examples_only", "UPDATE_READ_CONTRACT")
    keys = previous["train"] if previous else []
    check(contract["keys"] == keys and set(keys) <= set(body["train"])
          and not set(keys) & set(body["evaluate"]), "UPDATE_READ_SPLIT")
    expected = {}
    for key in keys:
        old = previous_payload["candidate"][key]["assessment"]["instruments"]
        new = payload["candidate"][key]["assessment"]["instruments"]
        check(old.keys() == new.keys(), "UPDATE_READ_ROSTER")
        expected[key] = {name: row["prediction_loss"] for name, row in old.items()}
        check(all(new[n]["instrument"] == v["instrument"]
                  and new[n]["prediction_loss"] <= v["prediction_loss"] for n, v in old.items()),
              "UPDATE_READ_FINAL_REGRESSION")
    check(contract["ceilings"] == expected and contract["predecessor"] ==
          (dict(candidate=previous["candidate"], pin=previous["pin"], payload=previous["payload"])
           if previous else None), "UPDATE_READ_ANCHOR")
    cursor, accepted = pin(start[3]), 0
    names = set(next(iter(payload["candidate"].values()))["assessment"]["instruments"])
    for step in proof["losses"]:
        row = step["retained_update"]
        check(row["rule"] == contract["rule"] and row["before"] == cursor
              and row["protected"] == len(keys)*len(names), "UPDATE_READ_PARAMETER_CHAIN")
        for p in row["projections"].values():
            projection(p, len(keys))
        if keys:
            check(set(row["projections"]) <= names and row["projections"], "UPDATE_READ_PROJECTION_ROSTER")
            for i, trial in enumerate(row["trials"]):
                check(trial["scale"] == (1., .5, .25, .125)[i]
                      and trial["scores"].keys() == expected.keys()
                      and all(trial["scores"][k].keys() == v.keys() for k, v in expected.items()),
                      "UPDATE_READ_TRIAL_ROSTER")
                values = [v for s in trial["scores"].values() for v in s.values()]
                check(all(np.isfinite(v) and v >= 0 for v in values), "UPDATE_READ_TRIAL_FINITE")
                passed = all(v <= expected[k][n] for k, s in trial["scores"].items() for n, v in s.items())
                check(trial["retained"] == passed
                      and (not passed or i == len(row["trials"])-1), "UPDATE_READ_TRIAL")
            if row["accepted"]:
                check(set(row["projections"]) == names and row["trials"][-1]["retained"]
                      and row["scale"] == row["trials"][-1]["scale"] and row["after"] != cursor,
                      "UPDATE_READ_ACCEPTANCE")
            else:
                check(row["scale"] == 0 and row["after"] == cursor, "UPDATE_READ_NOOP")
        else:
            check(row["accepted"] and not row["projections"] and not row["trials"], "UPDATE_READ_INITIAL")
        accepted += row["accepted"]
        cursor = row["after"]
    check(cursor == pin(final[3]) and proof["accepted_parameter_updates"] == accepted
          and proof["optimizer_attempts"] == len(proof["losses"]) == 8, "UPDATE_READ_FINAL")
    return dict(required=True, training_examples=len(keys), protected_slots=len(keys)*len(names),
                accepted_parameter_updates=accepted, optimizer_attempts=8, training_retention_passed=True)


def controls(data, model, before, gradients, generation, refuse):
    from model.retained_update import apply, validate
    from model.retention_projection import project
    saved = list(data.retained_keys)
    try:
        data.retained_keys = saved+[data.plan["evaluate"][0]]
        refuse(f"retained_withheld_{generation}", "RETAINED_UPDATE_SPLIT", lambda: validate(data))
    finally:
        data.retained_keys = saved
    refuse(f"retained_parameter_missing_{generation}", "RETAINED_UPDATE_PARAMETERS",
           lambda: apply(data, model, dict(list(before.items())[1:]), gradients))
    if gradients:
        matrix = next(iter(gradients.values())).clone()
        matrix[0, 0] = torch.nan
        direction = torch.zeros(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
        refuse(f"retained_nonfinite_gradient_{generation}", "RETENTION_PROJECTION_INPUT",
               lambda: project(matrix, direction))


def negative_readback(body, manifest, proof, payload, previous, previous_payload, start, final):
    if previous is None:
        return []
    controls = []
    for name in ("anchor", "parameter_chain", "accepted_count", "final_regression"):
        m, p, q = copy.deepcopy(manifest), copy.deepcopy(proof), copy.deepcopy(payload)
        if name == "anchor":
            key = next(iter(m["binding"]["training_retention"]["ceilings"]))
            lane = next(iter(m["binding"]["training_retention"]["ceilings"][key]))
            m["binding"]["training_retention"]["ceilings"][key][lane] += .1
        elif name == "parameter_chain":
            p["losses"][0]["retained_update"]["before"] = "foreign"
        elif name == "accepted_count":
            p["accepted_parameter_updates"] += 1
        else:
            key = previous["train"][0]
            lane = next(iter(q["candidate"][key]["assessment"]["instruments"]))
            q["candidate"][key]["assessment"]["instruments"][lane]["prediction_loss"] = \
                previous_payload["candidate"][key]["assessment"]["instruments"][lane]["prediction_loss"]+.1
        try:
            read(body, m, p, q, previous, previous_payload, start, final)
        except ValueError as exc:
            controls.append(dict(name=name, code=str(exc)))
        else:
            raise AssertionError(name)
    return controls
