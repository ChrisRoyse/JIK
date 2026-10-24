"""Independent scalar risk reconstruction from original typed native windows."""
import numpy as np
import torch

from audit.constellation_readback import edit_distance
from claims.codata_source import check


OBJECTIVE = "typed_native_region_expected_risk_v1"


def costs(inputs, record):
    units = record["region_context"]["definition"]["units"]
    request = record["request"]
    target = inputs.population.lookup[request["task"], request["turn"]+1]["unit"]
    result = {}
    for spec in inputs.specs:
        a, ptr = inputs.arrays[spec.name]
        truth = a[ptr[target]:ptr[target+1]].astype(np.float64)
        truth /= np.linalg.norm(truth, axis=1, keepdims=True)
        values = []
        for unit in units:
            v = a[ptr[unit]:ptr[unit+1]].astype(np.float64)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            values.append(edit_distance(np.clip(v@truth.T, -1, 1).tolist()))
        result[spec.name] = np.asarray(values)
    return result


def probabilities(evidence):
    a = np.exp(evidence-np.max(evidence))
    return a/a.sum()


def inspect(data, model, key, output):
    from model.constellation_risk import expected
    e = data.examples[key]
    reference = costs(data.session.inputs, e["record"])
    p = probabilities(output["constellation_regions"].detach().cpu().numpy()[0].astype(np.float64))
    # Differentiate only a detached copy of the real categorical evidence here.
    logits = output["constellation_regions"].detach().clone().requires_grad_(True)
    risk = expected(dict(output, constellation_regions=logits), e["region_costs"], e["allowed"], model.specs)
    rows = {}
    for spec in model.specs:
        name = spec.name
        observed = e["region_costs"][name][1].cpu().numpy()
        check(np.max(np.abs(reference[name]-observed)) < 1e-7, "RISK_READ_NATIVE_COST")
        value = float(p@reference[name])
        gradient = torch.autograd.grad(risk[name], logits, retain_graph=True)[0][0].cpu().numpy()
        derivative = p*(reference[name]-value)
        check(abs(float(risk[name].detach())-value) < 1e-6
              and np.max(np.abs(derivative-gradient)) < 1e-6, "RISK_READ_GRADIENT")
        rows[name] = dict(instrument=spec.identity, costs=observed.tolist(), risk=value,
                          gradient=gradient.tolist(), max_gradient_error=float(np.max(np.abs(derivative-gradient))))
    return dict(forecast=key, lanes=rows, regions=len(p))


def controls(data, model, key, output, generation, refuse):
    from model.constellation_risk import expected
    e = data.examples[key]
    values = e["region_costs"]
    name = model.specs[0].name
    invoke = lambda c, allowed=e["allowed"]: expected(output, c, allowed, model.specs)
    refuse(f"risk_missing_lane_{generation}", "REGION_RISK_ROSTER",
           lambda: invoke({k: v for k, v in values.items() if k != name}))
    refuse(f"risk_foreign_instrument_{generation}", "REGION_RISK_INSTRUMENT",
           lambda: invoke(dict(values, **{name: ("foreign", values[name][1])})))
    for label, number in (("nan", float("nan")), ("negative", -1.)):
        bad = values[name][1].clone()
        bad[0] = number
        refuse(f"risk_{label}_{generation}", "REGION_RISK_COST",
               lambda: invoke(dict(values, **{name: (values[name][0], bad)})))
    refuse(f"risk_wrong_size_{generation}", "REGION_RISK_INSTRUMENT",
           lambda: invoke(dict(values, **{name: (values[name][0], values[name][1][:-1])})))
    mask = e["allowed"].clone()
    mask[0] = False
    refuse(f"risk_mask_leak_{generation}", "REGION_RISK_MASK", lambda: invoke(values, mask))
    refuse(f"risk_withheld_{generation}", "REGION_RISK_SPLIT",
           lambda: data.recognition_risk(model, data.plan["evaluate"][0], output))
    saved = data.plan["region_risk"]
    try:
        data.plan["region_risk"] = "unknown"
        refuse(f"risk_unknown_objective_{generation}", "REGION_RISK_OBJECTIVE",
               lambda: data.recognition_risk(model, key, output))
    finally:
        data.plan["region_risk"] = saved


def read(inputs, record, evidence, sample, detail):
    reference = costs(inputs, record)
    p = probabilities(evidence)
    check(set(sample["lanes"]) == set(reference) and sample["regions"] == len(p), "RISK_READ_ROSTER")
    values = []
    for spec in inputs.specs:
        name = spec.name
        lane = sample["lanes"][name]
        observed = np.asarray(lane["costs"])
        gradient = np.asarray(lane["gradient"])
        expected = float(p@reference[name])
        check(lane["instrument"] == spec.identity and observed.shape == reference[name].shape
              and gradient.shape == reference[name].shape
              and np.max(np.abs(observed-reference[name])) < 1e-7
              and np.max(np.abs(gradient-p*(reference[name]-expected))) < 1e-5
              and abs(lane["risk"]-expected) < 1e-5
              and abs(detail[name]["region_risk"]-expected) < 1e-5, "RISK_READ_SAVED")
        values.append(expected)
    return sum(values)/len(values)
