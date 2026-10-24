"""Read the shared region objective independently of model/loss execution."""
import base64

import numpy as np
import torch

from audit.tool_native_recognition import scores
from audit.constellation_readback import edit_distance
from claims.codata_source import check
from audit import tool_region_risk


def logsumexp(v):
    high = np.max(v)
    return float(high+np.log(np.exp(v-high).sum()))


def inspect_start(data, model, key, output):
    evidence = output["constellation_regions"].detach().cpu().numpy()
    label = data.examples[key]["region"]
    return dict(forecast=key, evidence=evidence[0].tolist(),
                shared_loss=(logsumexp(evidence[0].astype(np.float64))-float(evidence[0, label]))
                            /len(model.specs))


def controls(data, model, key, output, generation, refuse):
    from model.constellation_learning import shared_region_loss
    e = data.examples[key]
    label = torch.tensor([e["region"]], dtype=torch.long, device=data.session.device)
    loss = lambda out, target=label, allowed=e["allowed"]: shared_region_loss(out, target, allowed, model.specs)
    missing = {k: v for k, v in output.items() if k != "constellation_regions"}
    refuse(f"shared_evidence_absent_{generation}", "LEARNING_REGION_EVIDENCE", lambda: loss(missing))
    nonfinite = output["constellation_regions"].detach().clone()
    nonfinite[:, label] = torch.nan
    refuse(f"shared_nonfinite_{generation}", "LEARNING_REGION_MASK",
           lambda: loss(dict(output, constellation_regions=nonfinite)))
    mask = e["allowed"].clone()
    mask[(e["region"]+1) % len(mask)] = False
    refuse(f"shared_mask_leak_{generation}", "LEARNING_REGION_MASK", lambda: loss(output, allowed=mask))
    outside = label+len(mask)
    refuse(f"shared_label_outside_{generation}", "LEARNING_REGION_LABEL", lambda: loss(output, target=outside))
    saved = data.plan["region_objective"]
    try:
        data.plan["region_objective"] = "independent_instrument_classification"
        refuse(f"old_objective_{generation}", "LEARNING_OBJECTIVE", lambda: data.loss(model))
    finally:
        data.plan["region_objective"] = saved


def read_loss(inputs, records, labels, hidden, start, proof):
    """Read the first actual update, with original windows and serialized outputs."""
    core, extra = start[2:4]
    supervision = {s["forecast"]: s for s in proof["supervision"]}
    evidence = {s["forecast"]: s for s in proof["objective"]}
    step = proof["losses"][0]
    risk_mode = proof.get("region_risk_objective")
    check(risk_mode in (None, tool_region_risk.OBJECTIVE), "RISK_READ_OBJECTIVE")
    risk_proof = {s["forecast"]: s for s in proof.get("region_risks", [])}
    check(set(risk_proof) == (set(evidence) if risk_mode else set()), "RISK_READ_ROSTER")
    check(set(evidence) == set(supervision) == set(proof["decision"]["train"])
          == set(step["details"]), "SHARED_READ_ROSTER")
    totals, rows = [], []
    for key, sample in evidence.items():
        r, label = records[key], labels[key]
        logits, _ = scores(inputs, r, core, extra, hidden[key])
        common = sum(v-logsumexp(v) for v in logits.values())
        check(np.max(np.abs(common-np.asarray(sample["evidence"]))) < 2e-5, "SHARED_READ_EVIDENCE")
        loss = (logsumexp(common)-common[label])/len(inputs.specs)
        check(abs(loss-sample["shared_loss"]) < 1e-5, "SHARED_READ_LOSS")
        request = r["request"]
        target = inputs.population.lookup[request["task"], request["turn"]+1]["unit"]
        placement = []
        for spec in inputs.specs:
            p = supervision[key]["predictions"][spec.name]
            prediction = np.frombuffer(base64.b64decode(p["base64"], validate=True), dtype="<f4")
            prediction = prediction.reshape(p["shape"]).astype(np.float64)
            a, ptr = inputs.arrays[spec.name]
            actual = a[ptr[target]:ptr[target+1]].astype(np.float64)
            prediction /= np.linalg.norm(prediction, axis=1, keepdims=True)
            actual /= np.linalg.norm(actual, axis=1, keepdims=True)
            value = edit_distance(np.clip(prediction@actual.T, -1, 1).tolist())
            detail = step["details"][key][spec.name]
            diagnostic = logsumexp(logits[spec.name])-logits[spec.name][label]
            check(abs(detail["region"]-diagnostic) < 1e-5
                  and abs(detail["shared_region"]-loss) < 1e-5
                  and abs(detail["placement"]-value) < 1e-5, "SHARED_READ_COMPONENT")
            placement.append(value)
        risk = tool_region_risk.read(inputs, r, common, risk_proof[key], step["details"][key]) if risk_mode else 0.
        total = loss+sum(placement)/len(placement)+risk
        totals.append(total)
        rows.append(dict(forecast=key, shared_loss=float(loss), region_risk=risk, total=float(total)))
    expected = sum(totals)/len(totals)
    check(abs(expected-step["loss"]) < 1e-5, "SHARED_READ_UPDATE")
    return dict(examples=rows, expected=expected, recorded=step["loss"],
                absolute_error=abs(expected-step["loss"]))
