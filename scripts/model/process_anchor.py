"""Termination forecasts from typed region mass, grounded only in prior executions."""
import math

import torch

from claims.codata_source import check
from claims.episodic_memory import digest


CLASSES = ("returncode_zero", "returncode_nonzero", "unobserved_region")


def bank(inputs, units):
    check(units and len(units) == len(set(units)), "PROCESS_ANCHOR_ROSTER")
    observed = getattr(inputs, "observed_regions", {})
    rows = []
    for unit in units:
        response = observed.get(str(unit))
        row = dict(unit=unit, outcome=2, observation=None)
        if response is not None:
            check(response["kind"] == "captured_process_output" and response["unit"] == unit,
                  "PROCESS_ANCHOR_REGION")
            provider = inputs.execution_inputs.for_observation(response["observation"])
            check(response["native"] == provider.checked(response["observation"])
                  and response["target"] == provider.observations.entries[response["observation"]]["target"]
                  and response["target"]["partition"] == "train", "PROCESS_ANCHOR_BINDING")
            actual = provider.observations.read(response["observation"])
            code = actual["result"]["returncode"]
            check(type(code) is int, "PROCESS_ANCHOR_RETURNCODE")
            row = dict(unit=unit, outcome=int(code != 0), observation=response["observation"],
                       observation_set=provider.observations.pin, native=response["native"],
                       returncode=code, finished=actual["finished"])
        rows.append(row)
    value = dict(schema="process-anchor-bank-v1", classes=list(CLASSES), rows=rows,
                 scope="prior_captured_termination_not_program_correctness")
    return dict(pin=digest(value), **value)


def tensors(output, labels, allowed):
    """Sum scalar probability mass by observed outcome; never mix native vectors."""
    check(labels.dtype == torch.long and labels.shape == allowed.shape
          and labels.device == allowed.device and allowed.dtype == torch.bool
          and bool(allowed.any()) and bool(((labels >= 0) & (labels < 3)).all()),
          "PROCESS_ANCHOR_LABELS")
    result = {}
    for name, logits in {**output["regions"], "common":output["constellation_regions"]}.items():
        check(logits.shape == (1,len(labels)) and logits.device == labels.device
              and bool(torch.isfinite(logits[:,allowed]).all())
              and bool(torch.isneginf(logits[:,~allowed]).all()), "PROCESS_ANCHOR_MASK")
        regions = torch.softmax(logits.to(torch.float64), dim=-1)[0]
        outcomes = torch.stack([regions[labels == k].sum() for k in range(3)])
        result[name] = (regions, outcomes)
    return result


def predict(output, description, allowed):
    check(description["pin"] == digest({k:v for k,v in description.items() if k != "pin"}),
          "PROCESS_ANCHOR_BANK_PIN")
    labels = torch.tensor([r["outcome"] for r in description["rows"]], dtype=torch.long,
                          device=allowed.device)
    values = tensors(output, labels, allowed)
    result = dict(schema="process-termination-prediction-v1", bank=description,
        instruments={n:dict(region_probabilities=r.detach().cpu().double().tolist(),
                            probabilities=p.detach().cpu().double().tolist())
                     for n,(r,p) in values.items() if n != "common"},
        common=dict(region_probabilities=values["common"][0].detach().cpu().double().tolist(),
                    probabilities=values["common"][1].detach().cpu().double().tolist()),
        calibrated=False, scientific_admission=False, serving=False,
        scope="termination_only_with_explicit_unsupported_mass")
    validate(result)
    return result


def validate(prediction):
    check(prediction["schema"] == "process-termination-prediction-v1"
          and prediction["calibrated"] is False and prediction["scientific_admission"] is False
          and prediction["serving"] is False, "PROCESS_ANCHOR_SCOPE")
    description = prediction["bank"]
    check(description["pin"] == digest({k:v for k,v in description.items() if k != "pin"})
          and description["classes"] == list(CLASSES), "PROCESS_ANCHOR_BANK_PIN")
    rows = description["rows"]
    check(rows and len({r["unit"] for r in rows}) == len(rows), "PROCESS_ANCHOR_ROSTER")
    for row in rows:
        if row["observation"] is None:
            check(row == dict(unit=row["unit"], outcome=2, observation=None),
                  "PROCESS_ANCHOR_UNOBSERVED_LABEL")
        else:
            check(type(row["returncode"]) is int and row["outcome"] == int(row["returncode"] != 0)
                  and row["native"]["observation"] == row["observation"], "PROCESS_ANCHOR_LABEL")
    for value in (*prediction["instruments"].values(), prediction["common"]):
        p, regions = value["probabilities"], value["region_probabilities"]
        check(len(p) == 3 and len(regions) == len(rows)
              and all(isinstance(v, (int,float)) and math.isfinite(v) and 0 <= v <= 1+1e-12
                      for v in (*p,*regions))
              and abs(sum(p)-1) < 1e-12 and abs(sum(regions)-1) < 1e-12, "PROCESS_ANCHOR_MASS")
        expected = [sum(v for row,v in zip(rows,regions) if row["outcome"] == k) for k in range(3)]
        check(all(abs(a-b) < 1e-12 for a,b in zip(p,expected)), "PROCESS_ANCHOR_PUSHFORWARD")


def score(prediction, observed, observation):
    validate(prediction)
    code = observed["result"]["returncode"]
    check(type(code) is int and all(r["observation"] != observation for r in prediction["bank"]["rows"]),
          "PROCESS_ANCHOR_TARGET_LEAK")
    truth = [float(k == int(code != 0)) for k in range(3)]
    rows = [r for r in prediction["bank"]["rows"] if r["observation"] is not None]
    latest = max(rows, key=lambda r:r["unit"]) if rows else None
    prior = [float(k == latest["outcome"]) for k in range(3)] if latest else [0.,0.,1.]
    def assess(value):
        p = value["probabilities"]
        brier = sum((x-y)**2 for x,y in zip(p,truth))
        return dict(brier=brier, uniform_brier=.5,
                    prior_observed_brier=sum((x-y)**2 for x,y in zip(prior,truth)),
                    unsupported_mass=p[2], zero_returncode_mass_interval=[p[0],min(1.,p[0]+p[2])])
    return dict(schema="process-termination-score-v1", observation=observation, returncode=code,
        outcome=int(code != 0), prediction=digest(prediction),
        instruments={n:assess(v) for n,v in prediction["instruments"].items()},
        common=assess(prediction["common"]), prior_observation=None if latest is None else latest["observation"],
        calibrated=False, task_correctness_established=False)
