"""Train-only, per-instrument retention checks around actual Adam proposals."""
import scipy
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation import score
from model.constellation_learning import ordered_loss, trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model.retention_projection import project
from model.tool_retention import metric


RULE = "retained_native_descent_v1"
SCALES = (1., .5, .25, .125)
MAX_RETAINED = 30


def bind(data):
    keys, ceilings, predecessor = [], {}, None
    if data.continuation_name is not None:
        prior = data.session.decisions[-1]
        check(prior["candidate"] == data.continuation_name, "RETAINED_UPDATE_PREDECESSOR")
        keys = list(prior["train"])
        check(keys and len(keys) <= MAX_RETAINED and len(set(keys)) == len(keys)
              and set(keys) <= set(data.plan["train"])
              and not set(keys) & set(data.plan["evaluate"]), "RETAINED_UPDATE_SPLIT")
        payload = data.session.journal.payload(prior["payload"])
        for key in keys:
            slots = payload["candidate"][key]["assessment"]["instruments"]
            check(set(slots) == {s.name for s in data.model.specs}, "RETAINED_UPDATE_ROSTER")
            ceilings[key] = {}
            for spec in data.model.specs:
                check(slots[spec.name]["instrument"] == spec.identity, "RETAINED_UPDATE_INSTRUMENT")
                ceilings[key][spec.name] = metric(slots[spec.name]["prediction_loss"])
        predecessor = dict(candidate=prior["candidate"], pin=prior["pin"], payload=prior["payload"])
    data.retained_keys, data.retained_ceilings = keys, ceilings
    data.binding["training_retention"] = dict(rule=RULE, scipy=scipy.__version__,
        keys=keys, ceilings=ceilings, predecessor=predecessor, scope="earlier_training_examples_only")


def validate(data):
    check(data.plan.get("retained_update") == RULE
          and data.plan.get("retention_scipy") == scipy.__version__, "RETAINED_UPDATE_RULE")
    saved = data.binding["training_retention"]
    check(data.retained_keys == saved["keys"] and data.retained_ceilings == saved["ceilings"]
          and set(data.retained_keys) <= set(data.plan["train"])
          and not set(data.retained_keys) & set(data.plan["evaluate"]), "RETAINED_UPDATE_SPLIT")


def lane_parameters(model):
    return {s.name: [(n, p) for n, p in trainable(model)
                    if n.startswith((f"query.{s.name}.", f"place.{s.name}."))]
            for s in model.specs}


def prepare(data, model):
    data.fresh()
    validate(data)
    if not data.retained_keys:
        return {}
    groups = lane_parameters(model)
    rows = {s.name: [] for s in model.specs}
    for key in data.retained_keys:
        output = data.forward(model, key)["predicted_constellation"]
        for i, spec in enumerate(model.specs):
            slot, target = output.slots[spec.name], data.examples[key]["target"][spec.name]
            loss = ordered_loss(slot.values[0, ~slot.padding[0]], target.values[0, ~target.padding[0]])
            parameters = [p for _, p in groups[spec.name]]
            gradient = torch.autograd.grad(loss, parameters, retain_graph=i+1 < len(model.specs),
                                           allow_unused=True)
            rows[spec.name].append(torch.cat([
                (g if g is not None else torch.zeros_like(p)).detach().reshape(-1).double()
                for g, p in zip(gradient, parameters)]))
    return {name: torch.stack(values) for name, values in rows.items()}


def scores(data, model):
    validate(data)
    result = {}
    with torch.no_grad():
        for key in data.retained_keys:
            e = data.examples[key]
            output = data.forward(model, key)["predicted_constellation"].pack(model.specs, model.config)
            measured = score(output, data.session.inputs, e["record"]["baseline_unit"],
                             e["target_unit"], model.config, target_inputs=e.get("target_inputs"))
            result[key] = {s.name: metric(measured["instruments"][s.name]["prediction_loss"])
                           for s in model.specs}
    return result


def retained(data, measured):
    check(set(measured) == set(data.retained_keys), "RETAINED_UPDATE_ROSTER")
    for key, row in measured.items():
        check(set(row) == set(data.retained_ceilings[key]), "RETAINED_UPDATE_ROSTER")
    return all(metric(value) <= data.retained_ceilings[key][name]
               for key, row in measured.items() for name, value in row.items())


def apply(data, model, before, gradients):
    validate(data)
    params = dict(trainable(model))
    check(set(before) == set(params), "RETAINED_UPDATE_PARAMETERS")
    before_pin = digest(tensor_fingerprint(before))
    if not data.retained_keys:
        return dict(rule=RULE, protected=0, accepted=True, scale=1.,
                    before=before_pin, after=digest(tensor_fingerprint(params)), projections={}, trials=[])
    check(set(gradients) == {s.name for s in model.specs}, "RETAINED_UPDATE_ROSTER")
    groups, deltas, projections = lane_parameters(model), {}, {}
    for name, group in groups.items():
        displacement = torch.cat([(p.detach()-before[n]).reshape(-1).double() for n, p in group])
        delta, report = project(gradients[name], displacement)
        projections[name] = report
        if delta is None:
            break
        start = 0
        for n, p in group:
            deltas[n] = delta[start:start+p.numel()].reshape(p.shape)
            start += p.numel()
    trials, accepted, scale = [], False, 0.
    with torch.no_grad():
        if len(deltas) == len(params):
            for scale in SCALES:
                for n, p in params.items():
                    p.copy_((before[n].double()+scale*deltas[n]).to(p.dtype))
                measured = scores(data, model)
                passed = retained(data, measured)
                trials.append(dict(scale=scale, scores=measured, retained=passed))
                if passed:
                    accepted = any(not torch.equal(p, before[n]) for n, p in params.items())
                    break
        if not accepted:
            scale = 0.
            for n, p in params.items():
                p.copy_(before[n])
            check(retained(data, scores(data, model)), "RETAINED_UPDATE_RESTORE")
    return dict(rule=RULE, protected=len(data.retained_keys)*len(model.specs), accepted=accepted,
                scale=scale, before=before_pin, after=digest(tensor_fingerprint(params)),
                projections=projections, trials=trials,
                reason="accepted_training_retention" if accepted else "rejected_no_parameter_change")
