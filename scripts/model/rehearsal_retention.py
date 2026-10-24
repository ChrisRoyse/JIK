"""Train-only native and outcome retention, including bank-transition repair."""
import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation import score
from model.constellation_learning import ordered_loss, trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model.retention_projection import project
from model.tool_retention import metric
from model import process_anchor, retained_update


RULE = "grounded_retention_fixed_ceiling_monotonic_repair_v1"


def measurements(item):
    outcome = item["termination"]["score"]
    return dict(native={n:metric(v["prediction_loss"]) for n,v in
                        item["assessment"]["instruments"].items()},
                termination={n:metric(v["brier"]) for n,v in
                             dict(outcome["instruments"], common=outcome["common"]).items()})


def bind(data):
    retained_update.bind(data)
    ceilings = {}
    prior = data.binding["training_retention"]["predecessor"]
    if prior is not None:
        payload = data.session.journal.payload(prior["payload"])
        for key in data.retained_keys:
            item = payload["candidate"][key]
            check(item["termination"]["score"]["scope"] ==
                  "known_target_rehearsal_not_prospective", "GROUNDED_RETENTION_SCOPE")
            ceilings[key] = measurements(item)
            check(ceilings[key]["native"] == data.retained_ceilings[key],
                  "GROUNDED_RETENTION_NATIVE")
    saved = dict(rule=RULE, keys=list(data.retained_keys), ceilings=ceilings,
                 predecessor=prior, scope="earlier_training_only_current_bank",
                 bank_transition="repair_pending_until_original_ceilings_met")
    data.binding["grounded_retention"] = saved
    data.grounded_retention_pin = digest(saved)
    validate(data)


def validate(data):
    retained_update.validate(data)
    saved = data.binding["grounded_retention"]
    names = {s.name for s in data.model.specs}
    check(data.plan.get("grounded_retention") == RULE and saved["rule"] == RULE
          and digest(saved) == data.grounded_retention_pin,
          "GROUNDED_RETENTION_BINDING")
    check(saved["keys"] == data.retained_keys
          and set(saved["ceilings"]) == set(data.retained_keys),
          "GROUNDED_RETENTION_SPLIT")
    for key,row in saved["ceilings"].items():
        check(set(row) == {"native", "termination"} and set(row["native"]) == names
              and set(row["termination"]) == names | {"common"}
              and row["native"] == data.retained_ceilings[key], "GROUNDED_RETENTION_ROSTER")
        for values in row.values():
            for value in values.values():
                metric(value)


def ordered(data):
    return [(key,kind,name) for key in data.retained_keys
            for kind,names in (("native", [s.name for s in data.model.specs]),
                               ("termination", [s.name for s in data.model.specs]+["common"]))
            for name in names]


def scores(data, model):
    validate(data)
    result = {}
    with torch.no_grad():
        for key in data.retained_keys:
            e = data.examples[key]
            output = data.forward(model, key)
            measured = score(output["predicted_constellation"].pack(model.specs, model.config),
                             data.session.inputs, e["record"]["baseline_unit"], e["target_unit"],
                             model.config, target_inputs=e["target_inputs"])
            p = process_anchor.predict(output, e["view"].execution_anchor_bank, e["allowed"])
            outcome = data.outcomes[key]
            result[key] = dict(native={n:metric(v["prediction_loss"]) for n,v in
                                       measured["instruments"].items()},
                termination={n:sum((x-float(i == outcome))**2 for i,x in enumerate(v["probabilities"]))
                             for n,v in dict(p["instruments"], common=p["common"]).items()})
    return result


def comparisons(data, measured):
    validate(data)
    ceilings = data.binding["grounded_retention"]["ceilings"]
    check(set(measured) == set(ceilings), "GROUNDED_RETENTION_ROSTER")
    for key,row in measured.items():
        check(set(row) == set(ceilings[key]) and all(set(v) == set(ceilings[key][kind])
              for kind,v in row.items()), "GROUNDED_RETENTION_ROSTER")
    return [dict(key=key, metric=kind, instrument=name, ceiling=ceilings[key][kind][name],
                 value=metric(measured[key][kind][name]),
                 retained=measured[key][kind][name] <= ceilings[key][kind][name])
            for key,kind,name in ordered(data)]


def advances(data, before, after):
    """An unresolved violation must improve; a satisfied constraint may not break."""
    comparisons(data, before)
    rows = comparisons(data, after)
    return all(row["retained"] or (before[row["key"]][row["metric"]][row["instrument"]]
                                    > row["ceiling"] and row["value"] <
                                    before[row["key"]][row["metric"]][row["instrument"]])
               for row in rows)


def prepare(data, model):
    data.fresh()
    validate(data)
    params = dict(trainable(model))
    rows = []
    for key in data.retained_keys:
        output = data.forward(model, key)
        e = data.examples[key]
        terms = []
        for spec in model.specs:
            slot, target = output["predicted_constellation"].slots[spec.name], e["target"][spec.name]
            terms.append(ordered_loss(slot.values[0, ~slot.padding[0]],
                                      target.values[0, ~target.padding[0]]))
        masses = process_anchor.tensors(output, data.labels[key], e["allowed"])
        truth = masses["common"][1].new_tensor([float(i == data.outcomes[key]) for i in range(3)])
        terms.extend((masses[name][1]-truth).square().sum()
                     for name in [s.name for s in model.specs]+["common"])
        for i,term in enumerate(terms):
            gradients = torch.autograd.grad(term, tuple(params.values()),
                                           retain_graph=i+1 < len(terms), allow_unused=True)
            # Parameter coordinates share an optimizer; instrument readings remain typed.
            rows.append(torch.cat([(g if g is not None else torch.zeros_like(p)).detach().reshape(-1).double()
                                   for g,p in zip(gradients, params.values())]))
    return dict(gradients=None if not rows else torch.stack(rows), scores=scores(data, model),
                before=digest(tensor_fingerprint(params)), order=ordered(data))


def apply(data, model, before, prepared):
    validate(data)
    params = dict(trainable(model))
    check(set(before) == set(params) and digest(tensor_fingerprint(before)) == prepared["before"]
          and prepared["order"] == ordered(data), "GROUNDED_RETENTION_PARAMETERS")
    report = dict(rule=RULE, protected=len(prepared["order"]), before=prepared["before"],
                  parameter_coordinates=sum(p.numel() for p in params.values()),
                  constraint_order=prepared["order"], before_scores=prepared["scores"],
                  before_comparisons=comparisons(data, prepared["scores"]))
    if not data.retained_keys:
        return dict(report, accepted=True, scale=1., projections={}, trials=[],
                    gradient_bytes=0, retained=True, after_scores={},
                    after=digest(tensor_fingerprint(params)))
    g = prepared["gradients"]
    check(g.shape == (len(prepared["order"]), report["parameter_coordinates"]),
          "GROUNDED_RETENTION_GRADIENTS")
    displacement = torch.cat([(p.detach()-before[n]).reshape(-1).double() for n,p in params.items()])
    delta, projection = project(g, displacement)
    trials, accepted, scale = [], False, 0.
    with torch.no_grad():
        if delta is not None:
            for scale in retained_update.SCALES:
                start = 0
                for n,p in params.items():
                    p.copy_((before[n].double()+scale*delta[start:start+p.numel()].reshape(p.shape)).to(p.dtype))
                    start += p.numel()
                measured = scores(data, model)
                passed = advances(data, prepared["scores"], measured)
                trials.append(dict(scale=scale, scores=measured, advances=passed,
                                   retained=all(r["retained"] for r in comparisons(data, measured))))
                if passed:
                    accepted = any(not torch.equal(p, before[n]) for n,p in params.items())
                    break
        if not accepted:
            scale = 0.
            for n,p in params.items():
                p.copy_(before[n])
            check(scores(data, model) == prepared["scores"], "GROUNDED_RETENTION_RESTORE")
    after = scores(data, model)
    kept = all(r["retained"] for r in comparisons(data, after))
    return dict(report, accepted=accepted, scale=scale, projections=projection, trials=trials,
                gradient_bytes=g.numel()*g.element_size(), after_scores=after, retained=kept,
                after=digest(tensor_fingerprint(params)),
                reason=("accepted_retained" if kept else "accepted_repair_pending") if accepted
                       else "rejected_no_parameter_change")


def assess(data, candidate, result):
    measured = {k:measurements(candidate[k]) for k in data.retained_keys}
    rows = comparisons(data, measured)
    kept = all(r["retained"] for r in rows)
    return dict(result, metric_gate_passed=result["metric_gate_passed"] and kept,
                grounded_training_retention=dict(rule=RULE, comparisons=rows, retained=kept),
                reasons=result["reasons"]+([] if kept else ["GROUNDED_TRAINING_RETENTION_PENDING"]))
