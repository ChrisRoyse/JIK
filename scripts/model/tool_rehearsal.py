"""Learn published outcomes without rewriting the inputs or historical evaluation."""
from pathlib import Path

import numpy as np
import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from model.constellation import ordered_distance, score
from model.constellation_learning import identity_pin
from model.tool_growth import GrowingToolLoop
from model.tool_learning import ToolLearningData, candidate_decision
from model import process_anchor, retained_update, tool_context_regions, tool_learner
from model import rehearsal_retention


RULE = "published_train_regions_original_history_v1"
OBJECTIVE = "native_risk_plus_lane_and_normalized_common_termination_brier_v1"


def code_pins():
    root = Path(__file__).parent
    return {"model/"+name:sha((root/name).read_bytes()) for name in
            ("tool_rehearsal.py", "rehearsal_training.py", "rehearsal_retention.py")}


def training_view(session, record, observations):
    request = record["request"]
    check(session.source.by_task[request["task"]]["partition"] == "train",
          "REHEARSAL_TRAIN_SPLIT")
    route = session.routing.route(request["reference"])
    desc = tool_context_regions.definition(session, request, record["support"], route)
    historical = record["region_context"]["definition"]
    check(desc["history"] == historical["history"]
          and desc.get("observed_echoes", {}) == historical.get("observed_echoes", {}),
          "REHEARSAL_INPUT_HISTORY")
    published = getattr(session.inputs, "observed_regions", {})
    shared = [u for u in desc["shared"] if str(u) not in published
              or published[str(u)]["observation"] in observations]
    desc.update(shared=shared, units=list(dict.fromkeys(
        shared+[p["reference"]["unit"] for p in desc["private"]])),
        rehearsal=dict(rule=RULE, head=session.head, forecast=digest(record),
                       historical_view=record["region_context"]["world"],
                       training_observations=sorted(observations)))
    return desc, tool_context_regions.make_view(session.inputs, session.model, desc,
                                               session.execution_inputs)


class RehearsalData(ToolLearningData):
    def __init__(self, session, *, train, evaluate, continuation=None):
        # The original constructor authenticates all history and computes historical costs.
        super().__init__(session, train=train, evaluate=evaluate)
        check(all("execution" in session.forecasts[k] and "region_context" in
                  session.forecasts[k]["record"] for k in self.keys), "REHEARSAL_CAPTURE_REQUIRED")
        self.plan.update(training_regions=RULE, outcome_objective=OBJECTIVE,
                         grounded_retention=rehearsal_retention.RULE,
                         scope="known_outcome_rehearsal_historical_evaluation")
        self.binding["code"] = dict(self.binding["code"], **code_pins())
        self.binding["training_views"] = {}
        observations = {session.forecasts[k]["execution"]["observation"] for k in train}
        self.labels, self.outcomes, self.actual = {}, {}, {}
        for key, e in self.examples.items():
            observation = session.forecasts[key]["execution"]["observation"]
            actual = session.observations.read(observation)
            self.actual[key] = (actual, observation)
            self.outcomes[key] = int(actual["result"]["returncode"] != 0)
            if key in train:
                desc, view = training_view(session, e["record"], observations)
                units = desc["units"]
                targets = [int(u) for u,r in session.inputs.observed_regions.items()
                           if r["observation"] == observation]
                check(len(targets) == 1 and targets[0] in units, "REHEARSAL_TARGET_NOT_ROUTED")
                index = units.index(targets[0])
                costs, reused, measured = {}, {}, {}
                old_positions = {u:i for i,u in enumerate(e["units"])}
                for spec in self.model.specs:
                    old = e["region_costs"][spec.name][1]
                    values, ptr = session.inputs.arrays[spec.name]
                    target = e["target"][spec.name]
                    native = target.values[0, ~target.padding[0]]
                    ref = view.references()[spec.name]
                    check(torch.equal(ref.values[index, ~ref.padding[index]], native),
                          "REHEARSAL_TARGET_NATIVE_JOIN")
                    target_values = native.cpu().numpy().astype(np.float64)
                    target_values /= np.linalg.norm(target_values, axis=1, keepdims=True)
                    lane, reused[spec.name], measured[spec.name] = [], 0, 0
                    for unit in units:
                        if unit in old_positions:
                            lane.append(old[old_positions[unit]].detach())
                            reused[spec.name] += 1
                            continue
                        rows = values[ptr[unit]:ptr[unit+1]].astype(np.float64)
                        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
                        check(len(rows)*len(target_values) <= self.model.config.max_attention_elements,
                              "TOOL_LEARNING_BUDGET")
                        distance = ordered_distance(np.clip(target_values@rows.T, -1, 1))
                        lane.append(old.new_tensor(distance))
                        measured[spec.name] += 1
                    costs[spec.name] = (spec.identity, torch.stack(lane))
                allowed = torch.ones(len(units), dtype=torch.bool, device=session.device)
                e.update(view=view, units=units, region=index, allowed=allowed,
                         cutoff=allowed.clone(), region_costs=costs)
                self.binding["training_views"][key] = dict(definition=desc, world=view.world_id,
                    reused_costs=reused, new_costs=measured, target_region=targets[0])
            bank = e["view"].execution_anchor_bank
            self.labels[key] = torch.tensor([r["outcome"] for r in bank["rows"]],
                                           dtype=torch.long, device=session.device)
            if key in evaluate:
                check(all(r["observation"] != observation for r in bank["rows"]),
                      "PROCESS_ANCHOR_TARGET_LEAK")
                check(e["view"].world_id == e["record"]["region_context"]["world"],
                      "REHEARSAL_EVALUATION_VIEW")
        if continuation is not None:
            tool_learner.bind(self, continuation)
        rehearsal_retention.bind(self)
        self.fresh()

    def recognition_risk(self, model, key, output):
        check(self.plan["outcome_objective"] == OBJECTIVE, "REHEARSAL_OBJECTIVE")
        native = super().recognition_risk(model, key, output)
        masses = process_anchor.tensors(output, self.labels[key], self.examples[key]["allowed"])
        truth = masses["common"][1].new_tensor([float(i == self.outcomes[key]) for i in range(3)])
        common = (masses["common"][1]-truth).square().sum()/len(model.specs)
        return {s.name:native[s.name]+(masses[s.name][1]-truth).square().sum()+common
                for s in model.specs}

    def evaluate(self, model):
        self.fresh()
        results = {}
        with torch.no_grad():
            for key,e in self.examples.items():
                output = self.forward(model, key)
                constellation = output["predicted_constellation"]
                packed = constellation.pack(model.specs, model.config)
                p = process_anchor.predict(output, e["view"].execution_anchor_bank, e["allowed"])
                actual, observation = self.actual[key]
                if key in self.plan["evaluate"]:
                    grade = process_anchor.score(p, actual, observation)
                else:
                    # Deliberately separate from the prospective scorer, which rejects own-target banks.
                    truth = self.outcomes[key]
                    def assess(v):
                        return dict(brier=sum((x-float(i == truth))**2 for i,x in
                                             enumerate(v["probabilities"])),
                                    unsupported_mass=v["probabilities"][2])
                    grade = dict(scope="known_target_rehearsal_not_prospective",
                        observation=observation, outcome=truth, calibrated=False,
                        instruments={n:assess(v) for n,v in p["instruments"].items()},
                        common=assess(p["common"]))
                results[key] = dict(region=constellation.regions[0], predictions=packed,
                    identity=identity_pin(output), assessment=score(packed, self.session.inputs,
                        e["record"]["baseline_unit"], e["target_unit"], model.config,
                        target_inputs=e["target_inputs"]), termination=dict(prediction=p, score=grade))
        return results


def termination_gate(data, baseline, candidate, result):
    comparisons, retained = {}, {}
    prior = None if data.continuation_name is None else data.session.decisions[-1]
    previous = {} if prior is None else data.session.journal.payload(prior["payload"])["candidate"]
    for key in data.plan["evaluate"]:
        comparisons[key] = {}
        new, old = candidate[key]["termination"]["score"], baseline[key]["termination"]["score"]
        for name in (*new["instruments"], "common"):
            a = new["common"] if name == "common" else new["instruments"][name]
            b = old["common"] if name == "common" else old["instruments"][name]
            comparisons[key][name] = dict(candidate=a["brier"], parent=b["brier"],
                uniform=a["uniform_brier"], last_observed=a["prior_observed_brier"],
                passed=a["brier"] < min(b["brier"],a["uniform_brier"],a["prior_observed_brier"]))
            if key in previous:
                old_score = previous[key]["termination"]["score"]
                before = old_score["common"] if name == "common" else old_score["instruments"][name]
                retained.setdefault(key,{})[name] = dict(previous=before["brier"],
                    candidate=a["brier"],passed=a["brier"] <= before["brier"])
    passed = all(v["passed"] for g in comparisons.values() for v in g.values())
    kept = all(v["passed"] for g in retained.values() for v in g.values())
    return dict(result, metric_gate_passed=result["metric_gate_passed"] and passed and kept,
        termination=dict(comparisons=comparisons, passed=passed, retention=retained, retained=kept),
        reasons=result["reasons"]+([] if passed else ["WITHHELD_TERMINATION_GATE_FAILED"])
                +([] if kept else ["RETAINED_TERMINATION_REGRESSION"]))


class RehearsalToolLoop(GrowingToolLoop):
    def consider_rehearsal(self, name, *, pin, train, evaluate, parent, continuation=None, expected=None):
        self._fresh()
        check(all(r["candidate"] != name for r in self.decisions), "TOOL_DUPLICATE_CANDIDATE")
        data = RehearsalData(self, train=train, evaluate=evaluate, continuation=continuation)
        body, payload = candidate_decision(data, name, pin, parent)
        result = termination_gate(data, payload["baseline"], payload["candidate"], payload["result"])
        result = rehearsal_retention.assess(data, payload["candidate"], result)
        payload["result"] = result
        body.update(kind="rehearsal_candidate", code=code_pins(), result=result,
                    payload=sha(canonical(payload)))
        if expected is None:
            self.journal.append(body, payload)
        else:
            check(body == expected and payload == self.journal.payload(expected["payload"]),
                  "REHEARSAL_DECISION_REPLAY")
        self.decisions.append(body)
        return body

    def _replay(self, body):
        if body.get("kind") == "rehearsal_candidate":
            check(body["code"] == code_pins(), "REHEARSAL_CODE_PIN")
            self.consider_rehearsal(body["candidate"], pin=body["pin"], train=body["train"],
                evaluate=body["evaluate"], parent=body["parent"],
                continuation=body.get("continuation"), expected=body)
        else:
            super()._replay(body)
