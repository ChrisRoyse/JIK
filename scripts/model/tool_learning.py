"""Incremental readout learning from eligible task outcomes in the durable loop."""
import base64
import json

import numpy as np
import torch

from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from model.constellation import ordered_distance
from model.constellation_learning import LearningData, RECIPE, frozen_pin, decision
from model.constellation_training_checkpoint import load
from model.constellation_world import ConstellationWorld
from model.world_checkpoint import load as load_world
from model.world_loop import WorldState
from model import tool_context_regions
from model import tool_retention
from model import constellation_risk
from model import retained_update


class ToolLearningData(LearningData):
    def __init__(self, session, *, train, evaluate, continuation=None):
        self.session, self.model = session, session.model
        self.optimizer_offset, self.continuation_name, self.predecessor = 0, None, None
        self.head, self.memory_head = session.head, session.memory.head
        self.plan = dict(**RECIPE, context_interpretation="ordered_native_query_recurrence_v1",
                         region_risk=constellation_risk.OBJECTIVE,
                         retained_update=retained_update.RULE, retention_scipy=retained_update.scipy.__version__,
                         train=list(train), evaluate=list(evaluate))
        self.keys = tuple(train)+tuple(evaluate)
        check(train and evaluate and len(set(self.keys)) == len(self.keys), "TOOL_LEARNING_SPLIT")
        self.fresh()
        items = {k: session.eligible(k) for k in self.keys}
        train_tasks = {items[k]["record"]["request"]["task"] for k in train}
        eval_tasks = {items[k]["record"]["request"]["task"] for k in evaluate}
        check(not train_tasks & eval_tasks and
              all(session.source.by_task[t]["partition"] == "train" for t in train_tasks) and
              all(session.source.by_task[t]["partition"] == "withheld" for t in eval_tasks),
              "TOOL_LEARNING_SPLIT")
        self.frozen = frozen_pin(self.model)
        self.binding = dict(head=self.head, memory_head=self.memory_head,
            model=session.active_checkpoint, contract=session.journal.pin,
            standard=self.model.standard_id, world=self.model.world_id, plan=self.plan,
            code=session.journal.contract["code"],
            evidence={k: dict(record=digest(v["record"]), result=digest(v["result"]),
                             target=v["target_episode"]) for k, v in items.items()})
        for key, item in items.items():
            if "execution" in item:
                self.binding["evidence"][key]["execution"] = item["execution"]
        if continuation is not None:
            from model.tool_learner import bind
            bind(self, continuation)
        retained_update.bind(self)
        self.examples = {}
        for key, item in items.items():
            record, target_unit = item["record"], item["target_unit"]
            target_inputs = session.inputs
            if "execution" in item:
                target_inputs = session.execution_inputs.for_observation(item["execution"]["observation"])
                target_unit = target_inputs.lookup[item["execution"]["observation"]]
            request, prior = record["request"], record["prior"]
            allowed = session.allowed.clone()
            view, units = None, session.inputs.world_units
            if "routing" in record and "region_context" not in record:
                router = session.routing_for(record["routing"]["kernel"])
                check(record["routing"] == router.route(request["reference"]) and
                      session.inputs.world_units[:len(router.inputs.world_units)] == router.inputs.world_units,
                      "TOOL_LEARNING_ROUTING")
                allowed.zero_()
                allowed[record["routing"]["allowed"]] = True
            if "region_context" in record:
                view, units = tool_context_regions.historical(session, record)
                allowed = torch.ones(len(units), dtype=torch.bool, device=session.device)
            raw = base64.b64decode(prior["hidden"], validate=True)
            shape = (len(self.model.specs), self.model.config.width)
            check(len(raw) == np.prod(shape)*4 and prior["sequence"] == request["sequence"],
                  "TOOL_PREACT")
            hidden = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(shape)).to(session.device)
            check(bool(torch.isfinite(hidden).all()), "TOOL_PREACT")
            state = WorldState(request["context"], self.model.standard_id, prior["sequence"],
                               hidden, prior["last_event"], self.model.world_id)
            # Label only after checked feedback; every candidate region predates the target.
            costs = np.where(allowed.cpu().numpy(), 0., np.inf)
            lane_costs = {}
            visible = set(allowed.nonzero().flatten().tolist())
            for spec in self.model.specs:
                lane = np.zeros(len(units), dtype=np.float32)
                values, ptr = session.inputs.arrays[spec.name]
                targets, target_ptr = target_inputs.arrays[spec.name]
                target = targets[target_ptr[target_unit]:target_ptr[target_unit+1]].astype(np.float64)
                target /= np.linalg.norm(target, axis=1, keepdims=True)
                for i, unit in enumerate(units):
                    if i not in visible:
                        continue
                    old = values[ptr[unit]:ptr[unit+1]].astype(np.float64)
                    old /= np.linalg.norm(old, axis=1, keepdims=True)
                    check(len(target)*len(old) <= self.model.config.max_attention_elements,
                          "TOOL_LEARNING_BUDGET")
                    distance = ordered_distance(np.clip(target@old.T, -1, 1))
                    costs[i] += distance
                    lane[i] = distance
                if key in train:
                    lane_costs[spec.name] = (spec.identity, torch.from_numpy(lane).to(session.device))
            self.examples[key] = dict(record=record, request=request, state=state,
                view=view, units=units, region_costs=lane_costs,
                allowed=allowed, cutoff=allowed.clone(), region=int(costs.argmin()),
                target_unit=target_unit, target_inputs=target_inputs,
                native=session.native_action(request).slots,
                target=target_inputs.batch([target_unit], self.model.config, session.device))
        self.fresh()

    def fresh(self):
        s = self.session
        s._fresh()
        check(s.head == self.head and s.memory.head == self.memory_head, "TOOL_LEARNING_STALE")
        s._bank_checked()
        for key in self.keys:
            s.eligible(key)

    def forward(self, model, key):
        e = self.examples[key]
        request = e["request"]
        check(torch.equal(e["allowed"], e["cutoff"]), "TOOL_LEARNING_CUTOFF")
        model = self.prediction_model(model, key)
        if e["view"] is not None:
            check(getattr(model, "context_history", None) is not None, "TOOL_LEARNING_HISTORY")
        return model(e["native"], (model.bind_state(e["state"]),), allowed=e["allowed"], phase=2,
                     event_ids=[digest(request)], evidence_origin=[request["origin"]],
                     history=getattr(model, "context_history", None))

    def prediction_model(self, model, key):
        view = self.examples[key]["view"]
        return model if view is None else tool_context_regions.attach(view, model)

    def recognition_risk(self, model, key, output):
        check(self.plan.get("region_risk") == constellation_risk.OBJECTIVE, "REGION_RISK_OBJECTIVE")
        check(key in self.plan["train"] and key not in self.plan["evaluate"], "REGION_RISK_SPLIT")
        e = self.examples[key]
        return constellation_risk.expected(output, e["region_costs"], e["allowed"], model.specs)


def candidate_path(session, name):
    check(isinstance(name, str) and name not in ("", ".", "..") and "/" not in name
          and "\\" not in name and session.candidate_root is not None, "TOOL_CANDIDATE_PATH")
    path = (session.candidate_root/name).resolve()
    check(path.parent == session.candidate_root, "TOOL_CANDIDATE_PATH")
    return path


def candidate_decision(data, name, pin, parent):
    session = data.session
    path = candidate_path(session, name)
    model, _, manifest = load(path, data, expected=pin)
    check(manifest["step"] == data.plan["steps"] and manifest["parent"] == parent,
          "TOOL_CANDIDATE_COMPLETENESS")
    baseline_model = session.model
    if parent != data.binding["model"]:
        history = [r for r in session.decisions if r["pin"] == parent]
        check(len(history) == 1, "TOOL_CANDIDATE_PARENT")
        previous = candidate_path(session, history[0]["candidate"])
        raw = (previous/"manifest.json").read_bytes()
        check(sha(raw) == parent == (previous/"COMMITTED").read_text(), "TOOL_CANDIDATE_PARENT")
        prior = json.loads(raw)
        baseline_model, _, _ = load_world(previous/"model", expected=prior["model"],
            standard=model.standard_id, world=prior["binding"]["world"], device=session.device)
        baseline_model = ConstellationWorld.from_world(baseline_model).eval()
        if baseline_model.world_id != model.world_id:
            from model.tool_growth import transport
            check(baseline_model.world_source_id in (
                      model.world_source_id, *getattr(session.inputs, "source_ancestry", ())),
                  "TOOL_CANDIDATE_PARENT_SOURCE")
            baseline_model, _ = transport(baseline_model, session.inputs, {})
        check(frozen_pin(baseline_model) == data.frozen, "TOOL_CANDIDATE_PARENT")
    baseline, candidate = data.evaluate(baseline_model), data.evaluate(model)
    result = decision(data, baseline, candidate, model)
    result = tool_retention.assess(data, result, candidate)
    payload = dict(baseline=baseline, candidate=candidate, result=result,
                   labels={k: e["region"] for k, e in data.examples.items()})
    body = dict(kind="candidate", candidate=name, pin=pin, parent=parent, train=data.plan["train"],
        evaluate=data.plan["evaluate"], binding=digest(data.binding), result=result,
        prior_head=session.head, payload=sha(canonical(payload)))
    if data.continuation_name is not None:
        body["continuation"] = data.continuation_name
    return body, payload
