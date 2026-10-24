"""Durable source experience, intrinsic phases and complete constellation forecasts."""
import base64
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import Memory, canonical, digest
from claims.event_journal import EventJournal
from claims.tool_evidence import ToolEvidence
from claims.process_observation import SCOPE
from claims.tool_temporal_source import CUTOFFS
from model.constellation import score
from model.constellation_world import ConstellationWorld
from model.loop_session import output_digest
from model.world_checkpoint import load
from model import tool_execution
from model import tool_conditions
from model import tool_enrollment
from model import process_anchor


def code_pins():
    root = Path(__file__).resolve().parents[1]
    return {n: sha((root/n).read_bytes()) for n in (
        "claims/event_journal.py", "claims/tool_evidence.py", "claims/tool_temporal_source.py",
        "claims/episodic_memory.py", "claims/process_observation.py", "claims/process_conditions.py",
        "claims/process_population.py", "model/process_input.py", "claims/condition_population.py",
        "model/condition_input.py", "model/tool_conditions.py", "model/tool_enrollment.py",
        "claims/forecast_commitment.py", "model/process_batches.py",
        "claims/condition_catalog.py", "model/condition_catalog.py",
        "model/tool_loop.py", "model/tool_execution.py", "model/tool_temporal_input.py",
        "model/tool_learning.py", "model/tool_learner.py", "model/tool_retention.py",
        "model/constellation_risk.py", "model/process_anchor.py",
        "model/retained_update.py", "model/retention_projection.py",
        "model/constellation.py", "model/constellation_world.py", "model/constellation_history.py")}


class ToolLoop:
    def __init__(self, path, source, inputs, checkpoint, *, model_pin, standard, world,
                 contexts, device, create=False, expected_head=None, expected_contract=None,
                 candidate_root=None, routing=None, extension=None, observations=None, execution_inputs=None,
                 recovery=False, condition_inputs=None):
        check(not recovery or (not create and expected_head is not None and expected_contract is not None),
              "TOOL_RECOVERY_HEAD_REQUIRED")
        base, initial, _ = load(checkpoint, expected=model_pin, standard=standard, world=world, device=device)
        check(all(s.sequence == 0 for s in initial) and base.specs == inputs.specs
              and base.world_source_id == inputs.source_id, "TOOL_MODEL_JOIN")
        check(contexts and len(set(contexts.values())) == len(contexts) and all(
            t in source.by_task and source.by_task[t]["status"] == "ready" for t in contexts.values()),
            "TOOL_CONTEXTS")
        self.model = ConstellationWorld.from_world(base).eval()
        self.source, self.inputs, self.device = source, inputs, device
        self.active_checkpoint = model_pin
        self.routing = routing
        check(routing is None or routing.inputs is inputs, "TOOL_KERNEL_INPUTS")
        self.route_memory = {}
        if routing is not None:
            self.route_memory = {digest(r): ToolEvidence(source).episode(r["task"], r["turn"])["id"]
                                 for r in routing.action_references()}
        self.contexts = dict(contexts)
        self.states = {c: self.model.start(c) for c in contexts}
        self.cursor, self.support = {c: 0 for c in contexts}, {c: {} for c in contexts}
        check(observations is None or observations.source is source, "TOOL_EXECUTION_SOURCE")
        self.observations, self.executions = observations, {}
        check(execution_inputs is None or (observations is not None
              and execution_inputs.observations is observations and execution_inputs.base is inputs),
              "TOOL_EXECUTION_NATIVE_SOURCE")
        self.execution_inputs = execution_inputs
        check(condition_inputs is None or condition_inputs.base is inputs, "TOOL_CONDITION_INPUT_SOURCE")
        self.condition_inputs, self.condition_bindings = condition_inputs, {}
        self.execution_enrollment = None
        self.execution_enrollments = []
        self.memory = Memory(ToolEvidence(source, observations,
            None if condition_inputs is None else condition_inputs.conditions))
        self.forecasts, self.preact, self.pending, self.decisions = {}, {}, {}, []
        self.dirty, self.replaying, self.replayed = set(), False, 0
        self.bank = {}
        self.candidate_root = Path(candidate_root).resolve() if candidate_root else None
        self.allowed = torch.ones(len(inputs.world_units), dtype=torch.bool, device=device)
        contract = dict(schema="tool-experience-loop-v1", source=source.pin, native=inputs.source_id,
            checkpoint=model_pin, standard=standard, world=world, contexts=contexts, code=code_pins(),
            evidence_scope="pinned_artifact_contents_only", mode="exposed_retrospective_replay",
            serving=False, scientific_admission=False)
        if routing is not None:
            contract.update(schema="tool-experience-loop-kernel-v1", routing=routing.identity)
        if extension is not None:
            contract["extension"] = extension
        if observations is not None:
            contract["process_observations"] = observations.pin
        if execution_inputs is not None:
            contract["execution_inputs"] = execution_inputs.identity
        if condition_inputs is not None:
            contract["condition_inputs"] = condition_inputs.identity
        self.journal = EventJournal(path, contract, create=create, payload_bound=base.config.checkpoint_bytes)
        try:
            if create:
                records = []
                bootstrap = list(inputs.population.initial_bank())
                if routing is not None:
                    bootstrap += list(routing.action_references())
                for ref in bootstrap:
                    ep = self.memory.checker.episode(ref["task"], ref["turn"])
                    records.extend(self._intake(ep))
                self.bank = {r["unit"]: self.memory.checker.episode(r["task"], r["turn"])["id"]
                             for r in inputs.population.initial_bank()}
                # Bootstrap separately: the event bound is unchanged.
                for record in records:
                    self.journal.append(dict(kind="memory", record=record))
            else:
                self.replaying = True
                self.journal.restore(self._replay, head=expected_head, contract=expected_contract)
                self.replaying = False
            if not recovery:
                self._bank_checked()
        except BaseException:
            if not self.journal.broken:
                self.close()
            raise

    @property
    def head(self):
        return self.journal.head

    def close(self):
        self.journal.close()

    def _fresh(self):
        if not self.replaying:
            self.journal.fresh()

    def _intake(self, ep):
        if ep["id"] in self.memory.episodes:
            self._support_checked(ep["id"])
            return []
        records = [self.memory.apply(dict(op="intake", episode=ep))]
        records.append(self.memory.apply(dict(op="check_source", episode=ep["id"])))
        receipt = self.memory.state[ep["id"]]["receipt"]["id"]
        records.append(self.memory.apply(dict(op="promote", episode=ep["id"], receipt=receipt)))
        return records

    def _checked(self, key):
        value = self.memory.state.get(key)
        check(value is not None and value["active"] and
              value["grounded_scope"] == "pinned_artifact_contents_only" and
              value["receipt"] == self.memory.checker.assess(self.memory.episodes[key]),
              "TOOL_EVIDENCE_UNAVAILABLE")
        return value["receipt"]

    def _bank_checked(self):
        for key in (*self.bank.values(), *self.route_memory.values()):
            self._support_checked(key)

    def routing_for(self, pin):
        check(self.routing is not None and self.routing.pin == pin, "TOOL_ROUTING_VERSION")
        return self.routing

    @staticmethod
    def record_support(record):
        echoes = record.get("region_context", {}).get("definition", {}).get("observed_echoes", {})
        condition = record["request"].get("execution_conditions")
        return (*record["support"].values(), *record.get("bank_support", {}).values(),
                *(v["episode"] for v in echoes.values()),
                *((condition["episode"],) if condition else ()))

    def _support_checked(self, key):
        ep = self.memory.episodes.get(key)
        if ep is not None and ep["claim_kind"] == "execution_precondition":
            state = self.memory.state[key]
            check(state["active"] and state["grounded_scope"] == "precommitted_execution_conditions"
                  and state["receipt"] == self.memory.checker.assess(ep), "TOOL_CONDITION_UNAVAILABLE")
            return state["receipt"]
        if ep is not None and ep["claim_kind"] == "event_occurrence":
            state = self.memory.state[key]
            check(state["active"] and state["grounded_scope"] == SCOPE
                  and state["receipt"] == self.memory.checker.assess(ep), "TOOL_EXECUTION_UNAVAILABLE")
            return state["receipt"]
        return self._checked(key)

    def needs_review(self, context):
        return context in self.dirty

    def _request(self, context, receipt, reference, execution=None):
        self._fresh()
        check(context in self.contexts and not self.needs_review(context), "TOOL_CONTEXT_REVIEW")
        for key in self.support[context].values():
            self._checked(key)
        self._bank_checked()
        state = self.states[context]
        phase, exchange = state.sequence % 5, state.sequence // 5
        action = 2+2*exchange
        turn = 1 if phase == 0 else action if phase in (1, 2) else action+1
        task = self.contexts[context]
        ref = self.inputs.population.lookup.get((task, turn))
        check(ref is not None and (reference is None or reference == ref), "TOOL_NEXT_REFERENCE")
        check(turn <= self.cursor[context]+1, "TOOL_AVAILABILITY")
        check(receipt == self.source.receipt(task, turn) if phase == 3 else receipt is None,
              "TOOL_FEEDBACK_RECEIPT")
        if phase == 3 and action in CUTOFFS:
            check(context in self.pending, "TOOL_UNCOMMITTED_ECHO")
        result = dict(context=context, task=task, turn=turn, phase=phase, sequence=state.sequence,
            reference=ref, receipt=receipt, prior_head=self.head, memory_head=self.memory.head,
            origin="generated_proposal" if phase in (1, 2) else "source_observation")
        condition = tool_conditions.request_binding(self, context, action, phase)
        if condition is not None:
            result["execution_conditions"] = condition
        if execution is not None:
            result["execution"] = tool_execution.echo_binding(self, context, execution)
            result["origin"] = "source_observation"
        if phase == 4:
            matches = [(key,item) for key,item in self.forecasts.items() if
                       item["record"]["request"]["context"] == context and
                       item["record"]["request"]["turn"]+1 == turn and "execution" in item]
            check(len(matches) <= 1, "TOOL_EXECUTION_CORRECTION_AMBIGUOUS")
            if matches:
                key, item = matches[0]
                result["execution_context"] = tool_execution.checked_echo(self, key, item["execution"])
        return result

    def _prediction(self, request, native):
        phase, context = request["phase"], request["context"]
        routing, allowed = None, self.allowed
        if self.routing is not None and phase == 2:
            routing = self.routing.route(request["reference"])
            allowed = torch.zeros_like(self.allowed)
            allowed[routing["allowed"]] = True
        with torch.no_grad():
            output = self.model(native.slots, (self.states[context],), allowed=allowed,
                phase=phase, event_ids=[digest(request)], evidence_origin=[request["origin"]])
        return output, routing, None

    def _advance(self, context, receipt, reference, expected=None, execution=None):
        request = self._request(context, receipt, reference, execution)
        phase, turn, task = request["phase"], request["turn"], request["task"]
        observation = execution if execution is not None else request.get("execution_context", {}).get("observation")
        native = (self.native_action(request)
                  if observation is None else self.execution_inputs.observed(observation, self.model.config, self.device))
        output, routing, region_context = self._prediction(request, native)
        state = output["states"][0]
        payload = {}
        if phase == 2 and turn in CUTOFFS:
            self.inputs.population.prefix(task, turn, partition=self.source.by_task[task]["partition"],
                                          observed_through=self.cursor[context])
            payload["forecast"] = dict(request=request, prior=self.preact[context],
                predictions=output["predicted_constellation"].pack(self.model.specs, self.model.config),
                recognition=output["recognition"],
                region=output["predicted_constellation"].regions[0],
                support={str(k): v for k, v in self.support[context].items()},
                bank_support={**{str(k): v for k, v in self.bank.items()}, **self.route_memory},
                world=self.model.world_id,
                baseline_unit=self.inputs.population.lookup[task, turn-1]["unit"])
            if routing is not None:
                payload["forecast"]["routing"] = routing
            if region_context is not None:
                payload["forecast"]["region_context"] = region_context
            if "execution_anchor" in output:
                payload["forecast"]["execution_anchor"] = output["execution_anchor"]
        if phase == 3 and context in self.pending:
            key = self.pending[context]
            record = self.forecasts[key]["record"]
            check(turn == record["request"]["turn"]+1, "TOOL_OUTCOME_ORDER")
            target = self.execution_inputs.for_observation(execution) if execution is not None else None
            payload["assessment"] = score(record["predictions"], self.inputs,
                record["baseline_unit"], request["reference"]["unit"] if execution is None
                    else target.lookup[execution], self.model.config, target_inputs=target)
            if execution is not None and "execution_anchor" in record:
                payload["execution_anchor_score"] = process_anchor.score(record["execution_anchor"],
                    self.observations.read(execution), execution)
        evidence = [self.support[context][self.cursor[context]]] if turn > self.cursor[context] and self.cursor[context] else []
        ep = self.memory.checker.episode(task, turn, evidence)
        if turn in self.support[context]:
            ep = self.memory.episodes[self.support[context][turn]]
        records = self._intake(ep)
        if execution is not None:
            attachment, extra = tool_execution.stage_echo(self, self.pending[context], request["execution"], ep["id"])
            records.extend(extra)
            payload["execution"] = attachment
        raw = state.hidden.detach().cpu().numpy().tobytes()
        body = dict(kind="advance", request=request, memory=records, episode=ep["id"],
            hidden=base64.b64encode(raw).decode(), hidden_sha256=sha(raw), output=output_digest(output),
            payload=sha(canonical(payload)) if payload else None)
        if expected is not None:
            check(body == expected and (not payload or payload == self.journal.payload(body["payload"])),
                  "TOOL_NEURAL_REPLAY")
            self.replayed += 1
        else:
            try:
                self.journal.append(body, payload if payload else None)
            except BaseException:
                self.journal.broken = True
                raise
        self.states[context] = state
        self.support[context][turn] = ep["id"]
        self.cursor[context] = max(turn, self.cursor[context])
        if phase == 1:
            self.preact[context] = dict(hidden=body["hidden"], sequence=state.sequence,
                                       last_event=state.last_event)
        if "forecast" in payload:
            key = digest(payload["forecast"])
            check(key not in self.forecasts and context not in self.pending, "TOOL_DUPLICATE_FORECAST")
            self.forecasts[key] = dict(record=payload["forecast"], status="pending", invalidated=False)
            self.pending[context] = key
        if "assessment" in payload:
            key = self.pending.pop(context)
            self.forecasts[key].update(status="observed", result=payload["assessment"],
                target_episode=ep["id"], target_unit=request["reference"]["unit"], receipt=receipt)
            if execution is not None:
                self.executions[key] = payload["execution"]
                self.forecasts[key]["execution"] = payload["execution"]
                if "execution_anchor_score" in payload:
                    self.forecasts[key]["execution_anchor_score"] = payload["execution_anchor_score"]
        return body

    def advance(self, context, *, receipt=None, reference=None, execution=None):
        return self._advance(context, receipt, reference, execution=execution)

    def bind_execution_conditions(self, context, key, *, expected=None):
        return tool_conditions.bind(self, context, key, expected=expected)

    def native_action(self, request):
        return tool_conditions.native_action(self, request)

    def enroll_execution(self, spec, *, expected=None):
        return tool_enrollment.enroll(self, spec, expected=expected)

    def observe_execution(self, key, observation, *, expected=None):
        return tool_execution.observe(self, key, observation, expected=expected)

    def execution_evidence(self, key):
        return tool_execution.checked(self, key)

    def eligible(self, key):
        self._fresh()
        check(key in self.forecasts, "TOOL_FORECAST_UNKNOWN")
        item = self.forecasts[key]
        check(item["status"] == "observed" and not item["invalidated"], "TOOL_LEARNING_INELIGIBLE")
        for ep in (*self.record_support(item["record"]), item["target_episode"]):
            self._support_checked(ep)
        check(item["receipt"] == self.source.receipt(item["record"]["request"]["task"],
              item["record"]["request"]["turn"]+1), "TOOL_FEEDBACK_RECEIPT")
        if "execution" in item:
            tool_execution.checked_echo(self, key, item["execution"])
        return item

    def invalidate(self, key, *, reason, expected=None):
        self._fresh()
        record = self.memory.apply(dict(op="invalidate", episode=key, reason=reason))
        body = dict(kind="invalidate", record=record)
        if expected is None:
            self.journal.append(body)
        else:
            check(body == expected, "TOOL_INVALIDATION_REPLAY")
        affected = set(record["event"]["affected"])
        self.dirty.update(c for c, refs in self.support.items() if affected.intersection(refs.values()))
        self.dirty.update(c for (c,_), r in self.condition_bindings.items() if r["episode"] in affected)
        for item in self.forecasts.values():
            if affected.intersection((*self.record_support(item["record"]), item.get("target_episode"),
                                      item.get("execution", {}).get("episode"))):
                item["invalidated"] = True
                self.dirty.add(item["record"]["request"]["context"])
        return body

    def consider(self, name, *, pin, train, evaluate, parent, continuation=None, expected=None):
        from model.tool_learning import ToolLearningData, candidate_decision
        self._fresh()
        check(not any(r["candidate"] == name for r in self.decisions), "TOOL_DUPLICATE_CANDIDATE")
        data = ToolLearningData(self, train=train, evaluate=evaluate, continuation=continuation)
        body, payload = candidate_decision(data, name, pin, parent)
        if expected is None:
            self.journal.append(body, payload)
        else:
            check(body == expected and payload == self.journal.payload(body["payload"]),
                  "TOOL_DECISION_REPLAY")
        self.decisions.append(body)
        return body

    def require_selection(self, name):
        self._fresh()
        found = [r for r in self.decisions if r["candidate"] == name]
        check(len(found) == 1 and found[0]["result"]["selected"], "TOOL_CANDIDATE_NOT_SELECTED")
        for key in (*found[0]["train"], *found[0]["evaluate"]):
            self.eligible(key)

    def _replay(self, body):
        kind = body.get("kind")
        if kind == "execution_enrollment":
            self.enroll_execution(body["spec"], expected=body)
        elif kind == "execution_conditions":
            self.bind_execution_conditions(body["context"], body["conditions"]["native"]["condition"], expected=body)
        elif kind == "memory":
            record = body["record"]
            command = {k: v for k, v in record["event"].items() if k not in ("result", "affected")}
            check(self.memory.apply(command) == record, "TOOL_MEMORY_REPLAY")
            # The initial bank is known before the first model event.
            self.bank = {r["unit"]: self.memory.checker.episode(r["task"], r["turn"])["id"]
                         for r in self.inputs.population.initial_bank()}
        elif kind == "advance":
            req = body["request"]
            self._advance(req["context"], req["receipt"], req["reference"], expected=body,
                          execution=req.get("execution", {}).get("observation"))
        elif kind == "invalidate":
            record = body["record"]["event"]
            self.invalidate(record["episode"], reason=record["reason"], expected=body)
        elif kind == "candidate":
            self.consider(body["candidate"], pin=body["pin"], train=body["train"],
                          evaluate=body["evaluate"], parent=body["parent"],
                          continuation=body.get("continuation"), expected=body)
        elif kind == "execution_observation":
            self.observe_execution(body["forecast"], body["observation"], expected=body)
        else:
            check(False, "TOOL_EVENT_KIND")

    def snapshot(self):
        self._fresh()
        result = dict(head=self.head, contract=self.journal.pin, events=self.journal.count,
            memory=self.memory.snapshot(), standard=self.model.standard_id, world=self.model.world_id,
            active_model=self.active_checkpoint, contexts=dict(self.contexts), cursor=dict(self.cursor),
            dirty=sorted(self.dirty),
            states={c: dict(sequence=s.sequence, last_event=s.last_event,
                    hidden=sha(s.hidden.detach().cpu().numpy().tobytes())) for c, s in self.states.items()},
            forecasts={k: dict(status=v["status"], invalidated=v["invalidated"],
                record=digest(v["record"]), result=digest(v.get("result"))) for k, v in self.forecasts.items()},
            decisions=self.decisions, serving=False, world_truth_established=False)
        if self.observations is not None:
            result["process_observations"] = dict(pin=self.observations.pin, forecasts=self.executions)
        if self.condition_inputs is not None:
            result["execution_conditions"] = [dict(context=c, action=t, **r)
                for (c,t),r in sorted(self.condition_bindings.items())]
        if self.execution_enrollment is not None:
            result["execution_enrollment"] = self.execution_enrollment
        if len(self.execution_enrollments) > 1:
            result["execution_enrollments"] = self.execution_enrollments.copy()
        return result
