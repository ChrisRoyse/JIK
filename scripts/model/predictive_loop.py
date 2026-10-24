"""Commit -> external echo -> score -> correction, in the scientific event chain.

The initial source has already-published editions. Its forecasts are explicitly
retrospective temporal replays, never relabelled prospective evidence. Scores
attest prediction error on source observations, not the truth of an explanation.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from claims.codata_source import DATES, check, sha
from claims.episodic_memory import canonical, digest
from model.loop_session import output_digest
from model.predictive_score import pack, score
from model.scientific_loop import ScientificLoop


class PredictiveLoop(ScientificLoop):
    def __init__(self, *args, expected_prediction_contract=None, **kwargs):
        self.forecasts, self.last_act = {}, {}
        self.prediction_initialized = False
        self.prediction_replayed = 0
        check(kwargs.pop("neural_replay", True), "PREDICTION_NEURAL_REPLAY_REQUIRED")
        attach = kwargs.get("attach", False)
        root = Path(__file__).parent
        self.prediction_code = {n: sha((root/n).read_bytes()) for n in ("predictive_loop.py", "predictive_score.py")}
        super().__init__(*args, **kwargs, neural_replay=True)
        try:
            if attach:
                check(not self.prediction_initialized, "PREDICTION_ALREADY_ATTACHED")
                self._append(dict(kind="prediction_init", contract=self.prediction_contract()))
                self.prediction_initialized = True
            check(self.prediction_initialized, "PREDICTION_MISSING")
            check(attach or digest(self.prediction_contract()) == expected_prediction_contract, "PREDICTION_CONTRACT")
        except BaseException:
            self.close()
            raise

    def prediction_contract(self):
        return dict(schema="scientific-native-prediction-loop-v1", loop=self.loop_pin, code=self.prediction_code,
                    mode="retrospective_temporal_replay", baseline="last_observed_source_row",
                    evidence_scope="pinned_scientific_reference_contents",
                    future_success_is_sufficient_for_truth=False, serving=False)

    def _forward(self, request):
        output = super()._forward(request)
        if request["phase"] == 2:
            self.last_act[request["context"]] = dict(request=request,
                predictions={n: pack(t[0]) for n, t in output["predicted_echo"].items()},
                output_sha256=output_digest(output))
        return output

    def _fresh(self):
        self._live()
        if not self._replaying:
            row = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            check(row is not None and row[0] == self.head, "STALE_LEDGER")

    def _commit_body(self, context, target_edition, issued_at):
        self._fresh()
        check(self.prediction_initialized and context in self.last_act and not self.needs_review(context),
              "PREDICTION_CONTEXT")
        act = self.last_act[context]
        request, state = act["request"], self.states[context]
        check(state.sequence == request["sequence"]+1 and state.last_event == digest(request),
              "PREDICTION_ACT_CHANGED")
        check(not any(v["record"]["context"] == context and v["status"] == "pending"
                      for v in self.forecasts.values()), "PREDICTION_DUPLICATE")
        check((self.edition, target_edition) == ("2018", "2022"), "PREDICTION_TARGET_ORDER")
        clock = datetime.fromisoformat(issued_at)
        check(clock.tzinfo is not None and clock.astimezone(timezone.utc).isoformat() == issued_at
              and clock.date().isoformat() >= DATES[target_edition], "PREDICTION_REPLAY_CLOCK")
        ep, receipt = self._checked(self.edition, request["quantity"])
        value = dict(context=context, quantity=request["quantity"], source_edition=self.edition,
                     target_edition=target_edition, baseline_unit=self.inputs.unit("row", self.edition, request["quantity"]),
                     episode=ep["id"], receipt=receipt, predictions=act["predictions"],
                     issued_at=issued_at, mode="retrospective_temporal_replay",
                     target_published= DATES[target_edition], source_manifest=self.source.pin,
                     world=self.model.world_id, standard=self.model.standard_id, model=self.loop_contract["checkpoint"],
                     act=act["request"], act_output_sha256=act["output_sha256"],
                     support=json.loads(canonical(self.support[context])),
                     context_sequence=state.sequence, context_last_event=state.last_event, prior_head=self.head)
        return dict(kind="prediction_commit", id=digest(value), record=value)

    def commit_forecast(self, context, *, target_edition):
        body = self._commit_body(context, target_edition, datetime.now(timezone.utc).isoformat())
        self._append(body)
        self._retain_commit(body)
        return body["id"]

    def _retain_commit(self, body):
        check(body["id"] not in self.forecasts, "PREDICTION_DUPLICATE")
        self.forecasts[body["id"]] = dict(record=json.loads(canonical(body["record"])), status="pending", invalidated=False,
                                         result=None, target_episode=None)

    def _resolve_body(self, forecast, receipt):
        self._fresh()
        check(forecast in self.forecasts, "PREDICTION_UNKNOWN")
        item = self.forecasts[forecast]
        check(item["status"] == "pending" and not item["invalidated"], "PREDICTION_UNAVAILABLE")
        record, context = item["record"], item["record"]["context"]
        state = self.states[context]
        check(state.sequence == record["context_sequence"] and state.last_event == record["context_last_event"],
              "PREDICTION_ACT_CHANGED")
        check(self.support[context] == record["support"], "PREDICTION_CONTEXT_EVIDENCE_CHANGED")
        check(self.edition == record["target_edition"], "PREDICTION_OUTCOME_NOT_ADOPTED")
        _, old_receipt = self._checked(record["source_edition"], record["quantity"])
        check(old_receipt == record["receipt"], "PREDICTION_PRIOR_EVIDENCE_CHANGED")
        # Any withdrawal affecting this context invalidates its pending forecast,
        # even if the same evidence is subsequently rechecked.
        for key, expected in self.support[context].items():
            current = self.memory.state[key]
            check(current["active"] and current["grounded_scope"] is not None and current["receipt"] == expected,
                  "PREDICTION_PRIOR_EVIDENCE_CHANGED")
        ep, checked = self._checked(self.edition, record["quantity"])
        check(receipt == checked["id"], "PREDICTION_OUTCOME_RECEIPT")
        unit = self.inputs.unit("row", self.edition, record["quantity"])
        assessment = score(record["predictions"], self.inputs, record["baseline_unit"], unit)
        request = dict(forecast=forecast, context=context, quantity=record["quantity"], edition=self.edition,
                       episode=ep["id"], checked_receipt=checked, unit=unit, phase=3, prior_head=self.head,
                       origin="execution_receipt", prior_sequence=state.sequence)
        native = self.inputs.batch([unit], self.model.config, self.device)
        allowed = self.inputs.allowed(self, self.device, historical=self._replaying)
        import torch
        with torch.no_grad():
            output = self.model(native, (state,), allowed=allowed, phase=3, event_ids=[digest(request)],
                                evidence_origin=["execution_receipt"])
        hidden = output["states"][0].hidden.detach().cpu().numpy().tobytes()
        body = dict(kind="prediction_observe", forecast=forecast, request=request, assessment=assessment,
                    hidden_sha256=sha(hidden), output_sha256=output_digest(output),
                    interpretation="prediction_error_on_published_reference_not_independent_physics_validation")
        return body, output

    def observe_forecast(self, forecast, *, receipt):
        body, output = self._resolve_body(forecast, receipt)
        self._append(body)
        self._retain_outcome(body, output)
        return body["assessment"]

    def _retain_outcome(self, body, output):
        item, request = self.forecasts[body["forecast"]], body["request"]
        context = request["context"]
        self.states[context] = output["states"][0]
        self._support(request)
        # This explicit outcome/review event closes the edition gap. A source
        # withdrawal cannot take this path because it invalidates the forecast.
        self.dirty.discard(context)
        item.update(status="scored", result=body["assessment"], target_episode=request["episode"])

    def _withdraw_forecasts(self, record):
        affected = set(record["event"].get("affected", ()))
        if record["event"]["op"] == "retire":
            affected.add(record["event"]["episode"])
        for item in self.forecasts.values():
            dependencies = set(item["record"]["support"]) | {item["record"]["episode"], item["target_episode"]}
            if affected.intersection(dependencies):
                item["invalidated"] = True

    def apply_memory(self, command):
        record = super().apply_memory(command)
        self._withdraw_forecasts(record)
        return record

    def _reset(self, context):
        super()._reset(context)
        for item in self.forecasts.values():
            if item["record"]["context"] == context and item["status"] == "pending":
                item["invalidated"] = True

    def grounding_status(self, forecast):
        check(forecast in self.forecasts, "PREDICTION_UNKNOWN")
        item = self.forecasts[forecast]
        reasons = ["NO_INDEPENDENT_PAST_EXPLANATION_CHECK", "NO_INDEPENDENT_PRESENT_CONSISTENCY_CHECK",
                   "RETROSPECTIVE_NOT_PROSPECTIVE", "PUBLICATION_CONTENTS_NOT_INDEPENDENT_WORLD_OBSERVATION"]
        if item["status"] != "scored":
            reasons.append("OUTCOME_PENDING")
        elif not item["result"]["better_on_every_instrument"]:
            reasons.append("PERSISTENCE_BASELINE_NOT_BEATEN")
        if item["invalidated"]:
            reasons.append("EVIDENCE_WITHDRAWN")
        return dict(status="insufficient_for_empirical_grounding", reasons=reasons,
                    forecast=forecast, source_scope="pinned_scientific_reference_contents",
                    prediction_usable_as_learning_evidence=item["status"] == "scored" and not item["invalidated"],
                    establishes_empirical_truth=False)

    def require_empirical_grounding(self, forecast):
        status = self.grounding_status(forecast)
        check(status["establishes_empirical_truth"], "PREDICTION_GROUNDING_INSUFFICIENT")

    def _replay(self, body):
        kind = body.get("kind")
        if kind == "prediction_init":
            check(not self.prediction_initialized and body == dict(kind=kind, contract=self.prediction_contract()),
                  "PREDICTION_INIT")
            self.prediction_initialized = True
        elif kind == "prediction_commit":
            record = body["record"]
            check(body == self._commit_body(record["context"], record["target_edition"], record["issued_at"]),
                  "PREDICTION_COMMIT_REPLAY")
            self._retain_commit(body)
        elif kind == "prediction_observe":
            expected, output = self._resolve_body(body["forecast"], body["request"]["checked_receipt"]["id"])
            check(body == expected, "PREDICTION_OUTCOME_REPLAY")
            self._retain_outcome(body, output)
            self.prediction_replayed += 1
        else:
            super()._replay(body)
            if kind == "memory":
                self._withdraw_forecasts(body["record"])

    def snapshot(self):
        return dict(**super().snapshot(), prediction_contract=digest(self.prediction_contract()),
                    forecasts={key: dict(context=v["record"]["context"], mode=v["record"]["mode"],
                        predictions={n: x["sha256"] for n, x in v["record"]["predictions"].items()},
                        status=v["status"], invalidated=v["invalidated"], result=v["result"],
                        grounding=self.grounding_status(key)) for key, v in sorted(self.forecasts.items())})
