"""Complete constellation forecasts in the existing evidence/identity event chain."""
from pathlib import Path

import torch

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation import score
from model.constellation_world import ConstellationWorld
from model.loop_session import output_digest
from model.predictive_loop import PredictiveLoop
from model.scientific_loop import ScientificLoop


class ConstellationLoop(PredictiveLoop):
    def __init__(self, *args, **kwargs):
        root = Path(__file__).parent
        self.constellation_code = {n: sha((root/n).read_bytes()) for n in
            ("constellation.py", "constellation_world.py", "constellation_loop.py")}
        super().__init__(*args, **kwargs)

    def prediction_contract(self):
        return {**super().prediction_contract(), "schema": "scientific-constellation-prediction-v1",
                "constellation_code": self.constellation_code,
                "representation": "one_region_separate_typed_ordered_native_windows",
                "metric": "per_instrument_ordered_cosine_edit_distance_v1"}

    def _ensure_model(self):
        if not isinstance(self.model, ConstellationWorld):
            self.model = ConstellationWorld.from_world(self.model)

    def _forward(self, request):
        self._ensure_model()
        output = ScientificLoop._forward(self, request)
        if request["phase"] == 2:
            constellation = output["predicted_constellation"]
            self.last_act[request["context"]] = dict(request=request,
                predictions=constellation.pack(self.model.specs, self.model.config),
                region=constellation.regions[0], output_sha256=output_digest(output))
        return output

    def _commit_body(self, context, target_edition, issued_at):
        body = super()._commit_body(context, target_edition, issued_at)
        body["record"]["constellation"] = dict(schema="typed-native-constellation-v1",
            region=self.last_act[context]["region"], bank=self.model.world_id,
            ordered_instruments=[s.identity for s in self.model.specs])
        body["id"] = digest(body["record"])
        return body

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
        for key, expected in self.support[context].items():
            current = self.memory.state[key]
            check(current["active"] and current["grounded_scope"] is not None and current["receipt"] == expected,
                  "PREDICTION_PRIOR_EVIDENCE_CHANGED")
        ep, checked = self._checked(self.edition, record["quantity"])
        check(receipt == checked["id"], "PREDICTION_OUTCOME_RECEIPT")
        unit = self.inputs.unit("row", self.edition, record["quantity"])
        assessment = score(record["predictions"], self.inputs, record["baseline_unit"], unit, self.model.config)
        request = dict(forecast=forecast, context=context, quantity=record["quantity"], edition=self.edition,
                       episode=ep["id"], checked_receipt=checked, unit=unit, phase=3, prior_head=self.head,
                       origin="execution_receipt", prior_sequence=state.sequence)
        native = self.inputs.batch([unit], self.model.config, self.device)
        allowed = self.inputs.allowed(self, self.device, historical=self._replaying)
        self._ensure_model()
        with torch.no_grad():
            output = self.model(native, (state,), allowed=allowed, phase=3, event_ids=[digest(request)],
                                evidence_origin=["execution_receipt"])
        hidden = output["states"][0].hidden.detach().cpu().numpy().tobytes()
        body = dict(kind="prediction_observe", forecast=forecast, request=request, assessment=assessment,
                    hidden_sha256=sha(hidden), output_sha256=output_digest(output),
                    interpretation="constellation_error_on_published_reference_not_independent_physics_validation")
        return body, output

    def snapshot(self):
        result = super().snapshot()
        for key, item in self.forecasts.items():
            result["forecasts"][key]["constellation"] = item["record"]["constellation"]
            result["forecasts"][key]["shapes"] = {n: v["shape"] for n, v in item["record"]["predictions"].items()}
        return result
