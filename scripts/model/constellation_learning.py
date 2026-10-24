"""Outcome-bound readout learning; complete typed windows remain the targets."""
import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.constellation import score
from model.loop_training_checkpoint import tensor_fingerprint
from model.world_loop import WorldState


RECIPE = dict(steps=8, learning_rate=.001, weight_decay=.01,
              optimizer="AdamW", selection="final_only",
              region_objective="shared_constellation_ce_per_instrument_v1",
              placement_conditioning="checked_region_label_with_historical_mask",
              scope="exposed_publication_replay_not_sealed_capability_evaluation")


def code_pins():
    root = Path(__file__).parent
    return {n: sha((root/n).read_bytes()) for n in
            ("constellation_learning.py", "constellation_history.py", "constellation_training_checkpoint.py",
             "constellation_learning_session.py")}


def frozen_pin(model):
    return digest(tensor_fingerprint({k: v for k, v in model.state_dict().items()
                                     if not k.startswith(("query.", "place."))}))


def configure(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("query.", "place.")))
    model.eval()
    return model


def trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def identity_pin(output):
    return digest(tensor_fingerprint(dict(regions=output["identity_regions"],
        slots=output["identity_slots"], verdict=output["predicted_verdict"],
        hidden=[s.hidden for s in output["states"]])))


def ordered_loss(prediction, target):
    """Differentiable ordered edit loss. No pooling, including unequal lengths."""
    cosines = (F.normalize(prediction, dim=-1) @ F.normalize(target, dim=-1).T).clamp(-1, 1)
    rows, cols = cosines.shape
    previous = [cosines.new_tensor(float(j)) for j in range(cols+1)]
    for i in range(rows):
        current = [cosines.new_tensor(float(i+1))]
        for j in range(cols):
            current.append(torch.stack((previous[j+1]+1, current[-1]+1,
                                        previous[j]+1-cosines[i, j])).min())
        previous = current
    return previous[-1]/max(rows, cols)


def shared_region_loss(output, label, allowed, instruments):
    """One categorical target for the constellation; native slots remain typed."""
    evidence = output.get("constellation_regions")
    check(isinstance(evidence, torch.Tensor) and evidence.ndim == 2
          and evidence.shape == (len(label), len(allowed))
          and set(output["regions"]) == {s.name for s in instruments},
          "LEARNING_REGION_EVIDENCE")
    check(allowed.dtype == torch.bool and label.dtype == torch.long
          and evidence.device == label.device == allowed.device
          and bool(((label >= 0) & (label < len(allowed))).all())
          and bool(allowed[label].all()), "LEARNING_REGION_LABEL")
    check(bool(torch.isfinite(evidence[:, allowed]).all())
          and bool(torch.isneginf(evidence[:, ~allowed]).all()), "LEARNING_REGION_MASK")
    return F.cross_entropy(evidence, label)/len(instruments)


class LearningData:
    def __init__(self, session, *, train, evaluate):
        session._ensure_model()
        self.session, self.model = session, session.model
        self.head, self.memory_head = session.head, session.memory.head
        self.plan = dict(**RECIPE, train=list(train), evaluate=list(evaluate))
        check(len(train) == len(evaluate) == 1 and not set(train) & set(evaluate),
              "LEARNING_SPLIT")
        self.keys = tuple(train)+tuple(evaluate)
        self.fresh()
        records = {k: session.forecasts[k]["record"] for k in self.keys}
        check(len({(r["quantity"], r["source_edition"], r["target_edition"])
                   for r in records.values()}) == len(records), "LEARNING_SPLIT")
        self.frozen = frozen_pin(self.model)
        self.binding = dict(head=self.head, memory_head=self.memory_head,
            prediction_contract=digest(session.prediction_contract()),
            model=session.loop_contract["checkpoint"], standard=self.model.standard_id,
            world=self.model.world_id, native_source=session.inputs.source_id,
            source=session.source.pin, plan=self.plan, code=code_pins(),
            evidence={k: dict(record=digest(records[k]),
                             outcome=digest(session.forecasts[k]["result"]),
                             target=session.forecasts[k]["target_episode"]) for k in self.keys})
        # One authenticated ledger scan per binding; no disk/source work per update.
        prior = {}
        for raw, in session.db.execute("SELECT body FROM events WHERE seq < ? ORDER BY seq", (session.count,)):
            body = json.loads(raw)
            if body["kind"] == "world_advance":
                request = body["request"]
                prior[(request["context"], request["sequence"]+1)] = body
        self.examples = {}
        for key, record in records.items():
            request = record["act"]
            body = prior.get((record["context"], request["sequence"]))
            check(body is not None and body["request"]["phase"] == 1, "LEARNING_PREACT")
            raw = base64.b64decode(body["hidden"], validate=True)
            check(sha(raw) == body["hidden_sha256"], "LEARNING_PREACT")
            shape = (len(self.model.specs), self.model.config.width)
            hidden = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(shape)).to(session.device)
            state = WorldState(record["context"], self.model.standard_id, request["sequence"],
                               hidden, digest(body["request"]), self.model.world_id)
            allowed = torch.zeros(len(session.inputs.world_units), dtype=torch.bool, device=session.device)
            for quantity in sorted(session.source.rows[record["source_edition"]]):
                session._checked(record["source_edition"], quantity)
                unit = session.inputs.unit("row", record["source_edition"], quantity)
                allowed[session.inputs.world_index[unit]] = True
            target_unit = session.inputs.unit("row", record["target_edition"], record["quantity"])
            self.examples[key] = dict(record=record, state=state, allowed=allowed, cutoff=allowed.clone(),
                region=session.inputs.world_index[record["baseline_unit"]], target_unit=target_unit,
                native=session.inputs.batch([request["unit"]], self.model.config, session.device),
                target=session.inputs.batch([target_unit], self.model.config, session.device))
        self.fresh()

    def fresh(self):
        s = self.session
        s._fresh()
        check(s.head == self.head and s.memory.head == self.memory_head, "LEARNING_STALE_BINDING")
        for key in self.keys:
            check(key in s.forecasts and s.grounding_status(key)["prediction_usable_as_learning_evidence"],
                  "LEARNING_EVIDENCE_INELIGIBLE")
            item = s.forecasts[key]
            for edition in ("source_edition", "target_edition"):
                s._checked(item["record"][edition], item["record"]["quantity"])

    def forward(self, model, key):
        e = self.examples[key]
        request = e["record"]["act"]
        check(torch.equal(e["allowed"], e["cutoff"]), "LEARNING_CUTOFF")
        check(bool(e["allowed"][e["region"]]), "LEARNING_TARGET_REGION_UNAVAILABLE")
        return model(e["native"], (e["state"],), allowed=e["allowed"],
                     phase=2, event_ids=[digest(request)], evidence_origin=[request["origin"]])

    def prediction_model(self, model, key):
        return model

    def recognition_risk(self, model, key, output):
        return {s.name: output["constellation_regions"].new_zeros(()) for s in model.specs}

    def loss(self, model):
        self.fresh()
        check(self.plan.get("region_objective") == RECIPE["region_objective"], "LEARNING_OBJECTIVE")
        total, details = [], {}
        for key in self.plan["train"]:
            output = self.forward(model, key)
            e = self.examples[key]
            label = torch.tensor([e["region"]], dtype=torch.long, device=model.core.initial.device)
            shared = shared_region_loss(output, label, e["allowed"], model.specs)
            risks = self.recognition_risk(model, key, output)
            placement_output = self.prediction_model(model, key).supervised_placement(output, label, allowed=e["allowed"],
                                                          event_ids=[output["states"][0].last_event])
            details[key] = {}
            placements = []
            for spec in model.specs:
                slot = placement_output.slots[spec.name]
                target = e["target"][spec.name]
                placement = ordered_loss(slot.values[0, ~slot.padding[0]],
                                         target.values[0, ~target.padding[0]])
                region = F.cross_entropy(output["regions"][spec.name],
                    torch.tensor([e["region"]], device=slot.values.device))
                placements.append(placement+risks[spec.name])
                details[key][spec.name] = dict(region=float(region.detach()), placement=float(placement.detach()),
                    shared_region=float(shared.detach()),
                    region_risk=float(risks[spec.name].detach()),
                    supervised_region=e["region"], inference_region=output["predicted_constellation"].regions[0])
            total.append(shared+torch.stack(placements).mean())
        return torch.stack(total).mean(), details

    def evaluate(self, model):
        self.fresh()
        results = {}
        with torch.no_grad():
            for key, e in self.examples.items():
                output = self.forward(model, key)
                constellation = output["predicted_constellation"]
                packed = constellation.pack(model.specs, model.config)
                results[key] = dict(region=constellation.regions[0], predictions=packed,
                    identity=identity_pin(output), assessment=score(packed, self.session.inputs,
                    e["record"]["baseline_unit"], e["target_unit"], model.config,
                    target_inputs=e.get("target_inputs")))
        return results


def decision(data, baseline, candidate, model):
    data.fresh()
    model.check_reference()
    check(frozen_pin(model) == data.frozen, "LEARNING_FROZEN_DRIFT")
    check(all(candidate[k]["identity"] == baseline[k]["identity"] for k in data.keys),
          "LEARNING_IDENTITY_REGRESSION")
    comparisons = {}
    for key in data.plan["evaluate"]:
        comparisons[key] = {}
        for spec in model.specs:
            new = candidate[key]["assessment"]["instruments"][spec.name]
            old = baseline[key]["assessment"]["instruments"][spec.name]
            comparisons[key][spec.name] = dict(candidate=new["prediction_loss"],
                parent=old["prediction_loss"], persistence=new["persistence_loss"],
                better_than_parent=new["prediction_loss"] < old["prediction_loss"],
                better_than_persistence=new["prediction_loss"] < new["persistence_loss"])
    qualified = all(v["better_than_parent"] and v["better_than_persistence"]
                    for group in comparisons.values() for v in group.values())
    # This source/panel is not admitted. Even a development win cannot activate it.
    return dict(comparisons=comparisons, metric_gate_passed=qualified, selected=False,
                reasons=([] if qualified else ["WITHHELD_PER_INSTRUMENT_GATE_FAILED"]) +
                        ["DEVELOPMENT_REPLAY_NOT_ADMITTED_CAPABILITY_EVIDENCE"],
                active_model=data.binding["model"], identity_retained=True,
                frozen_retained=True, useful_capability_established=False)
