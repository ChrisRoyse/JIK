"""Observed text prefixes extend private regions without inventing future observations."""
from pathlib import Path

import numpy as np
import torch

from claims.codata_source import check, sha
from model.constellation import ordered_distance, unpack
from model.constellation_learning import ordered_loss, shared_region_loss
from model.native import NativeInput
from model.native_identity_world import NativeIdentityWorld
from model.text_sequence_data import FLAGS, SequenceData, state_unpack
from model.text_sequence_session import TextSequenceSession
from model.tool_context_regions import attach
from senses.contract import digest
from text.constellation_corpus import file_sha


class PrefixSession(TextSequenceSession):
    def _predict(self, left):
        body, payload, state = super()._predict(left)
        view, definition = self.data.view(self.model, left)
        payload["private_region"] = dict(definition=definition, world=view.world_id)
        body["payload"] = digest(payload)
        return body, payload, state


class ContextSequenceData(SequenceData):
    def __init__(self, corpus, bank, config):
        super().__init__(corpus, bank, config)
        self.base_bind = self.bind
        self.bind = dict(schema="text-observed-prefix-regions-v1", base=self.base_bind,
            code=file_sha(Path(__file__)), attach_code=file_sha(Path(__file__).with_name("tool_context_regions.py")),
            scope="same_document_observed_prefix_only", flags=FLAGS)
        self.pin = digest(self.bind)
        size = sum(lane.count*lane.dims for lanes in self.inputs.lanes for lane in lanes.values())
        check(size <= config.max_native_elements, "TEXT_CONTEXT_CACHE_BUDGET")
        # Reuse immutable native measurements. One cold read, no per-view file/tokenizer work.
        self.native = {s.name: [] for s in self.inputs.specs}
        for source, index in self.sources:
            for name, (values, _) in bank.constellation(source, index).items():
                self.native[name].append(values)
        self.views, self.context_targets = {}, {}

    def region_indices(self, left, *, supplied=None):
        check(left in self.by_index, "TEXT_CONTEXT_SOURCE")
        group = self.by_index[left]
        prefix = group["indices"][:group["indices"].index(left)+1]
        check(supplied is None or supplied == prefix, "TEXT_CONTEXT_FUTURE_OR_OTHER")
        return list(dict.fromkeys(self.refs+prefix))

    def view(self, model, left):
        context = self.by_index[left]["context"]
        indices = self.region_indices(left)
        desc = dict(schema="text-private-native-regions-v1", context=context, left=left,
            through=self.sources[left][0]["char_end"], shared=model.world_id,
            rows=[self.sources[i][0]["atom_id"] for i in indices], input=self.inputs.binding)
        pin = digest(desc)
        cached = self.views.get(context)
        if cached is None or cached[0] != pin:
            references = {}
            for spec in model.specs:
                arrays = [self.native[spec.name][i] for i in indices]
                width = max(map(len, arrays))
                check(width <= model.config.max_native_tokens and len(indices)*width*spec.dims
                      <= model.config.max_native_elements, "TEXT_CONTEXT_REFERENCE_BUDGET")
                values = torch.zeros(len(indices), width, spec.dims, device=model.core.initial.device)
                padding = torch.ones(len(indices), width, dtype=torch.bool, device=values.device)
                for j, array in enumerate(arrays):
                    values[j, :len(array)] = torch.from_numpy(array.copy()).to(values.device)
                    padding[j, :len(array)] = False
                references[spec.name] = NativeInput(spec.identity, tuple(desc["rows"]), values, padding,
                    torch.zeros(len(indices), dtype=torch.bool, device=values.device))
            with torch.random.fork_rng(devices=[]):
                view = NativeIdentityWorld(model.core, references, world_source_id=pin)
            view.to(model.core.initial.device).eval()
            self.views[context] = (pin, view)
        view = attach(self.views[context][1], model)
        return view, desc

    def forward(self, model, prior, left, event):
        view, desc = self.view(model, left)
        allowed = torch.ones(len(desc["rows"]), dtype=torch.bool, device=model.core.initial.device)
        out = view(self.batch(left), (view.bind_state(prior),), allowed=allowed, phase=0,
            event_ids=[event], evidence_origin=["source_observation"], advance_phase=False)
        return {**out, "states": tuple(model.bind_state(s) for s in out["states"]),
            "private_states": out["states"], "private_view": view, "private_definition": desc}

    def reveal(self, left, right, predictions, probabilities):
        self.edge(left, right)
        indices = self.region_indices(left)
        if left not in self.context_targets:
            arrays = {s.name: self.array(s.name, right) for s in self.inputs.specs}
            distances = np.zeros(len(indices), dtype=np.float64)
            for name, target in arrays.items():
                for j, i in enumerate(indices):
                    reference = self.native[name][i].astype(np.float64)
                    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
                    distances[j] += ordered_distance(np.clip(target @ reference.T, -1, 1))
            self.context_targets[left] = arrays, int(distances.argmin())
        arrays, label = self.context_targets[left]
        check(len(probabilities) == len(indices) and np.isfinite(probabilities).all(), "TEXT_CONTEXT_LOGPROB")
        metrics = {}
        for spec in self.inputs.specs:
            target, current = arrays[spec.name], self.array(spec.name, left)
            pred = unpack(predictions[spec.name], spec, max_windows=self.config.max_native_tokens)
            metrics[spec.name] = dict(prediction=ordered_distance(np.clip(target @ pred.T, -1, 1)),
                persistence=ordered_distance(np.clip(target @ current.T, -1, 1)),
                target_windows=len(target), predicted_windows=len(pred),
                target_sha256=sha(target.astype("<f4").tobytes()))
        return dict(target=self.sources[right][0]["atom_id"], label=label, region_vocabulary=indices,
            region_nll_bits=float(-probabilities[label]/np.log(2)), correct_region=int(np.argmax(probabilities)) == label,
            instruments=metrics, metric="observed_prefix_source_region_logscore_and_native_edit_v1", flags=FLAGS)

    def loss(self, model, example):
        prior = state_unpack(example["prior"], model)
        out = self.forward(model, prior, example["left"], example["event"])
        view = out["private_view"]
        arrays, target_label = self.context_targets[example["left"]]
        check(target_label == example["label"] and self.successor[example["left"]] == example["right"], "TEXT_CONTEXT_LABEL")
        label = torch.tensor([target_label], dtype=torch.long, device=model.core.initial.device)
        allowed = torch.ones(len(out["private_definition"]["rows"]), dtype=torch.bool, device=label.device)
        shared = shared_region_loss(out, label, allowed, model.specs)
        local = {**out, "states": out["private_states"]}
        placed = view.supervised_placement(local, label, allowed=allowed, event_ids=[example["event"]])
        losses = []
        for spec in model.specs:
            slot = placed.slots[spec.name]
            target = torch.from_numpy(arrays[spec.name].copy()).to(slot.values).float()
            losses.append(ordered_loss(slot.values[0, ~slot.padding[0]], target))
        return shared+torch.stack(losses).mean()
