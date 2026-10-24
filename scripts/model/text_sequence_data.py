"""Source succession supervision, separately scoped from empirical world outcomes."""
import base64
from pathlib import Path

import numpy as np
import torch

from claims.codata_source import check, sha
from model.constellation import ordered_distance, unpack
from model.constellation_learning import ordered_loss, shared_region_loss
from model.native_episode_session import tensor
from model.native_identity_world import SensoryState
from model.text_episode_input import TextInputs
from senses.contract import digest
from text.constellation_corpus import file_sha


PURPOSE = "source_sequence_diagnostic"
FLAGS = dict(scientific_admission=False, serving=False, useful_capability_proven=False,
             empirical_world_forecast=False, sealed_evaluation=False, source_claims_are_truth=False)
PLAN = dict(epochs=8, steps=1520, learning_rate=.001, weight_decay=.01, selection="final_only",
            cycle_1=["general", "legal"], cycle_2=["general", "legal", "science"],
            scope=PURPOSE, target="next_source_constellation", flags=FLAGS)


def code_pins():
    root = Path(__file__).parent
    names = ("text_sequence_data.py", "text_sequence_session.py", "text_sequence_checkpoint.py",
             "constellation_learning.py", "native_episode_session.py")
    return {n: file_sha(root/n) for n in names}


def state_pack(state):
    return dict(context=state.context, standard=state.standard_id, world_id=state.world_id,
                sequence=state.sequence, last_event=state.last_event,
                world=tensor(state.hidden), identity=tensor(state.identity_hidden))


def state_unpack(value, model):
    def decode(data, shape):
        check(data["dtype"] == "float32-le" and data["shape"] == list(shape), "TEXT_STATE_SHAPE")
        raw = base64.b64decode(data["base64"], validate=True)
        check(sha(raw) == data["sha256"] and len(raw) == int(np.prod(shape))*4, "TEXT_STATE_BYTES")
        a = np.frombuffer(raw, dtype="<f4").reshape(shape)
        check(np.isfinite(a).all(), "TEXT_STATE_FINITE")
        return torch.from_numpy(a.copy()).to(model.core.initial.device)
    check(value["standard"] == model.standard_id and value["world_id"] == model.world_id
          and value["sequence"] == 0, "TEXT_STATE_CONTRACT")
    return SensoryState(value["context"], value["standard"], 0,
        decode(value["world"], model.core.initial.shape), value["last_event"], value["world_id"],
        decode(value["identity"], model.core.identity.initial.shape))


class SequenceData:
    """One pinned diagnostic population. Neither training authority nor a truth oracle."""
    def __init__(self, corpus, bank, config):
        sources, self.groups, offset = [], [], 0
        for i, doc in enumerate(corpus.documents):
            rows = corpus.document(i)
            if doc["domain"] in ("general", "legal", "science"):
                indices = list(range(len(sources), len(sources)+len(rows)))
                sources.extend((r, offset+j) for j, r in enumerate(rows))
                check(len({r["sequence"] for r in rows}) == 1, "TEXT_SEQUENCE_FIELD")
                self.groups.append(dict(context=doc["id"], domain=doc["domain"],
                    split="train" if doc["ordinal"] == 0 else "development", indices=indices))
            offset += len(rows)
        check(len(sources) == 203 and len(self.groups) == 6, "TEXT_PROOF_POPULATION")
        self.inputs = TextInputs(bank, sources, config)
        self.sources, self.bank, self.config = self.inputs.sources, bank, config
        self.refs = [i for g in self.groups if g["split"] == "train" for i in g["indices"]]
        self.ref_set = set(self.refs)
        check(len(self.refs) == 107, "TEXT_REFERENCE_POPULATION")
        self.by_index, self.successor = {}, {}
        for group in self.groups:
            for left, right in zip(group["indices"], group["indices"][1:]):
                self.edge(left, right)
                self.successor[left] = right
                self.by_index[left] = group
        self.bind = dict(schema="text-sequence-diagnostic-v1", input=self.inputs.binding,
            groups=self.groups, references=self.refs, plan=PLAN, code=code_pins(), flags=FLAGS,
            scope="retrospective_source_replay; training_vocabulary_known; no_future_world_claim")
        self.pin = digest(self.bind)
        self.reference_arrays = {s.name: [self.array(s.name, i) for i in self.refs] for s in self.inputs.specs}
        self.targets = {}

    def edge(self, left, right):
        check(type(left) is int and type(right) is int and 0 <= left < len(self.sources)
              and 0 <= right < len(self.sources), "TEXT_SUCCESSOR_INDEX")
        a, b = self.sources[left][0], self.sources[right][0]
        check(a["sequence"] == b["sequence"] and a["pos"]+1 == b["pos"]
              and a["char_end"] == b["char_start"] and a["byte_end"] == b["byte_start"],
              "TEXT_NOT_IMMEDIATE_SUCCESSOR")

    def array(self, name, index):
        a = self.bank.row(name, *self.sources[index])[0].astype(np.float64)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    def batch(self, index):
        return self.inputs.batch(index, through=self.sources[index][0]["char_end"])

    def reveal(self, left, right, predictions, probabilities):
        self.edge(left, right)
        if right not in self.targets:
            arrays = {s.name: self.array(s.name, right) for s in self.inputs.specs}
            # Reduce scalar match evidence only. No native vector pooling/concatenation.
            distances = np.zeros(len(self.refs), dtype=np.float64)
            for name, target in arrays.items():
                for j, reference in enumerate(self.reference_arrays[name]):
                    distances[j] += ordered_distance(np.clip(target @ reference.T, -1, 1))
            self.targets[right] = (arrays, int(distances.argmin()))
        arrays, label = self.targets[right]
        check(len(probabilities) == len(self.refs) and np.isfinite(probabilities).all(), "TEXT_REGION_LOGPROB")
        metrics = {}
        for spec in self.inputs.specs:
            target, current = arrays[spec.name], self.array(spec.name, left)
            pred = unpack(predictions[spec.name], spec, max_windows=self.config.max_native_tokens)
            metrics[spec.name] = dict(prediction=ordered_distance(np.clip(target @ pred.T, -1, 1)),
                persistence=ordered_distance(np.clip(target @ current.T, -1, 1)),
                target_windows=len(target), predicted_windows=len(pred),
                target_sha256=sha(target.astype("<f4").tobytes()))
        return dict(target=self.sources[right][0]["atom_id"], label=label,
            region_nll_bits=float(-probabilities[label]/np.log(2)),
            correct_region=int(np.argmax(probabilities)) == label, instruments=metrics,
            metric="source_sequence_logscore_and_typed_ordered_cosine_edit_v1", flags=FLAGS)

    def forward(self, model, prior, left, event):
        allowed = torch.ones(len(self.refs), dtype=torch.bool, device=model.core.initial.device)
        return model(self.batch(left), (prior,), allowed=allowed, phase=0, event_ids=[event],
                     evidence_origin=["source_observation"], advance_phase=False)

    def loss(self, model, example):
        prior = state_unpack(example["prior"], model)
        out = self.forward(model, prior, example["left"], example["event"])
        right = self.successor[example["left"]]
        check(right == example["right"] and self.targets[right][1] == example["label"], "TEXT_LEARNING_LABEL")
        label = torch.tensor([example["label"]], dtype=torch.long, device=model.core.initial.device)
        allowed = torch.ones(len(self.refs), dtype=torch.bool, device=label.device)
        shared = shared_region_loss(out, label, allowed, model.specs)
        placed = model.supervised_placement(out, label, allowed=allowed, event_ids=[example["event"]])
        losses = []
        for spec in model.specs:
            slot = placed.slots[spec.name]
            target = torch.from_numpy(self.targets[right][0][spec.name].copy()).to(slot.values).float()
            losses.append(ordered_loss(slot.values[0, ~slot.padding[0]], target))
        return shared+torch.stack(losses).mean()


def summarize(rows):
    check(rows, "TEXT_EMPTY_ASSESSMENT")
    names = sorted(rows[0]["instruments"])
    return dict(count=len(rows), region_nll_bits=float(np.mean([r["region_nll_bits"] for r in rows])),
        region_accuracy=float(np.mean([r["correct_region"] for r in rows])),
        instruments={n: {k: float(np.mean([r["instruments"][n][k] for r in rows]))
                         for k in ("prediction", "persistence")} for n in names})
