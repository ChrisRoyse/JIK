"""A fixed train-only native region vocabulary and exact observed-prefix inputs."""
from pathlib import Path

import numpy as np
import torch

from claims.codata_source import check, sha
from model.constellation import ordered_distance
from model.episode_input import EpisodeBatch
from model.native_prefix_transformer import PrefixSensoryCore
from model.text_dynamics import DynamicsData
from model.text_sequence_data import FLAGS
from senses.contract import digest
from text.constellation_corpus import file_sha


REGIONS = 8


class PrefixBatch(EpisodeBatch):
    def __init__(self, data, left, *, indices=None):
        super().__init__(data.inputs, left)
        self.data = data
        self.indices = data.prefix_indices(left) if indices is None else tuple(indices)
        self.frames = tuple(data.cached[i] for i in self.indices)

    def validate_prefix(self):
        check(all(type(i) is int for i in self.indices)
              and self.indices == self.data.prefix_indices(self.index), "SEMANTIC_PREFIX_CUTOFF")
        check(len(self.frames) == len(self.indices)
              and all(frame is self.data.cached[i] for i, frame in zip(self.indices, self.frames)),
              "SEMANTIC_PREFIX_BINDING")
        for frame in self.frames:
            check(tuple((n, x._version) for n, x in frame.items()) == self.data.versions[id(frame)],
                  "SEMANTIC_PREFIX_CHANGED")

    def validate(self, specs, config):
        super().validate(specs, config)
        self.validate_prefix()
        return self


class SemanticData(DynamicsData):
    def __init__(self, corpus, bank, config, *, transformer):
        super().__init__(corpus, bank, config)
        self.transformer = transformer
        self.fit = tuple(self.refs)
        normalized = {n: [a.astype(np.float64)/np.linalg.norm(a.astype(np.float64), axis=1, keepdims=True)
                           for a in values] for n, values in self.native.items()}
        def distance(i, j):
            return sum(ordered_distance(np.clip(values[i]@values[j].T, -1, 1)) for values in normalized.values())
        # O(R*N), fixed deterministic coverage. No all-pairs matrix, validation
        # fitting, native averaging, or supervised anchor input.
        selected, nearest = [], np.full(len(self.fit), np.inf)
        chosen = 0
        for _ in range(REGIONS):
            selected.append(self.fit[chosen])
            nearest = np.minimum(nearest, [distance(i, selected[-1]) for i in self.fit])
            for j, i in enumerate(self.fit):
                if i in selected:
                    nearest[j] = -1
            chosen = int(nearest.argmax())
        check(len(set(selected)) == REGIONS and set(selected) <= set(self.fit), "SEMANTIC_TRAIN_ONLY_BOOK")
        self.refs = selected
        self.ref_set = set(selected)
        codes = [int(np.argmin([distance(i, j) for j in selected])) for i in range(len(self.sources))]
        self.book = dict(schema="native-exemplar-region-vocabulary-v1", regions=REGIONS,
            method="train_only_farthest_first_ordered_native_distance", fit=list(self.fit), exemplars=selected,
            codes=codes, input=self.inputs.binding,
            native={n: [sha(self.native[n][i].astype("<f4").tobytes()) for i in selected] for n in self.native},
            train_occupancy=np.bincount([codes[i] for i in self.fit], minlength=REGIONS).tolist(), flags=FLAGS)
        self.book_pin = digest(self.book)
        self.shapes = []
        for group in self.groups:
            rows = group["indices"][:-1]
            windows = sum(len(self.native[n][i]) for n in self.native for i in rows)
            elements = sum(self.native[n][i].size for n in self.native for i in rows)
            tokens = windows+len(rows)*len(self.inputs.specs)
            check(tokens <= config.max_native_tokens and elements <= config.max_native_elements
                  and config.heads*tokens*tokens <= config.max_attention_elements, "SEMANTIC_PREFIX_BUDGET")
            self.shapes.append(dict(context=group["context"], atoms=len(rows), windows=windows, tokens=tokens,
                                    attention_elements=config.heads*tokens*tokens))
        self.context_targets = {left: ({n: v[right] for n, v in normalized.items()}, codes[right])
                                for left, right in self.successor.items()}
        paths = ("text_semantic_data.py", "native_prefix_transformer.py", "native_identity_world.py")
        self.bind = dict(schema="text-semantic-region-sequence-v1", base=self.bind, book=self.book,
            book_pin=self.book_pin, transformer=transformer, code={n: file_sha(Path(__file__).with_name(n)) for n in paths},
            target="fixed_region_then_complete_native_slots", flags=FLAGS)
        self.pin = digest(self.bind)
        self.cached, self.versions = {}, {}

    def prepare(self, device):
        check(not self.cached, "SEMANTIC_CACHE_ALREADY_PREPARED")
        check(sum(a.size for values in self.native.values() for a in values) <= self.config.max_native_elements,
              "SEMANTIC_CACHE_BUDGET")
        for i in range(len(self.sources)):
            frame = {n: torch.from_numpy(values[i].copy()).to(device) for n, values in self.native.items()}
            self.cached[i] = frame
            self.versions[id(frame)] = tuple((n, x._version) for n, x in frame.items())

    def prefix_indices(self, left):
        check(type(left) is int and left in self.by_index, "SEMANTIC_PREFIX_SOURCE")
        indices = self.by_index[left]["indices"]
        return tuple(indices[:indices.index(left)+1])

    def region_indices(self, left, *, supplied=None):
        prefix = self.prefix_indices(left)
        check(supplied is None or tuple(supplied) == prefix, "SEMANTIC_PREFIX_CUTOFF")
        return list(self.refs)

    def batch(self, index):
        return PrefixBatch(self, index).validate(self.inputs.specs, self.config)

    def representation_controls(self, model):
        from audit.text_semantic_checks import controls
        return controls(model, self)


def build_semantic(data, args):
    from audit.text_learning_support import build
    from model.loop_training_checkpoint import tensor_fingerprint
    from model.text_dynamics import configure
    import json
    print(json.dumps(dict(prefix_shapes=data.shapes, train_region_occupancy=data.book["train_occupancy"])), flush=True)
    model = build(data, args)
    data.shared_initial = digest(tensor_fingerprint(model.state_dict()))
    if data.transformer:
        model.core = PrefixSensoryCore(model.core).to(model.core.initial.device)
    check(sum(p.numel() for p in model.parameters()) <= model.config.max_parameters,
          "SEMANTIC_MODEL_PARAMETER_BUDGET")
    data.prepare(model.core.initial.device)
    return configure(model)
