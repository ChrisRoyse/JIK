"""Missing-slot relationships to fixed native landmarks; no serving authority.

All six outputs retain their instrument's coordinates and landmark identities.
The prediction API receives only the other five slots. Neither identity state,
prefix, target vectors nor target labels are available through that API.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from claims.codata_source import check, sha
from model.attention import Linear
from model.loop_training_checkpoint import tensor_fingerprint
from model.text_semantic_data import SemanticData, build_semantic
from model.text_sequence_data import FLAGS
from senses.contract import digest
from text.constellation_corpus import file_sha


RECIPE = dict(epochs=8, steps=16, learning_rate=.001, weight_decay=.01,
    temperature=.1, selection="final_only", train_atoms=[88, 107],
    training_atom_exposures=1560, missing_slot_examples=9360,
    objective="missing_instrument_native_landmark_distribution",
    controls="within_document_roll_of_training_input_constellations",
    flags={**FLAGS, "conditional_lens_information": False})


class MaskedData(SemanticData):
    def __init__(self, corpus, bank, config):
        super().__init__(corpus, bank, config, transformer=True)
        self.nodes = list(range(len(self.sources)))
        self.train_nodes = list(self.fit)
        self.lanes = tuple(s.name for s in self.inputs.specs)
        check(len(self.nodes) == 203 and len(self.fit) == 107 and len(self.lanes) == 6
              and all(len(v) == 1 for rows in self.native.values() for v in rows),
              "MASKED_PROOF_POPULATION")
        self.arrays = {}
        targets = []
        for name in self.lanes:
            raw = np.stack([a[0] for a in self.native[name]]).astype(np.float64)
            raw /= np.linalg.norm(raw, axis=1, keepdims=True)
            self.arrays[name] = raw.astype(np.float32)
            targets.append(np.clip(raw@raw[self.fit, :].T, -1, 1).astype(np.float32))
        self.teacher = np.stack(targets)
        self.bind = dict(schema="masked-native-relationships-v1", base=self.bind,
            recipe=RECIPE, nodes=self.nodes, train_nodes=self.train_nodes, lanes=list(self.lanes),
            native_window_comparisons=int(self.teacher.size), teacher=sha(self.teacher.tobytes()),
            source_atoms=[r[0]["atom_id"] for r in self.sources],
            code={n: file_sha(Path(__file__).with_name(n)) for n in
                ("text_masked_associations.py", "native_prefix_transformer.py")},
            driver=file_sha(Path(__file__).parents[1]/"audit/text_masked_learning.py"))
        self.pin = digest(self.bind)

    def indices(self, domains):
        return [i for g in self.groups if g["split"] == "train" and g["domain"] in domains
                for i in g["indices"]]

    def deranged(self, indices):
        mapping = {}
        for g in self.groups:
            rows = g["indices"]
            mapping.update(zip(rows, rows[1:]+rows[:1]))
        return [mapping[i] for i in indices]

    def observed(self, indices, missing):
        check(missing in self.lanes and indices and len(set(indices)) == len(indices)
              and all(type(i) is int and i in self.cached for i in indices), "MASKED_SOURCE_INDEX")
        # Deliberately do not retrieve or inspect the missing instrument's tensor.
        result = {}
        for name in self.lanes:
            if name == missing:
                continue
            for i in indices:
                check((name, self.cached[i][name]._version) in self.versions[id(self.cached[i])],
                      "MASKED_SOURCE_CHANGED")
            result[name] = torch.stack([self.cached[i][name] for i in indices])
        return result

    def loss(self, probe, indices, *, shuffled):
        check(set(indices) <= set(self.train_nodes), "MASKED_DEVELOPMENT_GRADIENT")
        observed_ids = self.deranged(indices) if shuffled else indices
        losses = []
        for lane, name in enumerate(self.lanes):
            scores = probe(self.observed(observed_ids, name), name)
            valid = torch.tensor(indices, device=scores.device)[:, None] != probe.landmark_ids[None]
            target = torch.from_numpy(self.teacher[lane, indices].copy()).to(scores.device)
            lt = F.log_softmax((target/RECIPE["temperature"]).masked_fill(~valid, -torch.inf), -1)
            ls = F.log_softmax((scores/RECIPE["temperature"]).masked_fill(~valid, -torch.inf), -1)
            losses.append((lt.exp()*(lt.masked_fill(~valid, 0)-ls.masked_fill(~valid, 0))).sum(-1).mean())
        return torch.stack(losses)


class MaskedRelationProbe(nn.Module):
    def __init__(self, world, data):
        super().__init__()
        self.config, self.specs = world.config, world.specs
        self.encoder = world.core.sequence_transformer
        self.projectors = world.core.projectors
        self.heads = nn.ModuleDict({s.name: Linear(self.config.width, s.dims) for s in self.specs})
        self.register_buffer("landmark_ids", torch.tensor(data.train_nodes, dtype=torch.long))
        for lane, s in enumerate(self.specs):
            self.register_buffer(f"keys_{lane}", torch.from_numpy(data.arrays[s.name][data.fit, :].copy()))
        self.frozen = digest(tensor_fingerprint(dict(self.named_buffers())))
        check(sum(p.numel() for p in self.parameters()) <= self.config.max_parameters,
              "MASKED_PARAMETER_BUDGET")

    def check_reference(self):
        check(digest(tensor_fingerprint(dict(self.named_buffers()))) == self.frozen,
              "MASKED_FROZEN_REFERENCE_CHANGED")

    def forward(self, observed, missing):
        names = tuple(s.name for s in self.specs)
        check(missing in names and set(observed) == set(names)-{missing}, "MASKED_OBSERVED_ROSTER")
        device = self.encoder.query.device
        batch = next(iter(observed.values())).shape[0]
        size = 2*len(names)-1
        check(batch > 0 and batch*size <= self.config.max_native_tokens
              and sum(x.numel() for x in observed.values()) <= self.config.max_native_elements
              and batch*self.config.heads*size*size <= self.config.max_attention_elements,
              "MASKED_INPUT_BUDGET")
        zero = self.encoder.position(torch.zeros(1, dtype=torch.long, device=device))
        tokens, query_position, cursor = [], None, 0
        for lane, spec in enumerate(self.specs):
            if spec.name != missing:
                x = observed[spec.name]
                check(x.shape == (batch, 1, spec.dims) and x.device == device and x.is_cuda
                      and x.dtype == torch.float32 and bool(torch.isfinite(x).all()), "MASKED_NATIVE_INPUT")
                projected = F.layer_norm(self.projectors[spec.name](x), (self.config.width,))
                tokens.append(projected+self.encoder.kind[lane]+2*zero)
                cursor += 1
            else:
                query_position = cursor
            query = self.encoder.query[lane]+self.encoder.kind[lane]+zero
            tokens.append(query[None].expand(batch, 1, -1))
            cursor += 1
        # Attention positions are typed tokens, not concatenated native coordinates.
        x = torch.cat(tokens, dim=1)
        mask = torch.zeros(size, size, device=device, dtype=torch.bool)
        for block in self.encoder.blocks:
            x = block(x, mask)
        state = self.encoder.norm(x)[:, query_position]
        native_query = F.normalize(self.heads[missing](state), dim=-1)
        keys = getattr(self, f"keys_{names.index(missing)}")
        return native_query@keys.T


def build_probe(data, args):
    world = build_semantic(data, args)
    data.identity_before = digest(tensor_fingerprint(world.core.identity.state_dict()))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261021)
        probe = MaskedRelationProbe(world, data).cuda().eval()
    for p in probe.parameters():
        p.requires_grad_(True)
    probe.check_reference()
    check(data.identity_before == digest(tensor_fingerprint(world.core.identity.state_dict())),
          "MASKED_IDENTITY_CHANGED")
    return probe
