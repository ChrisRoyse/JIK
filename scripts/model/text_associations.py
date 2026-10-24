"""Learn typed between-constellation associations, beyond immediate succession.

This is a bounded native-geometry diagnostic. Neighbor relations are neither
causal claims nor admitted motifs. Each instrument retains its own targets.
"""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from claims.codata_source import check
from model.constellation import ordered_distance
from model.constellation_learning import trainable
from model.text_dynamics import frozen_pin
from model.text_semantic_data import SemanticData
from senses.contract import digest
from text.constellation_corpus import file_sha


RECIPE = dict(epochs=8, steps=16, learning_rate=.001, weight_decay=.01,
    temperature=.1, association_weight=1., update="complete_training_population",
    objective="next_constellation_plus_six_typed_neighborhood_KL",
    scope="native_geometry_distillation_not_grounded_motif_admission",
    fit_nodes=[86, 104], development_nodes=93, gradient_node_exposures=1520)


class AssociationData(SemanticData):
    def __init__(self, corpus, bank, config):
        super().__init__(corpus, bank, config, transformer=True)
        self.nodes = sorted(self.successor)
        self.train_nodes = [i for i in self.nodes if self.by_index[i]["split"] == "train"]
        check(len(self.nodes) == 197 and len(self.train_nodes) == 104, "ASSOCIATION_POPULATION")
        self.node_pos = {n: i for i, n in enumerate(self.nodes)}
        self.reference_pos = {n: i for i, n in enumerate(self.train_nodes)}
        self.lanes = tuple(s.name for s in self.inputs.specs)
        normalized = {n: [a.astype(np.float64)/np.linalg.norm(a.astype(np.float64), axis=1, keepdims=True)
                           for a in values] for n, values in self.native.items()}
        self.teacher = np.empty((len(self.lanes), len(self.nodes), len(self.train_nodes)), np.float32)
        work = 0
        for lane, name in enumerate(self.lanes):
            values = normalized[name]
            for q, left in enumerate(self.nodes):
                for r, right in enumerate(self.train_nodes):
                    work += len(values[left])*len(values[right])
                    self.teacher[lane, q, r] = 1-ordered_distance(np.clip(values[left]@values[right].T, -1, 1))
        check(work == 122928, "ASSOCIATION_MEASURED_WINDOW_WORK")
        self.bind = dict(schema="text-association-learning-v1", base=self.bind, recipe=RECIPE,
            nodes=self.nodes, train_nodes=self.train_nodes, lanes=list(self.lanes),
            teacher_shape=list(self.teacher.shape), native_window_comparisons=work,
            teacher_sha256=__import__("hashlib").sha256(self.teacher.astype("<f4").tobytes()).hexdigest(),
            code=file_sha(Path(__file__)),
            driver=file_sha(Path(__file__).parents[1]/"audit/text_association_learning.py"))
        self.pin = digest(self.bind)

    def hidden(self, model, groups, examples=None):
        states, losses, indices = [], [], []
        for group in groups:
            if examples is not None:
                self.sequence([examples[i] for i in group["indices"][:-1]], group)
            prior = model.start(group["context"])
            for left in group["indices"][:-1]:
                event = digest(dict(association=self.pin, atom=left))
                out = self.forward(model, prior, left, event)
                prior = out["states"][0]
                states.append(out["placement_hidden"][0])
                indices.append(left)
                if examples is not None:
                    check(group["split"] == "train", "ASSOCIATION_DEVELOPMENT_GRADIENT")
                    losses.append(self.supervised(model, out, {**examples[left], "event": event}))
        return torch.stack(states), indices, torch.stack(losses).mean() if losses else None

    def relations(self, model, query, reference, qids, rids, *, shuffled=False):
        check(len(qids) == len(set(qids)) and len(rids) == len(set(rids))
              and set(qids) <= self.node_pos.keys() and set(rids) <= self.reference_pos.keys(),
              "ASSOCIATION_BOUND_NODES")
        device = query.device
        target = torch.from_numpy(self.teacher[:, [self.node_pos[i] for i in qids]][:,
            :, [self.reference_pos[i] for i in rids]].copy()).to(device)
        valid = torch.tensor(qids, device=device)[:, None] != torch.tensor(rids, device=device)[None]
        check(bool(valid.any(-1).all()), "ASSOCIATION_NO_CANDIDATES")
        scores = []
        for lane, name in enumerate(self.lanes):
            a = F.normalize(model.query[name](query[:, lane]), dim=-1)
            b = F.normalize(model.query[name](reference[:, lane]), dim=-1)
            scores.append(a@b.T)
        scores = torch.stack(scores)
        if shuffled:
            target = target.roll(1, -1)
        lt = F.log_softmax((target/RECIPE["temperature"]).masked_fill(~valid[None], -torch.inf), -1)
        ls = F.log_softmax((scores/RECIPE["temperature"]).masked_fill(~valid[None], -torch.inf), -1)
        kl = (lt.exp()*(lt.masked_fill(~valid[None], 0)-ls.masked_fill(~valid[None], 0))).sum(-1)
        return kl, target.masked_fill(~valid[None], -torch.inf), scores.masked_fill(~valid[None], -torch.inf)

    def evaluate(self, model):
        with torch.no_grad():
            hidden, ids, _ = self.hidden(model, self.groups)
            positions = {i: j for j, i in enumerate(ids)}
            refs = hidden[[positions[i] for i in self.train_nodes]]
            result = {}
            for group in self.groups:
                if group["split"] != "development":
                    continue
                nodes = group["indices"][:-1]
                query = hidden[[positions[i] for i in nodes]]
                kl, target, scores = self.relations(model, query, refs, nodes, self.train_nodes)
                shuffled = self.relations(model, query, refs, nodes, self.train_nodes, shuffled=True)[0]
                correct = target.argsort(dim=-1, descending=True, stable=True)[..., :5]
                ranked = scores.argsort(dim=-1, descending=True, stable=True)[..., :5]
                recall = (correct[..., :, None] == ranked[..., None, :]).any(-1).float().mean(-1)
                result[group["domain"]] = {name: dict(kl_bits=float(kl[j].mean()/np.log(2)),
                    neighbor_recall_at_5=float(recall[j].mean()),
                    shuffled_correspondence_kl_bits=float(shuffled[j].mean()/np.log(2)))
                    for j, name in enumerate(self.lanes)}
            return result

    def controls(self, model):
        with torch.no_grad():
            hidden, ids, _ = self.hidden(model, [self.groups[0]])
            ids, hidden = ids[:6], hidden[:6]
            base = self.relations(model, hidden, hidden, ids, ids)[0]
            order = [3, 0, 5, 1, 4, 2]
            permuted = self.relations(model, hidden[order], hidden[order],
                [ids[i] for i in order], [ids[i] for i in order])[0]
            check(torch.allclose(base[:, order], permuted, atol=2e-6, rtol=0), "ASSOCIATION_ORDER_DEPENDENCE")
            changed = self.relations(model, hidden, hidden, ids, ids, shuffled=True)[0]
            check(not torch.allclose(base, changed), "ASSOCIATION_CORRESPONDENCE_IGNORED")
            denied = []
            for name, bad in (("duplicate", [ids[0]]*len(ids)), ("unbound", [-1]+ids[1:])):
                try:
                    self.relations(model, hidden, hidden, bad, ids)
                except ValueError as exc:
                    check(str(exc) == "MNLM_CODATA_ASSOCIATION_BOUND_NODES", "ASSOCIATION_CONTROL_CODE")
                    denied.append(name)
                else:
                    raise ValueError("MNLM_ASSOCIATION_BAD_NODES_ACCEPTED")
            return dict(simultaneous_permutation_invariant=True, changed_correspondence=True,
                refused=denied, native_target_order="typed_per_instrument", tolerance=2e-6)

    def objective(self, model, optimizer, hidden, succession, relation):
        return succession+RECIPE["association_weight"]*relation, {}


def update(model, optimizer, data, examples, domains, frozen):
    groups = [g for g in data.groups if g["split"] == "train" and g["domain"] in domains]
    optimizer.zero_grad(set_to_none=True)
    hidden, ids, succession = data.hidden(model, groups, examples)
    relation = data.relations(model, hidden, hidden, ids, ids)[0].mean()
    loss, diagnostics = data.objective(model, optimizer, hidden, succession, relation)
    check(bool(torch.isfinite(loss)), "ASSOCIATION_NONFINITE")
    gradient, = torch.autograd.grad(relation, hidden, retain_graph=True)
    check(torch.isfinite(gradient).all() and gradient.abs().max() > 0, "ASSOCIATION_DISCONNECTED")
    loss.backward()
    parameters = [p for _, p in trainable(model)]
    check(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), "ASSOCIATION_GRADIENT_COVERAGE")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    check(torch.isfinite(norm), "ASSOCIATION_GRADIENT_NORM")
    optimizer.step()
    check(frozen_pin(model) == frozen, "ASSOCIATION_STANDARD_CHANGED")
    return dict(loss=float(loss.detach()), succession=float(succession.detach()),
        association_kl_nats=float(relation.detach()), association_gradient=float(gradient.abs().max()),
        nodes=len(ids), directed_associations=len(ids)*(len(ids)-1)*len(data.lanes), **diagnostics)
