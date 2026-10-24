"""Learn source-sequence dynamics from complete current-model document replays."""
from pathlib import Path

import torch

from claims.codata_source import check
from model.constellation_learning import ordered_loss, shared_region_loss, trainable
from model.loop_training_checkpoint import tensor_fingerprint
from model.text_context_regions import ContextSequenceData
from model.text_sequence_data import FLAGS
from senses.contract import digest
from text.constellation_corpus import file_sha


PLAN = dict(epochs=8, steps=40, learning_rate=.001, weight_decay=.01,
    cycle_1=["general", "legal"], cycle_2=["general", "legal", "science"],
    update="complete_document_bptt", selection="final_only", gradient_clip=1., flags=FLAGS)


def code_pins():
    root = Path(__file__).parent
    return {n: file_sha(root/n) for n in ("text_dynamics.py", "text_dynamics_checkpoint.py",
        "text_context_regions.py", "constellation_learning.py")}


def configure(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("core.identity."))
    return model.eval()


def frozen_pin(model):
    names = {n for n, _ in trainable(model)}
    return digest(tensor_fingerprint({n: v for n, v in model.state_dict().items() if n not in names}))


def optimizer_for(model):
    configure(model)
    return torch.optim.AdamW([p for _, p in trainable(model)], lr=PLAN["learning_rate"],
                            weight_decay=PLAN["weight_decay"], foreach=False, fused=False)


class DynamicsData(ContextSequenceData):
    def __init__(self, corpus, bank, config):
        super().__init__(corpus, bank, config)
        self.bind = dict(schema="text-sequence-dynamics-v1", base=self.bind, plan=PLAN,
                         code=code_pins(), flags=FLAGS)
        self.pin = digest(self.bind)
        self.receipts = {}

    def bind_examples(self, examples):
        check(len(examples) == 197 and {e["left"] for e in examples} == set(self.successor),
              "DYNAMICS_EVIDENCE_POPULATION")
        self.receipts = {e["left"]: digest(e) for e in examples}
        self.evidence = digest(examples)

    def loss(self, model, example):
        raise ValueError("MNLM_CODATA_DYNAMICS_SAVED_STATE_FORBIDDEN")

    def sequence(self, examples, group):
        check(group in self.groups and group["split"] == "train", "DYNAMICS_DEVELOPMENT_GRADIENT")
        check([e["left"] for e in examples] == group["indices"][:-1], "DYNAMICS_COMPLETE_SEQUENCE")
        check(all(self.receipts.get(e["left"]) == digest(e) for e in examples), "DYNAMICS_RECEIPT")

    def supervised(self, model, out, example):
        left = example["left"]
        arrays, target_label = self.context_targets[left]
        check(target_label == example["label"] and self.successor[left] == example["right"],
              "DYNAMICS_LABEL")
        label = torch.tensor([target_label], dtype=torch.long, device=model.core.initial.device)
        allowed = torch.ones(len(out["private_definition"]["rows"]), dtype=torch.bool, device=label.device)
        region = shared_region_loss(out, label, allowed, model.specs)
        placed = out["private_view"].supervised_placement({**out, "states": out["private_states"]},
            label, allowed=allowed, event_ids=[example["event"]])
        terms = []
        for spec in model.specs:
            slot = placed.slots[spec.name]
            target = torch.from_numpy(arrays[spec.name].copy()).to(slot.values)
            terms.append(ordered_loss(slot.values[0, ~slot.padding[0]], target))
        return region+torch.stack(terms).mean()

    def document_loss(self, model, examples, group, *, gradient_probe=False):
        self.sequence(examples, group)
        # Saved journal states are evidence only. Every state below is recomputed
        # with this version's parameters; no intra-document detach or stale prefix.
        state = model.start(group["context"])
        losses, first = [], None
        for example in examples:
            out = self.forward(model, state, example["left"], example["event"])
            state = out["states"][0]
            if first is None:
                first = state.hidden
            losses.append(self.supervised(model, out, example))
        trace = None
        if gradient_probe:
            gradient, = torch.autograd.grad(losses[-1], first, retain_graph=True)
            check(bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max() > 0),
                  "DYNAMICS_RECURRENCE_GRADIENT")
            trace = dict(transitions=len(losses), final_loss_to_first_state_max=float(gradient.abs().max()))
        return torch.stack(losses).mean(), trace


def update(model, optimizer, data, examples, group, *, gradient_probe=False):
    optimizer.zero_grad(set_to_none=True)
    loss, trace = data.document_loss(model, examples, group, gradient_probe=gradient_probe)
    check(bool(torch.isfinite(loss)), "DYNAMICS_NONFINITE")
    loss.backward()
    parameters = [p for _, p in trainable(model)]
    check(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
          "DYNAMICS_GRADIENT_COVERAGE")
    norm = torch.nn.utils.clip_grad_norm_(parameters, PLAN["gradient_clip"])
    check(bool(torch.isfinite(norm)), "DYNAMICS_GRADIENT_NORM")
    optimizer.step()
    return dict(loss=float(loss.detach()), gradient_norm=float(norm), recurrence=trace)


def train_cycle(model, optimizer, data, examples, domains, frozen):
    configure(model)
    check(domains in (PLAN["cycle_1"], PLAN["cycle_2"]), "DYNAMICS_DOMAINS")
    groups = [g for g in data.groups if g["split"] == "train" and g["domain"] in domains]
    by_left = {e["left"]: e for e in examples}
    results = []
    for epoch in range(PLAN["epochs"]):
        for group in groups:
            rows = [by_left[i] for i in group["indices"][:-1]]
            result = update(model, optimizer, data, rows, group,
                            gradient_probe=epoch == 0 and group == groups[0])
            check(frozen_pin(model) == frozen, "DYNAMICS_FROZEN_DRIFT")
            results.append(dict(epoch=epoch, domain=group["domain"], count=len(rows), **result))
        import json
        print(json.dumps(dict(epoch=epoch, domains=domains,
            losses=[r["loss"] for r in results[-len(groups):]])), flush=True)
    return dict(updates=results, steps=len(results), supervised_transitions=sum(r["count"] for r in results))
