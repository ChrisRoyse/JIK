"""A new, scoped ending-head fit; the parent predictor and Adam state stay frozen.

Inputs are separately keyed causal states per proposal and native instrument.
Only decision logits combine. This is not continuation of full-model AdamW.
"""
from copy import deepcopy

import torch
from torch.nn import functional as F
from model.native_population_training import cpu, fingerprint
from text.constellation_corpus import need

RECIPE = dict(lr=1., max_iter=10, max_eval=15, tolerance_grad=1e-9,
              tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')
REGULARIZATION = .001
GRADIENT_TOLERANCE = 1e-7


class EndingData:
    def __init__(self, rows, roster, proposals, working_bytes):
        self.rows, self.roster, self.proposals = deepcopy(rows), tuple(roster), tuple(proposals)
        need(len(rows) > 0 and all(set(r['states']) == set(proposals) for r in rows), 'ENDING_DATA_ROSTER')
        self.inputs = {p:{n:torch.stack([r['states'][p][n] for r in self.rows]).double()
                          for n in roster} for p in proposals}
        self.bytes = sum(v.numel()*v.element_size() for group in self.inputs.values() for v in group.values())
        need(self.bytes <= working_bytes, 'ENDING_DATA_MEMORY')
        need(all(bool(v.isfinite().all()) and v.ndim == 2 and not v.requires_grad
                 for group in self.inputs.values() for v in group.values()), 'ENDING_DATA_VALUES')
        self.labels = torch.tensor([int(r['end']) for r in rows], dtype=torch.long)
        self.meta = [{k:v for k,v in r.items() if k not in ('states','original_logits')} for r in rows]
        self.binding = fingerprint(self.meta)
        self.tensor_versions = self.versions()

    def versions(self):
        return tuple((id(v),v._version) for group in self.inputs.values() for v in group.values()) + ((id(self.labels),self.labels._version),)

    def fresh(self):
        need(self.binding == fingerprint(self.meta) and self.tensor_versions == self.versions(), 'ENDING_DATA_CHANGED')

    def fitting(self, mode, shuffled):
        self.fresh()
        need(mode in ('document_mean','sequence_sum') and type(shuffled) is bool, 'ENDING_FIT_RULE')
        need(all(r['side'] == 'train' for r in self.meta), 'ENDING_FIT_QUERY')
        groups = {}
        for i,r in enumerate(self.meta): groups.setdefault(r['document'],[]).append(i)
        labels = self.labels.clone(); weights = torch.empty(len(labels),dtype=torch.float64)
        for doc,ids in groups.items():
            need(sum(int(self.labels[i]) for i in ids) == 1 and bool(self.labels[ids[-1]]), 'ENDING_LABEL_SCHEDULE')
            weights[ids] = 1./len(groups)/(len(ids) if mode == 'document_mean' else 1.)
            if shuffled:
                order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(20261024+doc))
                labels[ids] = self.labels[ids][order]
        return dict(labels=labels, weights=weights, mode=mode, shuffled=shuffled,
                    pin=fingerprint((self.binding,labels,weights,mode,shuffled)))


def logits(heads, inputs, roster):
    return {p:torch.stack([heads[p][n](inputs[p][n]) for n in roster]).mean(0) for p in heads}


def objective(heads, data, fit):
    data.fresh()
    need(fit['pin'] == fingerprint((data.binding,fit['labels'],fit['weights'],fit['mode'],fit['shuffled'])),
         'ENDING_FIT_BINDING')
    need(all(r['side'] == 'train' for r in data.meta), 'ENDING_FIT_QUERY')
    values = logits(heads,data.inputs,data.roster)
    losses = [torch.sum(F.cross_entropy(v,fit['labels'],reduction='none')*fit['weights']) for v in values.values()]
    penalty = REGULARIZATION*sum(p.square().sum() for p in heads.parameters())
    return torch.stack(losses).mean()+penalty


def make(parent, saved=None, *, expected_pin=None):
    heads = deepcopy(parent.ending).double()
    heads.requires_grad_(True)
    opt = torch.optim.LBFGS(heads.parameters(), **RECIPE)
    if saved is not None:
        need(expected_pin is not None and fingerprint(saved) == expected_pin, 'ENDING_RESTORE_PIN')
        need(set(saved) == {'heads','optimizer'} and saved['optimizer']['state'], 'ENDING_RESTORE_HISTORY')
        heads.load_state_dict(saved['heads'],strict=True); opt.load_state_dict(deepcopy(saved['optimizer']))
        need(fingerprint(checkpoint(heads,opt)) == expected_pin, 'ENDING_RESTORE_EXACT')
        first = next(heads.parameters()); state = opt.state[first]
        need({'func_evals','n_iter','d','t','old_dirs','old_stps','ro','H_diag','prev_flat_grad','prev_loss'} <= set(state),
             'ENDING_RESTORE_HISTORY')
    return heads,opt


def checkpoint(heads, opt):
    return cpu(dict(heads=heads.state_dict(),optimizer=opt.state_dict()))


def step(heads, opt, data, fit):
    need(type(opt) is torch.optim.LBFGS and len(opt.param_groups) == 1, 'ENDING_OPTIMIZER')
    need(all(a is b for a,b in zip(opt.param_groups[0]['params'],heads.parameters(),strict=True)), 'ENDING_OPTIMIZER_OWNER')
    before = fingerprint(heads.state_dict()); evaluations = []
    def closure():
        opt.zero_grad(set_to_none=True); loss = objective(heads,data,fit)
        need(bool(torch.isfinite(loss)), 'ENDING_NONFINITE_LOSS'); loss.backward()
        gradients = {n:p.grad for n,p in heads.named_parameters()}
        need(all(g is not None and bool(g.isfinite().all()) for g in gradients.values()), 'ENDING_GRADIENTS')
        evaluations.append(dict(loss=float(loss.detach()),gradients=fingerprint(gradients)))
        return loss
    opt.step(closure)
    loss = closure(); gradients = cpu({n:p.grad for n,p in heads.named_parameters()})
    maximum = max(float(g.abs().max()) for g in gradients.values())
    after = fingerprint(heads.state_dict())
    status = 'converged' if maximum <= GRADIENT_TOLERANCE else ('stalled' if before == after else 'running')
    return dict(before=before,after=after,loss=float(loss.detach()),gradient_max=maximum,
                gradients=gradients,evaluations=evaluations,status=status,optimizer=fingerprint(opt.state_dict()))


def install(model, heads, *, parent_pin):
    need(fingerprint(model.state_dict()) == parent_pin, 'ENDING_INSTALL_PARENT')
    before = fingerprint({n:v for n,v in model.state_dict().items() if not n.startswith('ending.')})
    converted = {n:v.detach().float() for n,v in heads.state_dict().items()}
    model.ending.load_state_dict(converted,strict=True)
    need(before == fingerprint({n:v for n,v in model.state_dict().items() if not n.startswith('ending.')}),
         'ENDING_INSTALL_BACKBONE')
    need(fingerprint(model.ending.state_dict()) == fingerprint(converted), 'ENDING_INSTALL_WEIGHTS')
    return dict(parent=parent_pin,model=fingerprint(model.state_dict()),frozen_backbone=before,
                ending=fingerprint(converted),maximum_weight_cast_error=max(float((v.double()-converted[n].double()).abs().max())
                                                                          for n,v in heads.state_dict().items()))
