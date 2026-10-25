"""Increase renderer capacity over a frozen trained parent, with explicit new state."""
from copy import deepcopy
from dataclasses import asdict

import torch
from torch import nn

from model.answer_surface_stream import native_versions
from model.native_answer_sequence import versions
from model.native_population_training import cpu,fingerprint
from model.stream_surface import StreamSurfaceDecoder
from model.surface_device_runtime import evaluate
from text.constellation_corpus import need


class ProtectedSurfaceDecoder(nn.Module):
    def __init__(self,parent,config,*,seed=171):
        super().__init__();config.validate()
        need(config.vocabulary==parent.config.vocabulary and config.max_prefix==parent.config.max_prefix,
             'PROTECTED_SURFACE_ALPHABET')
        self.parent=deepcopy(parent).eval().requires_grad_(False)
        self.config=config;self.specs=tuple(parent.specs)
        with torch.random.fork_rng(devices=[]):
            # The new module is initialized on CPU. Do not seed another device's
            # generator, including a caller's existing CUDA training stream.
            torch.random.default_generator.manual_seed(seed)
            self.correction=StreamSurfaceDecoder(config,self.specs)
        self.correction.to(self.parent.tokens.weight.device)
        with torch.no_grad():
            self.correction.output.weight.zero_();self.correction.output.bias.zero_()
        self.parent_pin=fingerprint(self.parent.state_dict());self.parent_versions=versions(self.parent)
        self.definition=asdict(config),tuple(self.specs),repr(self.parent.config)

    @property
    def tokens(self):return self.correction.tokens

    def train(self,mode=True):
        super().train(mode);self.parent.eval();return self

    def fresh(self):
        need(not self.parent.training and all(not p.requires_grad and p.grad is None for p in self.parent.parameters())
             and versions(self.parent)==self.parent_versions,'PROTECTED_SURFACE_PARENT_CHANGED')
        need((asdict(self.config),tuple(self.specs),repr(self.parent.config))==self.definition
             and self.correction.config==self.config and self.correction.specs==self.specs,'PROTECTED_SURFACE_CONFIG_CHANGED')

    def validate_inputs(self,native):
        self.fresh();self.parent.validate_inputs(native);return self.correction.validate_inputs(native)

    def memories(self,native):
        raise ValueError('MNLM_PROTECTED_SURFACE_USE_COMPOSITE_STEPPER_OR_DIRECT')

    def forward(self,prefix,native,*,offsets):
        self.fresh();before=native_versions(native)
        with torch.no_grad():parent=self.parent(prefix,native,offsets=offsets)
        delta=self.correction(prefix,native,offsets=offsets)
        logits=parent['logits']+delta['logits']
        need(bool(logits.isfinite().all()) and before==native_versions(native),'PROTECTED_SURFACE_FORWARD')
        self.fresh();return dict(delta,logits=logits,purpose='engineering_protected_surface_correction')


def optimizer_for(model):
    model.fresh();return torch.optim.AdamW(model.correction.parameters(),lr=1e-4,foreach=False)


def check(model,optimizer,step):
    model.fresh();actual=[p for g in optimizer.param_groups for p in g['params']]
    expected=list(model.correction.parameters())
    need(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups)==1
         and optimizer.param_groups[0]['lr']==1e-4 and optimizer.param_groups[0]['foreach'] is False,
         'PROTECTED_SURFACE_OPTIMIZER_POLICY')
    need(len(actual)==len(expected) and all(a is b for a,b in zip(actual,expected)),'PROTECTED_SURFACE_OPTIMIZER_MEMBERSHIP')
    need(all(p.requires_grad and bool(p.isfinite().all()) for p in expected),'PROTECTED_SURFACE_PARAMETERS')
    need(len(optimizer.state)==(len(expected) if step else 0),'PROTECTED_SURFACE_MOMENT_COVERAGE')
    for parameter,value in optimizer.state.items():
        need(set(value)=={'step','exp_avg','exp_avg_sq'} and float(value['step'])==step
             and value['exp_avg'].shape==parameter.shape==value['exp_avg_sq'].shape
             and all(bool(t.isfinite().all()) for t in value.values()),'PROTECTED_SURFACE_MOMENTS')


def update(model,optimizer,examples,step):
    need(type(step) is int and step>=0 and 0<len(examples)<=128 and all(e.side=='train' for e in examples),
         'PROTECTED_SURFACE_NONTRAIN')
    check(model,optimizer,step);before=fingerprint(model.correction.state_dict())
    native=fingerprint([{n:x.values for n,x in e.native.items()} for e in examples])
    model.train();optimizer.zero_grad(set_to_none=True)
    scores,profile=evaluate(model,examples,'gathered',8,backward=True)
    gradients={n:p.grad for n,p in model.correction.named_parameters()}
    need(all(v is not None and bool(v.isfinite().all()) for v in gradients.values()),'PROTECTED_SURFACE_GRADIENT')
    branches={}
    for s in model.specs:
        for prefix in ['projectors.'+s.name+'.',*[f'blocks.{i}.branches.{s.name}.' for i in range(model.config.layers)]]:
            values={n:v for n,v in gradients.items() if n.startswith(prefix)}
            need(values and (step==0 or any(bool(v.abs().any()) for v in values.values())),'PROTECTED_SURFACE_DISCONNECTED')
            branches[prefix]=dict(pin=fingerprint(values),nonzero=any(bool(v.abs().any()) for v in values.values()))
    if step==0:
        need(all(not bool(v.abs().any()) for n,v in gradients.items() if not n.startswith('output.'))
             and any(bool(v.abs().any()) for n,v in gradients.items() if n.startswith('output.')),'PROTECTED_SURFACE_ZERO_START')
    gradient=fingerprint(gradients);optimizer.step();check(model,optimizer,step+1)
    need(native==fingerprint([{n:x.values for n,x in e.native.items()} for e in examples]),'PROTECTED_SURFACE_NATIVE_CHANGED')
    row=dict(step=step+1,before=before,after=fingerprint(model.correction.state_dict()),optimizer=fingerprint(optimizer.state_dict()),
             gradient=gradient,branches=branches,scores=fingerprint(scores),positions=profile['positions'],objective=profile['objective'],native=native)
    return row,profile,scores
