"""Preserve the parent computation and evaluate the constrained residual separately."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from claims.codata_source import check, sha
from claims.episodic_memory import digest
from model.attention import Linear
from model.loop_training_checkpoint import tensor_fingerprint
from model.response_basis import modules

RULE = 'separate_parent_and_projected_residual_v1'


def code_pin():
    return sha(Path(__file__).read_bytes())


class SeparatedLinear(nn.Module):
    def __init__(self, linear, projector):
        super().__init__()
        check(type(linear) is Linear and linear.bias is None, 'STABLE_RESPONSE_MODULE')
        check(linear.weight.dtype == torch.float32 and linear.weight.is_cuda,
              'STABLE_RESPONSE_RUNTIME')
        self.weight = linear.weight
        self.register_buffer('anchor',linear.weight.detach().clone())
        self.register_buffer('projector',projector.detach().to(linear.weight.device).clone())
        check(self.projector.dtype == torch.float64 and self.projector.shape ==
              (self.weight.shape[1],self.weight.shape[1]),'STABLE_RESPONSE_PROJECTOR')
        self.seal()

    def seal(self):
        self._versions = tuple((id(v),v.data_ptr(),v._version) for v in (self.anchor,self.projector))

    def checked(self):
        check(tuple((id(v),v.data_ptr(),v._version) for v in (self.anchor,self.projector)) == self._versions,
              'STABLE_RESPONSE_FROZEN_BUFFER')

    def forward(self, values):
        self.checked()
        check(values.dtype == torch.float32 and values.device == self.weight.device,
              'STABLE_RESPONSE_INPUT_RUNTIME')
        flat = values.contiguous().reshape(-1,self.weight.shape[1])
        parent = F.linear(flat,self.anchor)
        residual = F.linear(flat.double()@self.projector,self.weight.double()-self.anchor.double())
        return (parent+residual.to(parent.dtype)).reshape(*values.shape[:-1],self.weight.shape[0])


def install(model, basis):
    original = modules(model)
    check(list(original) == [r['name'] for r in basis.manifest['branches']], 'STABLE_RESPONSE_ROSTER')
    for i,(name,linear) in enumerate(original.items()):
        _,branch,instrument,_ = name.split('.')
        projector = torch.from_numpy(basis.arrays[f'p_{i}'])
        model.query[branch][instrument].up = SeparatedLinear(linear,projector)
    return buffers(model)


def buffers(model):
    values = {}
    for name,module in modules(model).items():
        check(type(module) is SeparatedLinear,'STABLE_RESPONSE_MISSING')
        module.checked()
        values[name] = dict(anchor=module.anchor,projector=module.projector)
    return digest(tensor_fingerprint(values))
