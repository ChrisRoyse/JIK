"""Keep the complete parent response fixed while learning a separate residual."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.loop_training_checkpoint import tensor_fingerprint
from model.response_basis import modules
from model.stable_response import SeparatedLinear

RULE = 'frozen_parent_response_plus_successor_residual_v1'
STATIC = ('anchor','projector','parent_weight','successor_projector')


def code_pin():
    return sha(Path(__file__).read_bytes())


class SuccessorLinear(nn.Module):
    def __init__(self,linear,projector):
        super().__init__()
        check(type(linear) is SeparatedLinear,'SUCCESSOR_RESIDUAL_PARENT_TYPE')
        linear.checked()
        self.weight = linear.weight
        for name in ('anchor','projector'):
            self.register_buffer(name,getattr(linear,name).detach().clone())
        self.register_buffer('parent_weight',linear.weight.detach().clone())
        self.register_buffer('successor_projector',projector.detach().to(linear.weight.device).clone())
        check(self.successor_projector.dtype == torch.float64 and
              self.successor_projector.shape == self.projector.shape,'SUCCESSOR_RESIDUAL_PROJECTOR')
        self.seal()

    def seal(self):
        self._versions = tuple((id(v),v.data_ptr(),v._version) for v in
                               (getattr(self,n) for n in STATIC))

    def checked(self):
        check(tuple((id(v),v.data_ptr(),v._version) for v in
                    (getattr(self,n) for n in STATIC)) == self._versions,
              'SUCCESSOR_RESIDUAL_FROZEN_BUFFER')

    def forward(self,values):
        self.checked()
        check(values.dtype == torch.float32 and values.device == self.weight.device,
              'SUCCESSOR_RESIDUAL_INPUT_RUNTIME')
        flat = values.contiguous().reshape(-1,self.weight.shape[1])
        anchor = F.linear(flat,self.anchor)
        previous = F.linear(flat.double()@self.projector,
                            self.parent_weight.double()-self.anchor.double())
        parent = anchor+previous.to(anchor.dtype)
        residual = F.linear(flat.double()@self.successor_projector,
                            self.weight.double()-self.parent_weight.double())
        return (parent+residual.to(parent.dtype)).reshape(*values.shape[:-1],self.weight.shape[0])


def install(model,basis):
    original = modules(model)
    check(list(original) == [r['name'] for r in basis.manifest['branches']],
          'SUCCESSOR_RESIDUAL_ROSTER')
    for i,(name,linear) in enumerate(original.items()):
        _,branch,instrument,_ = name.split('.')
        model.query[branch][instrument].up = SuccessorLinear(linear,torch.from_numpy(basis.arrays[f'p_{i}']))
    return buffers(model)


def buffers(model):
    values = {}
    for name,module in modules(model).items():
        check(type(module) is SuccessorLinear,'SUCCESSOR_RESIDUAL_MISSING')
        module.checked()
        values[name] = {n:getattr(module,n) for n in STATIC}
    return digest(tensor_fingerprint(values))
