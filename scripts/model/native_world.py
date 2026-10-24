"""Plastic typed native evidence alongside the retained frozen world geometry."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from claims.codata_source import check
from model.attention import Linear
from model.constellation_world import ConstellationWorld
from model.episode_input import EpisodeBatch
from model.window_maxsim import maxsim

RULE = 'typed_native_residual_world_v1'
BRANCH = '_native'


class NativeEvidence(nn.Module):
    def __init__(self, dims, width):
        super().__init__()
        self.down = Linear(dims, width)
        self.context = Linear(width, width)
        self.up = Linear(width, dims, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, values, context):
        return self.up(torch.tanh(self.down(F.normalize(values, dim=-1))+self.context(context)))


class NativeWorld(ConstellationWorld):
    def __init__(self, core, references, *, world_source_id):
        super().__init__(core, references, world_source_id=world_source_id)
        h = self.config.width
        extra = sum(2*s.dims*h+h*h+2*h for s in self.specs)
        expected = sum(p.numel() for p in self.parameters())+extra
        check(expected <= self.config.max_parameters, 'NATIVE_WORLD_PARAMETER_BUDGET')
        self.query[BRANCH] = nn.ModuleDict({s.name:NativeEvidence(s.dims,h) for s in self.specs})
        check(sum(p.numel() for p in self.parameters()) == expected, 'NATIVE_WORLD_PARAMETER_COUNT')

    @classmethod
    def from_world(cls, model):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261021)
            result = cls(model.core, model.references(), world_source_id=model.world_source_id)
        result.to(model.core.initial.device)
        missing, unexpected = result.load_state_dict(model.state_dict(), strict=False)
        check(not unexpected and all(n.startswith('query.'+BRANCH+'.') for n in missing),
              'NATIVE_WORLD_MIGRATION')
        check(isinstance(model, cls) == (not missing), 'NATIVE_WORLD_MIGRATION')
        for name in ('context_history', 'execution_anchor_bank'):
            if hasattr(model, name):
                setattr(result, name, getattr(model, name))
        result.train(model.training)
        result.check_reference()
        return result

    def forward(self, native, states, **kwargs):
        old = super().forward(native, states, **kwargs)
        allowed = kwargs['allowed']
        visible = allowed.nonzero().flatten()
        scores, residuals, support = {}, {}, None
        for i,spec in enumerate(self.specs):
            context = F.normalize(old['placement_hidden'][:,i],dim=-1)[:,None,:]
            keys = F.normalize(getattr(self,f'world_values_{i}')[visible],dim=-1)
            padding = getattr(self,f'world_padding_{i}')[visible]
            head = self.query[BRANCH][spec.name]
            observation = native[spec.name]
            def evidence(values, state, reference, *, _head=head, _padding=padding):
                return maxsim(_head(values.to(state.device),state),reference,_padding,self.config)
            if isinstance(native, EpisodeBatch):
                residual, count = None, 0
                for block in observation.blocks(self.config):
                    check(block.offset == count and not bool(block.padding.any()), 'NATIVE_WORLD_COVERAGE')
                    part = (checkpoint(evidence,block.values,context,keys,use_reentrant=False,
                                       preserve_rng_state=False) if torch.is_grad_enabled()
                            else evidence(block.values,context,keys)).sum(1)
                    residual = part if residual is None else residual+part
                    count += block.values.shape[1]
                check(count == observation.count and count > 0, 'NATIVE_WORLD_COVERAGE')
                residual = residual/count
            else:
                matches = evidence(observation.values,context,keys)
                residual = matches.masked_fill(observation.padding[:,:,None],0.).sum(1)
                residual = residual/(~observation.padding).sum(1)[:,None]
            check(bool(torch.isfinite(residual).all()), 'NATIVE_WORLD_NONFINITE')
            full = old['regions'][spec.name].clone()
            full[:,visible] = full[:,visible]+residual
            scores[spec.name], residuals[spec.name] = full, residual
            lane = F.log_softmax(full[:,visible],dim=-1)
            support = lane if support is None else support+lane
        common = torch.full_like(old['constellation_regions'],-torch.inf)
        common[:,visible] = support
        selected = visible[support.argmax(-1)]
        constellation = self._placement(old['placement_hidden'],selected,kwargs['event_ids'])
        return dict(old, regions=scores, constellation_regions=common,
            slots={n:s.values for n,s in constellation.slots.items()},
            predicted_echo={n:s.values for n,s in constellation.slots.items()},
            predicted_constellation=constellation, recognition=RULE,
            native_residuals=residuals, world_owner={s.name:selected for s in self.specs})
