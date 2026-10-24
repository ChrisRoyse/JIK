"""Per-instrument patch/source relations feed the existing common-region predictor."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from claims.codata_source import check
from model.attention import Linear
from model.constellation_world import ConstellationWorld
from model.field_world import FieldWorld
from model.native_world import NativeWorld
from model.repository_relation_input import RepositoryBatch, ROLES
from model.window_maxsim import maxsim

BRANCH = '_repository'
RULE = 'typed_repository_window_relations_v1'


class RepositoryEvidence(nn.Module):
    def __init__(self, dims, width):
        super().__init__()
        self.roles = nn.ModuleDict({r:Linear(dims, width) for r in ROLES})
        self.context = Linear(width, width)
        self.up = Linear(width, dims, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, values, context):
        action = self.roles['recorded_patch'](F.normalize(values['recorded_patch'], dim=-1))
        queries, bases = [], []
        for role in ROLES[:3]:
            raw = F.normalize(values[role], dim=-1)
            other = self.roles[role](raw)
            hidden = torch.tanh(action[:,:,None,:]+other[:,None,:,:]
                                +self.context(context)[:,None,None,:])
            queries.append(self.up(hidden.flatten(1,2)))
            bases.append(raw[:,None,:,:].expand(-1,action.shape[1],-1,-1).flatten(1,2))
        # Sequence concatenation within one instrument, never vector flattening across lenses.
        return torch.cat(queries, dim=1), torch.cat(bases, dim=1)


class RepositoryWorld(FieldWorld):
    @classmethod
    def from_world(cls, parent, *, references=None, world_source_id=None):
        check(type(parent) in (ConstellationWorld, NativeWorld, FieldWorld, cls), 'RELATION_PARENT_CLASS')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261021)
            result = cls(copy.deepcopy(parent.core), references or parent.references(),
                         world_source_id=world_source_id or parent.world_source_id)
            result.query = copy.deepcopy(parent.query)
            result.place = copy.deepcopy(parent.place)
            if BRANCH not in result.query:
                result.query[BRANCH] = nn.ModuleDict({s.name:RepositoryEvidence(s.dims, result.config.width)
                                                     for s in result.specs})
        result.legacy = parent.legacy if isinstance(parent, cls) else type(parent).__name__
        check(sum(p.numel() for p in result.parameters()) <= result.config.max_parameters,
              'RELATION_PARAMETER_BUDGET')
        result.to(parent.core.initial.device)
        reseal(result)
        for name in ('context_history', 'execution_anchor_bank'):
            if hasattr(parent, name):
                setattr(result, name, copy.deepcopy(getattr(parent, name)))
        result.train(parent.training)
        result.check_reference()
        return result

    def forward(self, native, states, **kwargs):
        legacy = {'ConstellationWorld':ConstellationWorld, 'NativeWorld':NativeWorld, 'FieldWorld':FieldWorld}
        if not isinstance(native, RepositoryBatch):
            return legacy[self.legacy].forward(self, native, states, **kwargs)
        native.checked(self.specs)
        base_class = NativeWorld if '_native' in self.query else ConstellationWorld
        old = base_class.forward(self, native, states, **kwargs)
        if native.description['intervention'] is not None:
            return dict(old, recognition='repository_action_withheld_baseline')
        visible = kwargs['allowed'].nonzero().flatten()
        scores, residuals, support = {}, {}, None
        for i,spec in enumerate(self.specs):
            values = {r:native.role(spec.name, r) for r in ROLES}
            count = values['recorded_patch'].shape[1]*sum(values[r].shape[1] for r in ROLES[:3])
            check(count <= self.config.max_native_tokens and count*spec.dims <= self.config.max_native_elements,
                  'RELATION_INTERACTION_BUDGET')
            context = F.normalize(old['placement_hidden'][:,i], dim=-1)
            query, base = self.query[BRANCH][spec.name](values, context)
            check(query.shape == base.shape and query.shape[1] == count, 'RELATION_INTERACTION_COVERAGE')
            keys = F.normalize(getattr(self, f'world_values_{i}')[visible], dim=-1)
            padding = getattr(self, f'world_padding_{i}')[visible]
            residual = (maxsim(base+query, keys, padding, self.config)
                        -maxsim(base, keys, padding, self.config)).mean(1)
            check(bool(torch.isfinite(residual).all()), 'RELATION_NONFINITE')
            full = old['regions'][spec.name].clone()
            full[:,visible] = full[:,visible]+residual
            scores[spec.name], residuals[spec.name] = full, residual
            lane = F.log_softmax(full[:,visible], dim=-1)
            support = lane if support is None else support+lane
        common = torch.full_like(old['constellation_regions'], -torch.inf)
        common[:,visible] = support
        selected = visible[support.argmax(-1)]
        constellation = self._placement(old['placement_hidden'], selected, kwargs['event_ids'])
        return dict(old, regions=scores, constellation_regions=common,
            slots={n:s.values for n,s in constellation.slots.items()},
            predicted_echo={n:s.values for n,s in constellation.slots.items()},
            predicted_constellation=constellation, recognition=RULE,
            repository_residuals=residuals, world_owner={s.name:selected for s in self.specs})


def reseal(model):
    for module in model.query.modules():
        if hasattr(module, 'seal'):
            module.seal()


def configure(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('query.'+BRANCH+'.'))
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
