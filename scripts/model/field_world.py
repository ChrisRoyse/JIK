"""Typed native relations among attested action, mechanism and available resources."""
import torch
from torch import nn
from torch.nn import functional as F

from claims.codata_source import check
from model.attention import Linear
from model.field_native import FieldBatch, ROLES
from model.native_world import NativeWorld
from model.window_maxsim import maxsim

RULE = 'typed_attested_field_interaction_v2'
BRANCH = '_fields'


class FieldEvidence(nn.Module):
    def __init__(self, dims, width):
        super().__init__()
        self.roles = nn.ModuleDict({r:Linear(dims,width) for r in ROLES})
        self.context = Linear(width,width)
        self.up = Linear(width,dims,bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, values, context):
        a,p,r = [self.roles[n](F.normalize(values[n],dim=-1)) for n in ROLES]
        # Every actual window triple stays separate until scalar reference scoring.
        pairs = p[:,:,None,:]*r[:,None,:,:]
        hidden = torch.tanh(a[:,:,None,None,:]+pairs[:,None,:,:,:]
                            +self.context(context)[:,None,None,None,:])
        return self.up(hidden.flatten(1,3))


class FieldWorld(NativeWorld):
    def __init__(self, core, references, *, world_source_id):
        super().__init__(core,references,world_source_id=world_source_id)
        h = self.config.width
        expected = sum(p.numel() for p in self.parameters())+sum(4*s.dims*h+h*h+4*h for s in self.specs)
        check(expected <= self.config.max_parameters,'FIELD_WORLD_PARAMETER_BUDGET')
        self.query[BRANCH] = nn.ModuleDict({s.name:FieldEvidence(s.dims,h) for s in self.specs})
        check(sum(p.numel() for p in self.parameters()) == expected,'FIELD_WORLD_PARAMETER_COUNT')

    @classmethod
    def from_world(cls, model):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261021)
            result = cls(model.core,model.references(),world_source_id=model.world_source_id)
        result.to(model.core.initial.device)
        missing,unexpected = result.load_state_dict(model.state_dict(),strict=False)
        check(not unexpected and all(n.startswith('query.'+BRANCH+'.') for n in missing),
              'FIELD_WORLD_MIGRATION')
        check(isinstance(model,cls) == (not missing),'FIELD_WORLD_MIGRATION')
        for name in ('context_history','execution_anchor_bank'):
            if hasattr(model,name):
                setattr(result,name,getattr(model,name))
        result.train(model.training)
        result.check_reference()
        return result

    def forward(self,native,states,**kwargs):
        check(isinstance(native,FieldBatch),'FIELD_NATIVE_REQUIRED')
        native.checked(self.specs)
        old = super().forward(native,states,**kwargs)
        visible = kwargs['allowed'].nonzero().flatten()
        scores,residuals,support = {},{},None
        for i,spec in enumerate(self.specs):
            values = {r:native.role(spec.name,r) for r in ROLES}
            count = 1
            for value in values.values():
                count *= value.shape[1]
            check(count <= self.config.max_native_tokens and count*spec.dims <= self.config.max_native_elements,
                  'FIELD_WORLD_INTERACTION_BUDGET')
            context = F.normalize(old['placement_hidden'][:,i],dim=-1)
            query = self.query[BRANCH][spec.name](values,context)
            keys = F.normalize(getattr(self,f'world_values_{i}')[visible],dim=-1)
            padding = getattr(self,f'world_padding_{i}')[visible]
            # Anchor the zero-delta residual at each actual program reading. A
            # zero query would tie every reference window and give an arbitrary
            # first-step MaxSim subgradient. Keep every window triple separate.
            a,p,r = [values[n].shape[1] for n in ROLES]
            base = F.normalize(values['program'],dim=-1)[:,None,:,None,:]
            base = base.expand(-1,a,-1,r,-1).reshape(query.shape)
            residual = (maxsim(base+query,keys,padding,self.config)
                        - maxsim(base,keys,padding,self.config)).mean(1)
            check(bool(torch.isfinite(residual).all()),'FIELD_WORLD_NONFINITE')
            full = old['regions'][spec.name].clone()
            full[:,visible] = full[:,visible]+residual
            scores[spec.name],residuals[spec.name] = full,residual
            lane = F.log_softmax(full[:,visible],dim=-1)
            support = lane if support is None else support+lane
        common = torch.full_like(old['constellation_regions'],-torch.inf)
        common[:,visible] = support
        selected = visible[support.argmax(-1)]
        constellation = self._placement(old['placement_hidden'],selected,kwargs['event_ids'])
        return dict(old,regions=scores,constellation_regions=common,
            slots={n:s.values for n,s in constellation.slots.items()},
            predicted_echo={n:s.values for n,s in constellation.slots.items()},
            predicted_constellation=constellation,recognition=RULE,field_residuals=residuals,
            world_owner={s.name:selected for s in self.specs})
