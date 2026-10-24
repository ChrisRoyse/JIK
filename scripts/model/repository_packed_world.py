"""Length-bucketed native banks; global region scores, unchanged typed windows."""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation import Constellation
from model.constellation_history import interpret
from model.episode_input import validate_observation
from model.identity_loop import reference_hash
from model.native import NativeInput
from model.repository_relation_input import RepositoryBatch,ROLES
from model.repository_relation_world import RepositoryWorld
from model.window_maxsim import maxsim
from model.world_loop import WorldState


class PackedRepositoryWorld(RepositoryWorld):
    def __init__(self,model,regions,representatives,*,world_source_id):
        nn.Module.__init__(self)
        self.core,self.query,self.place=model.core,model.query,model.place
        self.config,self.specs=model.config,model.specs
        self.standard_id,self.source_id=model.standard_id,model.source_id
        self.world_source_id=world_source_id
        self.reference_rows={s.name:tuple(regions) for s in self.specs}
        check(0<len(regions)<=709 and len(set(regions))==len(regions)==len(representatives),
              'PACKED_BANK_ROSTER')
        groups=defaultdict(list)
        for index,e in enumerate(representatives):
            lengths=tuple(int((~e['target'][s.name].padding[0]).sum()) for s in self.specs)
            check(all(n>0 for n in lengths),'PACKED_BANK_EMPTY')
            groups[tuple((n-1).bit_length() for n in lengths)].append(index)
        self.groups=list(groups.values());self.locations={}
        planned=sum(len(indices)*max(int((~representatives[i]['target'][s.name].padding[0]).sum())
                                     for i in indices)*s.dims for indices in self.groups for s in self.specs)
        check(planned<=self.config.max_native_elements,'PACKED_BANK_BUDGET')
        self.scratch_elements=max(len(regions)*max(int((~e['target'][s.name].padding[0]).sum())
                                  for e in representatives)*s.dims for s in self.specs)
        check(self.scratch_elements<=self.config.max_native_elements,'PACKED_LANE_BUDGET')
        total=0
        for group,indices in enumerate(self.groups):
            for offset,index in enumerate(indices):self.locations[index]=(group,offset)
            self.register_buffer(f'indices_{group}',torch.tensor(indices,device=model.core.initial.device))
            for lane,s in enumerate(self.specs):
                rows=[representatives[i]['target'][s.name] for i in indices]
                values=[r.values[0,~r.padding[0]] for r in rows]
                width=max(len(v) for v in values)
                bank=torch.stack([F.pad(v,(0,0,0,width-len(v))) for v in values]).detach()
                padding=torch.stack([torch.arange(width,device=bank.device)>=len(v) for v in values])
                total+=bank.numel()
                self.register_buffer(f'values_{group}_{lane}',bank)
                self.register_buffer(f'padding_{group}_{lane}',padding)
        check(total==planned,'PACKED_BANK_ACCOUNTING')
        self.elements=total;self.world_id=self.reference_digest();self.bank_seal=self.stamps()
        self.training=model.training
        for name in ('context_history','execution_anchor_bank'):
            if hasattr(model,name):setattr(self,name,getattr(model,name))
        self.check_reference()

    def stamps(self):
        return {n:(id(t),t.data_ptr(),t._version) for n,t in self.named_buffers(recurse=False)}

    def reference_digest(self):
        hashes=[]
        for group,indices in enumerate(self.groups):
            check(torch.equal(getattr(self,f'indices_{group}'),torch.tensor(indices,device=self.core.initial.device)),
                  'PACKED_BANK_INDICES')
            refs={}
            for lane,s in enumerate(self.specs):
                values=getattr(self,f'values_{group}_{lane}');padding=getattr(self,f'padding_{group}_{lane}')
                rows=tuple(self.reference_rows[s.name][i] for i in indices)
                refs[s.name]=NativeInput(s.identity,rows,values,padding,
                                        torch.zeros(len(rows),dtype=torch.bool,device=values.device))
                refs[s.name].check(s,self.config)
            hashes.append(reference_hash(self.specs,refs,self.world_source_id))
        return digest(dict(source=self.world_source_id,rows=self.reference_rows,groups=self.groups,hashes=hashes))

    def check_reference(self):
        self.core.check_reference()
        check(self.stamps()==self.bank_seal and self.reference_digest()==self.world_id,'PACKED_BANK_CHANGED')
        check(self.locations=={i:(g,o) for g,indices in enumerate(self.groups) for o,i in enumerate(indices)},
              'PACKED_BANK_LOCATIONS')

    def dense_lane(self,lane,visible):
        """One bounded scratch lane preserves the original GEMM/MaxSim tie arithmetic."""
        width=max(getattr(self,f'values_{g}_{lane}').shape[1] for g in range(len(self.groups)))
        check(len(visible)*width*self.specs[lane].dims<=self.config.max_native_elements,'PACKED_LANE_BUDGET')
        values=torch.zeros((len(visible),width,self.specs[lane].dims),device=self.core.initial.device)
        padding=torch.ones(values.shape[:2],dtype=torch.bool,device=values.device)
        lookup=torch.full((len(self.locations),),-1,dtype=torch.long,device=values.device)
        lookup[visible]=torch.arange(len(visible),device=values.device)
        for group in range(len(self.groups)):
            positions=lookup[getattr(self,f'indices_{group}')];keep=positions>=0
            if not bool(keep.any()):continue
            source=getattr(self,f'values_{group}_{lane}');mask=getattr(self,f'padding_{group}_{lane}')
            values[positions[keep],:source.shape[1]]=source[keep]
            padding[positions[keep],:source.shape[1]]=mask[keep]
        check(not bool(padding.all(1).any()),'PACKED_LANE_COVERAGE')
        return values,padding

    def forward(self,native,states,*,allowed,phase,event_ids,evidence_origin,history=None,advance_phase=True):
        check(isinstance(native,RepositoryBatch),'PACKED_BANK_INPUT')
        native.checked(self.specs);validate_observation(native,self.specs,self.config)
        check(self.stamps()==self.bank_seal,'PACKED_BANK_CHANGED')
        check(all(isinstance(s,WorldState) and s.world_id==self.world_id for s in states),'CONSTELLATION_CONTEXT')
        rows=next(iter(self.reference_rows.values()))
        check(allowed.dtype==torch.bool and allowed.shape==(len(rows),) and bool(allowed.any())
              and allowed.device==self.core.initial.device,'CONSTELLATION_ALLOWED')
        mode={} if advance_phase is True else dict(advance_phase=advance_phase)
        output=self.core(native,states,phase=phase,event_ids=event_ids,evidence_origin=evidence_origin,**mode)
        hidden=interpret(self,torch.stack([s.hidden for s in output['states']]),states,history)
        scores={};visible=allowed.nonzero().flatten();support=None
        for lane,s in enumerate(self.specs):
            obs=native[s.name];context=F.normalize(hidden[:,lane],dim=-1)[:,None,:]
            observed=F.normalize(self.core.projectors[s.name](obs.values),dim=-1)
            query=F.normalize(self.query[s.name](context+observed),dim=-1)
            residual=self.query['_native'][s.name](obs.values,context) if '_native' in self.query else None
            action=base=None
            if native.description['intervention'] is None:
                roles={r:native.role(s.name,r) for r in ROLES}
                count=roles['recorded_patch'].shape[1]*sum(roles[r].shape[1] for r in ROLES[:3])
                check(count<=self.config.max_native_tokens and count*s.dims<=self.config.max_native_elements,
                      'RELATION_INTERACTION_BUDGET')
                action,base=self.query['_repository'][s.name](roles,context[:,0])
                check(action.shape==base.shape and action.shape[1]==count,'RELATION_INTERACTION_COVERAGE')
            # Bind closure defaults: backward runs after the outer lane loop finishes.
            # Recompute scratch keys in backward instead of retaining six padded lanes.
            def score_lane(query,residual,action,base,*,_lane=lane,_spec=s,_visible=visible,
                           _padding=obs.padding,_count=(~obs.padding).sum(1)):
                raw,padding=self.dense_lane(_lane,_visible)
                projected=F.normalize(self.core.projectors[_spec.name](raw),dim=-1)
                matches=maxsim(query,projected,padding,self.config)
                score=matches.masked_fill(_padding[:,:,None],0.).sum(1)/_count[:,None]
                keys=F.normalize(raw,dim=-1)
                if residual is not None:
                    matches=maxsim(residual,keys,padding,self.config)
                    score=score+matches.masked_fill(_padding[:,:,None],0.).sum(1)/_count[:,None]
                if action is not None:
                    score=score+(maxsim(base+action,keys,padding,self.config)-maxsim(base,keys,padding,self.config)).mean(1)
                return score
            score=(checkpoint(score_lane,query,residual,action,base,use_reentrant=False,preserve_rng_state=False)
                   if torch.is_grad_enabled() and action is not None and action.requires_grad
                   else score_lane(query,residual,action,base))
            full=torch.full((len(states),len(rows)),-torch.inf,device=hidden.device)
            full[:,visible]=score
            check(bool(torch.isfinite(full[:,visible]).all()),'PACKED_BANK_NONFINITE')
            scores[s.name]=full
            term=F.log_softmax(full[:,visible],dim=-1)
            support=term if support is None else support+term
        selected=visible[support.argmax(-1)];common=torch.full_like(next(iter(scores.values())),-torch.inf)
        common[:,visible]=support
        constellation=self._placement(hidden,selected,event_ids)
        return dict(output,states=tuple(self.bind_state(s) for s in output['states']),
            identity_regions=output['regions'],identity_slots=output['slots'],regions=scores,
            constellation_regions=common,slots={n:s.values for n,s in constellation.slots.items()},
            predicted_echo={n:s.values for n,s in constellation.slots.items()},predicted_constellation=constellation,
            recognition='typed_repository_length_buckets_v1',placement_hidden=hidden,
            world_owner={s.name:selected for s in self.specs},world_bank=self.world_id)

    def _placement(self,hidden,selected,event_ids):
        rows=next(iter(self.reference_rows.values()));slots={}
        for lane,s in enumerate(self.specs):
            values=[]
            for index in selected.tolist():
                group,offset=self.locations[index]
                raw=getattr(self,f'values_{group}_{lane}')[offset]
                mask=getattr(self,f'padding_{group}_{lane}')[offset]
                values.append(raw[~mask])
            width=max(len(v) for v in values)
            book=torch.stack([F.pad(v,(0,0,0,width-len(v))) for v in values])
            padding=torch.stack([torch.arange(width,device=book.device)>=len(v) for v in values])
            base=F.normalize(book,dim=-1)
            state=hidden[:,lane,None,:]+self.core.projectors[s.name](book)
            residual=torch.tanh(self.place[s.name](state))
            residual=.1*residual/torch.linalg.vector_norm(residual,dim=-1,keepdim=True).clamp_min(1.)
            predicted=F.normalize(base+residual,dim=-1).masked_fill(padding[:,:,None],0.)
            slots[s.name]=NativeInput(s.identity,tuple('prediction:'+e for e in event_ids),predicted,padding,
                                     torch.zeros(len(selected),dtype=torch.bool,device=book.device))
        return Constellation(slots,'predicted',tuple(rows[i] for i in selected.tolist())).validate(self.specs,self.config)
