"""Compare packed banks with the retained dense path on real bounded bank prefixes."""
import copy
from dataclasses import replace

import torch

from claims.codata_source import check
from claims.episodic_memory import digest
from model.constellation_learning import identity_pin
from model.native import NativeInput
from model.repository_packed_world import PackedRepositoryWorld
from model.repository_relation_world import RepositoryWorld
from model.tool_context_regions import attach
from senses.contract import SenseError


def dense_view(data,model,stage):
    representatives=[next(data.examples[k] for k in stage['keys'] if data.examples[k]['region']==r)
                     for r in stage['regions']]
    elements=sum(len(representatives)*max(e['target'][s.name].values.shape[1] for e in representatives)*s.dims
                 for s in data.specs)
    if elements>model.config.max_native_elements:return None
    refs={}
    for s in data.specs:
        width=max(e['target'][s.name].values.shape[1] for e in representatives)
        values=torch.zeros((len(representatives),width,s.dims),device=data.device)
        padding=torch.ones(values.shape[:2],dtype=torch.bool,device=data.device)
        for index,e in enumerate(representatives):
            raw=e['target'][s.name].values[0,~e['target'][s.name].padding[0]]
            values[index,:len(raw)]=raw;padding[index,:len(raw)]=False
        refs[s.name]=NativeInput(s.identity,tuple(stage['regions']),values,padding,
                                torch.zeros(len(representatives),dtype=torch.bool,device=data.device))
    return attach(RepositoryWorld.from_world(model,references=refs,world_source_id=digest(stage['description'])),model)


def audit(data,model):
    comparisons=[];gradient_checks=[]
    def run(view,entry,allowed,intervention):
        batch=entry['baseline'] if intervention else entry['batch']
        return view(batch,(view.start('repository:'+entry['task']),),allowed=allowed,phase=0,
                    event_ids=[batch.pin],evidence_origin=['source_observation'],advance_phase=False)
    for group in data.order:
        stage=data.stage(group,model);packed=attach(stage['view'],model)
        old=dense_view(data,model,stage)
        if old is None:continue
        key=max(stage['keys'],key=lambda k:sum(v.values.numel() for v in data.examples[k]['batch'].values()))
        entry=data.examples[key]
        for mode in ('full','withheld','restricted'):
            allowed=stage['allowed'].clone()
            if mode=='restricted':allowed[1::2]=False
            with torch.no_grad():
                a=run(old,entry,allowed,mode=='withheld');b=run(packed,entry,allowed,mode=='withheld')
            check(identity_pin(a)==identity_pin(b),'PACKED_PARITY_IDENTITY')
            check(a['predicted_constellation'].regions==b['predicted_constellation'].regions,'PACKED_PARITY_REGION')
            maxdiff=0.
            for name in (*a['regions'],'common'):
                x=a['constellation_regions'] if name=='common' else a['regions'][name]
                y=b['constellation_regions'] if name=='common' else b['regions'][name]
                check(torch.equal(torch.isfinite(x),torch.isfinite(y)),'PACKED_PARITY_ALLOWED')
                diff=float((x[:,allowed]-y[:,allowed]).abs().max());maxdiff=max(maxdiff,diff)
                check(diff<=1e-5,'PACKED_PARITY_SCORE')
            for s in data.specs:
                x=a['predicted_constellation'].slots[s.name];y=b['predicted_constellation'].slots[s.name]
                check(torch.equal(x.padding,y.padding) and torch.allclose(x.values,y.values,rtol=0,atol=1e-6),
                      'PACKED_PARITY_PLACEMENT')
            comparisons.append(dict(group=group,key=key,mode=mode,max_score_absolute_difference=maxdiff))
        parameters=[p for p in model.parameters() if p.requires_grad]
        try:
            stage['view']=old;left=data.losses(model,key,group)
            ga=torch.autograd.grad(left,parameters)
            stage['view']=packed;right=data.losses(model,key,group)
            gb=torch.autograd.grad(right,parameters)
        finally:stage['view']=packed
        check(abs(float((left-right).detach()))<=1e-5,'PACKED_PARITY_LOSS')
        check(all(torch.allclose(x,y,rtol=1e-4,atol=1e-6) for x,y in zip(ga,gb,strict=True)),
              'PACKED_PARITY_GRADIENT')
        gradient_checks.append(dict(group=group,loss_absolute_difference=abs(float((left-right).detach())),
                                   gradient_max_absolute_difference=max(float((x-y).abs().max()) for x,y in zip(ga,gb,strict=True))))
        del old
    check(len(comparisons)==18 and len(gradient_checks)==6,'PACKED_PARITY_COVERAGE')
    controls=[]
    def refuse(name,code,fn):
        try:fn()
        except (ValueError,SenseError) as error:check(code in str(error),'PACKED_PARITY_CONTROL:'+str(error))
        else:raise AssertionError('PACKED_PARITY_NEGATIVE_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    first=data.stage(data.order[0],model);view=first['view'];entry=data.examples[first['keys'][0]]
    broken=copy.deepcopy(view);broken.bank_seal=broken.stamps()
    getattr(broken,'values_0_0')[0,0,0]+=1
    refuse('changed_bank','PACKED_BANK_CHANGED',broken.check_reference)
    refuse('empty_allowed','CONSTELLATION_ALLOWED',lambda:run(view,entry,torch.zeros_like(first['allowed']),False))
    representatives=[next(data.examples[k] for k in first['keys'] if data.examples[k]['region']==r) for r in first['regions']]
    duplicate=list(first['regions']);duplicate[-1]=duplicate[0]
    refuse('duplicate_region','PACKED_BANK_ROSTER',lambda:PackedRepositoryWorld(model,duplicate,representatives,world_source_id='0'*64))
    smaller=copy.copy(model);smaller.config=replace(model.config,max_native_elements=view.elements-1)
    refuse('bank_over_budget','PACKED_BANK_BUDGET',lambda:PackedRepositoryWorld(smaller,first['regions'],representatives,world_source_id='0'*64))
    return dict(comparisons=comparisons,gradients=gradient_checks,controls=controls,
                score_atol=1e-5,gradient_atol=1e-6,gradient_rtol=1e-4,
                scope='floating arithmetic parity only; exact retention gate has zero tolerance')
