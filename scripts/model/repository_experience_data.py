"""Require every supported repository intake in the connected learning history."""
import json
from pathlib import Path
import torch

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.repository_cohort_input import CohortNative
from model.repository_depth_input import DepthNative
from model.repository_growth_data import GrowthData
from model.repository_relation_data import RelationData,distances
from model.repository_relation_input import bind,withhold_action
from model.repository_packed_world import PackedRepositoryWorld
from senses.trajectory_population import read_pinned

KINDS = {'repository_cohort_intake':('repository_cohort_intakes',CohortNative),
         'repository_depth_intake':('repository_depth_intakes',DepthNative)}


def code_pins():
    root=Path(__file__).parent
    pins={'model/'+n:sha((root/n).read_bytes()) for n in
          ('repository_experience_data.py','repository_growth_data.py','repository_packed_world.py')}
    pins['audit/repository_packed_readback.py']=sha((root.parent/'audit/repository_packed_readback.py').read_bytes())
    return pins


def roster(session,populations):
    configured={digest(p['native']):p for p in populations}
    check(len(configured)==len(populations)>0,'EXPERIENCE_DATA_DUPLICATE_POPULATION')
    check(all(p['kind'] in KINDS for p in populations),'EXPERIENCE_DATA_KIND')
    check(all(hasattr(session,v[0]) for v in KINDS.values()),'EXPERIENCE_DATA_STATE')
    groups,seen,prefixes=[],set(),set()
    for seq,raw in session.journal.db.execute('SELECT seq,body FROM events ORDER BY seq'):
        body=json.loads(raw);kind=body.get('kind')
        if kind not in KINDS:continue
        spec=body['spec'];key=digest(spec);saved=getattr(session,KINDS[kind][0])
        check(key in saved and saved[key]['event']==body and key not in seen,
              'EXPERIENCE_DATA_JOURNAL_JOIN')
        native={k:v for k,v in spec.items() if k!='partition'};population=digest(native)
        check(population in configured,'EXPERIENCE_DATA_MISSING_POPULATION')
        check(configured[population]['kind']==kind,'EXPERIENCE_DATA_KIND')
        check(spec['partition'] in ('learn_1','learn_2'),'EXPERIENCE_DATA_PARTITION')
        check(spec['partition']=='learn_1' or population in prefixes,'EXPERIENCE_DATA_PREFIX')
        if spec['partition']=='learn_1':prefixes.add(population)
        groups.append(dict(id='intake-'+str(seq),seq=seq,key=key,population=population,spec=spec,kind=kind))
        seen.add(key)
    saved_keys=[k for field,_ in KINDS.values() for k in getattr(session,field)]
    check(len(saved_keys)==len(set(saved_keys)) and seen==set(saved_keys) and
          {g['population'] for g in groups}==set(configured),'EXPERIENCE_DATA_COMPLETE_ROSTER')
    return groups,configured


class ExperienceData(GrowthData):
    def __init__(self,session,populations):
        self.session,self.parent=session,session.model
        self.specs,self.device=self.parent.specs,self.parent.core.initial.device
        self.groups,configured=roster(session,populations)
        self.providers,self.examples,self.stages={},{},{}
        self.order=[g['id'] for g in self.groups]
        proofs={k:read_pinned(Path(v['proof']),v['proof_pin']) for k,v in configured.items()}
        for key,p in configured.items():
            check(proofs[key]['native']==p['native'],'EXPERIENCE_DATA_NATIVE_PROOF')
            receipts={}
            for group in self.groups:
                if group['population']!=key:continue
                saved=getattr(session,KINDS[group['kind']][0])[group['key']]['bindings']
                check(not receipts.keys() & saved.keys(),'EXPERIENCE_DATA_DUPLICATE_CONDITION')
                receipts.update({k:{s:v[s] for s in ('input','echo')} for k,v in saved.items()})
            check(receipts==proofs[key]['receipts'],'EXPERIENCE_DATA_RECEIPT_ROSTER')
        for group in self.groups:
            provider=KINDS[group['kind']][1](**group['spec'])
            self.providers[group['id']]=provider
            saved=getattr(session,KINDS[group['kind']][0])[group['key']]['bindings']
            expected={k for k,v in provider.population.fields.items()
                      if v['partition']==group['spec']['partition']}
            check(expected==set(saved) and expected,'EXPERIENCE_DATA_PARTITION_ROSTER')
            for key in sorted(saved):
                check(key not in self.examples,'EXPERIENCE_DATA_DUPLICATE_CONDITION')
                source=provider.population.fields[key]
                episode=provider.population.observations.episode(key)
                check(saved[key]['episode']==episode['id'] and
                      session.memory.episodes[episode['id']]==episode and
                      session.memory.state[episode['id']]['active'],'EXPERIENCE_DATA_ACTIVE_EPISODE')
                batch=bind(provider,key,self.parent,saved[key]['input'])
                target,desc=provider.observed(key,'echo',source['echo_available_after'],
                                              self.parent.config,self.device,self.specs)
                check(desc==saved[key]['echo']['description'],'EXPERIENCE_DATA_TARGET_DESCRIPTION')
                for s in self.specs:
                    slot=target.slots[s.name]
                    a=slot.values[0,~slot.padding[0]].detach().cpu().numpy().astype('<f4')
                    check(dict(instrument=s.identity,shape=list(a.shape),sha256=sha(a.tobytes()),
                        dtype='float32-le',absent=bool(slot.absent[0]))==saved[key]['echo']['slots'][s.name],
                        'EXPERIENCE_DATA_TARGET_NATIVE')
                absent=all(bool(v.absent[0]) for v in target.slots.values())
                check(absent==(desc.get('absence')=='empty_retained_streams'),'EXPERIENCE_DATA_ABSENCE')
                region=None if absent else digest(saved[key]['echo']['slots'])
                entry=provider.population.observations.entries[key]
                self.examples[key]=dict(batch=batch,baseline=withhold_action(batch,self.specs),
                    target=target.slots,region=region,partition=group['id'],outcome=entry['outcome'],
                    task=source['task'],receipt=saved[key],absent=absent,population=group['population'])
            provider.fresh()
        self.binding=dict(schema='repository-experience-learning-v1',head=session.head,
            active_model=session.snapshot()['active_model'],groups=self.groups,populations=populations,
            adapter_code=code_pins(),examples={k:dict(input=e['batch'].pin,region=e['region'],
                outcome=e['outcome'],partition=e['partition'],absent=e['absent'],
                episode=e['receipt']['episode']) for k,e in self.examples.items()},
            scope='exposed_experience_not_heldout_or_prospective',scientific_admission=False,serving=False)
        self.binding_pin=digest(self.binding)

    def stage(self,partition,model):
        check(partition in self.order,'GROWTH_DATA_PARTITION')
        if partition in self.stages:return self.stages[partition]
        visible=set(self.order[:self.order.index(partition)+1])
        keys=[k for k,e in self.examples.items() if e['partition'] in visible]
        regions=list(dict.fromkeys(self.examples[k]['region'] for k in keys if not self.examples[k]['absent']))
        representatives=[next(self.examples[k] for k in keys if self.examples[k]['region']==r) for r in regions]
        check(regions,'GROWTH_DATA_EMPTY_BANK')
        desc=dict(binding=self.binding_pin,partition=partition,regions=regions,keys=keys)
        view=PackedRepositoryWorld(model,regions,representatives,world_source_id=digest(desc))
        labels=[]
        for region in regions:
            outcomes={self.examples[k]['outcome'] for k in keys if self.examples[k]['region']==region}
            labels.append(next(iter(outcomes)) if len(outcomes)==1 else 2)
        costs,persistence={},{}
        for key in keys:
            e=self.examples[key]
            if e['absent']:continue
            costs[key],persistence[key]={},{}
            for s in self.specs:
                lane=[distances(e['target'][s.name],r['target'][s.name],model.config) for r in representatives]
                costs[key][s.name]=torch.tensor(lane,dtype=torch.float32,device=self.device)
                persistence[key][s.name]=distances(e['target'][s.name],e['batch'][s.name],model.config)
        self.stages[partition]=dict(view=view,keys=keys,regions=regions,description=desc,costs=costs,
            persistence=persistence,labels=torch.tensor(labels,device=self.device),
            allowed=torch.ones(len(regions),dtype=torch.bool,device=self.device))
        return self.stages[partition]

    def fresh(self):
        check(self.binding['adapter_code']==code_pins(),'EXPERIENCE_DATA_CODE_CHANGED')
        check(roster(self.session,self.binding['populations'])[0]==self.groups,'EXPERIENCE_DATA_ORDER_CHANGED')
        for stage in self.stages.values():stage['view'].check_reference()
        RelationData.fresh(self)
