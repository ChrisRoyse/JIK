"""Require every supported repository intake in the connected learning history."""
import json
from pathlib import Path

from claims.codata_source import check,sha
from claims.episodic_memory import digest
from model.repository_cohort_input import CohortNative
from model.repository_depth_input import DepthNative
from model.repository_demonstration_input import DemonstrationNative
from model.repository_experience_data import ExperienceData,code_pins as prior_code_pins
from model.repository_relation_data import RelationData
from model.repository_relation_input import bind,withhold_action
from senses.trajectory_population import read_pinned

KINDS = {'repository_cohort_intake':('repository_cohort_intakes',CohortNative),
         'repository_depth_intake':('repository_depth_intakes',DepthNative),
         'repository_demonstration_intake':('repository_demonstration_intakes',DemonstrationNative)}


def code_pins():
    return dict(prior_code_pins(),**{'model/repository_teaching_data.py':sha(Path(__file__).read_bytes())})


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


class TeachingData(ExperienceData):
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
                    target=target.slots,region=region,partition=group['id'],outcome=entry['outcome'],action_origin=entry.get('action_origin','recorded_source_attempt'),
                    task=source['task'],receipt=saved[key],absent=absent,population=group['population'])
            provider.fresh()
        self.binding=dict(schema='repository-teaching-learning-v1',head=session.head,
            active_model=session.snapshot()['active_model'],groups=self.groups,populations=populations,
            adapter_code=code_pins(),examples={k:dict(input=e['batch'].pin,region=e['region'],
                outcome=e['outcome'],partition=e['partition'],absent=e['absent'],
                episode=e['receipt']['episode'],action_origin=e['action_origin']) for k,e in self.examples.items()},
            scope='exposed_experience_not_heldout_or_prospective',scientific_admission=False,serving=False)
        self.binding_pin=digest(self.binding)

    def fresh(self):
        check(self.binding['adapter_code']==code_pins(),'EXPERIENCE_DATA_CODE_CHANGED')
        check(roster(self.session,self.binding['populations'])[0]==self.groups,'EXPERIENCE_DATA_ORDER_CHANGED')
        for stage in self.stages.values():stage['view'].check_reference()
        RelationData.fresh(self)
