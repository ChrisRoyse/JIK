"""Append fixed repository cohorts after depth experience with one ancestry replay."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import repository_depth_memory as depth, repository_depth_open
from audit.fsv_identity_loop import configure
from audit.repository_growth_memory import retained
from audit.repository_memory_readback import read
from audit.state_cycle_open import prepare
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_cohort_population import CohortPopulation
from lens.lenscommon import run_cli,take_lease
from model.repository_experience_data import ExperienceData,roster
from senses.contract import SenseError
from senses.trajectory_population import read_pinned


def authenticate(config):
    check(config['schema']=='repository-breadth-memory-v1','BREADTH_MEMORY_SCHEMA')
    parent=Path(config['source'])
    proof=read_pinned(Path(config['proof']),config['proof_pin'])
    for name,pin in config['parent_files'].items():
        check(sha((parent/name).read_bytes())==pin,'BREADTH_MEMORY_PARENT')
    check(proof['journal']==config['parent_files']['experience.sqlite3'] and
          proof['snapshot']==config['parent_files']['snapshot-final.json'] and
          proof['cold_replay_equal'],'BREADTH_MEMORY_PARENT_PROOF')
    for name,pin in proof['immutable_after'].items():
        check(sha(Path(name).read_bytes())==pin,'BREADTH_MEMORY_ANCESTOR')
    if config['parent_kind']=='repository_depth_memory':
        check(proof['schema']=='repository-depth-memory-readback-v1','BREADTH_MEMORY_PARENT_KIND')
        return depth.authenticate(read(parent/'config.json'))
    check(config['parent_kind']=='repository_breadth_memory' and
          proof['schema']=='repository-breadth-memory-readback-v1','BREADTH_MEMORY_PARENT_KIND')
    return authenticate(read(parent/'config.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=('intake','cold'),required=True)
    a=p.parse_args();start=time.perf_counter();config=read(a.config)
    root,parent=Path(config['root']),Path(config['source'])
    recipes=authenticate(config);cohorts=config['cohorts']
    check(cohorts and len({digest(c['native']) for c in cohorts})==len(cohorts),
          'BREADTH_MEMORY_DUPLICATE_SPEC')
    proofs=[]
    for c in cohorts:
        proof=read_pinned(Path(c['native_proof']),c['native_proof_pin'])
        check(proof['native']==c['native'] and proof['source_actions']>0 and
              not proof['scientific_admission'] and not proof['learned_repair'],
              'BREADTH_MEMORY_NATIVE_PROOF')
        proofs.append(proof)
    old=read(parent/'snapshot-final.json')
    if a.stage=='intake':
        prepare(config,recipes,root);write_committed(root/'config.json',config)
    else:
        check(read(root/'config.json')==config,'BREADTH_MEMORY_CONFIG')
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    take_lease('28','repository-breadth-memory-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure();authenticated=time.perf_counter()
    session,sessions=repository_depth_open.open_memory(recipes,root,
        old if a.stage=='intake' else read(root/'snapshot-final.json'))
    opened=time.perf_counter();controls=[]
    print(json.dumps(dict(stage=a.stage,replayed_events=session.journal.count)),flush=True)

    def refuse(name,code,fn):
        before=digest(session.snapshot())
        try:
            fn()
        except (SenseError,ValueError) as exc:
            check(code in str(exc),'BREADTH_MEMORY_CONTROL_CODE:'+name+':'+str(exc))
        else:
            raise AssertionError('BREADTH_MEMORY_CONTROL_ACCEPTED:'+name)
        check(before==digest(session.snapshot()),'BREADTH_MEMORY_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code,before=before,after=before))

    if a.stage=='intake':
        for index,(c,proof) in enumerate(zip(cohorts,proofs,strict=True)):
            spec=c['native'];pop=CohortPopulation(spec['population'],spec['population_pin'])
            check(all(any(f['partition']==part for f in pop.fields.values())
                      for part in ('learn_1','learn_2')),'BREADTH_MEMORY_EMPTY_PARTITION')
            refuse(f'{index}:missing_prefix','COHORT_MEMORY_PREFIX',
                   lambda:session.observe_cohort(dict(**spec,partition='learn_2')))
            refuse(f'{index}:reserved_training','COHORT_MEMORY_PARTITION',
                   lambda:session.observe_cohort(dict(**spec,partition='reserved_evaluation')))
            key=next(iter(pop.observations.entries))
            forged=copy.deepcopy(pop.observations.episode(key))
            forged['content']['outcome']=1-forged['content']['outcome']
            refuse(f'{index}:forged_outcome','COHORT_OBSERVATION_EPISODE',
                   lambda:pop.observations.assess(forged))
            for part in ('learn_1','learn_2'):
                chosen=dict(**spec,partition=part);body=session.observe_cohort(chosen)
                payload=session.journal.payload(body['payload'])
                keys={k for k,v in pop.fields.items() if v['partition']==part}
                check(set(payload['bindings'])==keys and all(
                    payload['bindings'][k][s]==proof['receipts'][k][s]
                    for k in keys for s in ('input','echo')),'BREADTH_MEMORY_NATIVE_JOIN')
                refuse(f'{index}:duplicate_{part}','COHORT_MEMORY_DUPLICATE',
                       lambda:session.observe_cohort(chosen))
                prefix=f'cohort-{index}-{part}'
                write_committed(root/(prefix+'-intake.json'),dict(event=body,payload=payload))
                write_committed(root/(prefix+'-snapshot.json'),session.snapshot())
                retained(old,session.snapshot())
                print(json.dumps(dict(cohort=index,partition=part,actions=len(keys),
                                      events=session.journal.count)),flush=True)
        write_committed(root/'snapshot-final.json',session.snapshot())
    final=session.snapshot();retained(old,final)
    check(final==read(root/'snapshot-final.json'),'BREADTH_MEMORY_COLD')
    added=set(final['memory']['states'])-set(old['memory']['states'])
    keys={digest(dict(**c['native'],partition=part)) for c in cohorts for part in ('learn_1','learn_2')}
    intakes=final['repository_cohort_intakes']
    check(set(intakes)-set(old.get('repository_cohort_intakes',{}))==keys and
          added=={e for k in keys for e in intakes[k]['episodes'].values()} and
          len(added)==sum(p['source_actions'] for p in proofs) and
          final['events']==old['events']+len(keys),'BREADTH_MEMORY_COVERAGE')
    intake_done=time.perf_counter()
    data=ExperienceData(session,config['populations'])
    refuse('consumer_missing_population','EXPERIENCE_DATA_MISSING_POPULATION',
           lambda:roster(session,config['populations'][:-1]))
    refuse('consumer_duplicate_population','EXPERIENCE_DATA_DUPLICATE_POPULATION',
           lambda:roster(session,config['populations']+[config['populations'][0]]))
    tasks={e['task'] for e in data.examples.values()}
    consumer=dict(binding=data.binding,examples=len(data.examples),projects=len(tasks),
        groups=len(data.groups),outcomes={str(o):sum(e['outcome']==o for e in data.examples.values())
                                         for o in (0,1)},
        mixed_outcome_projects=sum(len({e['outcome'] for e in data.examples.values()
                                       if e['task']==t})>1 for t in tasks))
    check(digest(session.snapshot())==digest(final),'BREADTH_MEMORY_CONSUMER_MUTATED')
    if a.stage=='intake':
        write_committed(root/'consumer.json',consumer)
    else:
        check(consumer==read(root/'consumer.json'),'BREADTH_MEMORY_CONSUMER_COLD')
    for s in sessions:
        s.close()
    finish=time.perf_counter()
    result=dict(stage=a.stage,head=final['head'],events=final['events'],added=len(added),
        source_head=old['head'],source_events=old['events'],controls=controls,
        old_contexts_and_models_unchanged=True,old_intakes_unchanged=True,
        native_measured=True,learned_repair=False,scientific_admission=False,serving=False,
        snapshot=sha((root/'snapshot-final.json').read_bytes()),
        journal=sha((root/'experience.sqlite3').read_bytes()),cold_replay_equal=True,
        consumer=sha((root/'consumer.json').read_bytes()),
        phases=dict(authentication=authenticated-start,replay=opened-authenticated,
                    intake=intake_done-opened,consumer=finish-intake_done),
        wall_seconds=finish-start,acceptance='author-checked')
    write_committed(root/(a.stage+'-stage.json'),result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    run_cli(main)
