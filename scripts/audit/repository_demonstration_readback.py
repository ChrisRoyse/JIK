"""Independently reconstruct demonstration provenance, native receipts and teaching data."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_demonstration_native import reconstruct
from audit.repository_memory_readback import chain,read
from audit.repository_depth_learning_readback import input_pin
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_demonstration_population import DemonstrationPopulation
from claims.repository_observation import SCOPE
from model.repository_demonstration_memory import code_pins
from model.repository_teaching_data import code_pins as teaching_pins
from senses.trajectory_population import read_pinned


def audit(root,output):
    start=time.perf_counter();config=read(root/'config.json')
    check(config['schema']=='repository-demonstration-memory-v1','DEMONSTRATION_READER_SCHEMA')
    parent=Path(config['source']);proof=read_pinned(Path(config['proof']),config['proof_pin'])
    immutable={str(parent/n):pin for n,pin in config['parent_files'].items()}
    immutable.update(proof['immutable_after'])
    for name,pin in tuple(immutable.items()):
        if Path(name).is_relative_to(parent/'candidates'):
            immutable[str(root/Path(name).relative_to(parent))]=pin
    before={n:sha(Path(n).read_bytes()) for n in immutable}
    check(before==immutable,'DEMONSTRATION_READER_PARENT_BYTES')
    old,final=read(parent/'snapshot-final.json'),read(root/'snapshot-final.json')
    oc,old_rows=chain(parent/'experience.sqlite3');nc,rows=chain(root/'experience.sqlite3')
    cohorts=config['cohorts'];count=2*len(cohorts)
    check(count>0 and oc==nc and rows[:-count]==old_rows and
          len(rows)==final['events']==old['events']+count and
          old_rows[-1][2]==old['head'] and rows[-1][2]==final['head'],'DEMONSTRATION_READER_CHAIN')
    previous,head=old['head'],old['memory']['head']
    last_records=next(pay['memory_records'] for _,pay,_ in reversed(old_rows)
                      if pay and pay.get('memory_records'))
    sequence=last_records[-1]['sequence']+1
    states=copy.deepcopy(old['memory']['states'])
    intakes=copy.deepcopy(old.get('repository_demonstration_intakes',{}))
    episodes,counts,controls={},{},[]
    for index,c in enumerate(cohorts):
        spec=c['native'];pop=DemonstrationPopulation(spec['population'],spec['population_pin'])
        ob=pop.observations;native_proof=read_pinned(Path(c['native_proof']),c['native_proof_pin'])
        check(native_proof['native']==spec,'DEMONSTRATION_READER_NATIVE_PROOF')
        controls.extend(native_proof['controls'])
        for offset,part in enumerate(('learn_1','learn_2')):
            body,payload,event_pin=rows[len(old_rows)+2*index+offset]
            chosen=dict(**spec,partition=part);prefix=f'cohort-{index}-{part}'
            check(body==dict(kind='repository_demonstration_intake',spec=chosen,prior_head=previous,
                            payload=digest(payload),code=code_pins()) and
                  read(root/(prefix+'-intake.json'))==dict(event=body,payload=payload),
                  'DEMONSTRATION_READER_EVENT')
            selected=sorted((k for k,v in pop.fields.items() if v['partition']==part),
                            key=lambda k:(pop.fields[k]['input_available_by'],k))
            check(selected,'DEMONSTRATION_READER_EMPTY_PARTITION');expected,ids,records={},{},[]
            for key in selected:
                ep=ob.episode(key);check(ep['id'] not in states,'DEMONSTRATION_READER_REUSED_EPISODE')
                receipt=ob.assess(ep);ids[key]=ep['id'];episodes[ep['id']]=ep
                for event in (dict(op='intake',episode=ep),
                              dict(op='check_source',episode=ep['id'],result=receipt),
                              dict(op='promote',episode=ep['id'],receipt=receipt['id'])):
                    record=dict(sequence=sequence,previous=head,event=event)
                    head=digest(record);record['sha256']=head;records.append(record);sequence+=1
                states[ep['id']]=dict(active=True,grounded_scope=SCOPE,review='source_checked',receipt=receipt)
                expected[key]=dict(episode=ep['id'],**{side:reconstruct(pop,Path(spec['native']),
                    spec['native_pin'],key,side) for side in ('input','echo')})
                check(all(expected[key][s]==native_proof['receipts'][key][s] for s in ('input','echo')),
                      'DEMONSTRATION_READER_NATIVE_JOIN')
            check(payload==dict(episodes=ids,bindings=expected,memory_records=records,
                observations=ob.pin,partition=part,native_measured=True,native_consumer='typed_constellation',
                independent_actions=False,reference_demonstrations=True,learned_repair=False,scientific_admission=False,serving=False),
                'DEMONSTRATION_READER_RECONSTRUCTION')
            check(digest(chosen) not in intakes,'DEMONSTRATION_READER_REUSED_INTAKE')
            intakes[digest(chosen)]=dict(event=body,episodes=ids,bindings=expected)
            expected_memory=dict(old['memory'],head=head,states=states,
                active_episodes=sorted(set(old['memory']['active_episodes'])|set(episodes)),
                grounded_world_claims=sorted(set(old['memory']['grounded_world_claims'])|set(episodes)))
            expected_snapshot=dict(old,head=event_pin,events=old['events']+2*index+offset+1,
                                   memory=expected_memory,repository_demonstration_intakes=intakes)
            check(read(root/(prefix+'-snapshot.json'))==expected_snapshot,'DEMONSTRATION_READER_INCREMENT_STATE')
            previous=event_pin;counts[prefix]=len(ids)
    check(final==expected_snapshot,'DEMONSTRATION_READER_FINAL_STATE')
    cold=read(root/'cold-stage.json')
    check(cold['snapshot']==sha((root/'snapshot-final.json').read_bytes()) and
          cold['journal']==sha((root/'experience.sqlite3').read_bytes()) and
          cold['head']==final['head'] and cold['events']==len(rows) and cold['cold_replay_equal'],
          'DEMONSTRATION_READER_COLD')
    consumer=read(root/'consumer.json')
    check(cold['consumer']==sha((root/'consumer.json').read_bytes()),'DEMONSTRATION_READER_CONSUMER_PIN')
    configured={digest(p['native']):p for p in config['populations']}
    check(len(configured)==len(config['populations']),'DEMONSTRATION_READER_DUPLICATE_POPULATION')
    from claims.repository_depth_population import DepthPopulation
    from claims.repository_cohort_population import CohortPopulation
    factories={'repository_cohort_intake':CohortPopulation,'repository_depth_intake':DepthPopulation,
               'repository_demonstration_intake':DemonstrationPopulation}
    populations={k:factories[p['kind']](p['native']['population'],p['native']['population_pin'])
                 for k,p in configured.items()}
    groups,examples,tasks=[],{},{}
    for seq,(body,payload,_) in enumerate(rows):
        kind=body.get('kind')
        if kind not in factories:continue
        spec=body['spec'];native={k:v for k,v in spec.items() if k!='partition'}
        popkey=digest(native);check(popkey in configured,'DEMONSTRATION_READER_CONSUMER_POPULATION')
        pop=populations[popkey];group='intake-'+str(seq)
        groups.append(dict(id=group,seq=seq,key=digest(spec),population=popkey,spec=spec,kind=kind))
        for k,r in sorted(payload['bindings'].items()):
            field=pop.fields[k];outcome=pop.observations.entries[k]['outcome']
            check(k not in examples and final['memory']['states'][r['episode']]['active'],
                  'DEMONSTRATION_READER_CONSUMER_EPISODE')
            absent=all(s['absent'] for s in r['echo']['slots'].values())
            examples[k]=dict(input=input_pin(r['input'],field['missing']),
                region=None if absent else digest(r['echo']['slots']),outcome=outcome,
                partition=group,absent=absent,episode=r['episode'],
                action_origin=pop.observations.entries[k].get('action_origin','recorded_source_attempt'))
            tasks.setdefault(field['task'],set()).add(outcome)
    check(consumer['reference_demonstrations']==sum(e['action_origin']=='supplied_reference_repair'
          for e in examples.values()),'DEMONSTRATION_READER_ORIGIN_COUNTS')
    binding=consumer['binding']
    check(binding['schema']=='repository-teaching-learning-v1' and
          binding['adapter_code']==teaching_pins() and not binding['scientific_admission'] and
          not binding['serving'] and binding['scope']=='exposed_experience_not_heldout_or_prospective',
          'DEMONSTRATION_READER_CONSUMER_SCOPE')
    check(binding['head']==final['head'] and binding['active_model']==final['active_model'] and
          binding['populations']==config['populations'] and binding['examples']==examples and
          binding['groups']==groups and consumer['examples']==len(examples) and
          consumer['groups']==len(groups) and consumer['projects']==len(tasks) and
          consumer['outcomes']=={str(o):sum(e['outcome']==o for e in examples.values()) for o in (0,1)} and
          consumer['mixed_outcome_projects']==sum(len(o)>1 for o in tasks.values()),
          'DEMONSTRATION_READER_CONSUMER_RECONSTRUCTION')
    controls.extend(read(root/'intake-stage.json')['controls'])
    with tempfile.TemporaryDirectory(prefix='demonstration-memory-control-',dir=root.parent) as tmp:
        clone=Path(tmp)/'journal.sqlite3';shutil.copyfile(root/'experience.sqlite3',clone)
        with sqlite3.connect(clone) as db:
            db.execute('UPDATE payloads SET data=? WHERE hash=?',(b'{}',rows[-1][0]['payload']))
        try:
            chain(clone)
        except ValueError as exc:
            check('REPOSITORY_READER_PAYLOAD' in str(exc),'DEMONSTRATION_READER_CONTROL_CODE')
        else:
            raise AssertionError('DEMONSTRATION_READER_CHANGED_PAYLOAD_ACCEPTED')
        controls.append(dict(name='changed_persisted_payload',code='REPOSITORY_READER_PAYLOAD'))
    after={n:sha(Path(n).read_bytes()) for n in immutable}
    check(before==after,'DEMONSTRATION_READER_MUTATED')
    result=dict(schema='repository-demonstration-memory-readback-v1',events=len(rows),head=final['head'],
        source_events=len(old_rows),source_head=old['head'],snapshot=cold['snapshot'],journal=cold['journal'],
        memory_head=head,partitions=counts,source_actions=len(episodes),controls=controls,
        consumer=cold['consumer'],consumer_examples=len(examples),consumer_projects=len(tasks),
        consumer_outcomes=consumer['outcomes'],mixed_outcome_projects=consumer['mixed_outcome_projects'],
        immutable_before=before,immutable_after=after,cold_replay_equal=True,old_state_unchanged=True,
        old_intakes_unchanged=True,cohorts=cohorts,wall_seconds=time.perf_counter()-start,
        number_status='measured',acceptance='author-checked',learned_repair=False,
        reference_demonstrations=True,independent_candidate_attempts=0,
        scientific_admission=False,serving=False)
    pin=write_committed(output,result)
    print(json.dumps(dict(pin=pin,events=len(rows),head=final['head'],actions=len(episodes),
                         controls=len(controls),wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.output)
