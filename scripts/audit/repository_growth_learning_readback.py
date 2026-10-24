"""Read source joins, probability arithmetic and transactional bytes independently."""
import argparse
from collections import Counter
import copy
import io
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_memory_readback import chain
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_cohort_population import CohortPopulation
from model.loop_training_checkpoint import tensor_fingerprint
from senses.trajectory_population import read_pinned
import torch


def read(path):return json.loads(path.read_bytes())


def input_pin(receipt,gaps):
    desc=receipt['description'];fields,absent={},{}
    roles=('task_report','original_source','candidate_source','recorded_patch')
    for lane,source in desc['fields'].items():
        rows,cursor=[],0
        for field in source:
            start=cursor
            for window in field['windows']:
                check(window['window_start']==cursor,'GROWTH_READ_WINDOW_JOIN')
                cursor=window['window_end']
            rows.append(dict(role=field['role'],start=start,end=cursor,source=field))
        check(cursor==receipt['slots'][lane]['shape'][0],'GROWTH_READ_WINDOW_COUNT')
        fields[lane]=rows;absent[lane]=[r for r in roles if r not in {f['role'] for f in rows}]
    return digest(dict(schema='repository-relation-input-v1',source=desc,fields=fields,
                       absent_roles=absent,source_gaps=gaps,intervention=None))


def source(config,result):
    root=Path(config['source']);proof=read_pinned(Path(config['proof']),config['proof_pin'])
    check(sha((root/'experience.sqlite3').read_bytes())==proof['journal'] and
          sha((root/'snapshot-final.json').read_bytes())==proof['snapshot'],'GROWTH_READ_SOURCE')
    snap=read(root/'snapshot-final.json');_,rows=chain(root/'experience.sqlite3')
    check(rows[-1][2]==snap['head']==result['binding']['head'] and len(rows)==snap['events'] and
          snap['active_model']==result['binding']['active_model'],'GROWTH_READ_HEAD')
    configured={digest(p['native']):p for p in config['populations']}
    check(len(configured)==len(config['populations']),'GROWTH_READ_POPULATIONS')
    populations={k:CohortPopulation(p['native']['population'],p['native']['population_pin'])
                 for k,p in configured.items()}
    proofs={k:read_pinned(Path(p['proof']),p['proof_pin']) for k,p in configured.items()}
    groups,examples,projects,receipts=[],{},set(),{k:{} for k in configured}
    for seq,(event,payload,_) in enumerate(rows):
        if event.get('kind')!='repository_cohort_intake':continue
        spec=event['spec'];native={k:v for k,v in spec.items() if k!='partition'}
        popkey=digest(native);check(popkey in configured,'GROWTH_READ_POPULATION')
        pop=populations[popkey];group='intake-'+str(seq);key=digest(spec)
        groups.append(dict(id=group,seq=seq,key=key,population=popkey,spec=spec))
        saved=snap['repository_cohort_intakes'][key]
        check(saved['event']==event and saved['bindings']==payload['bindings'],'GROWTH_READ_INTAKE')
        selected={k for k,v in pop.fields.items() if v['partition']==spec['partition']}
        check(selected==set(payload['bindings']),'GROWTH_READ_PARTITION')
        for k in sorted(selected):
            receipt=payload['bindings'][k];ep=pop.observations.episode(k)
            check(receipt['episode']==ep['id'] and snap['memory']['states'][ep['id']]['active'],
                  'GROWTH_READ_EPISODE')
            retained={side:receipt[side] for side in ('input','echo')}
            check(retained==proofs[popkey]['receipts'][k] and k not in examples,'GROWTH_READ_NATIVE')
            receipts[popkey][k]=retained;projects.add(pop.fields[k]['task'])
            absent=all(v['absent'] for v in receipt['echo']['slots'].values())
            examples[k]=dict(input=input_pin(receipt['input'],pop.fields[k]['missing']),
                region=None if absent else digest(receipt['echo']['slots']),
                outcome=pop.observations.entries[k]['outcome'],partition=group,absent=absent,
                episode=ep['id'],task=pop.fields[k]['task'])
    check(groups==result['binding']['groups'] and set(examples)==set(result['binding']['examples']),
          'GROWTH_READ_GROUPS')
    for k,e in examples.items():
        saved=result['binding']['examples'][k]
        check({n:saved[n] for n in e if n!='task'}=={n:v for n,v in e.items() if n!='task'},
              'GROWTH_READ_EXAMPLE')
    check(all(receipts[k]==proofs[k]['receipts'] for k in configured),'GROWTH_READ_COMPLETE_NATIVE')
    maxima={}
    for population in receipts.values():
        for receipt in population.values():
            for lane,slot in receipt['input']['slots'].items():
                maxima[lane]=max(maxima.get(lane,0),slot['shape'][0])
    check(result['census']['examples']==len(examples) and result['census']['projects']==len(projects) and
          result['census']['native_target_absent']==sum(e['absent'] for e in examples.values()) and
          result['census']['max_input_windows']==maxima,
          'GROWTH_READ_CENSUS')
    order=[g['id'] for g in groups];banks={}
    for i,group in enumerate(order):
        keys=[k for k,e in examples.items() if e['partition'] in order[:i+1]]
        regions=list(dict.fromkeys(examples[k]['region'] for k in keys if not examples[k]['absent']))
        labels=[]
        for r in regions:
            values={examples[k]['outcome'] for k in keys if examples[k]['region']==r}
            labels.append(next(iter(values)) if len(values)==1 else 2)
        banks[group]=dict(keys=keys,regions=regions,labels=labels)
    check(result['census']['groups']==[dict(id=g,examples=len(b['keys']),regions=len(b['regions']))
          for g,b in banks.items()],'GROWTH_READ_BANKS')
    return examples,banks,proof


def probabilities(rows,examples,bank):
    check(set(rows)==set(bank['keys']),'GROWTH_READ_SCORE_ROSTER')
    lanes=None
    for key,row in rows.items():
        e=examples[key]
        check(row['outcome']==e['outcome'] and row['native_target_absent']==e['absent'] and
              row['task']==e['task'],'GROWTH_READ_SCORE_SOURCE')
        if lanes is None:lanes=set(row['lanes'])
        check(set(row['lanes'])==lanes and len(lanes)==7,'GROWTH_READ_LANES')
        for name,value in row['lanes'].items():
            p=value['probabilities']
            check(len(p)==len(bank['regions']) and all(math.isfinite(v) and 0<=v<=1 for v in p)
                  and abs(sum(p)-1)<1e-6,'GROWTH_READ_PROBABILITIES')
            # Independent double arithmetic versus persisted CUDA float32 reductions.
            # This tolerance audits rounding only; retention below has zero tolerance.
            anchor=[sum(v for v,label in zip(p,bank['labels'],strict=True) if label==i) for i in range(3)]
            check(all(abs(x-y)<1e-6 for x,y in zip(anchor,value['anchor'],strict=True)),
                  'GROWTH_READ_ANCHOR')
            check(value['brier']==sum((v-float(i==e['outcome']))**2 for i,v in enumerate(value['anchor'])),
                  'GROWTH_READ_BRIER')
            if not e['absent']:
                index=bank['regions'].index(e['region'])
                check(value['target_probability']==p[index] and
                      value['correct']==(max(range(len(p)),key=p.__getitem__)==index),'GROWTH_READ_REGION')
                if name!='common':
                    check(all(math.isfinite(value[n]) and -1e-6<=value[n]<=2.000001
                          for n in ('native_expected_loss','native_placement_loss','persistence_loss')),
                          'GROWTH_READ_NATIVE_SCORE')
    return lanes


def metrics(result,examples,banks):
    check(set(result['before'])==set(result['after'])==set(banks),'GROWTH_READ_BANK_ROSTER')
    for phase in ('before','after'):
        for group,bank in banks.items():probabilities(result[phase][group],examples,bank)
    last=list(banks)[-1];old=set(banks[list(banks)[-2]]['keys'])
    probabilities(result['withheld_action'],examples,banks[last])
    before,after={},{}
    for phase,target in (('before',before),('after',after)):
        for group,rows in result[phase].items():
            for key,row in rows.items():
                if key not in old:continue
                for name,lane in row['lanes'].items():
                    for metric in ('brier','native_expected_loss','native_placement_loss'):
                        if metric in lane:target[group+'|'+key+'|'+name+'|'+metric]=lane[metric]
    check(before and before.keys()==after.keys(),'GROWTH_READ_RETENTION_ROSTER')
    regressions={k:dict(before=v,after=after[k],delta=after[k]-v) for k,v in before.items() if after[k]>v}
    verdict=result['decision']
    accepted=not regressions and verdict['objective_after']<verdict['objective_before']
    check(verdict['regressions']==regressions and verdict['retained_checks']==len(before) and
          verdict['accepted']==accepted and not verdict['serving'] and not verdict['scientific_admission'],
          'GROWTH_READ_DECISION')
    return dict(retained_examples=len(old),retained_checks=len(before),regressions=len(regressions),
                accepted=accepted)


def checkpoints(root,result):
    manifests,states={},{}
    for name,pin in result['checkpoints'].items():
        folder=root/name;body=(folder/'manifest.json').read_bytes();m=json.loads(body)
        check(sha(body)==pin==(folder/'COMMITTED').read_text(),'GROWTH_READ_CHECKPOINT_PIN')
        raw=(folder/'delta.pt').read_bytes()
        check(dict(bytes=len(raw),sha256=sha(raw))==m['payload'] and m['binding']==result['binding'],
              'GROWTH_READ_CHECKPOINT_BYTES')
        check(not m['serving'] and not m['scientific_admission'] and
              all(sha((Path(__file__).resolve().parents[1]/'model'/n).read_bytes())==v
                  for n,v in m['code'].items()),'GROWTH_READ_CHECKPOINT_CODE')
        states[name]=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);manifests[name]=m
        check(digest(tensor_fingerprint(states[name]['optimizer']))==m['fingerprint']['optimizer'],
              'GROWTH_READ_OPTIMIZER')
    check(len({m['frozen'] for m in manifests.values()})==1 and
          manifests['initial']['step']==0 and manifests['initial']['predecessor'] is None and
          manifests['proposal']['step']==1 and
          manifests['proposal']['predecessor']==result['checkpoints']['initial'] and
          manifests['final']['predecessor']==result['checkpoints']['proposal'] and
          manifests['final']['step']==int(result['decision']['accepted']),'GROWTH_READ_CHECKPOINT_CHAIN')
    expected='proposal' if result['decision']['accepted'] else 'initial'
    check(tensor_fingerprint(states['final'])==tensor_fingerprint(states[expected]) and
          manifests['final']['fingerprint']==result['final_fingerprint'] and
          manifests['proposal']['fingerprint']==result['proposal']['fingerprint'],
          'GROWTH_READ_TRANSACTION')
    a=tensor_fingerprint(states['initial']['branch']);b=tensor_fingerprint(states['proposal']['branch'])
    changed=['query._repository.'+n for n in a if a[n]!=b[n]]
    check(changed and changed==result['changed_parameters'] and
          sum(v.numel() for v in states['initial']['branch'].values())==result['census']['parameters'],
          'GROWTH_READ_CHANGED_PARAMETERS')
    return dict(checkpoints=result['checkpoints'],changed_tensors=len(changed),state_equal_to=expected,
                frozen_parent=manifests['initial']['frozen'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter();root=args.root
    config=read(root/'config.json');result=read(root/'result.json')
    check(not any(result[n] for n in ('serving','scientific_admission','learned_capability')),
          'GROWTH_READ_SCOPE')
    scripts=Path(__file__).resolve().parents[1]
    check(result['driver_sha256']==sha((scripts/'audit/repository_growth_learning.py').read_bytes()) and
          all(sha((scripts/n).read_bytes())==pin for n,pin in result['binding']['adapter_code'].items()),
          'GROWTH_READ_CODE')
    examples,banks,proof=source(config,result);scores=metrics(result,examples,banks)
    outcomes=Counter(e['outcome'] for e in examples.values())
    projects={task:Counter(e['outcome'] for e in examples.values() if e['task']==task)
              for task in {e['task'] for e in examples.values()}}
    def entropy(counts):
        total=sum(counts.values())
        return -sum(n/total*math.log2(n/total) for n in counts.values())
    population=dict(outcomes=dict(outcomes),projects={k:dict(v) for k,v in projects.items()},
        mixed_outcome_projects=sum(len(v)>1 for v in projects.values()),
        empirical_outcome_entropy_bits=entropy(outcomes),
        empirical_within_project_outcome_entropy_bits=sum(sum(v.values())/len(examples)*entropy(v)
                                                        for v in projects.values()),
        scope='descriptive exposed-label census, not a population bound or an admission estimator')
    transaction=checkpoints(root,result);controls=[]
    def refuse(name,code,fn):
        try:fn()
        except ValueError as error:check(code in str(error),'GROWTH_READ_CONTROL:'+str(error))
        else:raise AssertionError('GROWTH_READ_NEGATIVE_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    group=next(iter(banks));key=banks[group]['keys'][0]
    altered=copy.deepcopy(result);del altered['before'][group][key]
    refuse('dropped_example','GROWTH_READ_SCORE_ROSTER',lambda:metrics(altered,examples,banks))
    altered=copy.deepcopy(result);altered['before'][group][key]['outcome']=2
    refuse('forged_outcome','GROWTH_READ_SCORE_SOURCE',lambda:metrics(altered,examples,banks))
    altered=copy.deepcopy(result);altered['before'][group][key]['lanes']['common']['brier']+=.01
    refuse('forged_metric','GROWTH_READ_BRIER',lambda:metrics(altered,examples,banks))
    altered=copy.deepcopy(result);altered['decision']['accepted']=not altered['decision']['accepted']
    refuse('forged_acceptance','GROWTH_READ_DECISION',lambda:metrics(altered,examples,banks))
    altered=copy.deepcopy(result);altered['checkpoints']['initial']='0'*64
    refuse('wrong_checkpoint','GROWTH_READ_CHECKPOINT_PIN',lambda:checkpoints(root,altered))
    stages={}
    for stage in ('learn','cold'):
        report=read(root/(stage+'-readback.json'))
        check(report['result']==sha((root/'result.json').read_bytes()) and
              report['journal']==proof['journal'] and report['snapshot']==proof['snapshot'] and
              len(report['controls'])==8 and
              abs(sum(report['phases'].values())-report['wall_seconds'])<1e-9,'GROWTH_READ_STAGE')
        stages[stage]=dict(pin=sha((root/(stage+'-readback.json')).read_bytes()),
                           wall_seconds=report['wall_seconds'],phases=report['phases'])
    check(sha((Path(config['source'])/'experience.sqlite3').read_bytes())==proof['journal'],
          'GROWTH_READ_SOURCE_UNCHANGED')
    report=dict(schema='repository-growth-learning-readback-v1',result=sha((root/'result.json').read_bytes()),
        source=dict(journal=proof['journal'],snapshot=proof['snapshot']),scores=scores,transaction=transaction,
        population=population,
        stages=stages,controls=controls,wall_seconds=time.perf_counter()-started,
        number_status='measured',acceptance='author-checked',scientific_admission=False,serving=False,
        learned_capability=False)
    pin=write_committed(args.output,report)
    print(json.dumps(dict(pin=pin,**scores,controls=len(controls),wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':main()
