"""Read back native joins, delta checkpoint bytes, split boundaries and measured scores."""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_cohort_native import reconstruct
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_cohort_population import CohortPopulation
from model.loop_training_checkpoint import tensor_fingerprint
from senses.contract import SenseError
from senses.trajectory_population import read_pinned

import torch


def read(path):
    return json.loads(path.read_bytes())


def summary(rows):
    lanes = list(next(iter(rows.values()))['lanes'])
    result = {}
    for lane in lanes:
        allrows = [r['lanes'][lane] for r in rows.values()]
        item = {}
        for metric in ('brier','target_probability','correct','native_expected_loss','native_placement_loss','persistence_loss'):
            values = [r[metric] for r in allrows if metric in r]
            if values:
                item[metric] = dict(mean=sum(values)/len(values),n=len(values))
        pairs = []
        for task in sorted({r['task'] for r in rows.values()}):
            success = [r['lanes'][lane]['anchor'][1] for r in rows.values() if r['task']==task and r['outcome']==1]
            failure = [r['lanes'][lane]['anchor'][1] for r in rows.values() if r['task']==task and r['outcome']==0]
            pairs.extend(a-b for a in success for b in failure)
        item['within_task_success_failure_margins'] = pairs
        result[lane] = item
    return result


def check_stage(result,examples,partition):
    d = result['stages'][partition]['description']
    keys = [k for k,e in examples.items() if e['partition']=='learn_1' or partition=='learn_2']
    regions = list(dict.fromkeys(examples[k]['region'] for k in keys if not examples[k]['absent']))
    check(d['keys']==keys and d['regions']==regions and d['partition']==partition,'RELATION_READER_SPLIT')
    labels=[]
    for region in regions:
        seen={examples[k]['outcome'] for k in keys if examples[k]['region']==region}
        labels.append(next(iter(seen)) if len(seen)==1 else 2)
    for variant in ('before','after','withheld_action'):
        rows = result['stages'][partition][variant]
        check(set(rows)==set(keys),'RELATION_READER_SCORE_ROSTER')
        for key,row in rows.items():
            check(row['outcome']==examples[key]['outcome'] and row['native_target_absent']==examples[key]['absent'],
                  'RELATION_READER_TARGET')
            for value in row['lanes'].values():
                p = value['probabilities']
                check(len(p)==len(regions) and abs(sum(p)-1)<1e-5 and all(0<=v<=1 for v in p),
                      'RELATION_READER_PROBABILITIES')
                a = value['anchor']
                recounted=[sum(p[j] for j,label in enumerate(labels) if label==i) for i in range(3)]
                check(all(abs(x-y)<1e-6 for x,y in zip(a,recounted,strict=True)), 'RELATION_READER_ANCHOR_JOIN')
                brier = sum((v-float(i==row['outcome']))**2 for i,v in enumerate(a))
                check(len(a)==3 and abs(sum(a)-1)<1e-5 and abs(brier-value['brier'])<1e-12,
                      'RELATION_READER_BRIER')
    return {v:summary(result['stages'][partition][v]) for v in ('before','after','withheld_action')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    t=time.perf_counter()
    root=args.root
    config,result=read(root/'config.json'),read(root/'result.json')
    native=config['native']
    proof=read_pinned(Path(config['native_proof']),config['native_proof_pin'])
    memory=read_pinned(Path(config['proof']),config['proof_pin'])
    parent=Path(config['source'])
    check(sha((parent/'experience.sqlite3').read_bytes())==memory['journal'] and
          sha((parent/'snapshot-final.json').read_bytes())==memory['snapshot'],'RELATION_READER_PARENT')
    pop=CohortPopulation(native['population'],native['population_pin'])
    reconstructed={}
    for key in result['binding']['examples']:
        sides={side:reconstruct(pop,Path(native['native']),native['native_pin'],key,side) for side in ('input','echo')}
        check(sides==proof['receipts'][key],'RELATION_READER_NATIVE_JOIN')
        desc=sides['input']['description']
        fields={}
        for name,items in desc['fields'].items():
            cursor=0;rows=[]
            for f in items:
                start=cursor
                for w in f['windows']:
                    check(w['window_start']==cursor,'RELATION_READER_CONTIGUITY')
                    cursor=w['window_end']
                rows.append(dict(role=f['role'],start=start,end=cursor,source=f))
            fields[name]=rows
        roles=('task_report','original_source','candidate_source','recorded_patch')
        absent_roles={n:[r for r in roles if r not in {f['role'] for f in fs}] for n,fs in fields.items()}
        b=dict(schema='repository-relation-input-v1',source=desc,fields=fields,absent_roles=absent_roles,
               source_gaps=pop.fields[key]['missing'],intervention=None)
        absent=all(v['absent'] for v in sides['echo']['slots'].values())
        reconstructed[key]=dict(input=digest(b),region=None if absent else digest(sides['echo']['slots']),
            outcome=pop.observations.entries[key]['outcome'],partition=pop.fields[key]['partition'],absent=absent)
    check(reconstructed==result['binding']['examples'],'RELATION_READER_INPUT_BINDING')
    check(result['binding']['episodes']=={k:pop.observations.episode(k)['id'] for k in reconstructed},
          'RELATION_READER_EPISODE_JOIN')
    # Independently reconstruct the fixed partition/key order, not the reported roster.
    order=sorted(reconstructed,key=lambda k:(reconstructed[k]['partition'],k))
    reconstructed={k:reconstructed[k] for k in order}
    measured={part:check_stage(result,reconstructed,part) for part in ('learn_1','learn_2')}
    checkpoints={}
    steps={'initial':0,'learn_1-mid':4,'learn_1-final':8,'learn_2-mid':12,'learn_2-final':16}
    for name,pin in result['checkpoints'].items():
        folder=root/name
        m=read_pinned(folder/'manifest.json',pin)
        check((folder/'COMMITTED').read_text()==pin and m['step']==steps[name] and
              m['code']==result['code'] and m['binding']==result['binding'],'RELATION_READER_CHECKPOINT')
        raw=(folder/'delta.pt').read_bytes()
        check(dict(bytes=len(raw),sha256=sha(raw))==m['payload'],'RELATION_READER_PAYLOAD')
        value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        check(all(int(s['step'])==m['step'] for s in value['optimizer']['state'].values()),'RELATION_READER_CLOCK')
        check(bool(value['optimizer']['state'])==(m['step']>0),'RELATION_READER_CLOCK')
        if name=='initial':
            check(all(not bool(v.any()) for n,v in value['branch'].items() if n.endswith('up.weight')),
                  'RELATION_READER_INITIAL')
        if name=='learn_2-final':
            check(all(tensor_fingerprint(v)==result['final_parameters']['query._repository.'+n]
                      for n,v in value['branch'].items()),'RELATION_READER_FINAL_WEIGHTS')
        checkpoints[name]=dict(pin=pin,step=m['step'],payload=m['payload'])
    changed=[n for n,v in result['final_parameters'].items() if v!=result['parent_parameters'][n]]
    check(set(changed)==set(result['changed_parameters']) and changed and
          all(n.startswith('query._repository.') for n in changed),'RELATION_READER_CHANGED_SCOPE')
    cold=read(root/'cold-readback.json');learn=read(root/'learn-readback.json')
    check(cold['cold']['resumed_both_increments'] and cold['cold']['rollback_equal'] and
          cold['cold']['parent_retained_equal'] and cold['result']==sha((root/'result.json').read_bytes()),
          'RELATION_READER_COLD')
    for r in (cold,learn):
        check(abs(sum(r['phases'].values())-r['wall_seconds'])<1e-6 and min(r['phases'].values())>=0,
              'RELATION_READER_TIMING')
    controls=[]
    def reject(name,code,fn):
        try:fn()
        except (ValueError,SenseError) as exc:
            check(code in str(exc),'RELATION_READER_CONTROL_CODE')
        else:raise AssertionError('RELATION_READER_CONTROL_ACCEPTED')
        controls.append(dict(name=name,code=code))
    bad=copy.deepcopy(result)
    bad['stages']['learn_1']['description']['keys'].append(next(k for k,e in reconstructed.items() if e['partition']=='learn_2'))
    reject('future_partition','RELATION_READER_SPLIT',lambda:check_stage(bad,reconstructed,'learn_1'))
    bad=copy.deepcopy(result)
    first=next(iter(bad['stages']['learn_1']['after'].values()))
    first['outcome']=1-first['outcome']
    reject('forged_outcome','RELATION_READER_TARGET',lambda:check_stage(bad,reconstructed,'learn_1'))
    reject('wrong_checkpoint_pin','MNLM_UCR_INPUT_DIGEST',lambda:read_pinned(root/'initial/manifest.json','0'*64))
    prior=result['stages']['learn_1']['after'];later=result['stages']['learn_2']['retained_previous']
    regressions={}
    for key,row in prior.items():
        for lane,v in row['lanes'].items():
            for metric in ('brier','native_expected_loss','native_placement_loss'):
                if metric in v and later[key]['lanes'][lane][metric]>v[metric]:
                    regressions.setdefault(key,{})[lane+':'+metric]=later[key]['lanes'][lane][metric]-v[metric]
    report=dict(number_status='measured',acceptance='author-checked',result=sha((root/'result.json').read_bytes()),
        native_input_joins=len(reconstructed),checkpoints=checkpoints,summary=measured,
        previous_increment_regressions=regressions,controls=controls,source_unchanged=True,
        serving=False,scientific_admission=False,learned_repair=False,heldout=False,
        action_intervention_limit='withheld patch and candidate; original-file selection remains action-dependent',
        wall_seconds=time.perf_counter()-t)
    write_committed(args.output,report)
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output.read_bytes()),
        controls=len(controls),regression_examples=len(regressions),wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':main()
