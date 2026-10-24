"""Independent byte/metric recount of repository retention, without model inference."""
import argparse
import copy
import io
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_cohort_native import reconstruct
from audit.repository_relation_readback import check_stage,summary
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_cohort_population import CohortPopulation
from model.loop_training_checkpoint import tensor_fingerprint
from senses.contract import SenseError
from senses.trajectory_population import read_pinned

import torch


def read(p):return json.loads(p.read_bytes())


def measures(rows):
    return {key+'|'+lane+'|'+metric:float(v[metric])
            for key,row in rows.items() for lane,v in row['lanes'].items()
            for metric in ('brier','native_expected_loss','native_placement_loss') if metric in v}


def compare(before,after):
    check(before.keys()==after.keys(),'RETENTION_READER_ROSTER')
    return {k:after[k]-v for k,v in before.items() if after[k]>v}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter();root=a.root
    config,result=read(root/'config.json'),read(root/'result.json')
    if 'fixed_root' in config:
        fixed=read_pinned(Path(config['fixed_root'])/'result.json',config['fixed_pin'])
        for stage in ('learn','cold'):
            replay=read(root/('fixed-rejection-'+stage+'.json'))
            for key in ('accepted','code','before','after','before_scores','after_scores','trials','projection'):
                check(replay[key]==fixed['attempts'][0][key],'RETENTION_READER_FIXED_REPLAY')
    previous=Path(config['previous'])
    prior=read_pinned(previous/'result.json',config['previous_pin'])
    c=read(previous/'config.json');native=c['native']
    native_proof=read_pinned(Path(c['native_proof']),c['native_proof_pin'])
    pop=CohortPopulation(native['population'],native['population_pin'])
    examples=prior['binding']['examples']
    for key,e in examples.items():
        actual={s:reconstruct(pop,Path(native['native']),native['native_pin'],key,s) for s in ('input','echo')}
        check(actual==native_proof['receipts'][key],'RETENTION_READER_NATIVE_JOIN')
        check(e['outcome']==pop.observations.entries[key]['outcome'] and
              e['partition']==pop.fields[key]['partition'],'RETENTION_READER_TARGET')
    contract=result['contract']
    baseline=prior['stages']['learn_1']['after']
    ceilings=measures(baseline)
    check(contract['ceiling']==ceilings and contract['baseline']==digest(baseline) and
          contract['binding']==digest(prior['binding']) and contract['policy']['tolerance']==0.,
          'RETENTION_READER_BASELINE')
    expected_keys=[k for k,e in examples.items() if e['partition']=='learn_1']
    expected_new=[k for k,e in examples.items() if e['partition']=='learn_2']
    # Source-ordered lists are independent of JSON's canonical key ordering.
    check(contract['keys']==sorted(expected_keys) and contract['new']==sorted(expected_new),
          'RETENTION_READER_SPLIT')
    stages={'learn_2':dict(description=prior['stages']['learn_2']['description'],
        before=result['before'],after=result['after'],withheld_action=result['withheld_action'])}
    ordered={k:examples[k] for k in sorted(examples,key=lambda k:(examples[k]['partition'],k))}
    checked_summary=check_stage(dict(stages=stages),ordered,'learn_2')
    known=compare(ceilings,measures(prior['stages']['learn_2']['retained_previous']))
    check(known and set(known)==set(result['previous_regressions']),'RETENTION_READER_NEGATIVE')
    force=result['forced_rejection']
    check(not force['accepted'] and force['before']==force['after'] and
          force['proposal_fingerprint']!=force['before'],'RETENTION_READER_ATOMIC_CONTROL')
    names=list(result['checkpoints'])
    check(set(names)=={'initial'}|{f'attempt-{i:02}' for i in range(1,len(result['attempts'])+1)},
          'RETENTION_READER_CHECKPOINT_ROSTER')
    snapshots={}
    for name,pin in result['checkpoints'].items():
        d=root/name;m=read_pinned(d/'manifest.json',pin)
        check((d/'COMMITTED').read_text()==pin and m['contract']==contract and m['code']==result['code'] and
              not m['serving'] and not m['scientific_admission'],'RETENTION_READER_CHECKPOINT_CONTRACT')
        for code,pin in m['code'].items():
            check(sha((Path(__file__).resolve().parents[1]/'model'/code).read_bytes())==pin,
                  'RETENTION_READER_CODE_PIN')
        raw=(d/'delta.pt').read_bytes()
        check(dict(bytes=len(raw),sha256=sha(raw))==m['payload'],'RETENTION_READER_PAYLOAD')
        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        check(all(int(v['step'])==m['step'] for v in state['optimizer']['state'].values()),
              'RETENTION_READER_OPTIMIZER_CLOCK')
        snapshots[name]=dict(manifest=m,state=state)
    clock=8;cursor=ceilings;fp=snapshots['initial']['manifest']['fingerprint']
    check(force['before']==fp and snapshots['initial']['manifest']['step']==clock,'RETENTION_READER_INITIAL')
    accepted=0
    for index,row in enumerate(result['attempts'],1):
        check(row['attempt']==index and row['before']==fp and row['before_scores']==cursor,
              'RETENTION_READER_STEP_CHAIN')
        check(not compare(cursor,row['after_scores']),'RETENTION_READER_REGRESSION')
        if row['accepted']:
            check(row['trials'] and row['trials'][-1]['gate_passed'] and
                  not row['trials'][-1]['forced_rejection'] and
                  row['trials'][-1]['scores']==row['after_scores'] and
                  row['trials'][-1]['new_loss']<row['new_loss_before'] and row['after']!=fp,
                  'RETENTION_READER_GAIN')
            clock+=1;accepted+=1
        else:
            check(row['after']==fp and row['after_scores']==cursor and index==len(result['attempts']),
                  'RETENTION_READER_REFUSAL')
        for trial in row['trials']:
            violations=compare(cursor,trial['scores'])
            check(set(violations)==set(trial['regressions']),'RETENTION_READER_TRIAL')
            if trial['gate_passed']:
                check(not violations and trial['new_loss']<row['new_loss_before'],'RETENTION_READER_TRIAL_GATE')
            if 'next_scale' in trial:
                roots=[];possible=True
                for key,change in violations.items():
                    slope=row['directional'].get(key)
                    if slope is None or slope>=0:
                        possible=False;break
                    curvature=(change-trial['scale']*slope)/trial['scale']**2
                    if not math.isfinite(curvature) or curvature<=0:
                        possible=False;break
                    roots.append(-slope/curvature)
                expected_scale=(.5*min(trial['scale'],*roots) if roots else trial['scale']*.5) if possible else None
                check(expected_scale==trial['next_scale'],'RETENTION_READER_CONTRACTION')
        m=snapshots[f'attempt-{index:02}']['manifest']
        check(m['step']==clock==row['optimizer_step'] and m['attempt']==index and
              m['fingerprint']==row['after'],'RETENTION_READER_CHECKPOINT_CHAIN')
        check(abs(sum(row['phases'].values())-row['wall_seconds'])<1e-6,'RETENTION_READER_STEP_TIMING')
        cursor,fp=row['after_scores'],row['after']
    check(cursor==measures(result['retained_after']) and fp==result['final_fingerprint'],
          'RETENTION_READER_FINAL')
    final=snapshots[f"attempt-{len(result['attempts']):02}"]['state']
    initial=snapshots['initial']['state']
    changed=[]
    for name,tensor in final['branch'].items():
        n='query._repository.'+name
        check(tensor_fingerprint(tensor)==result['final_parameters'][n] and
              tensor_fingerprint(initial['branch'][name])==result['initial_parameters'][n],
              'RETENTION_READER_PARAMETERS')
    for name,v in result['final_parameters'].items():
        if v!=result['initial_parameters'][name]:changed.append(name)
    check(set(changed)==set(result['changed_parameters']) and
          all(n.startswith('query._repository.') for n in changed) and bool(changed)==bool(accepted),
          'RETENTION_READER_CHANGED_SCOPE')
    for stage in ('learn','cold'):
        r=read(root/(stage+'-readback.json'))
        check(r['source_unchanged'] and r['result']==sha((root/'result.json').read_bytes()) and
              abs(sum(r['phases'].values())-r['wall_seconds'])<1e-6,'RETENTION_READER_RECEIPT')
        if stage=='cold':check(all(r['cold'].values()),'RETENTION_READER_COLD')
    controls=[]
    def reject(name,code,fn):
        try:fn()
        except (SenseError,ValueError) as exc:check(code in str(exc),'RETENTION_READER_CONTROL_CODE')
        else:raise AssertionError('RETENTION_READER_CONTROL_ACCEPTED')
        controls.append(dict(name=name,code=code))
    missing=dict(cursor);missing.pop(next(iter(missing)))
    reject('missing_score','RETENTION_READER_ROSTER',lambda:compare(ceilings,missing))
    bad=copy.deepcopy(stages);next(iter(bad['learn_2']['after'].values()))['outcome']=2
    reject('forged_outcome','RELATION_READER_TARGET',lambda:check_stage(dict(stages=bad),ordered,'learn_2'))
    reject('wrong_checkpoint_pin','MNLM_UCR_INPUT_DIGEST',lambda:read_pinned(root/'initial/manifest.json','0'*64))
    proof=read_pinned(Path(c['proof']),c['proof_pin']);source=Path(c['source'])
    check(sha((source/'experience.sqlite3').read_bytes())==proof['journal'] and
          sha((source/'snapshot-final.json').read_bytes())==proof['snapshot'],'RETENTION_READER_SOURCE')
    report=dict(number_status='measured',acceptance='author-checked',result=sha((root/'result.json').read_bytes()),
        native_joins=len(examples),retained_checks=len(ceilings),regressions=compare(ceilings,cursor),
        rejected_known_regressions=known,accepted_updates=accepted,attempts=len(result['attempts']),
        changed_parameter_tensors=len(changed),summary=checked_summary,retained_summary=summary(result['retained_after']),
        new_example_summary={v:summary({k:result[v][k] for k in contract['new']}) for v in ('before','after')},
        controls=controls,source_unchanged=True,serving=False,scientific_admission=False,heldout=False,
        useful_new_capability_demonstrated=False,wall_seconds=time.perf_counter()-started)
    write_committed(a.output,report)
    print(json.dumps(dict(output=str(a.output),sha256=sha(a.output.read_bytes()),accepted=accepted,
                         regressions=len(report['regressions']),seconds=report['wall_seconds'])))


if __name__=='__main__':main()
