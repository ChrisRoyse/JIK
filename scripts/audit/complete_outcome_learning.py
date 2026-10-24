"""Physical scoped-outcome updates on retained native action/context measurements."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from torch.nn import functional as F
from audit import complete_outcome_source as source
from audit.complete_answer_baselines import reject
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.complete_answer_outcome import OutcomeInputs,ARMS,make,update,predict,validate
from model.native_population_training import cpu,fingerprint
from text.constellation_corpus import file_sha,need


def metrics(value, logits):
    result={}
    for split,native in logits.items():
        keys=[k for k,r in value['rows'].items() if r['split']==split]
        labels=torch.tensor([value['rows'][k]['outcome'] for k in keys],dtype=torch.float64)
        lanes={**native,'combined':torch.stack(list(native.values()),1).double().mean(1)}
        result[split]={}
        for name,x in lanes.items():
            x=x.double();p=x.sigmoid();brier=(p-labels).square()
            loss=F.binary_cross_entropy_with_logits(x,labels,reduction='none')
            groups={}
            for group in sorted({value['rows'][k]['group'] for k in keys}):
                ids=[i for i,k in enumerate(keys) if value['rows'][k]['group']==group]
                groups[group]=dict(n=len(ids),brier=float(brier[ids].mean()),log_loss=float(loss[ids].mean()),
                    correct=int(((x[ids]>0)==labels[ids].bool()).sum()))
            result[split][name]=dict(n=len(keys),brier=float(brier.mean()),log_loss=float(loss.mean()),
                correct=int(((x>0)==labels.bool()).sum()),groups=groups,
                equal_group_brier=sum(r['brier'] for r in groups.values())/len(groups))
    return result


def baselines(value):
    train=[r['outcome'] for r in value['rows'].values() if r['split']=='train']; rate=sum(train)/len(train)
    result={}
    for split in ('train','query'):
        rows=[r for r in value['rows'].values() if r['split']==split];y=torch.tensor([r['outcome'] for r in rows],dtype=torch.float64)
        result[split]={name:dict(n=len(rows),probability=p,brier=float(((y-p)**2).mean()),
            correct=sum((p>0.5)==bool(r['outcome']) for r in rows),
            groups={g:dict(n=sum(r['group']==g for r in rows),brier=sum((r['outcome']-p)**2 for r in rows if r['group']==g)/sum(r['group']==g for r in rows))
                    for g in sorted({r['group'] for r in rows})}) for name,p in (('always_failure',0.),('train_base_rate',rate))}
    return result


def controls(s,data):
    out={};v=s['value'];failed=next(k for k in data.train if data.rows[k]['outcome']==0)
    out['query_fit']=reject(lambda:data.batch([data.query[0]],'actual',fitting=True),'OUTCOME_FIT_QUERY')
    out['failed_as_success']=reject(lambda:s['supervisor'].training_target(failed,purpose='successful_patch'),'SUPERVISION_FAILED_ACTION')
    row=deepcopy(data.rows[failed]);data.rows[failed]['outcome']=1
    out['changed_label']=reject(data.fresh,'OUTCOME_CHANGED_BINDING');data.rows[failed]=deepcopy(row)
    data.rows[failed]['context'].reverse()
    out['changed_order']=reject(data.fresh,'OUTCOME_CHANGED_BINDING');data.rows[failed]=deepcopy(row)
    data.rows[failed]['context'].pop()
    out['missing_field']=reject(data.fresh,'OUTCOME_CHANGED_BINDING');data.rows[failed]=deepcopy(row)
    data.rows[failed]['context'][0]['role']='outcome'
    out['unknown_role']=reject(data.fresh,'OUTCOME_CHANGED_BINDING');data.rows[failed]=deepcopy(row)
    # Use a separate owned copy for a deliberate mutation, leaving the training input untouched.
    changed=OutcomeInputs(v['rows'],v['fields'],v['roster']);first=next(iter(changed.fields.values()))
    next(iter(first.values())).add_(1)
    out['native_mutation']=reject(changed.fresh,'OUTCOME_CHANGED_NATIVE');del changed
    caller=deepcopy(v['rows']);caller[failed]['outcome']=1;data.fresh();out['caller_owned_rows']=True
    model,opt=make(v['dimensions'],0);out['missing_optimizer_moments']=reject(lambda:validate(model,opt,1),'OUTCOME_OPTIMIZER_STATE')
    wrong,_=make(v['dimensions'],0);out['wrong_optimizer_owner']=reject(lambda:validate(wrong,opt,0),'OUTCOME_OPTIMIZER_OWNER')
    donor=next(k for k in data.train if data.rows[k]['outcome']==1)
    assessed=s['supervisor'].assess(failed,s['supervisor'].text[donor])
    need(assessed['requires_new_execution'] and assessed['execution'] is None and not assessed['accepted'],'OUTCOME_DONOR_TRANSFER')
    out['donor_execution_does_not_transfer']=True
    need(all(not r['generation']['allowed'] for r in data.rows.values() if r['outcome']==0),'OUTCOME_FAILURE_GENERATION')
    out['all_failed_repairs_excluded_from_positive_generation']=True
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    s=source.load(a.root);v=s['value'];data=OutcomeInputs(v['rows'],v['fields'],v['roster'])
    phases['source_native_authentication']=time.perf_counter()-tick;tick=time.perf_counter()
    write_committed(a.output/'before.json',dict(code=s['code'],inputs=s['inputs'],native=data.native,rows=data.seal,
        source_representation=v['source_representation'],native_banks=s['native_banks']))
    torch.save(v,a.output/'inputs.pt');write_committed(a.output/'routes.json',v['route'])
    negative=controls(s,data);write_committed(a.output/'controls.json',negative)
    alias=defaultdict(list)
    for key,r in data.rows.items():
        identity=digest([[(v['field_records'][f['field']]['text_sha256'],f['role']) for f in r['context']],r['answer_sha256']])
        alias[identity].append(key)
    conflicts=[dict(identity=i,tasks=ks,outcomes=[data.rows[k]['outcome'] for k in ks]) for i,ks in alias.items()
               if len({data.rows[k]['outcome'] for k in ks})>1]
    write_committed(a.output/'conflicts.json',conflicts)
    phases['persist_and_negative_controls']=time.perf_counter()-tick;tick=time.perf_counter()
    runs=[]
    for seed in range(3):
        initial=None
        for arm in ARMS:
            directory=a.output/(str(seed)+'-'+arm);directory.mkdir();model,opt=make(v['dimensions'],seed)
            start=fingerprint(model.state_dict());need(initial is None or initial==start,'OUTCOME_INITIAL_MATCH');initial=start
            torch.save(cpu(model.state_dict()),directory/'initial.pt');before=predict(model,data,arm);ledger=[]
            run_start=time.perf_counter()
            for step in range(20):
                ledger.append(update(model,opt,data,arm,step))
                if step==9:torch.save(cpu(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=10)),directory/'mid.pt')
            after=predict(model,data,arm)
            torch.save(dict(before=before,after=after,ledger=ledger),directory/'trace.pt')
            torch.save(cpu(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=20)),directory/'final.pt')
            record=dict(seed=seed,arm=arm,initial=start,final=fingerprint(model.state_dict()),
                before=metrics(v,before),after=metrics(v,after),seconds=time.perf_counter()-run_start,
                train_updates=20,train_presentations=20*len(data.train))
            runs.append(record);write_committed(directory/'result.json',record)
            print(json.dumps(dict(seed=seed,arm=arm,query_brier=record['after']['query']['combined']['brier'],seconds=record['seconds'])),flush=True)
    phases['training_and_predictions']=time.perf_counter()-tick;tick=time.perf_counter()
    data.fresh(full=True);s['fresh']();phases['final_readback']=time.perf_counter()-tick
    elapsed=time.perf_counter()-began;need(.9<=sum(phases.values())/elapsed<=1.001,'OUTCOME_PHASES')
    report=dict(complete=True,schema='complete-answer-outcome-probe-v1',code=s['code'],inputs=s['inputs'],
        files={str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()},
        runs=runs,baselines=baselines(v),controls=negative,conflicts=conflicts,
        counts=dict(train=len(data.train),query=len(data.query),unique_fields=len(data.fields),
                    windows=sum(len(x) for values in data.fields.values() for x in values.values()),native_bytes=s['bytes'],
                    routes=len(v['route']),excluded_prior_fields=len(v['excluded']),updates=240),
        phases=phases,seconds=elapsed,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu_threads=1,gpu_used=False,frozen_encoder_calls=0,new_executions=0,
        complete_answer_generator_trained=False,queued_objective_changed=False,serving=False,
        scientific_admission=False,scope='retained execution outcomes; 2 QUERY groups; development exposed',acceptance='author-checked')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=elapsed)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
