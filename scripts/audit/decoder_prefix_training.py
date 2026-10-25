"""Paired fixed decoder-prefix intervention on a previously sealed real cohort."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit.decoder_prefix_cohort import code,save
from audit.surface_condition_inputs import measured_native
from audit.trained_answer_render import observe
from claims.process_conditions import write_committed
from model import decoder_prefix_mask as masking
from model import surface_inversion_training as inversion
from model.answer_surface_stream import render_atom
from model.contracts import ModelConfig,Instrument
from model.legal_surface_inputs import SurfaceExample
from model.native_population_training import cpu,fingerprint
from model.predicted_surface_training import continuation
from model.surface_device_runtime import batches
from senses.contract import digest,SenseError
from text.constellation_corpus import file_sha,need
from text.constellation_input import stamp

COHORT='decoder-prefix-cohort-v1-20261025'
PIN='c5d592a6b5d462e9a906351e8db3e96fbba351ff003af0b9bf7a2d50f16f1bfe'
ARMS=('initial','ordinary','masked')


def load(root):
    directory=root/COHORT;path=directory/'report.json';need(file_sha(path)==PIN,'PREFIX_COHORT_PIN')
    r=json.loads(path.read_bytes());stamps={path:stamp(path)};loaded={}
    for name,ref in r['files'].items():
        path=directory/name;stamps[path]=stamp(path);need(file_sha(path)==ref['sha256'],'PREFIX_INPUT_PIN')
        loaded[name]=torch.load(path,map_location='cpu',weights_only=True)
        need(fingerprint(loaded[name])==ref['state'],'PREFIX_INPUT_STATE')
    config=ModelConfig(**r['config']);specs=tuple(Instrument(**s).validate() for s in r['instruments'])
    need(r['complete'] and r['no_forward'] and r['initial_step']==51150 and r['missing_shape_matched_donors']==0,
         'PREFIX_COHORT_SCOPE')
    rows=loaded['observations.pt'];initial=loaded['initial.pt']
    model,opt=continuation(config,specs,initial,'cpu',step=51150)
    examples=[SurfaceExample(x['occurrence'],x['uid'],x['side'],bytes(x['raw'].tolist()),measured_native(x,model)) for x in rows]
    train=[e for e in examples if e.side=='train'];evals=[e for e in examples if e.side!='train']
    trainrows=[r for r in rows if r['side']=='train'];evrows=[r for r in rows if r['side']!='train'];donors=[]
    need(len(train)==1024 and len(evals)==128 and len(rows)==1152,'PREFIX_COHORT_COUNTS')
    for row,index in zip(rows,r['donor_train_indices']):
        d=trainrows[index]
        need(d['component']!=row['component'] and d['uid']!=row['uid']
             and {n:tuple(v.shape) for n,v in d['native'].items()}=={n:tuple(v.shape) for n,v in row['native'].items()},
             'PREFIX_DONOR_COMPONENT_SHAPE')
        if row['side']!='train':donors.append(train[index])
    def fresh():
        need(all(stamp(p)==s for p,s in stamps.items()),'PREFIX_INPUT_CHANGED')
        need(fingerprint(initial)==loaded_initial,'PREFIX_INITIAL_CHANGED')
    loaded_initial=fingerprint(initial);fresh()
    return dict(report=r,rows=rows,config=config,specs=specs,initial=initial,train=train,evals=evals,
                evalrows=evrows,donors=donors,fresh=fresh)


def new(s):return continuation(s['config'],s['specs'],s['initial'],'cpu',step=51150)


def rejection(fn,code):
    try:fn()
    except (ValueError,RuntimeError,SenseError) as e:
        need(code in str(e),'PREFIX_WRONG_REFUSAL');return str(e)
    raise ValueError('MNLM_PREFIX_MISSING_REFUSAL')


def qualify(s):
    left,lo=new(s);right,ro=new(s);examples=s['train'][:128]
    a,_=inversion.update(left,lo,examples,51150)
    b,_,payload=masking.update(right,ro,examples,51150,rate=0.)
    need(all(a[k]==b[k] for k in b),'PREFIX_ZERO_UPDATE_CHANGED')
    need(fingerprint(left.state_dict())==fingerprint(right.state_dict())
         and fingerprint(lo.state_dict())==fingerprint(ro.state_dict()),'PREFIX_ZERO_STATE_CHANGED')
    batch=next(batches(examples[:2],right.config,right.specs,'gathered',128))
    rng=torch.random.get_rng_state().clone();mask=masking.mask_for(batch,51150,.4)
    need(torch.equal(rng,torch.random.get_rng_state()),'PREFIX_MASK_RNG')
    need(torch.equal(mask,masking.mask_for(batch,51150,.4)) and bool(mask.any()),'PREFIX_MASK_DETERMINISM')
    controls={}
    controls['nontrain']=rejection(lambda:masking.update(right,ro,s['evals'][:1],51151,rate=.4),'PREFIX_MASK_NONTRAIN')
    controls['bad_rate']=rejection(lambda:masking.mask_for(batch,51151,.41),'PREFIX_MASK_POLICY')
    controls['wrong_layout']=rejection(lambda:masking.forward(right,batch,mask.float()),'PREFIX_MASK_LAYOUT')
    bad=mask.clone();bad[0,0]=True
    controls['bos_mask']=rejection(lambda:masking.forward(right,batch,bad),'PREFIX_MASK_NONBYTE_OR_PADDING')
    right.eval();controls['eval_mask']=rejection(lambda:masking.forward(right,batch,mask),'PREFIX_MASK_TRAIN_ONLY')
    right.train();badbatch=replace(batch,native={})
    # Forward fails after the scoped hook is installed; cleanup must still run.
    try:masking.forward(right,badbatch,mask)
    except SenseError as e:
        need('MNLM_MODEL_ROSTER' in str(e),'PREFIX_WRONG_NATIVE_REFUSAL');controls['missing_native']=str(e)
    else:raise ValueError('PREFIX_MISSING_NATIVE_ACCEPTED')
    need(not right.tokens._forward_hooks,'PREFIX_FAILED_HOOK_LEAK')
    bad=deepcopy(s['initial']);bad['optimizer']['state']={}
    controls['missing_moments']=rejection(lambda:continuation(s['config'],s['specs'],bad,'cpu',step=51150),'INVERSION_MOMENT_COVERAGE')
    return dict(zero_update=a,controls=controls,zero_state_exact=True,mask_rng_unchanged=True,
                hook_cleanup=True,zero_payload=fingerprint(payload))


def free_summary(values,rows):
    result={}
    for side in ('development','query'):
        for presence in (None,False,True):
            selected=[(v,r) for v,r in zip(values,rows) if r['side']==side and (presence is None or r['train_present']==presence)]
            counts=dict(atoms=len(selected),ended=0,censored=0,refused=0,exact=0,fits=0,exact_fits=0)
            for v,r in selected:
                raw=bytes(r['raw'].tolist());events=v['events'];end=bool(events and events[-1]['kind']=='atom_end')
                output=bytes(e['token'] for e in events if 'token' in e and e['token']<256)
                exact=end and output==raw;fits=len(raw)+1<=96
                counts['ended']+=int(end);counts['censored']+=int(v['censored']);counts['exact']+=int(exact)
                counts['fits']+=int(fits);counts['exact_fits']+=int(exact and fits)
                counts['refused']+=int(bool(events and events[-1]['kind']=='refuse'))
            result[side+('/all' if presence is None else '/train_present' if presence else '/train_absent')]=counts
    return result


def summarize(paired,rows):
    result={}
    for side in ('development','query'):
        indices=[i for i,r in enumerate(rows) if r['side']==side];positions=sum(len(rows[i]['raw'])+1 for i in indices)
        totals={k:sum(float(paired[k][i]['losses'].double().sum()) for i in indices) for k in paired}
        result[side]=dict(positions=positions,correct_nll=totals['correct']/positions,
            shuffled_nll=totals['shuffled']/positions,margin=(totals['shuffled']-totals['correct'])/positions,
            correct_better=sum(float(paired['correct'][i]['losses'].double().sum())<float(paired['shuffled'][i]['losses'].double().sum()) for i in indices),
            atoms=len(indices))
    return result


def evaluate(model,s,label):
    paired,profile=inversion.paired(model,s['evals'],s['donors']);free=[]
    model.eval()
    for i,e in enumerate(s['evals']):
        free.append(observe(render_atom(model,e.native,capture_logits=True),96))
        if (i+1)%32==0:print(json.dumps(dict(arm=label,free_atoms=i+1)),flush=True)
    need(not model.tokens._forward_hooks,'PREFIX_EVALUATION_HOOK')
    return dict(paired=paired,free=free),dict(teacher=summarize(paired,s['evalrows']),free=free_summary(free,s['evalrows'])),profile


def run(a):
    began=tick=time.perf_counter();c0=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    s=load(a.root);qualified=qualify(s);phases=dict(inputs_and_qualification=time.perf_counter()-tick)
    files={};summary={};records={}
    for arm in ARMS:
        tick=time.perf_counter();model,opt=new(s);rows=[];refs=[];ledger=digest(dict(cohort=PIN,arm=arm));binding=dict(cohort=PIN,arm=arm)
        if arm!='initial':
            for number,start in enumerate(range(0,len(s['train']),128)):
                examples=s['train'][start:start+128];step=51150+number
                if arm=='ordinary':row,profile=inversion.update(model,opt,examples,step);payload={}
                else:row,profile,payload=masking.update(model,opt,examples,step,rate=.4)
                gradients=cpu({n:p.grad for n,p in model.named_parameters()})
                after,moments=inversion.check_state(model,opt,step+1)
                ledger=digest(dict(parent=ledger,update=row))
                snapshot=dict(row=row,after=after,optimizer=moments,gradients=gradients,payload=payload,ledger=ledger)
                name=f'{arm}-step-{number+1}.pt';files[name]=save(a.output/name,snapshot)
                rows.append(dict(row=row,profile=profile,path=name,ledger=ledger))
                if number in (3,7):
                    ref=inversion.checkpoint(a.output/f'{arm}-checkpoint-{number+1}.pt',model,opt,binding,start+128,step+1,ledger)
                    refs.append(ref)
                    if number==3:
                        model,opt=new(s);inversion.restore(ref,model,opt,binding,size=len(s['train']))
                print(json.dumps(dict(arm=arm,step=step+1,positions=row['positions'])),flush=True)
        phases[arm+'_training']=time.perf_counter()-tick;tick=time.perf_counter()
        result,summary[arm],profile=evaluate(model,s,arm)
        name=arm+'-evaluation.pt';files[name]=save(a.output/name,result)
        records[arm]=dict(updates=rows,checkpoints=refs,evaluation_profile=profile)
        phases[arm+'_evaluation']=time.perf_counter()-tick
    s['fresh']();need(not torch.cuda.is_initialized(),'PREFIX_CPU_ONLY')
    r=dict(complete=True,cohort=PIN,files=files,code=code(),qualification=qualified,records=records,summary=summary,
        phases=phases,wall=time.perf_counter()-began,cpu=time.process_time()-c0,
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,device='cpu',explicit_cpu_engineering=True,
        policy=dict(mask=.4,epochs=1,atoms=128,steps=8,initial_step=51150,observer=96,training_context='true_native'),
        acceptance='author-checked',scientific_admission=False,untouched_test=False,promoted=False)
    pin=write_committed(a.output/'report.json',r);print(json.dumps(dict(pin=pin,summary=summary)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.is_dir():write_committed(a.output/'failure.json',dict(error=type(e).__name__,message=str(e)))
        raise
