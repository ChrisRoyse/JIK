"""Cold optimizer/trajectory replay plus independent count-law and tail-baseline checks."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit import open_window_answer as producer
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from model.native_request_input import NativeRequest,legal
from model.native_requested_answer import update
from model.native_placement_checked import validate_optimizer
from text.constellation_corpus import file_sha,need


def count_math(head,hidden,count):
    h=hidden.detach().clone().requires_grad_();counts=torch.tensor([count],dtype=torch.int64)
    result=head.loss(h,counts);parameters=(h,*head.parameters())
    actual=torch.autograd.grad(result['loss'],parameters)
    positions=torch.arange(1,count+1);pieces=[]
    for lo in range(0,count,head.TILE):pieces.append(head.logits(h,positions[lo:lo+head.TILE]))
    logits=torch.cat(pieces,1);values=logits.detach().double().numpy()[0]
    # Independent probability product in log space, with one terminal observation.
    reference=sum(float(np.logaddexp(0,z)) for z in values[:-1])+float(np.logaddexp(0,-values[-1]))
    loss_error=abs(float(result['loss'].detach())-reference)
    need(loss_error<=1e-12,'WINDOW_READBACK_LIKELIHOOD')
    derivative=logits.detach().double().sigmoid();derivative[:,-1]-=1
    expected=torch.autograd.grad(logits,parameters,grad_outputs=derivative.to(logits.dtype))
    errors=[float((a-b).abs().max()) for a,b in zip(actual,expected)]
    scale=max(1.,*(float(v.abs().max()) for v in expected))
    need(max(errors)<=8*torch.finfo(torch.float32).eps*scale,'WINDOW_READBACK_LIKELIHOOD_GRADIENT')
    # Independently enumerate the admitted physical range, then bound every
    # unenumerated probability by the remaining survival mass.
    survival=0.;best=-math.inf;chosen=None;ordinal=1
    with torch.no_grad():
        for lo in range(1,head.limit+1,head.TILE):
            p=torch.arange(lo,min(lo+head.TILE,head.limit+1))
            for score in head.logits(hidden,p).double().numpy()[0]:
                probability=survival-float(np.logaddexp(0,-score))
                if probability>best:best=probability;chosen=ordinal
                survival-=float(np.logaddexp(0,score));ordinal+=1
        forecast=head.forecast(hidden)
    need(survival<=best and int(forecast['counts'][0])==chosen,'WINDOW_READBACK_GLOBAL_MODE')
    return dict(count=count,nll=reference,nll_error=loss_error,max_gradient_error=max(errors),
                gradient_bound=8*torch.finfo(torch.float32).eps*scale,mode=chosen,remaining_log_mass=survival)


def compare(rows,roster):
    result={}
    for mode in ('correct','wrong_operation'):
        for proposal in sorted({r['proposal'] for r in rows}):
            selected=[r for r in rows if r['mode']==mode and r['proposal']==proposal]
            entry=dict(tasks=len(selected),new_ended=sum(r['metrics']['ended'] for r in selected),
                parent_ended=sum(r['parent_metrics']['ended'] for r in selected),
                new_exact_end=sum(r['metrics']['ended_at_reference_boundary'] for r in selected),
                parent_exact_end=sum(r['parent_metrics']['ended_at_reference_boundary'] for r in selected),
                refusals=sum(r['refusal'] is not None for r in selected),per_slot={})
            for n in roster:
                a=[v for r in selected for v in r['metrics']['per_slot'][n]]
                b=[v for r in selected for v in r['parent_metrics']['per_slot'][n]]
                need(len(a)==len(b) and a,'WINDOW_READBACK_COMPARISON_DENOMINATOR')
                entry['per_slot'][n]=dict(atoms=len(a),new=sum(a)/len(a),parent=sum(b)/len(b),delta=(sum(a)-sum(b))/len(a))
            result[mode+':'+proposal]=entry
    return result


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        r=producer.sequence.runtime.pinned(a.input/'report.json',a.pin)
        need(r['complete'],'WINDOW_READBACK_INCOMPLETE')
        for n,h in r['files'].items():need(file_sha(a.input/n)==h,'WINDOW_READBACK_PAYLOAD')
        s,b,data,case=producer.source(a);need(producer.code(s)==r['code'],'WINDOW_READBACK_CODE')
        target,pin=producer.extension(b);need(pin==r['extension_pin'],'WINDOW_READBACK_EXTENSION')
        old=data['labels'].model;frozen=json.loads((a.input/'frozen.json').read_bytes())
        need(fingerprint(target.state_dict())==fingerprint(old.state_dict())==frozen['state'],'WINDOW_READBACK_FROZEN')
        # Direct baseline formula, independent of the extension's implementation.
        tails={}
        for i,n in enumerate(target.roster):
            size=case['counts'][n][0];expected=torch.empty(size,b['dimensions'][n]);prior=getattr(old,'position_'+str(i))
            expected[:len(prior)]=prior;expected[len(prior):]=getattr(old,'center_'+str(i))
            actual=target.baseline(i,size)
            need(torch.equal(actual,expected),'WINDOW_READBACK_TRAIN_MEAN_TAIL');tails[n]=fingerprint(actual)
        phases['source_and_tail_recount']=time.perf_counter()-tick
        train=[d for d in data['items'] if d['side']=='train'];held={d['document']:d for d in data['items'] if d['side']!='train'}
        results={};comparisons={}
        for architecture in producer.sequence.runtime.original.ARCHITECTURES:
            tick=time.perf_counter();base=a.input/architecture;model,opt=producer.make(b,target,pin,architecture)
            initial=torch.load(base/'initial.pt',map_location='cpu',weights_only=True)
            need(fingerprint(dict(model=model.state_dict(),optimizer=opt.state_dict()))==fingerprint(initial),'WINDOW_READBACK_INITIAL')
            ledger=[];resumed=False
            for step,d in enumerate(train):
                item,request=producer.old.item(d)
                record,grad=update(model,opt,item,step,dict(b['policy'],device='cpu'),request=request,target=target)
                expected=torch.load(base/f'update-{step+1}.pt',map_location='cpu',weights_only=True)
                actual=dict(record=record,gradients=grad,model=model.state_dict(),optimizer=opt.state_dict())
                need(fingerprint(actual)==fingerprint(expected),'WINDOW_READBACK_UPDATE')
                ledger.append(dict(step=step+1,state=fingerprint(actual)))
                if step==1:
                    model,opt=producer.make(b,target,pin,architecture)
                    model.load_state_dict(expected['model'],strict=True);opt.load_state_dict(expected['optimizer'])
                    validate_optimizer(model,opt,dict(b['policy'],device='cpu'),step+1);resumed=True
            capacity=torch.load(base/'capacity.pt',map_location='cpu',weights_only=True)
            need(fingerprint(producer.capacity(model,target,b,train[0],case))==fingerprint(capacity),'WINDOW_READBACK_CAPACITY')
            math_rows={}
            for proposal,heads in producer.count_heads(model).items():
                row=capacity['proposals'][proposal]
                math_rows[proposal]={n:count_math(head,row['hidden'][:,i],int(row['counts'][n][0]))
                                     for i,(n,head) in enumerate(heads.items())}
            rows=torch.load(base/'evaluation.pt',map_location='cpu',weights_only=True)
            parent=torch.load(a.root/producer.BASE[0]/architecture/'evaluation.pt',map_location='cpu',weights_only=True)
            parent_index={(v['document'],v['mode'],v['proposal']):v for v in parent};trajectory_pins=[]
            for row in rows:
                d=held[row['document']];desc=legal(2)
                if row['mode']=='wrong_operation':desc['operation']='respond_to_source_question'
                result=producer.observe(model,target,tuple(d['frames'][:2]),row['proposal'],20261024+d['document'],
                                        len(d['frames'])-1,NativeRequest(desc))
                need(fingerprint(result)==fingerprint(dict(events=row['events'],refusal=row['refusal'])),'WINDOW_READBACK_TRAJECTORY')
                metric=producer.measured(result,d['frames'][2:],model.roster,d['frames'][1])
                need(metric==row['metrics'] and row['parent_metrics']==parent_index[(row['document'],row['mode'],row['proposal'])]['metrics'],
                     'WINDOW_READBACK_SCORE')
                trajectory_pins.append(fingerprint(result))
            comparison=compare(rows,model.roster);comparisons[architecture]=comparison
            write_committed(a.output/(architecture+'.json'),dict(ledger=ledger,trajectories=trajectory_pins,count_math=math_rows,comparison=comparison))
            results[architecture]=dict(updates=len(ledger),midpoint_restore_exact=resumed,rollouts=len(rows),
                all_cold_states_exact=True,all_trajectories_exact=True,capacity_shapes_exact=True,
                max_nll_error=max(v['nll_error'] for group in math_rows.values() for v in group.values()),
                max_gradient_error=max(v['max_gradient_error'] for group in math_rows.values() for v in group.values()))
            phases[architecture]=time.perf_counter()-tick;print(json.dumps(dict(architecture=architecture,**results[architecture])),flush=True)
        tick=time.perf_counter();data['close']();need(producer.code(s)==r['code'] and not torch.cuda.is_initialized(),'WINDOW_READBACK_FINAL')
        write_committed(a.output/'comparisons.json',comparisons);phases['final_authentication']=time.perf_counter()-tick
        elapsed=time.perf_counter()-began;need(.9<=sum(phases.values())/elapsed<=1.001,'WINDOW_READBACK_PHASES')
        result=dict(complete=True,source=dict(path=str(a.input),pin=a.pin),results=results,code=r['code'],tails=tails,
            phases=phases,seconds=elapsed,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            scientific_admission=False,capability_demonstrated=False,acceptance='author-checked')
        h=write_committed(a.output/'report.json',result);print(json.dumps(dict(pin=h,seconds=elapsed)),flush=True)
    except BaseException as e:write_committed(a.output/'failure.json',dict(error=repr(e),phases=phases));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=producer.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
