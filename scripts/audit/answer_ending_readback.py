"""Cold causal histories, autograd readout calculus and scalar ending metrics."""
import argparse
from copy import deepcopy
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
from audit import answer_ending_diagnostics as source
from audit.answer_surface_stream import rejected
from audit.breadth_answer_readback import check_ages
from audit.full_history_readback import same
from claims.process_conditions import write_committed
from model.native_population_training import cpu,fingerprint
from text.constellation_corpus import file_sha,need


def make(s,key):
    _,m=source.source.source.models(s,0 if key.startswith('initial/') else 91540);value=s['states'][key]
    m.load_state_dict(value['model'],strict=True);o=torch.optim.AdamW(m.parameters(),lr=s['policy']['learning_rate'],
        weight_decay=s['policy']['weight_decay'],foreach=False,fused=False)
    o.load_state_dict(deepcopy(value['optimizer']));check_ages(m,o,value['binding'],value['cursor']);m.eval()
    return m,o


@torch.no_grad()
def history(s,m,key,row):
    item,request=s['items_ready'][row['document']];c,a=len(item['context']),len(item['answers'])
    stream=m.stream(request);cache=None;previous=None;hs=[];scores=[];inputs=[]
    rng=torch.Generator().manual_seed(20261024+item['document'])
    for pos in range(c+a):
        if pos<c:frame=item['context'][pos]
        elif row['history']=='reference':frame=item['answers'][pos-c]
        else:frame=previous
        h,cache=stream.step((frame,),context='cold-end-history',offset=pos,state=cache,detach_history=True)
        if pos<c-1:continue
        index=pos-c+1
        logits=torch.stack([m.ending['assembled'][n](h[:,i]) for i,n in enumerate(m.roster)]).mean(0)[0]
        hs.append(cpu(h[0]));scores.append(cpu(logits));inputs.append(fingerprint(frame))
        if row['history']=='generated' and index<a:
            forecast=m.forecaster.predictor.forecast(h);noise=torch.randn(1,1,s['policy']['noise_width'],generator=rng)
            value=m.forecaster.placement['assembled'](h,forecast['codes'],forecast['counts'],noise,target=s['target'])[0]
            previous={n:t[0].detach() for n,t in value.items()}
    hidden=torch.stack(hs);end=torch.tensor([False]*a+[True])
    actual=dict(document=item['document'],side=item['side'],history=row['history'],binding=item['binding'],
        source_native=item['native_pin'],source_labels=item['label_pin'],checkpoint=fingerprint(s['states'][key]),
        hidden={n:hidden[:,i,:].clone() for i,n in enumerate(m.roster)},stop=torch.stack(scores),end=end,input_pins=inputs)
    actual['pin']=source.probe.seal(actual);same(actual,row,'END_COLD_HISTORY');return actual


def fit_readback(rows,saved):
    need(all(r['side']=='train' and r['history']=='reference' for r in rows),'END_COLD_TRAIN')
    roster=tuple(rows[0]['hidden']);width=rows[0]['hidden'][roster[0]].shape[1];x={n:torch.cat([r['hidden'][n] for r in rows]).double() for n in roster}
    ys=[];ws=[];positions={}
    for row in rows:
        length=len(row['end']);position=length-1
        if saved['shuffled']:position=int(np.random.default_rng(20261024+row['document']).integers(length-1))
        y=torch.zeros(length,dtype=torch.float64);y[position]=1;ys.append(y);ws.append(torch.full((length,),1./(len(rows)*length),dtype=torch.float64))
        positions[row['document']]=position
    y=torch.cat(ys);w=torch.cat(ws);d=len(roster)*width+1;theta=np.zeros(d)
    prior=float(w.numpy()@y.numpy());need(abs(prior-float(w@y))<=1e-15,'END_COLD_PRIOR');theta[-1]=math.log(prior/(1-prior))
    need(saved['source']=={r['document']:r['pin'] for r in rows} and saved['label_positions']==positions
         and saved['ridge']==.001 and saved['tolerance']==1e-9 and saved['roster']==roster and saved['width']==width,'END_COLD_FIT_BINDING')
    def objective(t):
        z=sum(torch.mv(v,t[i*width:(i+1)*width])/len(roster) for i,v in enumerate(x.values()))+t[-1]
        return torch.dot(w,torch.nn.functional.softplus(z)-y*z)+.0005*torch.dot(t[:-1],t[:-1])
    canonical={n:v.numpy() for n,v in x.items()};maximum=0.
    for iteration,entry in enumerate(saved['trace']):
        need(np.array_equal(theta,entry['theta'].numpy()),'END_COLD_ITERATE')
        t=torch.from_numpy(theta.copy()).requires_grad_(True);loss=objective(t)
        derivative=torch.autograd.grad(loss,t)[0];curvature=torch.autograd.functional.hessian(objective,t,vectorize=True)
        obj,g,h=source.probe.statistics(theta,canonical,y.numpy(),w.numpy())
        error=max(abs(float(loss.detach())-obj),float(np.max(np.abs(derivative.numpy()-g))),float(np.max(np.abs(curvature.numpy()-h))))
        need(error<=1e-12,'END_COLD_AUTOGRAD_CALCULUS');maximum=max(maximum,error)
        need(obj==entry['objective'] and np.array_equal(g,entry['gradient'].numpy()) and fingerprint(torch.from_numpy(h))==entry['curvature'],
             'END_COLD_TRACE')
        if np.max(np.abs(g))<=1e-9:
            need(iteration==len(saved['trace'])-1 and entry['step']==0.,'END_COLD_CONVERGENCE');break
        direction=-np.linalg.solve(h,g);slope=float(g@direction);need(slope<0 and np.isfinite(direction).all(),'END_COLD_DIRECTION');step=1.
        while source.probe.statistics(theta+step*direction,canonical,y.numpy(),w.numpy())[0]>obj+1e-4*step*slope:
            step*=.5;need(step>=np.finfo(np.float64).eps,'END_COLD_LINE_SEARCH')
        need(entry['step']==step,'END_COLD_STEP');theta=theta+step*direction
    else:raise ValueError('END_COLD_NO_CONVERGENCE')
    need(np.array_equal(theta,saved['theta'].numpy()),'END_COLD_FIT_RESULT')
    return maximum


def scores(train,row,saved):
    roster=tuple(row['hidden']);length=len(row['end']);prior=sum(1./len(t['end']) for t in train)/len(train)
    base=saved['base'];need(base['constant']==prior and base['lengths']==[len(t['end'])-1 for t in train],'END_COLD_BASE')
    result=dict(current=(row['stop'][:,1].double()-row['stop'][:,0].double()).tolist(),constant=[math.log(prior/(1-prior))]*length,
        length=[],centroid=[0.]*length)
    for i in range(length):
        risk=sum(len(t['end'])-1>=i for t in train);events=sum(len(t['end'])-1==i for t in train)
        result['length'].append(math.log((events+1)/(risk-events+1)))
    for n in roster:
        end=torch.stack([t['hidden'][n][-1] for t in train]).double().mean(0)
        continuation=torch.cat([t['hidden'][n][:-1] for t in train]).double().mean(0)
        need(torch.allclose(end,torch.tensor(base['centroids'][n]['end'],dtype=torch.float64),atol=1e-12,rtol=0)
             and torch.allclose(continuation,torch.tensor(base['centroids'][n]['continuation'],dtype=torch.float64),atol=1e-12,rtol=0),'END_COLD_CENTROIDS')
        for i,v in enumerate(row['hidden'][n].double()):
            delta=float(torch.dot(v,end)/(v.norm()*end.norm())-torch.dot(v,continuation)/(v.norm()*continuation.norm()))
            result['centroid'][i]+=delta/len(roster)
    for method,field in [('linear','linear'),('linear_shuffled','shuffled')]:
        theta=saved[field]['theta'];width=saved[field]['width'];z=torch.full((length,),float(theta[-1]),dtype=torch.float64)
        for i,n in enumerate(roster):z+=torch.mv(row['hidden'][n].double(),theta[i*width:(i+1)*width])/len(roster)
        result[method]=z.tolist()
    return result


def metric(rows,values):
    answer={}
    for method in values[0]:
        documents=[];brier=nll=0.;pos=[];neg=[];correct=early=censored=tp=fp=tn=fn=0
        for row,value in zip(rows,values):
            zs=value[method];length=len(row['end']);first=next((i for i,z in enumerate(zs) if z>0),None);bs=ls=0.
            for i,z in enumerate(zs):
                y=int(i==length-1);p=math.exp(-(max(-z,0)+math.log1p(math.exp(-abs(z)))))
                bs+=(p-y)**2;ls+=max(z,0)+math.log1p(math.exp(-abs(z)))-y*z
                (pos if y else neg).append(z)
                if z>0:
                    if y:tp+=1
                    else:fp+=1
                elif y:fn+=1
                else:tn+=1
            bs/=length;ls/=length;brier+=bs/len(rows);nll+=ls/len(rows)
            correct+=first==length-1;early+=first is not None and first<length-1;censored+=first is None
            documents.append(dict(document=row['document'],first_end=first,expected_end=length-1,brier=bs,log_loss=ls))
        wins=sum(1. if a>b else .5 if a==b else 0. for a in pos for b in neg)
        answer[method]=dict(documents=documents,brier=brier,log_loss=nll,auroc=wins/(len(pos)*len(neg)),
            correct_end=correct,early_end=early,censored=censored,tp=tp,fp=fp,tn=tn,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn))
    return answer


def compare(a,b):
    if isinstance(a,dict):need(a.keys()==b.keys(),'END_COLD_METRIC_KEYS');return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,list):need(len(a)==len(b),'END_COLD_METRIC_LENGTH');return max([compare(x,y) for x,y in zip(a,b)] or [0.])
    if isinstance(a,float):need(abs(a-b)<=1e-12,'END_COLD_METRIC_FLOAT');return abs(a-b)
    need(a==b,'END_COLD_METRIC_VALUE');return 0.


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};results={};decisions=fits=0;maximum=0.
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'END_COLD_REPORT_PIN');report=json.loads((a.input/'report.json').read_bytes());need(report['complete'],'END_COLD_REPORT')
    for name,pin in report['files'].items():need(file_sha(a.input/name)==pin,'END_COLD_BYTES')
    s=source.load(a.root);need(s['code']==report['code'],'END_COLD_CODE');index=json.loads((a.input/'index.json').read_bytes());expected=json.loads((a.input/'results.json').read_bytes())
    phases['authentication']=time.perf_counter()-tick
    for entry in index:
        tick=time.perf_counter();key=entry['state'];m,o=make(s,key);before=fingerprint((m.state_dict(),o.state_dict()))
        need(entry['checkpoint']==fingerprint(s['states'][key]),'END_COLD_CHECKPOINT');saved=torch.load(a.input/entry['file'],map_location='cpu',weights_only=True)
        rows=[history(s,m,key,r) for r in saved['rows']];decisions+=sum(len(r['end']) for r in rows);train=[r for r in rows if r['history']=='reference' and r['side']=='train']
        for field in ('linear','shuffled'):maximum=max(maximum,fit_readback(train,saved[field]));fits+=1
        readings=[scores(train,r,saved) for r in rows]
        for read,ref in zip(readings,saved['scores']):maximum=max(maximum,compare(read,{n:v.tolist() for n,v in ref.items()}))
        results[key]={}
        for kind in ('reference','generated'):
            for side in ('train','development','query'):
                chosen=[i for i,r in enumerate(rows) if r['history']==kind and r['side']==side]
                value=metric([rows[i] for i in chosen],[readings[i] for i in chosen]);maximum=max(maximum,compare(value,expected[key][kind+'/'+side]));results[key][kind+'/'+side]=value
        need(fingerprint((m.state_dict(),o.state_dict()))==before,'END_COLD_STATE_MUTATED');phases[key]=time.perf_counter()-tick
        print(json.dumps(dict(state=key,decisions=decisions)),flush=True)
    tick=time.perf_counter();negative={}
    for field in ('theta','source'):
        wrong=deepcopy(saved['linear'])
        if field=='theta':wrong['theta'][0]+=1
        else:wrong['source'][next(iter(wrong['source']))]='0'*64
        code='END_COLD_FIT_RESULT' if field=='theta' else 'END_COLD_FIT_BINDING'
        negative[field]=rejected(lambda:fit_readback(train,wrong),code)
    wrong=deepcopy(entry);wrong['checkpoint']='0'*64
    negative['checkpoint']=rejected(lambda:need(wrong['checkpoint']==fingerprint(s['states'][key]),'END_COLD_CHECKPOINT'),'END_COLD_CHECKPOINT')
    s['fresh']();need((decisions,fits)==(3200,16),'END_COLD_CENSUS');write_committed(a.output/'results.json',results)
    phases['final_controls']=time.perf_counter()-tick;seconds=time.perf_counter()-began;need(.9<=sum(phases.values())/seconds<=1.001,'END_COLD_PHASES')
    result=dict(complete=True,acceptance='author-checked',source=dict(path=str(a.input),pin=a.pin),code=s['code'],decisions=decisions,fits=fits,
        all_histories_exact=True,all_fits_exact=True,independent_autograd_and_scalar_metrics=True,maximum_numeric_error=maximum,
        model_optimizer_unchanged=True,controls=negative,results_pin=file_sha(a.output/'results.json'),phases=phases,seconds=seconds,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,promoted=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.source.source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
