"""Cold causal-state, optimizer and autonomous-answer reconstruction."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from torch.nn import functional as F
from audit import ending_fit_source as source
from audit.full_history_readback import autonomous
from audit.answer_surface_stream import rejected
from claims.process_conditions import write_committed
from model import answer_ending_fit as learner
from model.native_population_training import cpu,fingerprint
from model.native_request_input import NativeRequest,legal
from text.constellation_corpus import file_sha,need


def same(a,b,code):need(fingerprint(a)==fingerprint(b),code)


def contexts(s,model,saved):
    rows=[];request=NativeRequest(legal(2));model.eval()
    with torch.no_grad():
        for doc in sorted(s['items']):
            d=s['items'][doc];cache=None;stream=model.stream(request)
            for pos in range(len(d['frames'])):
                h,cache=stream.step((d['frames'][pos],),context='cold-ending:'+str(doc),offset=pos,state=cache,detach_history=True)
                if pos==0:continue
                projected=model.forecaster.contexts(h)
                rows.append(dict(document=doc,side=d['side'],component=d['component'],position=pos,
                    end=pos==len(d['frames'])-1,native=d['native_pin'],request=request.pin,
                    states={p:{n:cpu(v[0,i]) for i,n in enumerate(model.roster)} for p,(_,v) in projected.items()},
                    original_logits={p:cpu(model.stop_logits(v,p)[0]) for p,(_,v) in projected.items()}))
    same(rows,saved,'ENDING_COLD_CAUSAL_STATES')
    return rows


def fit_arrays(rows,model,mode,shuffled,saved):
    rows=[r for r in rows if r['side']=='train'];groups={}
    for i,r in enumerate(rows):groups.setdefault(r['document'],[]).append(i)
    labels=torch.tensor([int(r['end']) for r in rows],dtype=torch.long);weights=torch.empty(len(rows),dtype=torch.float64)
    for doc,ids in groups.items():
        weights[ids]=1/len(groups)/(len(ids) if mode=='document_mean' else 1)
        if shuffled:
            perm=torch.randperm(len(ids),generator=torch.Generator().manual_seed(20261024+doc))
            labels[ids]=labels[ids][perm]
    same(labels,saved['labels'],'ENDING_COLD_LABELS');same(weights,saved['weights'],'ENDING_COLD_WEIGHTS')
    metadata=[{k:v for k,v in r.items() if k not in ('states','original_logits')} for r in rows]
    need(saved['pin']==fingerprint((fingerprint(metadata),labels,weights,mode,shuffled)),'ENDING_COLD_FIT_BINDING')
    values={p:{n:torch.stack([r['states'][p][n] for r in rows]).double() for n in model.roster} for p in model.proposals}
    return values,labels,weights


def replay(heads,opt,inputs,labels,weights,saved):
    before=fingerprint(heads.state_dict());seen=[]
    def closure():
        opt.zero_grad(set_to_none=True);losses=[]
        for p in heads:
            scores=torch.stack([F.linear(inputs[p][n].contiguous(),heads[p][n].weight,None)+heads[p][n].bias
                                for n in heads[p]]).mean(0)
            losses.append((F.cross_entropy(scores,labels,reduction='none')*weights).sum())
        loss=torch.stack(losses).mean()+.001*sum(p.square().sum() for p in heads.parameters())
        loss.backward();seen.append(dict(loss=float(loss.detach()),gradients=fingerprint({n:p.grad for n,p in heads.named_parameters()})))
        return loss
    opt.step(closure);loss=closure();gradients={n:p.grad for n,p in heads.named_parameters()}
    r=saved['record'];same(seen,r['evaluations'],'ENDING_COLD_EVALUATIONS');same(gradients,r['gradients'],'ENDING_COLD_GRADIENTS')
    maximum=max(float(g.abs().max()) for g in gradients.values());after=fingerprint(heads.state_dict())
    status='converged' if maximum<=1e-7 else ('stalled' if before==after else 'running')
    need(before==r['before'] and after==r['after'] and maximum==r['gradient_max'] and status==r['status']
         and float(loss.detach())==r['loss'] and fingerprint(opt.state_dict())==r['optimizer'],'ENDING_COLD_STEP')
    same(dict(heads=heads.state_dict(),optimizer=opt.state_dict()),saved['state'],'ENDING_COLD_CHECKPOINT')
    need(fingerprint(saved['state'])==saved['state_pin'],'ENDING_COLD_CHECKPOINT_PIN')


def reference(rows,model,heads,saved):
    converted={n:v.detach().float() for n,v in heads.state_dict().items()}
    same(converted,saved['ending'],'ENDING_COLD_CAST_WEIGHTS')
    body=fingerprint({n:v for n,v in model.state_dict().items() if not n.startswith('ending.')})
    need(fingerprint(model.state_dict())==saved['installation']['parent'],'ENDING_COLD_PARENT')
    model.ending.load_state_dict(converted,strict=True)
    need(body==saved['installation']['frozen_backbone'] and
         body==fingerprint({n:v for n,v in model.state_dict().items() if not n.startswith('ending.')})
         and fingerprint(model.state_dict())==saved['installation']['model'],'ENDING_COLD_INSTALL')
    values={};doubles={}
    with torch.no_grad():
        for p in model.proposals:
            values[p]=torch.cat([torch.stack([F.linear(r['states'][p][n][None].contiguous(),model.ending[p][n].weight,None)
                                  +model.ending[p][n].bias for n in model.roster]).mean(0) for r in rows])
            doubles[p]=torch.stack([heads[p][n](torch.stack([r['states'][p][n] for r in rows]).double())
                                    for n in model.roster]).mean(0)
    same(values,saved['logits'],'ENDING_COLD_REFERENCE_LOGITS');same(doubles,saved['double_logits'],'ENDING_COLD_REFERENCE_DOUBLE')
    cast=max(float((values[p].double()-doubles[p]).abs().max()) for p in values)
    return values,cast


def check_metrics(rows,values,reported):
    for side,proposals in reported.items():
        ids=[i for i,r in enumerate(rows) if r['side']==side];y=np.array([int(rows[i]['end']) for i in ids]);docs=len({rows[i]['document'] for i in ids})
        for p,r in proposals.items():
            scores=values[p][ids].double().numpy();centered=scores-scores.max(1,keepdims=True)
            logs=centered-np.log(np.exp(centered).sum(1,keepdims=True));prob=np.exp(logs[:,1]);loss=-logs[np.arange(len(ids)),y]
            end=prob[y==1];cont=prob[y==0]
            expected=dict(decision_nll=float(loss.mean()),whole_sequence_nll=float(loss.sum()/docs),brier=float(((prob-y)**2).mean()),
                auroc=float(((end[:,None]>cont[None]).astype(float)+.5*(end[:,None]==cont[None])).mean()))
            for k,v in expected.items():need(abs(v-r[k])<1e-12,'ENDING_COLD_METRIC:'+k)
            need(r['n']==len(ids) and r['documents']==docs and r['end_labels']==int(y.sum())
                 and r['correct']==int((scores.argmax(1)==y).sum()) and r['end_true_positives']==int((end>.5).sum())
                 and r['premature']==int((cont>.5).sum()),'ENDING_COLD_REFERENCE_COUNTS')
            for key,arr in [('end_probability_range',end),('continue_probability_range',cont)]:
                need(max(abs(r[key][0]-arr.min()),abs(r[key][1]-arr.max()))<1e-12,'ENDING_COLD_PROBABILITY_RANGE')


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};results={};negative={};updates=rollouts=0
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'ENDING_COLD_PIN');r=json.loads((a.input/'report.json').read_bytes())
    need(r['complete'],'ENDING_COLD_COMPLETE')
    for n,h in r['files'].items():need(file_sha(a.input/n)==h,'ENDING_COLD_BYTES')
    s=source.load(a.root);need(s['code']==r['code'],'ENDING_COLD_CODE');phases['authentication']=time.perf_counter()-tick
    load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    for arch in source.prior.prior.ARCHITECTURES:
        tick=time.perf_counter();model=source.make(s,arch);rows=contexts(s,model,load(a.input/arch/'contexts.pt'))
        original=load(a.input/arch/'original.pt');same(original['logits'],{p:torch.stack([x['original_logits'][p] for x in rows]) for p in model.proposals},'ENDING_COLD_ORIGINAL')
        check_metrics(rows,original['logits'],original['metrics']);phases[arch+'_contexts']=time.perf_counter()-tick
        for mode in ('document_mean','sequence_sum'):
            for shuffled in (False,True):
                tick=time.perf_counter();key=arch+'/'+mode+('/shuffled' if shuffled else '/actual');ref=r['results'][key]
                model=source.make(s,arch);model.eval();heads=deepcopy(model.ending).double();opt=torch.optim.LBFGS(heads.parameters(),**learner.RECIPE)
                fit=load(a.input/key/'fit.pt');inputs,labels,weights=fit_arrays(rows,model,mode,shuffled,fit)
                for cursor in range(1,ref['updates']+1):
                    saved=load(a.input/key/f'update-{cursor}.pt');replay(heads,opt,inputs,labels,weights,saved);updates+=1
                    if cursor==1:
                        # A real new optimizer restores all LBFGS history, not just weights.
                        restored=deepcopy(model.ending).double();restored.load_state_dict(saved['state']['heads'],strict=True)
                        resume=torch.optim.LBFGS(restored.parameters(),**learner.RECIPE);resume.load_state_dict(deepcopy(saved['state']['optimizer']))
                        same(dict(heads=restored.state_dict(),optimizer=resume.state_dict()),saved['state'],'ENDING_COLD_RESTORE')
                        heads,opt=restored,resume
                need(saved['record']['status']==ref['status'] and saved['record']['gradient_max']==ref['gradient_max'],'ENDING_COLD_TERMINAL')
                phases[key+'_fit']=time.perf_counter()-tick;tick=time.perf_counter()
                target=load(a.input/key/'reference.pt');values,error=reference(rows,model,heads,target)
                need(error==ref['maximum_logit_cast_error'],'ENDING_COLD_CAST_ERROR');check_metrics(rows,values,ref['reference'])
                rate=float((weights*labels).sum()/weights.sum());need(rate==ref['train_hazard'],'ENDING_COLD_BASE_RATE')
                constant={p:torch.tensor([1-rate,rate],dtype=torch.float64).log().repeat(len(rows),1) for p in model.proposals}
                same(constant,target['constant_logits'],'ENDING_COLD_CONSTANT');check_metrics(rows,constant,ref['constant'])
                observed=load(a.input/key/'autonomous.pt');results[key]=autonomous(s,model,observed);rollouts+=len(observed)
                phases[key+'_reference_autonomous']=time.perf_counter()-tick
                if key=='shared/document_mean/actual':
                    bad=deepcopy(saved['state']);next(iter(bad['optimizer']['state'].values())).pop('prev_flat_grad')
                    negative['missing_history']=rejected(lambda:learner.make(model,bad,expected_pin=fingerprint(bad)),'ENDING_RESTORE_HISTORY')
                    bad=deepcopy(target['ending']);next(iter(bad.values())).add_(1)
                    negative['head_corruption']=rejected(lambda:same(bad,target['ending'],'ENDING_COLD_CORRUPT_HEAD'),'ENDING_COLD_CORRUPT_HEAD')
    tick=time.perf_counter();s['fresh']();need(rollouts==1280 and updates==sum(v['updates'] for v in r['results'].values()),'ENDING_COLD_DENOMINATOR')
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'ENDING_COLD_PHASES')
    out=dict(complete=True,acceptance='author-checked',source=dict(path=str(a.input),pin=a.pin),code=s['code'],
        updates=updates,rollouts=rollouts,all_contexts_exact=True,all_optimizer_updates_exact=True,
        all_autonomous_trajectories_exact=True,all_reference_metrics_recounted=True,results=results,controls=negative,
        phases=phases,seconds=seconds,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        promoted=False,capability_demonstrated=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',out),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
