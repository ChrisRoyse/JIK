"""Fit real complete-answer ending heads; retain every failed control/outcome."""
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
import torch
from audit import ending_fit_source as source
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import answer_ending_fit as learner
from model.native_population_training import cpu,fingerprint
from text.constellation_corpus import file_sha,need


def canonical_logits(model, data):
    with torch.no_grad():
        return {p:torch.cat([model.stop_logits(torch.stack([data.inputs[p][n][i].float() for n in data.roster])[None],p)
                           for i in range(len(data.meta))]) for p in model.proposals}


def metrics(data, values):
    result={}
    for side in ('train','development','query'):
        ids=[i for i,r in enumerate(data.meta) if r['side']==side];labels=data.labels[ids]
        documents=len({data.meta[i]['document'] for i in ids});out={}
        for proposal,score in values.items():
            logits=score[ids].double();p=logits.softmax(-1)[:,1]
            loss=-logits.log_softmax(-1)[torch.arange(len(ids)),labels]
            end=p[labels.bool()];cont=p[~labels.bool()]
            out[proposal]=dict(n=len(ids),documents=documents,end_labels=int(labels.sum()),
                decision_nll=float(loss.mean()),whole_sequence_nll=float(loss.sum()/documents),
                brier=float((p-labels).square().mean()),correct=int((logits.argmax(-1)==labels).sum()),
                end_true_positives=int((end>.5).sum()),premature=int((cont>.5).sum()),
                end_probability_range=[float(end.min()),float(end.max())],
                continue_probability_range=[float(cont.min()),float(cont.max())],
                auroc=float(((end[:,None]>cont[None]).double()+.5*(end[:,None]==cont[None])).mean()))
        result[side]=out
    return result


def controls(s, model, rows):
    original=fingerprint(model.state_dict());heads,opt=learner.make(model)
    data=learner.EndingData([r for r in rows if r['side']=='train'],model.roster,model.proposals,s['policy']['working_bytes'])
    query=learner.EndingData([r for r in rows if r['side']=='query'],model.roster,model.proposals,s['policy']['working_bytes'])
    out={'query_fit':rejected(lambda:query.fitting('sequence_sum',False),'ENDING_FIT_QUERY')}
    fit=data.fitting('sequence_sum',True);fit['labels'][0]=1-fit['labels'][0]
    out['altered_labels']=rejected(lambda:learner.objective(heads,data,fit),'ENDING_FIT_BINDING')
    next(iter(next(iter(data.inputs.values())).values())).add_(1)
    out['altered_context']=rejected(data.fresh,'ENDING_DATA_CHANGED')
    empty=learner.checkpoint(heads,opt)
    out['missing_optimizer']=rejected(lambda:learner.make(model,empty,expected_pin=fingerprint(empty)), 'ENDING_RESTORE_HISTORY')
    out['wrong_parent']=rejected(lambda:learner.install(model,heads,parent_pin='0'*64),'ENDING_INSTALL_PARENT')
    need(fingerprint(model.state_dict())==original,'ENDING_CONTROL_CHANGED_PARENT')
    need(all(p.grad is None for p in heads.parameters()),'ENDING_CONTROL_GRADIENT')
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};results={};negative={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    s=source.load(a.root);phases['authentication']=time.perf_counter()-tick
    write_committed(a.output/'source.json',dict(parent=source.PARENT,cold=source.COLD,code=s['code'],
        policy=s['policy'],recipe=learner.RECIPE,regularization=learner.REGULARIZATION,
        gradient_tolerance=learner.GRADIENT_TOLERANCE,train=s['train']))
    for architecture in source.prior.prior.ARCHITECTURES:
        tick=time.perf_counter();model=source.make(s,architecture);model.eval();parent=fingerprint(model.state_dict())
        rows=source.collect(s,model);directory=a.output/architecture;directory.mkdir();persist(directory/'contexts.pt',rows)
        all_data=learner.EndingData(rows,model.roster,model.proposals,s['policy']['working_bytes'])
        train=learner.EndingData([r for r in rows if r['side']=='train'],model.roster,model.proposals,s['policy']['working_bytes'])
        original=canonical_logits(model,all_data)
        need(fingerprint(original)==fingerprint({p:torch.stack([r['original_logits'][p] for r in rows]) for p in model.proposals}),
             'ENDING_CACHED_ORIGINAL_LOGITS')
        persist(directory/'original.pt',dict(logits=original,metrics=metrics(all_data,original)))
        negative[architecture]=controls(s,model,rows)
        phases[architecture+'_contexts_controls']=time.perf_counter()-tick
        for mode in ('document_mean','sequence_sum'):
            for shuffled in (False,True):
                key=architecture+'/'+mode+('/shuffled' if shuffled else '/actual');dest=a.output/key;dest.mkdir(parents=True)
                tick=time.perf_counter();model=source.make(s,architecture);model.eval()
                need(fingerprint(model.state_dict())==parent,'ENDING_PARENT_REUSE');heads,opt=learner.make(model)
                fit=train.fitting(mode,shuffled);persist(dest/'fit.pt',fit)
                ledger=[];cursor=0
                while True:
                    record=learner.step(heads,opt,train,fit);cursor+=1
                    checkpoint=learner.checkpoint(heads,opt)
                    persist(dest/f'update-{cursor}.pt',dict(state=checkpoint,state_pin=fingerprint(checkpoint),record=record))
                    ledger.append({k:v for k,v in record.items() if k!='gradients'})
                    if record['status']!='running':break
                phases[key+'_fit']=time.perf_counter()-tick;tick=time.perf_counter()
                installation=learner.install(model,heads,parent_pin=parent)
                values=canonical_logits(model,all_data)
                with torch.no_grad():double_values=cpu(learner.logits(heads,all_data.inputs,all_data.roster))
                cast_error=max(float((values[p].double()-double_values[p]).abs().max()) for p in values)
                rate=float((fit['weights']*fit['labels']).sum()/fit['weights'].sum())
                constant={p:torch.tensor([1-rate,rate],dtype=torch.float64).log().repeat(len(rows),1) for p in model.proposals}
                persist(dest/'reference.pt',dict(logits=values,double_logits=double_values,constant_logits=constant,
                    ending=model.ending.state_dict(),installation=installation))
                before_rollout=fingerprint(model.state_dict());evaluation=source.prior.evaluate(s,model)
                need(fingerprint(model.state_dict())==before_rollout,'ENDING_ROLLOUT_MUTATION')
                persist(dest/'autonomous.pt',evaluation);phases[key+'_reference_autonomous']=time.perf_counter()-tick
                results[key]=dict(status=record['status'],updates=cursor,evaluations=sum(len(r['evaluations']) for r in ledger),
                    gradient_max=record['gradient_max'],objective=record['loss'],rollouts=len(evaluation),
                    fit_pin=fit['pin'],parent=parent,installation=installation,maximum_logit_cast_error=cast_error,
                    labels_changed=int((fit['labels']!=train.labels).sum()),train_hazard=rate,
                    reference=metrics(all_data,values),constant=metrics(all_data,constant))
                write_committed(dest/'ledger.json',ledger);write_committed(dest/'result.json',results[key])
                print(json.dumps(dict(arm=key,status=record['status'],updates=cursor,gradient_max=record['gradient_max'],
                    correct_end=sum(r['metrics']['ended_at_reference_boundary'] for r in evaluation if r['mode']=='correct'))),flush=True)
    tick=time.perf_counter();s['fresh']();need(not torch.cuda.is_initialized(),'ENDING_CPU_SCOPE')
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(r['rollouts'] for r in results.values())==1280,'ENDING_ROLLOUT_TOTAL')
    need(.9<=sum(phases.values())/seconds<=1.001,'ENDING_PHASES')
    report=dict(complete=True,schema='ending-head-fit-v1',acceptance='author-checked',code=s['code'],results=results,
        files={str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()},
        controls=negative,phases=phases,seconds=seconds,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        native_encoder_calls=0,promoted=False,scientific_admission=False,
        scope='exposed legal document continuation; ending heads only; parent Adam state preserved, new LBFGS state retained')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
