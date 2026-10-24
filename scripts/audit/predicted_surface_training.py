"""Matched TRAIN-only decoder adaptation to actual frozen predictor readings."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import predicted_surface_source as source
from audit.answer_surface_stream import observe, rejected
from claims.process_conditions import write_committed
from model.answer_surface_stream import render_atom
from model.native_population_training import cpu, fingerprint
from model import predicted_surface_training as adaptation
from model import surface_inversion_training as inversion
from model.surface_device_runtime import evaluate
from text.constellation_corpus import canonical, digest, file_sha, need


def persist(path,value):
    with path.open('xb') as f:torch.save(cpu(value),f);f.flush();os.fsync(f.fileno())


def scored(model,group,*,true_cache):
    views,metadata=source.evaluation_views(group);model.eval();results={}
    with torch.no_grad():
        for doc,conditions in views.items():
            scores={}
            for condition,examples in conditions.items():
                if condition=='true' and doc in true_cache:scores[condition]=true_cache[doc]
                else:
                    values,_=evaluate(model,examples,'gathered',128);scores[condition]=values
                    if condition=='true':true_cache[doc]=values
            free=[observe(render_atom(model,e.native)) for e in conditions['predicted']]
            results[doc]=dict(metadata=metadata[doc],scores=scores,free=free)
    return results


def summarize(value):
    result={}
    for doc,row in value.items():
        positions=row['metadata']['positions']
        means={n:sum(float(x['losses'].double().sum()) for x in values)/positions
               for n,values in row['scores'].items()}
        result[doc]=dict(atoms=row['metadata']['atoms'],positions=positions,nats_per_position=means,
            shuffled_minus_predicted=means['shuffled']-means['predicted'],
            predicted_minus_true=means['predicted']-means['true'],
            free_atom_eos=sum(r['events'][-1]['kind']=='atom_end' for r in row['free']),
            free_refusals=sum(r['events'][-1]['kind']=='refuse' for r in row['free']),
            free_censored=sum(r['censored'] for r in row['free']),semantic_validity=False)
    return result


def fit(state,groups,key,out,*,predicted):
    out.mkdir();model,opt=adaptation.continuation(state['config'],state['specs'],state['checkpoint'],'cpu',step=1)
    binding=dict(code=state['code'],source=state['trace_pin'],predictor=key,
        initial=fingerprint((state['checkpoint']['after'],state['checkpoint']['optimizer'])),
        train=source.TRAIN,rule='half_true_half_predicted' if predicted else 'two_true_views',
        policy=inversion.POLICY,scope='CPU_engineering_qualification')
    ledger=digest(canonical(binding));rows=[];checkpoints=[]
    for cursor,doc in enumerate(source.TRAIN):
        row,profile=adaptation.update(model,opt,groups[key][doc],cursor+1,predicted=predicted)
        ledger=digest(canonical(dict(previous=ledger,row=row)))
        ref=inversion.checkpoint(out/f'checkpoint-{cursor+2}.pt',model,opt,binding,cursor+1,cursor+2,ledger)
        rows.append(dict(document=doc,update=row,profile=profile));checkpoints.append(ref)
        if cursor==0:
            before=fingerprint((model.state_dict(),opt.state_dict()))
            restored,new_opt=adaptation.continuation(state['config'],state['specs'],state['checkpoint'],'cpu',step=1)
            position,step,chain=inversion.restore(ref,restored,new_opt,binding,size=2)
            need((position,step,chain)==(1,2,ledger)
                 and fingerprint((restored.state_dict(),new_opt.state_dict()))==before,'ADAPTATION_RESTART')
            model,opt=restored,new_opt
    persist(out/'updates.pt',rows);write_committed(out/'checkpoints.json',checkpoints)
    write_committed(out/'binding.json',binding)
    return model,opt,dict(updates=2,positions=sum(r['update']['positions'] for r in rows),
        final=fingerprint(model.state_dict()),optimizer=fingerprint(opt.state_dict()),checkpoint=checkpoints[-1])


def controls(state,groups,traces):
    key=next(iter(groups));pair=groups[key][source.TRAIN[0]][0];held=groups[key][source.EVALUATION[0]]
    model,opt=adaptation.continuation(state['config'],state['specs'],state['checkpoint'],'cpu',step=1)
    initial=fingerprint((model.state_dict(),opt.state_dict()));out={}
    out['held_gradient']=rejected(lambda:adaptation.update(model,opt,held,1,predicted=True),'PREDICTED_SURFACE_NONTRAIN')
    bad=deepcopy(state['checkpoint']);bad['optimizer']['state']={}
    out['missing_moments']=rejected(lambda:adaptation.continuation(state['config'],state['specs'],bad,'cpu',step=1),
                                     'INVERSION_MOMENT_COVERAGE')
    out['wrong_initial_step']=rejected(lambda:adaptation.continuation(state['config'],state['specs'],state['checkpoint'],
        'cpu',step=2),'INVERSION_MOMENT_STATE')
    out['duplicate_target']=rejected(lambda:adaptation.views([pair,pair],predicted=True),'PREDICTED_SURFACE_DUPLICATE_TARGET')
    out['invalid_mixture']=rejected(lambda:adaptation.views([pair],predicted=1),'PREDICTED_SURFACE_MIXTURE')
    row=next(r for r in traces if r['architecture']+'/'+r['proposal']==key and r['document']==source.TRAIN[0])
    event=row['events'][0]
    arguments=dict(position=0,proposal=row['proposal'],trace_pin=state['trace_pin'],region_pin=state['region_pin'],
        region_sizes=tuple(state['bundle']['sizes']),source_sha=pair.receipt['source_sha'],
        document=row['document'],source_index=pair.receipt['source_index'],fitting=True)
    def make(e=event,**overrides):
        return adaptation.paired(pair.observed,e,state['specs'],state['config'],**dict(arguments,**overrides))
    out['source_bytes']=rejected(lambda:make(source_sha='0'*64),'PREDICTED_SURFACE_SOURCE_BYTES')
    out['wrong_proposal']=rejected(lambda:make(proposal='not-the-source-proposal'),'PREDICTED_SURFACE_PROPOSAL')
    bad=deepcopy(event);bad['index']=1
    out['wrong_order']=rejected(lambda:make(bad),'ANSWER_SURFACE_POSITION')
    bad=deepcopy(event);bad['counts'][next(iter(bad['counts']))]+=1
    out['wrong_count']=rejected(lambda:make(bad),'ANSWER_SURFACE_COUNT')
    bad=deepcopy(event);bad['raw']=b'future'
    out['future_input']=rejected(lambda:make(bad),'ANSWER_SURFACE_EVENT_FIELDS')
    bad=deepcopy(traces);bad[0]['events']=[]
    out['missing_prediction']=rejected(lambda:source.assemble(state,bad,state['trace_pin']),'ADAPTATION_MISSING_PREDICTION')
    value=make();next(iter(value.predicted.native.values())).values.add_(1)
    out['native_changed']=rejected(value.fresh,'PREDICTED_SURFACE_NATIVE_CHANGED')
    value=make();value.receipt['position']=9
    out['binding_changed']=rejected(value.fresh,'PREDICTED_SURFACE_BINDING_CHANGED')
    need(fingerprint((model.state_dict(),opt.state_dict()))==initial,'ADAPTATION_CONTROL_MUTATION')
    original=fingerprint((state['checkpoint']['after'],state['checkpoint']['optimizer']))
    separate,separate_opt=adaptation.continuation(state['config'],state['specs'],state['checkpoint'],'cpu',step=1)
    next(iter(separate_opt.state.values()))['exp_avg'].add_(1)
    need(fingerprint((state['checkpoint']['after'],state['checkpoint']['optimizer']))==original,
         'ADAPTATION_INITIAL_MOMENT_ALIAS')
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();cpu0=time.process_time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    state=source.load(a.root);phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    traces,weights=source.predictions(state);persist(a.output/'train-predictions.pt',traces)
    state['trace_pin']=file_sha(a.output/'train-predictions.pt')
    groups,receipts=source.assemble(state,traces,state['trace_pin'])
    write_committed(a.output/'pairs.json',receipts);write_committed(a.output/'predictors.json',weights)
    persist(a.output/'initial.pt',dict(after=state['checkpoint']['after'],optimizer=state['checkpoint']['optimizer']))
    phases['TRAIN_predictions_and_all_source_joins']=time.perf_counter()-tick;tick=time.perf_counter()
    initial,_=adaptation.continuation(state['config'],state['specs'],state['checkpoint'],'cpu',step=1)
    initial_scores={};cache={};summary={};evaluation=a.output/'evaluation';evaluation.mkdir()
    for key,group in groups.items():
        values=scored(initial,group,true_cache=cache);initial_scores[key]=values
        persist(evaluation/('initial-'+key.replace('/','--')+'.pt'),values)
        summary[key]=dict(initial=summarize(values))
    phases['initial_evaluation']=time.perf_counter()-tick;tick=time.perf_counter()
    baseline_key=next(iter(groups));model,opt,baseline=fit(state,groups,baseline_key,a.output/'observed',predicted=False)
    cache={}
    for key,group in groups.items():
        values=scored(model,group,true_cache=cache);persist(evaluation/('observed-'+key.replace('/','--')+'.pt'),values)
        summary[key]['observed']=summarize(values)
    phases['observed_control_train_and_evaluation']=time.perf_counter()-tick;arms={}
    for key,group in groups.items():
        tick=time.perf_counter();name=key.replace('/','--')
        model,opt,arms[key]=fit(state,groups,key,a.output/name,predicted=True)
        values=scored(model,group,true_cache={});persist(evaluation/('mixed-'+name+'.pt'),values)
        summary[key]['mixed']=summarize(values)
        phases[name]=time.perf_counter()-tick
        print(json.dumps(dict(proposal=key,updates=arms[key]['updates'],summary=summary[key])),flush=True)
    tick=time.perf_counter();negative=controls(state,groups,traces);state['fresh']()
    latest=dict(after=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()))
    later,later_opt=adaptation.continuation(state['config'],state['specs'],latest,'cpu',step=3)
    need(fingerprint((later.state_dict(),later_opt.state_dict()))==fingerprint((model.state_dict(),opt.state_dict())),
         'ADAPTATION_LATER_CONTINUATION')
    need(not torch.cuda.is_initialized(),'ADAPTATION_CPU_PROOF_GPU')
    write_committed(a.output/'scores.json',summary)
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['controls_source_readback_hashes']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'ADAPTATION_PHASE_COVERAGE')
    result=dict(complete=True,acceptance='author-checked',number_status='measured',code=state['code'],files=files,
        source=source.REFS,trace_pin=state['trace_pin'],baseline_key=baseline_key,baseline=baseline,arms=arms,
        training_documents=source.TRAIN,evaluation_documents=source.EVALUATION,train_error_scope='in_sample',
        total_updates=baseline['updates']+sum(v['updates'] for v in arms.values()),
        total_positions=baseline['positions']+sum(v['positions'] for v in arms.values()),
        predictor_decisions=sum(len(r['events']) for r in traces),controls=negative,
        scientific_admission=False,capability_demonstrated=False,promoted_to_serving=False,
        phases=phases,seconds=seconds,cpu_seconds=time.process_time()-cpu0,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    need(result['total_updates']==18 and result['total_positions']==7218,'ADAPTATION_TRAIN_DENOMINATOR')
    pin=write_committed(a.output/'report.json',result);print(json.dumps(dict(complete=True,pin=pin,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.authority.ROOT)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
