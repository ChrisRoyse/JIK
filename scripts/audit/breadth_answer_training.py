"""Real broad-to-answer transfer, preserved Adam history and matched control."""
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
from audit import breadth_answer_source as source
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import breadth_answer_transfer as learner
from model.native_population_training import cpu,fingerprint
from model.native_request_input import NativeRequest,legal
from text.constellation_corpus import file_sha,need


def equivalence(s,parent,child):
    parent.eval();child.eval();rows=[]
    with torch.no_grad():
        for doc,d in sorted(s['items'].items()):
            a=parent.stream();b=child.stream(NativeRequest(legal(2)));ca=cb=None
            for position,frame in enumerate(d['frames']):
                x,ca=a.step((frame,),context='parent',offset=position,state=ca,detach_history=True)
                y,cb=b.step((frame,),context='child',offset=position,state=cb,detach_history=True)
                need(fingerprint(x)==fingerprint(y),'BROAD_ANSWER_INITIAL_HIDDEN')
                pa=parent.region(x);pb=child.forecaster.predictor.region(y)
                need(fingerprint(pa)==fingerprint(pb),'BROAD_ANSWER_INITIAL_REGION')
                rows.append(dict(document=doc,position=position,hidden=fingerprint(x),region=fingerprint(pa)))
    need(len(rows)==215,'BROAD_ANSWER_INITIAL_CENSUS')
    return rows


def controls(s):
    parent,model=source.models(s,91540);saved=s['parents'][91540];pin=fingerprint(saved)
    opt,binding=learner.install(model,parent,saved,s['policy'],source_pin=pin)
    initial=learner.checkpoint(model,opt,binding,0,s['policy']);initial_pin=fingerprint(initial);negative={}
    def bad_restore(value,code):
        return rejected(lambda:learner.restore(model,opt,value,binding,s['policy'],pin=fingerprint(value)),code)
    bad=deepcopy(initial);bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
    negative['missing_moment']=bad_restore(bad,'TRANSFER_MOMENT_CENSUS')
    bad=deepcopy(initial);next(iter(bad['optimizer']['state'].values()))['step'].add_(1)
    negative['wrong_age']=bad_restore(bad,'TRANSFER_AGE')
    bad=deepcopy(initial);next(iter(bad['optimizer']['state'].values()))['exp_avg'].add_(1)
    negative['changed_moment']=bad_restore(bad,'TRANSFER_MOMENT_ORIGIN')
    bad=deepcopy(initial);bad['binding']['ages'][next(iter(bad['binding']['ages']))]+=1
    negative['changed_binding']=bad_restore(bad,'TRANSFER_RESTORE_BINDING')
    for kind in ['architecture','target']:
        p,m=source.models(s,91540);before=fingerprint(m.state_dict())
        if kind=='architecture':m.forecaster.architecture='own_slot'
        else:m.forecaster.predictor._target+=1
        negative[kind]=rejected(lambda:learner.install(m,p,saved,s['policy'],source_pin=pin),'TRANSFER_'+kind.upper())
        need(fingerprint(m.state_dict())==before,'BROAD_CONTROL_INSTALL_MUTATION')
    item,request=source.prior.prior.item(s,s['train'][0])
    def run(value=item,req=request,policy=None):
        return learner.update(model,opt,value,0,policy or s['policy'],binding,request=req,target=s['target'])
    query=next(k for k,d in s['items'].items() if d['side']=='query')
    negative['query_fit']=rejected(lambda:run(source.prior.prior.item(s,query)[0]),'CONDITIONED_NONTRAIN')
    for field,code in [('native','CONDITIONED_NATIVE_PIN'),('labels','CONDITIONED_LABEL_PIN'),('binding','CONDITIONED_SOURCE_BINDING')]:
        bad=deepcopy(item)
        if field=='native':next(iter(bad['answers'][0].values())).add_(1)
        elif field=='labels':bad['codes'][0,0]=(bad['codes'][0,0]+1)%4
        else:bad['provenance']['observed']+=1
        negative[field]=rejected(lambda:run(bad),code)
    negative['request']=rejected(lambda:run(req=NativeRequest(dict(legal(2),operation='respond_to_source_question'))),'FULL_HISTORY_REQUEST')
    negative['memory']=rejected(lambda:run(policy=dict(s['policy'],working_bytes=1)),'GRADIENT_STORAGE_MEMORY')
    need(initial_pin==fingerprint(learner.checkpoint(model,opt,binding,0,s['policy'])),'BROAD_CONTROL_CHANGED_STATE')
    need(all(p.grad is None for p in model.parameters()),'BROAD_CONTROL_RETAINED_GRADIENTS')
    return negative


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};results={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    s=source.load(a.root)
    write_committed(a.output/'source.json',dict(parent=source.PARENT,cold=source.COLD,files=s['parent_files'],code=s['code'],
        policy=s['policy'],train=s['train'],census=s['census'],pretraining_updates=91540,pretraining_targets=6455636))
    phases['authentication']=time.perf_counter()-tick;tick=time.perf_counter()
    negative=controls(s);write_committed(a.output/'controls.json',negative);phases['controls']=time.perf_counter()-tick
    head_pins=[]
    for parent_step in (0,91540):
        name='initial' if parent_step==0 else 'breadth';out=a.output/name;out.mkdir();tick=time.perf_counter()
        parent,model=source.models(s,parent_step);saved=s['parents'][parent_step]
        opt,binding=learner.install(model,parent,saved,s['policy'],source_pin=fingerprint(saved))
        head_pins.append(binding['new_heads']);need(len(set(head_pins))==1,'BROAD_ANSWER_MATCHED_NEW_HEADS')
        persist(out/'initial.pt',learner.checkpoint(model,opt,binding,0,s['policy']))
        stream_rows=equivalence(s,parent,model);write_committed(out/'equivalence.json',stream_rows)
        phases[name+'_transfer_equivalence']=time.perf_counter()-tick;tick=time.perf_counter()
        initial=source.prior.evaluate(s,model);persist(out/'initial-evaluation.pt',initial)
        phases[name+'_initial_evaluation']=time.perf_counter()-tick;tick=time.perf_counter();ledger=[];model.train()
        for cursor,doc in enumerate(s['train']):
            item,request=source.prior.prior.item(s,doc)
            record,gradients,early=learner.update(model,opt,item,cursor,s['policy'],binding,request=request,target=s['target'])
            state=learner.checkpoint(model,opt,binding,cursor+1,s['policy'])
            persist(out/f'update-{cursor+1}.pt',dict(state=state,state_pin=fingerprint(state),record=record,gradients=gradients,early=early))
            ledger.append({k:v for k,v in record.items() if k!='records'})
            if cursor==1:
                _,model=source.models(s,parent_step);fresh_parent,_=source.models(s,parent_step)
                opt,newbinding=learner.install(model,fresh_parent,saved,s['policy'],source_pin=fingerprint(saved))
                need(newbinding==binding,'BROAD_ANSWER_RESTORE_BINDING')
                learner.restore(model,opt,state,binding,s['policy'],pin=fingerprint(state));model.train()
        write_committed(out/'ledger.json',ledger);phases[name+'_training']=time.perf_counter()-tick;tick=time.perf_counter()
        final=source.prior.evaluate(s,model);persist(out/'final-evaluation.pt',final)
        phases[name+'_final_evaluation']=time.perf_counter()-tick
        results[name]=dict(parent_step=parent_step,binding=binding,updates=len(ledger),decisions=sum(r['decisions'] for r in ledger),
            final=fingerprint(model.state_dict()),optimizer=fingerprint(opt.state_dict()),rollouts=len(initial)+len(final),
            peak_saved_storage=max(r['storage']['peak_saved_storage_bytes'] for r in ledger),
            all_storage_released=all(r['storage']['remaining_saved_storage_bytes']==0 for r in ledger))
        print(json.dumps(dict(arm=name,updates=len(ledger),final=results[name]['final'])),flush=True)
    tick=time.perf_counter();s['fresh']();need(not torch.cuda.is_initialized(),'BROAD_ANSWER_CPU_SCOPE')
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(sum(v['updates'] for v in results.values())==10 and sum(v['rollouts'] for v in results.values())==160,'BROAD_ANSWER_DENOMINATOR')
    need(.9<=sum(phases.values())/seconds<=1.001,'BROAD_ANSWER_PHASES')
    result=dict(complete=True,acceptance='author-checked',code=s['code'],results=results,controls=negative,
        files={str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()},
        phases=phases,seconds=seconds,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        native_encoder_calls=0,source_retraining=0,promoted=False,capability_demonstrated=False,
        scope='shared broad-pretrained transfer; exposed legal complete continuations; CPU qualification')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
