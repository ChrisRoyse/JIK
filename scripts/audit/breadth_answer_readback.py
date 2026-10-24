"""Independent named-state migration and complete-answer update/readback."""
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
from audit import breadth_answer_source as source
from audit.full_history_readback import autonomous,same
from audit.answer_surface_stream import rejected
from claims.process_conditions import write_committed
from model.native_population_training import cpu,fingerprint
from model.native_request_input import NativeRequest,legal
from text.constellation_corpus import file_sha,need


def snapshot(model,opt,binding,cursor):
    return dict(schema='broad-answer-checkpoint-v1',model=model.state_dict(),optimizer=opt.state_dict(),binding=binding,cursor=cursor)


def check_ages(model,opt,binding,cursor):
    names=dict(model.named_parameters());need(list(names)==binding['names'],'BROAD_COLD_NAMES')
    need(binding['pin']==fingerprint({k:v for k,v in binding.items() if k!='pin'}),'BROAD_COLD_BINDING')
    present={n for n,p in names.items() if p in opt.state};expected={n for n in names if binding['ages'][n]+cursor>0}
    need(present==expected and len(opt.state)==len(expected),'BROAD_COLD_MOMENT_CENSUS')
    for n in present:
        p=names[n];v=opt.state[p]
        need(set(v)=={'step','exp_avg','exp_avg_sq'} and float(v['step'])==binding['ages'][n]+cursor,'BROAD_COLD_AGE')
        need(all(v[k].shape==p.shape and v[k].dtype==p.dtype and v[k].device==p.device
                 for k in ('exp_avg','exp_avg_sq')) and all(bool(x.isfinite().all()) for x in v.values()),'BROAD_COLD_MOMENTS')
        if cursor==0:need(fingerprint(v)==binding['moment_origins'][n],'BROAD_COLD_ORIGIN')


def reconstruct(s,parent_step,saved):
    parent,model=source.models(s,parent_step);original=cpu(model.state_dict());native=s['parents'][parent_step]
    parameter_order=list(dict(parent.named_parameters()));ids=native['optimizer']['param_groups'][0]['params']
    need(len(ids)==len(parameter_order)==59 and len(set(ids))==59,'BROAD_COLD_SOURCE_IDS')
    retained=[n for n in parameter_order if not n.startswith('counts.')]
    need(len(retained)==47 and all(n.split('.')[0] in ('encoder','projectors','region') for n in retained),'BROAD_COLD_MAPPING')
    mapping={n:'forecaster.predictor.'+n for n in retained};names=dict(model.named_parameters())
    state=model.state_dict();kept={n:v for n,v in native['model'].items() if not n.startswith('counts.')}
    for n,v in kept.items():state['forecaster.predictor.'+n]=v.clone()
    model.load_state_dict(state,strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=s['policy']['learning_rate'],weight_decay=s['policy']['weight_decay'],foreach=False,fused=False)
    moments={n:native['optimizer']['state'].get(i) for n,i in zip(parameter_order,ids)}
    for n,destination in mapping.items():
        if parent_step:opt.state[names[destination]]=deepcopy(moments[n])
        else:need(moments[n] is None,'BROAD_COLD_INITIAL_MOMENT')
    binding=dict(schema='broad-answer-transfer-v1',source=fingerprint(native),source_step=parent_step,mapping=mapping,
        retired=[n for n in parameter_order if n not in retained],
        ages={n:parent_step if n in mapping.values() else 0 for n in names},names=list(names),
        moment_origins={destination:fingerprint(moments[n]) for n,destination in mapping.items() if parent_step},
        inherited=fingerprint(kept),new_heads=fingerprint({n:v for n,v in original.items() if n not in {'forecaster.predictor.'+k for k in kept}}))
    binding['pin']=fingerprint(binding);check_ages(model,opt,binding,0)
    same(snapshot(model,opt,binding,0),saved,'BROAD_COLD_INITIAL_STATE')
    return parent,model,opt,binding


def equivalence(s,parent,model,rows):
    count=0;parent.eval();model.eval()
    with torch.no_grad():
        for doc in sorted(s['items']):
            streams=[parent.stream(),model.stream(NativeRequest(legal(2)))];states=[None,None]
            for pos,frame in enumerate(s['items'][doc]['frames']):
                values=[]
                for i,stream in enumerate(streams):
                    h,states[i]=stream.step((frame,),context='independent-transfer',offset=pos,state=states[i],detach_history=True)
                    values.append(h)
                same(values[0],values[1],'BROAD_COLD_TRANSFER_HIDDEN')
                left=parent.region(values[0]);right=model.forecaster.predictor.region(values[1])
                same(left,right,'BROAD_COLD_TRANSFER_REGION')
                need(rows[count]==dict(document=doc,position=pos,hidden=fingerprint(values[0]),region=fingerprint(left)),
                     'BROAD_COLD_TRANSFER_RECORD');count+=1
    need(count==len(rows)==215,'BROAD_COLD_TRANSFER_CENSUS')


def replay(s,model,opt,binding,item,request,cursor,saved):
    record=saved['record'];need(fingerprint(model.state_dict())==record['before'],'BROAD_COLD_BEFORE')
    check_ages(model,opt,binding,cursor);opt.zero_grad(set_to_none=True)
    zero_initial=cursor==0 and all(torch.count_nonzero(p).item()==0 for p in model.ending.parameters())
    c,a=len(item['context']),len(item['answers']);first={};handles=[]
    def capture(name):
        def hook(module,args,output):
            if name not in first:first[name]=output
        return hook
    for name,projection in model.forecaster.predictor.projectors.items():handles.append(projection.register_forward_hook(capture(name)))
    noise=torch.randn(a,s['policy']['samples'],s['policy']['noise_width'],
                      generator=torch.Generator().manual_seed(s['policy']['noise_seed']+cursor))
    stream=model.stream(request);cache=None;losses=[];rows=[]
    try:
        for pos,frame in enumerate((*item['context'],*item['answers'])):
            h,cache=stream.step((frame,),context='cold-broad-answer',offset=pos,state=cache,detach_history=False)
            if pos<c-1:continue
            index=pos-c+1;end=index==a;stop=model.stop_loss(h,end);loss=stop['loss']/(a+1)
            ref=record['records'][len(rows)]
            need(ref['position']==pos and ref['answer_index']==(None if end else index) and ref['end']==end,'BROAD_COLD_TARGET_POSITION')
            same(stop,ref['stop'],'BROAD_COLD_STOP')
            if not end:
                value=model.forecaster.losses(h,item['codes'][index:index+1],{n:v[index:index+1] for n,v in item['counts'].items()},
                    (item['answers'][index],),noise[index:index+1],target=s['target'])
                need(float(value['loss'].detach())==ref['continuation'],'BROAD_COLD_NATIVE_LOSS');loss=loss+value['loss']/a
            else:need(ref['continuation'] is None,'BROAD_COLD_END')
            need(float(loss.detach())==ref['weighted_loss'],'BROAD_COLD_LOSS');losses.append(loss);rows.append(ref)
        total=torch.stack(losses).sum()
        ends=torch.autograd.grad(losses[-1],tuple(first.values()),retain_graph=True)
        earlier=torch.autograd.grad(total,tuple(first.values()),retain_graph=True)
        credit={n:dict(value=cpu(v),requires_grad=True,end_gradient=cpu(e),gradient=cpu(g))
                for (n,v),e,g in zip(first.items(),ends,earlier)}
        need(zero_initial==record['zero_initial_end'] and all(bool(g.abs().any()) for g in earlier),'BROAD_COLD_CONTEXT_GRADIENT')
        need(all((not bool(g.any())) if zero_initial else bool(g.abs().any()) for g in ends),'BROAD_COLD_END_DERIVATIVE')
        same(credit,saved['early'],'BROAD_COLD_EARLY');total.backward()
        gradients={n:p.grad for n,p in model.named_parameters()};same(gradients,saved['gradients'],'BROAD_COLD_GRADIENTS')
        need(fingerprint(gradients)==record['gradients'],'BROAD_COLD_GRADIENT_PIN')
        for prefix,pin in record['branch_gradients'].items():
            need(fingerprint({n:v for n,v in gradients.items() if n.startswith(prefix)})==pin,'BROAD_COLD_BRANCH')
        opt.step();check_ages(model,opt,binding,cursor+1)
        same(snapshot(model,opt,binding,cursor+1),saved['state'],'BROAD_COLD_UPDATED_STATE')
        need(fingerprint(saved['state'])==saved['state_pin'] and fingerprint(model.state_dict())==record['after']
             and fingerprint(opt.state_dict())==record['optimizer'],'BROAD_COLD_UPDATED_PIN')
        need(len(rows)==record['decisions']==a+1,'BROAD_COLD_DECISIONS')
        return len(rows)
    finally:
        for hook in handles:hook.remove()
        first.clear()


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};results={};updates=decisions=rollouts=0
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'BROAD_COLD_REPORT_PIN');r=json.loads((a.input/'report.json').read_bytes())
    need(r['complete'],'BROAD_COLD_INCOMPLETE')
    for n,h in r['files'].items():need(file_sha(a.input/n)==h,'BROAD_COLD_BYTES')
    s=source.load(a.root);need(s['code']==r['code'],'BROAD_COLD_CODE');phases['authentication']=time.perf_counter()-tick
    load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    for name,step in [('initial',0),('breadth',91540)]:
        base=a.input/name;tick=time.perf_counter();parent,model,opt,binding=reconstruct(s,step,load(base/'initial.pt'))
        need(binding==r['results'][name]['binding'],'BROAD_COLD_REPORTED_BINDING')
        equivalence(s,parent,model,json.loads((base/'equivalence.json').read_bytes()))
        phases[name+'_transfer_equivalence']=time.perf_counter()-tick;tick=time.perf_counter()
        observed=load(base/'initial-evaluation.pt');results[name+'/initial']=autonomous(s,model,observed);rollouts+=len(observed)
        phases[name+'_initial_evaluation']=time.perf_counter()-tick;tick=time.perf_counter();model.train()
        for cursor,doc in enumerate(s['train']):
            item,request=source.prior.prior.item(s,doc);saved=load(base/f'update-{cursor+1}.pt')
            decisions+=replay(s,model,opt,binding,item,request,cursor,saved);updates+=1
            if cursor==1:
                _,model=source.models(s,step);model.load_state_dict(saved['state']['model'],strict=True)
                opt=torch.optim.AdamW(model.parameters(),lr=s['policy']['learning_rate'],weight_decay=s['policy']['weight_decay'],foreach=False,fused=False)
                opt.load_state_dict(deepcopy(saved['state']['optimizer']));check_ages(model,opt,binding,cursor+1);model.train()
                same(snapshot(model,opt,binding,cursor+1),saved['state'],'BROAD_COLD_RESTORED')
        phases[name+'_training']=time.perf_counter()-tick;tick=time.perf_counter()
        observed=load(base/'final-evaluation.pt');results[name+'/final']=autonomous(s,model,observed);rollouts+=len(observed)
        need(fingerprint(model.state_dict())==r['results'][name]['final']
             and fingerprint(opt.state_dict())==r['results'][name]['optimizer'],'BROAD_COLD_FINAL')
        phases[name+'_final_evaluation']=time.perf_counter()-tick
    tick=time.perf_counter();negative={}
    for field in ('model','optimizer','early','gradients'):
        altered=deepcopy(saved[field] if field in saved else saved['state'][field]);original=saved[field] if field in saved else saved['state'][field]
        if field=='optimizer':next(iter(altered['state'].values()))['exp_avg'].add_(1)
        elif field=='early':next(iter(altered.values()))['gradient'].add_(1)
        else:next(iter(altered.values())).add_(1)
        negative[field]=rejected(lambda:same(altered,original,'BROAD_COLD_CORRUPT_'+field),'BROAD_COLD_CORRUPT_'+field)
    s['fresh']();need((updates,decisions,rollouts)==(10,132,160),'BROAD_COLD_TOTAL')
    phases['final_controls_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'BROAD_COLD_PHASES')
    out=dict(complete=True,acceptance='author-checked',source=dict(path=str(a.input),pin=a.pin),code=s['code'],
        updates=updates,decisions=decisions,rollouts=rollouts,all_transfers_exact=True,all_updates_exact=True,all_trajectories_exact=True,
        all_native_scores_recounted=True,optimizer_ages_preserved=True,results=results,controls=negative,
        phases=phases,seconds=seconds,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,promoted=False,capability_demonstrated=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',out),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
