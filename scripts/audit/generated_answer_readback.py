"""Cold replay of feedback choices, full gradients, Adam ages and autonomous answers."""
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
from audit import generated_answer_training as source
from audit.answer_surface_stream import rejected
from audit.breadth_answer_readback import check_ages
from audit.full_history_readback import autonomous,same
from claims.process_conditions import write_committed
from model.native_population_training import cpu,fingerprint
from text.constellation_corpus import file_sha,need


def make(s,arm,saved=None):
    value=s['saved'][arm] if saved is None else saved
    _,model=source.source.models(s,0 if arm=='initial' else 91540)
    model.load_state_dict(value['model'],strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=s['policy']['learning_rate'],weight_decay=s['policy']['weight_decay'],foreach=False,fused=False)
    opt.load_state_dict(deepcopy(value['optimizer']));check_ages(model,opt,value['binding'],value['cursor'])
    return model,opt


def snapshot(s,model,opt,arm,cursor):
    return dict(schema='broad-answer-checkpoint-v1',model=model.state_dict(),optimizer=opt.state_dict(),
                binding=s['saved'][arm]['binding'],cursor=cursor)


def replay(s,m,opt,arm,item,request,cursor,probability,saved):
    r=saved['record'];need(r['before']==fingerprint(m.state_dict()) and r['probability']==probability,'FEEDBACK_COLD_BEFORE')
    check_ages(m,opt,s['saved'][arm]['binding'],cursor);opt.zero_grad(set_to_none=True)
    first={};hooks=[];losses=[];records=[]
    def capture(name):
        def hook(module,args,result):
            if name not in first:first[name]=result
        return hook
    for n,projection in m.forecaster.predictor.projectors.items():hooks.append(projection.register_forward_hook(capture(n)))
    c,a=len(item['context']),len(item['answers']);p=s['policy']
    flags=(torch.rand(a,generator=torch.Generator().manual_seed(p['seed']+2000000+cursor))<probability).tolist()
    samples=torch.randn(a,p['samples'],p['noise_width'],generator=torch.Generator().manual_seed(p['noise_seed']+cursor))
    feedback_samples=torch.randn(a,1,p['noise_width'],generator=torch.Generator().manual_seed(p['noise_seed']+1000000+cursor))
    stream=m.stream(request);cache=None;supplied=None;generated=overrides=0
    try:
        for pos in range(c+a):
            frame=item['context'][pos] if pos<c else supplied
            h,cache=stream.step((frame,),context='cold-feedback',offset=pos,state=cache,detach_history=False)
            if pos<c-1:continue
            index=pos-c+1;end=index==a;ref=r['records'][index]
            need(ref['position']==pos and ref['answer_index']==(None if end else index) and ref['end']==end
                 and ref['input_pin']==fingerprint(frame),'FEEDBACK_COLD_POSITION')
            stop=m.stop_loss(h,end);same(stop,ref['stop'],'FEEDBACK_COLD_STOP');loss=stop['loss']/(a+1)
            if end:need(ref['feedback'] is None and ref['continuation'] is None,'FEEDBACK_COLD_END')
            else:
                fit=m.forecaster.losses(h,item['codes'][index:index+1],{n:v[index:index+1] for n,v in item['counts'].items()},
                    (item['answers'][index],),samples[index:index+1],target=s['target'])
                need(float(fit['loss'].detach())==ref['continuation'],'FEEDBACK_COLD_NATIVE_LOSS');loss=loss+fit['loss']/a
                if flags[index]:
                    with torch.no_grad():
                        estimate=m.forecaster.predictor.forecast(h.detach())
                        value=m.forecaster.placement['assembled'](h.detach(),estimate['codes'],estimate['counts'],
                            feedback_samples[index:index+1],target=s['target'])[0]
                        supplied={n:t[0].detach() for n,t in value.items()}
                    feedback=dict(source='generated_unverified',slots=fingerprint(supplied),codes=cpu(estimate['codes']),
                        counts=cpu(estimate['counts']),noise=fingerprint(feedback_samples[index:index+1]),
                        detached=all(not t.requires_grad for t in supplied.values()))
                    need(feedback['detached'],'FEEDBACK_COLD_DETACHED');generated+=1
                else:supplied=item['answers'][index];feedback=dict(source='reference',slots=fingerprint(supplied))
                overridden=bool(stop['proposals']['assembled']['logits'].argmax(-1).item());overrides+=int(overridden)
                feedback['training_continuation_override']=overridden;same(feedback,ref['feedback'],'FEEDBACK_COLD_CANDIDATE')
            need(float(loss.detach())==ref['weighted_loss'],'FEEDBACK_COLD_LOSS');losses.append(loss);records.append(ref)
        total=torch.stack(losses).sum();end_grads=torch.autograd.grad(losses[-1],tuple(first.values()),retain_graph=True)
        context_grads=torch.autograd.grad(total,tuple(first.values()),retain_graph=True)
        early={n:dict(value=cpu(v),gradient=cpu(g),end_gradient=cpu(e)) for (n,v),g,e in zip(first.items(),context_grads,end_grads)}
        need(set(first)==set(m.roster) and all(bool(g.abs().any()) and bool(e.abs().any()) for g,e in zip(context_grads,end_grads)),
             'FEEDBACK_COLD_CONTEXT_CREDIT')
        same(early,saved['early'],'FEEDBACK_COLD_EARLY');need(fingerprint(early)==r['early'],'FEEDBACK_COLD_EARLY_PIN')
        total.backward();grads={n:p.grad for n,p in m.named_parameters()}
        need(fingerprint(grads)==r['gradients'],'FEEDBACK_COLD_GRADIENTS')
        for prefix,pin in r['branch_gradients'].items():
            group={n:g for n,g in grads.items() if n.startswith(prefix)}
            need(group and any(bool(v.abs().any()) for v in group.values()) and fingerprint(group)==pin,'FEEDBACK_COLD_BRANCH')
        opt.step();check_ages(m,opt,s['saved'][arm]['binding'],cursor+1)
        need(fingerprint(m.state_dict())==r['after'] and fingerprint(opt.state_dict())==r['optimizer'],'FEEDBACK_COLD_UPDATE')
        need(len(records)==r['decisions']==a+1 and r['cursor']==cursor+1,'FEEDBACK_COLD_CENSUS')
        need(r['storage']['remaining_saved_storage_bytes']==0 and r['storage']['peak_saved_storage_bytes']<=p['working_bytes'],
             'FEEDBACK_COLD_STORAGE_RECEIPT')
        return dict(decisions=a+1,generated=generated,overrides=overrides,peak=r['storage']['peak_saved_storage_bytes'])
    finally:
        for hook in hooks:hook.remove()
        first.clear()


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};updates=decisions=rollouts=0;results={};census={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'FEEDBACK_COLD_REPORT_PIN');r=json.loads((a.input/'report.json').read_bytes())
    need(r['complete'],'FEEDBACK_COLD_REPORT')
    for name,pin in r['files'].items():need(file_sha(a.input/name)==pin,'FEEDBACK_COLD_BYTES')
    s=source.load(a.root);need(s['code']==r['code'],'FEEDBACK_COLD_CODE');phases['authentication']=time.perf_counter()-tick
    load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    for arm in ('initial','breadth'):
        for mode,probability in [('reference',0.),('half',.5),('generated',1.)]:
            tick=time.perf_counter();base=a.input/(arm+'-'+mode);m,opt=make(s,arm);m.train();ledger=[]
            initial=load(base/'initial.pt');recipe=dict(schema='generated-answer-recipe-v1',parent=fingerprint(s['saved'][arm]),
                parent_step=5,arm=arm,mode=mode,probability=probability,epochs=4,train=s['train'],policy=s['policy'],code=s['code'],
                source_native={k:s['items'][k]['native_pin'] for k in s['train']},source_labels={k:s['items'][k]['label_pin'] for k in s['train']})
            same(initial,dict(schema='generated-answer-checkpoint-v1',recipe=recipe,ledger=[],state=snapshot(s,m,opt,arm,5)),
                 'FEEDBACK_COLD_INITIAL')
            counts=dict(updates=0,decisions=0,generated_inputs=0,training_overrides=0,peak_saved_storage=0)
            for epoch in range(4):
                for doc in s['train']:
                    cursor=5+len(ledger);item,request=s['items_ready'][doc];saved=load(base/f'update-{cursor+1}.pt')
                    row=replay(s,m,opt,arm,item,request,cursor,probability,saved);record=saved['record']
                    ledger.append({k:record[k] for k in ('cursor','document','after','optimizer','gradients','early')})
                    counts['updates']+=1;counts['decisions']+=row['decisions'];counts['generated_inputs']+=row['generated']
                    counts['training_overrides']+=row['overrides'];counts['peak_saved_storage']=max(counts['peak_saved_storage'],row['peak'])
                cp=load(base/f'epoch-{epoch+1}.pt')
                same(cp,dict(schema='generated-answer-checkpoint-v1',recipe=recipe,ledger=ledger,state=snapshot(s,m,opt,arm,5+len(ledger))),
                     'FEEDBACK_COLD_CHECKPOINT')
                if epoch==0:
                    m,opt=make(s,arm,cp['state']);same(snapshot(s,m,opt,arm,5+len(ledger)),cp['state'],'FEEDBACK_COLD_RESTART');m.train()
                if epoch in (0,1,3):
                    observed=load(base/f'evaluation-{epoch+1}.pt');results[arm+'/'+mode+'/'+str(epoch+1)]=autonomous(s,m,observed)
                    rollouts+=len(observed);m.train()
            for name,value in counts.items():need(value==r['results'][arm+'/'+mode][name],'FEEDBACK_COLD_TOTAL_'+name)
            need(r['results'][arm+'/'+mode]['final']==fingerprint(m.state_dict()) and
                 r['results'][arm+'/'+mode]['optimizer']==fingerprint(opt.state_dict()),'FEEDBACK_COLD_FINAL')
            census[arm+'/'+mode]=counts;updates+=counts['updates'];decisions+=counts['decisions']
            phases[arm+'_'+mode]=time.perf_counter()-tick;print(json.dumps(dict(arm=arm,mode=mode,updates=counts['updates'])),flush=True)
    tick=time.perf_counter();negative={}
    for name in ('recipe','optimizer','feedback'):
        wrong=deepcopy(cp if name!='feedback' else saved)
        if name=='recipe':wrong['recipe']['probability']=.5
        elif name=='optimizer':next(iter(wrong['state']['optimizer']['state'].values()))['exp_avg'].add_(1)
        else:wrong['record']['records'][0]['feedback']['slots']='0'*64
        original=cp if name!='feedback' else saved
        negative[name]=rejected(lambda:same(wrong,original,'FEEDBACK_COLD_CORRUPT_'+name),'FEEDBACK_COLD_CORRUPT_'+name)
    s['fresh']();need((updates,decisions,rollouts)==(120,1584,720),'FEEDBACK_COLD_DENOMINATOR')
    write_committed(a.output/'results.json',dict(results=results,census=census))
    phases['final_controls_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'FEEDBACK_COLD_PHASES')
    report=dict(complete=True,acceptance='author-checked',source=dict(path=str(a.input),pin=a.pin),code=s['code'],
        updates=updates,decisions=decisions,rollouts=rollouts,all_feedback_exact=True,all_updates_exact=True,all_trajectories_exact=True,
        optimizer_ages_preserved=True,controls=negative,results_pin=file_sha(a.output/'results.json'),phases=phases,seconds=seconds,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,promoted=False,capability_demonstrated=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
