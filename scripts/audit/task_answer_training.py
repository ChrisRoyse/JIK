"""Complete task-answer training, with native labels reused from their cold publication."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--cpu-proof' in sys.argv or '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1]=='plan':
    os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_answer_sequence as sequence
from claims.process_conditions import write_committed
from lenscommon import take_lease,release_lease
from model.answer_training_schedule import AnswerSchedule
from model.native_conditioned_answer import document_item
from model.native_device_transport import transfer
from model.native_population_training import cpu,fingerprint
from model.native_request_input import NativeRequest,describe,legal
from model.native_requested_answer import update
from model import task_answer_training as training
from text.constellation_corpus import canonical,digest,file_sha,need

ROOT=sequence.ROOT
PLAN=('answer-native-plan-v1-20261023','22993ed7704f1d31ca48df404c0af14acff1d466624b412a9c55be2132cf1098')
PROOF=('open-window-answer-cpu-v2-20261024','a62c0402a911f7e7c7cafaf3f88e760c1fd6c332159931e10923028a3d3ae339')
COLD=('open-window-answer-readback-v2-20261024','abcd6121a90bc83118995db0877fc380f29d0581a18542d9d3b4f9a63b166242')
CENSUS=('canonical-answer-consumer-plan-v2-20261024','9311040852b8301bc172efc8e16b0ac1133010393d8212724d3ddc4d2b3eaf50')
NEW=('model/task_answer_training.py','audit/task_answer_training.py','audit/task_answer_training_readback.py','audit/task_answer_training_queue.py')
ARCHITECTURES=('shared','own_slot','whole_proposals')
EPOCHS=4


def pinned(root,pin):return sequence.runtime.pinned(Path(root)/'report.json',pin)


def source(root):
    prior=pinned(root/PROOF[0],PROOF[1]);cold=pinned(root/COLD[0],COLD[1]);census=pinned(root/CENSUS[0],CENSUS[1])
    need(prior['complete'] and cold['complete'] and cold['source']['pin']==PROOF[1]
         and all(r['all_cold_states_exact'] and r['all_trajectories_exact'] for r in cold['results'].values()),'TASK_TRAIN_PRIOR')
    base=Path(__file__).resolve().parents[1];code=dict(prior['code'],**census['code'])
    for n,h in code.items():need(file_sha(base/n)==h,'TASK_TRAIN_PRIOR_SOURCE')
    code.update({n:file_sha(base/n) for n in NEW})
    s,b=sequence.training.source_bundle(root);schedule=AnswerSchedule(root/PLAN[0],PLAN[1])
    return b,schedule,code,census


def donor_plan(schedule):
    # Matching reads public request structure, never private answer lengths/values.
    groups={};result={}
    for key,row in sorted(schedule.rows.items()):
        d=describe(schedule,key)
        signature=canonical(dict(operation=d['operation'],position=d['position_kind'],family=row['family'],
            fields=[{n:v for n,v in f.items() if n!='clock'} for f in d['fields']]))
        groups.setdefault(signature,[]).append(key)
    for group in groups.values():
        for key in group:
            row=schedule.rows[key]
            choices=[k for k in group if k!=key and schedule.rows[k]['group']!=row['group']]
            result[key]=min(choices,key=lambda k:digest(canonical(['task-context-donor-v1',key,k]))) if choices else None
    return result


def plan(root,output):
    b,s,code,c=source(root);donors=donor_plan(s);train=sorted(k for k,v in s.rows.items() if v['split']=='train')
    query=sorted(k for k,v in s.rows.items() if v['split']=='query')
    need(len(train)==319 and len(query)==170,'TASK_TRAIN_PLAN_CENSUS')
    write_committed(output/'schedule.json',dict(train=train,query=query,donors=donors))
    return dict(schema='complete-task-answer-plan-v1',complete=True,code=code,source=PLAN,summary=c['summary'],
        epochs=EPOCHS,updates_per_architecture=EPOCHS*len(train),decisions_per_architecture=EPOCHS*c['summary']['train']['decisions'],
        architectures=ARCHITECTURES,unavailable_query_controls=sum(donors[k] is None for k in query),
        label_recomputation=0,checkpoint_interval=64,files={'schedule.json':file_sha(output/'schedule.json')},
        scientific_admission=False,capability_demonstrated=False,acceptance='author-checked')


def data(a,b,s,target):
    if a.mode=='proof':
        original=sequence.training.load_data(argparse.Namespace(root=a.root,scope='small'),b)
        rows={str(r['document']):r for r in original['items']}
        def item(key):
            v=document_item(rows[key],2);request=NativeRequest(legal(2));v['provenance']['request']=request.pin
            v['binding']=fingerprint((v['document'],v['side'],v['provenance']));return v,request
        train=sorted(k for k,r in rows.items() if r['side']=='train');query=sorted(k for k in rows if k not in train)
        donors={k:next(t for t in train if rows[t]['component']!=rows[k]['component']) for k in query}
        return dict(item=item,train=train,query=query,donors=donors,epochs=1,close=original['close'],
            binding=dict(scope='real_legal_runtime_qualification',source=PROOF[1]),families={k:'legal_continuation' for k in rows})
    need(all(getattr(a,n,None) is not None for n in ('consumer','consumer_pin','consumer_cold','consumer_cold_pin')),
         'TASK_TRAIN_NATIVE_PUBLICATION_REQUIRED')
    result=training.task_inputs(s,a.consumer,a.consumer_pin,a.consumer_cold,a.consumer_cold_pin,target)
    result.update(train=sorted(k for k,r in s.rows.items() if r['split']=='train'),
        query=sorted(k for k,r in s.rows.items() if r['split']=='query'),donors=donor_plan(s),epochs=EPOCHS,
        families={k:r['family'] for k,r in s.rows.items()})
    return result


def observe(model,target,item,request,proposal,seed):
    device=next(model.parameters()).device
    generator=model.rollout(transfer(item['context'],device),request=request,target=target,proposal=proposal,
                            context='complete-task-answer',seed=seed)
    events=[];refusal=None
    try:
        # Observer-only scoring horizon. The model receives no length/target data.
        for _ in range(len(item['answers'])+1):
            row=next(generator);events.append(cpu(row))
            if row['kind']=='end':break
    except ValueError as exc:
        allowed=('WINDOW_END_PHYSICAL_MEMORY_EXHAUSTED_NO_TRUNCATION','PLACEMENT_OUTPUT_MEMORY',
                 'RQ_BATCH_PHYSICAL_MEMORY','PREFIX_STREAM_CURRENT_MEMORY','PREFIX_STREAM_CACHE_MEMORY')
        need(any(v in str(exc) for v in allowed),'TASK_TRAIN_UNEXPECTED_REFUSAL:'+str(exc));refusal=str(exc)
    finally:generator.close()
    return dict(events=events,refusal=refusal)


def metrics(result,item,roster):
    events=result['events'];future=item['answers'];copy=item['context'][-1]
    if events:r=sequence.metrics(events,future,roster,copy)
    else:r=dict(expected_atoms=len(future),generated_atoms=0,ended=False,ended_at_reference_boundary=False,
        censored=True,matched_positions=0,missing=len(future),extra=0,per_slot={n:[-1.]*len(future) for n in roster},
        copy_current={n:[sequence.native_similarity(copy[n],f[n]) for f in future] for n in roster},
        missing_similarity=-1.,semantic_validity_established=False)
    return dict(r,refusal=result['refusal'])


def evaluate(model,target,d,output,*,shuffled):
    output.mkdir();rows=[]
    for key in d['query']:
        item,request=d['item'](key);contexts={'correct':(item,request)};donor=d['donors'][key]
        if shuffled and donor is not None:
            other,other_request=d['item'](donor)
            need(len(item['context'])==len(other['context']),'TASK_TRAIN_DONOR_SHAPE')
            # Move the donor's observed slots and their attested clocks together.
            contexts['shuffled']=(dict(item,context=other['context']),other_request)
        payload=[]
        for mode,(given,req) in contexts.items():
            for proposal in model.proposals:
                result=observe(model,target,given,req,proposal,int(digest(canonical([key,'answer-noise']))[:12],16))
                scored=metrics(result,item,model.roster)
                payload.append(dict(task=key,family=d['families'][key],mode=mode,donor=donor if mode=='shuffled' else None,
                    proposal=proposal,request=req.pin,**result,metrics=scored))
        path=output/(key+'.pt');persist(path,payload)
        rows.append(dict(task=key,sha256=file_sha(path),rollouts=len(payload),control_available=donor is not None))
    write_committed(output/'index.json',rows);return rows


def persist(path,value):
    with path.open('xb') as f:torch.save(cpu(value),f);f.flush();os.fsync(f.fileno())


def binding(b,target_pin,code,d,architecture,device):
    return dict(source=d['binding'],target=target_pin,code=code,architecture=architecture,
        policy=dict(b['policy'],device=device),epochs=d['epochs'],train=d['train'],updates=len(d['train'])*d['epochs'],
        device=device,torch=str(torch.__version__))


def rejection(code,fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError) as exc:
        need(code in str(exc),'TASK_TRAIN_WRONG_REFUSAL');return dict(expected=code,observed=str(exc))
    raise ValueError('TASK_TRAIN_NEGATIVE_ACCEPTED')


def controls(b,target,pin,architecture,device,bound,model,opt,ledger,d):
    saved=training.checkpoint(model,opt,bound,ledger);state=fingerprint(saved)
    def load(value):
        m,o=training.make(b,target,pin,architecture,device);return training.restore(value,m,o,bound)
    broken=deepcopy(saved);broken['cursor']+=1
    wrong=deepcopy(saved);wrong['binding']=dict(bound,target='changed')
    weights=deepcopy(saved);weights['optimizer']['state']={}
    mutated=deepcopy(saved);next(iter(mutated['model'].values())).add_(1)
    out=[rejection('TASK_TRAIN_CHECKPOINT_CURSOR',lambda:load(broken)),
         rejection('TASK_TRAIN_CHECKPOINT_BINDING',lambda:load(wrong)),
         rejection('TASK_TRAIN_CHECKPOINT_STATE',lambda:load(weights)),
         rejection('TASK_TRAIN_CHECKPOINT_STATE',lambda:load(mutated))]
    held,req=d['item'](d['query'][0])
    out.append(rejection('CONDITIONED_NONTRAIN',lambda:update(model,opt,held,len(ledger),bound['policy'],request=req,target=target)))
    descriptor=json.loads(req.descriptor);descriptor['answer_length']=len(held['answers'])
    out.append(rejection('REQUEST_INPUT_SCHEMA',lambda:NativeRequest(descriptor)))
    need(fingerprint(training.checkpoint(model,opt,bound,ledger))==state,'TASK_TRAIN_CONTROL_MUTATION');return out


def environment(a):
    device='cpu' if a.cpu_proof else 'cuda';lease=False
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    if device=='cuda':
        take_lease('29','complete-task-answer');lease=True
        try:
            need(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                 and torch.cuda.is_available(),'TASK_TRAIN_GPU_BUSY_OR_UNAVAILABLE')
        except BaseException:release_lease();raise
    else:need(a.mode=='proof','TASK_TRAIN_NO_CPU_FULL')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return device,lease


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};lease=False
    try:
        if a.mode=='plan':
            r=plan(a.root,a.output);r['seconds']=time.perf_counter()-began
            print(write_committed(a.output/'report.json',r),flush=True);return
        device,lease=environment(a);b,s,code,_=source(a.root);target,pin=training.extension(b)
        d=data(a,b,s,target);target=target.to(device)
        if a.mode=='full':
            q=pinned(a.qualification,a.qualification_pin)
            need(q['complete'] and q['source_mode']=='proof' and q['device']=='cuda' and q['code']==code
                 and q['all_updates_exact'] and q['all_trajectories_exact'],'TASK_TRAIN_CUDA_QUALIFICATION')
        phases['source']=time.perf_counter()-tick;results={}
        for arch in ARCHITECTURES:
            tick=time.perf_counter();out=a.output/arch;out.mkdir();model,opt=training.make(b,target,pin,arch,device)
            bound=binding(b,pin,code,d,arch,device);ledger=[]
            persist(out/'initial.pt',training.checkpoint(model,opt,bound,ledger))
            before=evaluate(model,target,d,out/'initial-evaluation',shuffled=False)
            phases[arch+'_initial_evaluation']=time.perf_counter()-tick;tick=time.perf_counter()
            with (out/'updates.jsonl').open('xb') as log:
                for epoch in range(d['epochs']):
                    for key in d['train']:
                        item,request=d['item'](key);step=len(ledger)
                        record,_=update(model,opt,item,step,bound['policy'],request=request,target=target)
                        row=dict(step=step+1,epoch=epoch,task=key,update=training.json_value(record));ledger.append(row)
                        log.write(canonical(row)+b'\n')
                        boundary=(step+1)%64==0 or key==d['train'][-1] or step==1 or a.mode=='proof'
                        if boundary:
                            log.flush();os.fsync(log.fileno());cp=training.checkpoint(model,opt,bound,ledger)
                            path=out/f'checkpoint-{step+1:06d}.pt';persist(path,cp)
                            write_committed(out/f'checkpoint-{step+1:06d}.json',dict(path=path.name,sha256=file_sha(path),cursor=step+1))
                        if step==1:
                            cp=torch.load(path,map_location='cpu',weights_only=True);model,opt=training.make(b,target,pin,arch,device)
                            ledger=training.restore(cp,model,opt,bound)
                    print(json.dumps(dict(architecture=arch,epoch=epoch+1,updates=len(ledger))),flush=True)
            phases[arch+'_training']=time.perf_counter()-tick;tick=time.perf_counter()
            negative=controls(b,target,pin,arch,device,bound,model,opt,ledger,d);write_committed(out/'controls.json',negative)
            after=evaluate(model,target,d,out/'final-evaluation',shuffled=True)
            results[arch]=dict(updates=len(ledger),initial_rollouts=sum(v['rollouts'] for v in before),
                final_rollouts=sum(v['rollouts'] for v in after),parameters=sum(p.numel() for p in model.parameters()),
                final_model=fingerprint(model.state_dict()),optimizer=fingerprint(opt.state_dict()),controls=len(negative))
            phases[arch+'_final_evaluation']=time.perf_counter()-tick
        tick=time.perf_counter();d['close']()
        need(all(file_sha(Path(__file__).resolve().parents[1]/n)==h for n,h in code.items()),'TASK_TRAIN_FINAL_CODE')
        files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
        phases['final_authentication']=time.perf_counter()-tick;elapsed=time.perf_counter()-began
        need(.9<=sum(phases.values())/elapsed<=1.001,'TASK_TRAIN_PHASE_COVERAGE')
        r=dict(complete=True,mode=a.mode,device=device,code=code,files=files,binding=d['binding'],results=results,
            extension_pin=pin,epochs=d['epochs'],phases=phases,seconds=elapsed,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            consumer=None if a.mode=='proof' else dict(path=str(a.consumer),pin=a.consumer_pin,cold=str(a.consumer_cold),cold_pin=a.consumer_cold_pin),
            scientific_admission=False,capability_demonstrated=False,acceptance='author-checked')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',r),results=results,seconds=elapsed)),flush=True)
    except BaseException as exc:write_committed(a.output/'failure.json',dict(error=repr(exc),phases=phases));raise
    finally:
        if lease:release_lease()


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--mode',choices=['plan','proof','full'],required=True);p.add_argument('--cpu-proof',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    for n in ('consumer','consumer-cold','qualification'):
        p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-pin')
    return p


if __name__=='__main__':run(parser().parse_args())
