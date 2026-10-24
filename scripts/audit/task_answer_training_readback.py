"""Cold complete-answer updates, checkpoints, rollouts and denominator/score recount."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--cpu-proof' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import task_answer_training as producer
from claims.process_conditions import write_committed
from lenscommon import release_lease
from model import task_answer_training as training
from model.native_population_training import fingerprint
from model.native_requested_answer import update
from text.constellation_corpus import file_sha,need


def independent_score(result,item,roster):
    rows=[v for v in result['events'] if v['kind']=='constellation'];ended=bool(result['events']) and result['events'][-1]['kind']=='end'
    future=item['answers'];scores={}
    for n in roster:
        values=[]
        for i,truth in enumerate(future):
            if i>=len(rows):values.append(-1.);continue
            x,y=rows[i]['slots'][n].double(),truth[n].double()
            x=x/torch.linalg.vector_norm(x,dim=1)[:,None];y=y/torch.linalg.vector_norm(y,dim=1)[:,None]
            similarities=(x@y.T).clamp(-1,1)
            values.append(float(torch.minimum(similarities.max(1).values.mean(),similarities.max(0).values.mean())))
        scores[n]=values
    return dict(expected_atoms=len(future),generated_atoms=len(rows),ended=ended,
        ended_at_reference_boundary=ended and len(rows)==len(future),missing=max(0,len(future)-len(rows)),
        extra=max(0,len(rows)-len(future)),per_slot=scores)


def score_equal(actual,expected):
    for k,v in actual.items():
        if k!='per_slot':need(v==expected[k],'TASK_COLD_DENOMINATOR')
        else:
            for n,values in v.items():
                need(len(values)==len(expected[k][n]) and max(abs(a-b) for a,b in zip(values,expected[k][n]))<=1e-12,
                     'TASK_COLD_NATIVE_SCORE')


def read_evaluation(base,model,target,data,*,shuffled):
    index=json.loads((base/'index.json').read_bytes());need([r['task'] for r in index]==data['query'],'TASK_COLD_QUERY_COVERAGE')
    aggregate={};rollouts=0
    for entry in index:
        key=entry['task'];path=base/(key+'.pt');need(file_sha(path)==entry['sha256'],'TASK_COLD_TRAJECTORY_BYTES')
        rows=torch.load(path,map_location='cpu',weights_only=True);item,request=data['item'](key);donor=data['donors'][key]
        expected=[(mode,p) for mode in (['correct','shuffled'] if shuffled and donor is not None else ['correct']) for p in model.proposals]
        need([(r['mode'],r['proposal']) for r in rows]==expected and len(rows)==entry['rollouts'],'TASK_COLD_CONTROL_COVERAGE')
        for row in rows:
            given,req=item,request
            if row['mode']=='shuffled':
                other,req=data['item'](donor);given=dict(item,context=other['context'])
            need(row['request']==req.pin and row['family']==data['families'][key],'TASK_COLD_REQUEST_IDENTITY')
            result=producer.observe(model,target,given,req,row['proposal'],int(producer.digest(producer.canonical([key,'answer-noise']))[:12],16))
            need(fingerprint(result)==fingerprint(dict(events=row['events'],refusal=row['refusal'])),'TASK_COLD_ROLLOUT')
            check=independent_score(result,item,model.roster);score_equal(check,row['metrics']);rollouts+=1
            group='|'.join((row['family'],row['mode'],row['proposal']))
            v=aggregate.setdefault(group,dict(tasks=0,expected_atoms=0,generated_atoms=0,exact_end=0,no_end=0,refusals=0,
                missing=0,extra=0,per_slot={n:dict(total=0.,copy_total=0.,atoms=0) for n in model.roster}))
            for k in ('expected_atoms','generated_atoms','missing','extra'):v[k]+=check[k]
            v['tasks']+=1;v['exact_end']+=int(check['ended_at_reference_boundary']);v['no_end']+=int(not check['ended'])
            v['refusals']+=int(result['refusal'] is not None)
            for n in model.roster:
                cell=v['per_slot'][n];cell['total']+=sum(check['per_slot'][n]);cell['atoms']+=len(item['answers'])
                cell['copy_total']+=sum(producer.sequence.native_similarity(item['context'][-1][n],f[n]) for f in item['answers'])
    for group in aggregate.values():
        for v in group['per_slot'].values():v.update(mean=v['total']/v['atoms'],copy_mean=v['copy_total']/v['atoms'])
    return dict(rollouts=rollouts,groups=aggregate)


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();lease=False;phases={}
    try:
        source=producer.pinned(a.input,a.pin);need(source['complete'],'TASK_COLD_INCOMPLETE')
        a.mode=source['mode'];a.consumer=a.consumer_pin=a.consumer_cold=a.consumer_cold_pin=None
        if source['consumer']:
            r=source['consumer'];a.consumer=Path(r['path']);a.consumer_pin=r['pin'];a.consumer_cold=Path(r['cold']);a.consumer_cold_pin=r['cold_pin']
        device,lease=producer.environment(a);need(device==source['device'],'TASK_COLD_RUNTIME')
        for n,h in source['files'].items():need(file_sha(a.input/n)==h,'TASK_COLD_PAYLOAD')
        b,s,code,_=producer.source(a.root);need(code==source['code'],'TASK_COLD_CODE')
        target,pin=training.extension(b);data=producer.data(a,b,s,target);target=target.to(device)
        need(pin==source['extension_pin'] and data['binding']==source['binding'],'TASK_COLD_SOURCE')
        phases['authentication']=time.perf_counter()-tick;results={}
        for arch in producer.ARCHITECTURES:
            tick=time.perf_counter();base=a.input/arch;model,opt=training.make(b,target,pin,arch,device)
            binding=producer.binding(b,pin,code,data,arch,device)
            initial=torch.load(base/'initial.pt',map_location='cpu',weights_only=True)
            need(fingerprint(training.checkpoint(model,opt,binding,[]))==fingerprint(initial),'TASK_COLD_INITIAL')
            before=read_evaluation(base/'initial-evaluation',model,target,data,shuffled=False)
            expected=[json.loads(line) for line in (base/'updates.jsonl').read_bytes().splitlines()];ledger=[]
            need(len(expected)==binding['updates'],'TASK_COLD_UPDATE_DENOMINATOR')
            for i,row in enumerate(expected):
                key=data['train'][i%len(data['train'])];epoch=i//len(data['train'])
                item,request=data['item'](key);record,_=update(model,opt,item,i,binding['policy'],request=request,target=target)
                actual=dict(step=i+1,epoch=epoch,task=key,update=training.json_value(record))
                need(actual==row,'TASK_COLD_UPDATE');ledger.append(actual)
                path=base/f'checkpoint-{i+1:06d}.pt'
                if path.exists():
                    receipt=json.loads(path.with_suffix('.json').read_bytes());need(receipt==dict(path=path.name,sha256=file_sha(path),cursor=i+1),'TASK_COLD_CHECKPOINT_RECEIPT')
                    value=torch.load(path,map_location='cpu',weights_only=True)
                    need(fingerprint(value)==fingerprint(training.checkpoint(model,opt,binding,ledger)),'TASK_COLD_CHECKPOINT_STATE')
                    if i==1:
                        model,opt=training.make(b,target,pin,arch,device);ledger=training.restore(value,model,opt,binding)
            after=read_evaluation(base/'final-evaluation',model,target,data,shuffled=True)
            need(fingerprint(model.state_dict())==source['results'][arch]['final_model']
                 and fingerprint(opt.state_dict())==source['results'][arch]['optimizer'],'TASK_COLD_FINAL_STATE')
            write_committed(a.output/(arch+'.json'),dict(initial=before,final=after))
            results[arch]=dict(updates=len(ledger),initial_rollouts=before['rollouts'],final_rollouts=after['rollouts'],
                all_updates_exact=True,all_trajectories_exact=True,restored_at=2)
            phases[arch]=time.perf_counter()-tick
            print(json.dumps(dict(architecture=arch,**results[arch])),flush=True)
        tick=time.perf_counter();data['close']()
        need(all(file_sha(Path(__file__).resolve().parents[1]/n)==h for n,h in code.items()),'TASK_COLD_FINAL_SOURCE')
        phases['final_authentication']=time.perf_counter()-tick;elapsed=time.perf_counter()-began
        need(.9<=sum(phases.values())/elapsed<=1.001,'TASK_COLD_PHASE_COVERAGE')
        r=dict(complete=True,input=dict(path=str(a.input),pin=a.pin),source_mode=source['mode'],device=device,code=code,results=results,
            all_updates_exact=True,all_trajectories_exact=True,phases=phases,seconds=elapsed,
            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            scientific_admission=False,capability_demonstrated=False,acceptance='author-checked')
        print(json.dumps(dict(pin=write_committed(a.output/'report.json',r),seconds=elapsed)),flush=True)
    except BaseException as exc:write_committed(a.output/'failure.json',dict(error=repr(exc),phases=phases));raise
    finally:
        if lease:release_lease()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=producer.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-proof',action='store_true');run(p.parse_args())
