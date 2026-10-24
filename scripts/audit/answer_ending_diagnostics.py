"""Frozen full-answer ending signals, five TRAIN-only rules and shuffled readout."""
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
from audit import generated_answer_training as source
from audit.answer_surface_stream import rejected
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import answer_ending_probe as probe
from model import breadth_answer_transfer as transfer
from model.native_population_training import cpu,fingerprint
from text.constellation_corpus import file_sha,need

PARENT=('generated-answer-training-v1-20261024','dd6a3cfabe0d8bcaf64bffe18db198a67487767078fa9d7a9844e4e0f89d21fe')
COLD=('generated-answer-readback-v1-20261024','d21e3b0eb99b5db0d4ba6d28f782c241f69c41d8cce7805fa8f9d65616291081')
NEW=('model/answer_ending_probe.py','audit/answer_ending_diagnostics.py','audit/answer_ending_readback.py')


def load(root):
    s=source.load(root);reports=[]
    for name,pin in (PARENT,COLD):
        p=root/name/'report.json';need(file_sha(p)==pin,'END_DIAGNOSTIC_PARENT');reports.append(json.loads(p.read_bytes()))
    r,c=reports;need(r['complete'] and c['complete'] and c['source']['pin']==PARENT[1] and r['code']==s['code']
        and c['all_updates_exact'] and c['all_feedback_exact'] and c['all_trajectories_exact'],'END_DIAGNOSTIC_AUTHORITY')
    for name,pin in r['files'].items():need(file_sha(root/PARENT[0]/name)==pin,'END_DIAGNOSTIC_PARENT_BYTES')
    s['code'].update({n:file_sha(Path(__file__).resolve().parents[1]/n) for n in NEW})
    s['states']={};s['rollouts']={};s['boundaries']=[]
    for doc,item in s['items'].items():
        # The inherited placement bundle stores UIDs for successor targets [start+1:stop].
        need(item['stop']-item['start']==len(item['frames']) and len(item['uids'])==len(item['frames'])-1,
             'END_DIAGNOSTIC_SOURCE_BOUNDARY')
        s['boundaries'].append({n:item[n] for n in ['document','side','start','stop','component','native_pin','label_pin']})
    for arm in ('initial','breadth'):
        for mode in ('parent','reference','half','generated'):
            key=arm+'/'+mode
            if mode=='parent':
                s['states'][key]=s['saved'][arm];evaluation=root/source.PARENT[0]/arm/'final-evaluation.pt'
            else:
                base=root/PARENT[0]/(arm+'-'+mode)
                s['states'][key]=torch.load(base/'epoch-4.pt',map_location='cpu',weights_only=True)['state'];evaluation=base/'evaluation-4.pt'
            rows=torch.load(evaluation,map_location='cpu',weights_only=True)
            s['rollouts'][key]={v['document']:v for v in rows if v['mode']=='correct'}
            need(len(s['rollouts'][key])==10,'END_DIAGNOSTIC_ROLLOUT_CENSUS')
    return s


def make(s,key):
    model,opt=source.make(s,key.split('/')[0]);saved=s['states'][key]
    transfer.restore(model,opt,saved,saved['binding'],s['policy'],pin=fingerprint(saved));model.eval()
    return model,opt


@torch.no_grad()
def document(s,model,key,item,request,history):
    c,a=len(item['context']),len(item['answers']);stream=model.stream(request);state=None;inputs=[];hidden=[];stops=[]
    rng=torch.Generator().manual_seed(20261024+item['document']);candidate=None;reused=s['rollouts'][key].get(item['document'])
    need(history in ('reference','generated'),'END_DIAGNOSTIC_HISTORY')
    if reused is not None:
        need(reused['refusal'] is None and len(reused['events'])==a+1 and all(e['kind']=='constellation' for e in reused['events']),
             'END_DIAGNOSTIC_EARLY_END_REQUIRES_NEW_PROTOCOL')
    for pos in range(c+a):
        if pos<c:frame=item['context'][pos]
        elif history=='reference':frame=item['answers'][pos-c]
        elif reused is not None:frame=reused['events'][pos-c]['slots']
        else:frame=candidate
        h,state=stream.step((frame,),context='ending-diagnostic',offset=pos,state=state,detach_history=True)
        if pos<c-1:continue
        index=pos-c+1;stop=model.stop_loss(h,index==a)['proposals']['assembled']['logits'][0]
        hidden.append(cpu(h[0]));stops.append(cpu(stop));inputs.append(fingerprint(frame))
        if history=='generated' and reused is not None:
            need(torch.equal(stop,reused['events'][index]['stop_logits'][0]),'END_DIAGNOSTIC_PRIOR_LOGITS')
        if history=='generated' and reused is None and index<a:
            forecast=model.forecaster.predictor.forecast(h)
            noise=torch.randn(1,1,s['policy']['noise_width'],generator=rng)
            value=model.forecaster.placement['assembled'](h,forecast['codes'],forecast['counts'],noise,target=s['target'])[0]
            candidate={n:x[0].detach() for n,x in value.items()}
    hidden=torch.stack(hidden);ending=torch.zeros(a+1,dtype=torch.bool);ending[-1]=True
    row=dict(document=item['document'],side=item['side'],history=history,binding=item['binding'],
        source_native=item['native_pin'],source_labels=item['label_pin'],checkpoint=fingerprint(s['states'][key]),
        hidden={n:hidden[:,i,:].clone() for i,n in enumerate(model.roster)},stop=torch.stack(stops),end=ending,input_pins=inputs)
    row['pin']=probe.seal(row);probe.check([row]);return row


def controls(rows):
    train=[r for r in rows if r['side']=='train' and r['history']=='reference'];q=next(r for r in rows if r['side']=='query')
    out={'query_fit':rejected(lambda:probe.fit([q]),'END_PROBE_NONTRAIN')}
    for name in ('hidden','boundary','source'):
        bad=deepcopy(train[0])
        if name=='hidden':next(iter(bad['hidden'].values()))[0,0]+=1
        elif name=='boundary':bad['end'][0]=True
        else:bad['source_native']='0'*64
        out[name]=rejected(lambda:probe.check([bad],True),'END_PROBE_INPUT_PIN')
    bad=deepcopy(train[0]);bad['end'][0]=True;bad['pin']=probe.seal(bad)
    out['resigned_boundary']=rejected(lambda:probe.check([bad],True),'END_PROBE_BOUNDARY')
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={};results={};index=[];decisions=payload=0
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True));s=load(a.root)
    write_committed(a.output/'source.json',dict(parent=PARENT,cold=COLD,code=s['code'],boundaries=s['boundaries'],
        checkpoints={k:fingerprint(v) for k,v in s['states'].items()},ridge=probe.RIDGE,tolerance=probe.TOL))
    phases['authentication']=time.perf_counter()-tick
    for key in s['states']:
        tick=time.perf_counter();m,o=make(s,key);original=fingerprint((m.state_dict(),o.state_dict()));rows=[]
        for history in ('reference','generated'):
            for doc,(item,request) in s['items_ready'].items():rows.append(document(s,m,key,item,request,history))
        decisions+=sum(len(r['end']) for r in rows);size=sum(x.numel()*x.element_size() for r in rows for x in r['hidden'].values());payload+=size
        need(size<=s['policy']['cache_bytes'],'END_DIAGNOSTIC_CACHE');phases[key+'/histories']=time.perf_counter()-tick;tick=time.perf_counter()
        train=[r for r in rows if r['side']=='train' and r['history']=='reference'];linear=probe.fit(train);shuffled=probe.fit(train,True)
        base=probe.baselines(train);scores=[probe.score(r,base,linear,shuffled) for r in rows]
        negative=controls(rows);phases[key+'/probe_fits']=time.perf_counter()-tick;tick=time.perf_counter()
        values={}
        for history in ('reference','generated'):
            for side in ('train','development','query'):
                chosen=[i for i,r in enumerate(rows) if r['history']==history and r['side']==side]
                values[history+'/'+side]=probe.metrics([rows[i] for i in chosen],[scores[i] for i in chosen])
        saved=dict(rows=rows,base=base,linear=linear,shuffled=shuffled,scores=[{n:torch.from_numpy(v) for n,v in x.items()} for x in scores])
        path=a.output/(key.replace('/','-')+'.pt');persist(path,saved)
        need(fingerprint((m.state_dict(),o.state_dict()))==original,'END_DIAGNOSTIC_MODEL_MUTATION')
        results[key]=values;index.append(dict(state=key,file=path.name,pin=file_sha(path),checkpoint=fingerprint(s['states'][key]),
            cache_bytes=size,controls=negative,iterations={n:len(v['trace'])-1 for n,v in [('linear',linear),('shuffled',shuffled)]}))
        phases[key+'/persistence']=time.perf_counter()-tick;print(json.dumps(dict(state=key,decisions=decisions)),flush=True)
    tick=time.perf_counter();s['fresh']();need(decisions==3200 and payload==2457600,'END_DIAGNOSTIC_CENSUS')
    need(not torch.cuda.is_initialized(),'END_DIAGNOSTIC_CPU');write_committed(a.output/'results.json',results);write_committed(a.output/'index.json',index)
    files={str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    phases['final_authentication']=time.perf_counter()-tick;seconds=time.perf_counter()-began;need(.9<=sum(phases.values())/seconds<=1.001,'END_DIAGNOSTIC_PHASES')
    report=dict(complete=True,acceptance='author-checked',code=s['code'],files=files,decisions=decisions,hidden_bytes=payload,
        states=len(index),fits=2*len(index),model_optimizer_unchanged=True,promoted=False,native_encoder_calls=0,
        phases=phases,seconds=seconds,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.source.ROOT);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
