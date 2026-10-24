"""Independent replay and NumPy scores for transfer-component diagnostics."""
import argparse
from collections import defaultdict
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
from audit import answer_transfer_diagnostics as source
from audit.answer_surface_stream import rejected
from audit.breadth_answer_readback import check_ages
from audit.full_history_readback import same
from claims.process_conditions import write_committed
from model.native_population_training import fingerprint
from text.constellation_corpus import file_sha,need


def array(x):return x.detach().cpu().numpy().astype(np.float64)


def close(a,b):
    need(np.isfinite(a) and np.isfinite(b) and abs(float(a)-float(b))<1e-12,'TRANSFER_COMPONENT_NUMPY_SCORE')


def similarity(x,y):
    a,b=array(x),array(y);a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
    dot=np.clip(a@b.T,-1,1)
    return float(min(dot.max(0).mean(),dot.max(1).mean()))


def nll(logits,label):
    x=array(logits).reshape(-1);return float(np.logaddexp.reduce(x)-x[label])


def energy(samples,target,variances,reported):
    distances={};pairs={};result={}
    for n in target:
        x,y=array(samples[n]),array(target[n]);scale=np.sqrt(y.size*float(variances[n]))
        distances[n]=np.linalg.norm((x-y[None]).reshape(len(x),-1),axis=1)/scale
        pairs[n]=np.array([np.linalg.norm(x[i]-x[j])/scale for i in range(len(x)) for j in range(i+1,len(x))])
        result[n]=float(distances[n].mean()-.5*pairs[n].mean());close(result[n],reported['lanes'][n][0])
    joint=float(np.linalg.norm(np.stack(list(distances.values())),axis=0).mean()/np.sqrt(len(target))
                -.5*np.linalg.norm(np.stack(list(pairs.values())),axis=0).mean()/np.sqrt(len(target)))
    close(joint,reported['joint'][0]);close(np.mean(list(result.values()))+joint,reported['loss'])
    return result,joint


@torch.no_grad()
def recount(s,model,item,request,saved,autonomous,hazard):
    need(saved['document']==item['document'] and saved['side']==item['side'] and saved['binding']==item['binding'],
         'TRANSFER_COMPONENT_DOCUMENT')
    source.reference(item);stream=model.stream(request);cache=None;frames=(*item['context'],*item['answers'])
    head=model.forecaster.predictor;placement=model.forecaster.placement['assembled'];values=defaultdict(list)
    c,a=len(item['context']),len(item['answers']);cursor=0;generator=torch.Generator().manual_seed(20261024+item['document'])
    need(len(saved['rows'])==a+1,'TRANSFER_COMPONENT_DENOMINATOR')
    observed=[] if autonomous is None else [r for r in autonomous['events'] if r['kind']=='constellation']
    for pos,frame in enumerate(frames):
        h,cache=stream.step((frame,),context='cold-transfer-components',offset=pos,state=cache,detach_history=True)
        if pos<c-1:continue
        row=saved['rows'][cursor];end=cursor==a
        need((row['position'],row['index'],row['end'],row['hidden'])==(pos,cursor,end,fingerprint(h)),'TRANSFER_COMPONENT_POSITION')
        stop=model.stop_loss(h,end);same(stop,row['stop'],'TRANSFER_COMPONENT_STOP')
        logits=stop['proposals']['assembled']['logits'];x=array(logits)[0];p=float(np.exp(x[1]-np.logaddexp.reduce(x)))
        stop_nll=nll(logits,int(end));close(stop_nll,stop['loss'])
        values['end_nll'].append(stop_nll);values['end_brier'].append((p-int(end))**2)
        values['constant_brier'].append((hazard-int(end))**2);values['end_correct'].append(int(x.argmax()==int(end)))
        if end:values['at_end_probability'].append(p);cursor+=1;continue
        codes=item['codes'][cursor:cursor+1];counts={n:v[cursor:cursor+1] for n,v in item['counts'].items()}
        future=item['answers'][cursor];predicted=head.forecast(h);same(predicted,row['forecast'],'TRANSFER_COMPONENT_PREDICTION')
        same((codes,counts),(row['codes'],row['counts']),'TRANSFER_COMPONENT_LABEL_TARGET')
        seed=20261024+item['document']*1024+cursor
        noise=torch.randn(1,s['policy']['samples'],s['policy']['noise_width'],generator=torch.Generator().manual_seed(seed))
        one=torch.randn(1,1,s['policy']['noise_width'],generator=generator)
        need(seed==row['noise_seed'] and fingerprint(noise)==row['noise'] and fingerprint(one)==row['candidate_noise'],
             'TRANSFER_COMPONENT_NOISE')
        fit=model.forecaster.losses(h,codes,counts,(future,),noise,target=s['target'])['proposals']['assembled']
        same(fit,row['fit'],'TRANSFER_COMPONENT_FIT')
        for level,logit in enumerate(fit['discrete']['logits']):
            loss=nll(logit,int(codes[0,level]));close(loss,fit['discrete']['region'][level])
            values['region_nll/'+str(level)].append(loss)
            values['region_accuracy/'+str(level)].append(int(predicted['codes'][0,level]==codes[0,level]))
        values['region_exact'].append(int(torch.equal(predicted['codes'],codes)))
        e,joint=energy(fit['samples'][0],future,placement.variances(),fit['native']);values['energy_joint'].append(joint)
        for i,n in enumerate(model.roster):
            window=head.counts[n].loss(h[:,i],counts[n]);same(window,row['window'][n],'TRANSFER_COMPONENT_WINDOW')
            x=array(window['logits'])[0];w=int(counts[n][0]);count_nll=float(np.logaddexp(0,x[:w-1]).sum()+np.logaddexp(0,-x[w-1]))
            close(count_nll,window['loss']);close(count_nll,fit['discrete']['counts'][n])
            values['count_nll/'+n].append(count_nll);values['count_accuracy/'+n].append(int(predicted['counts'][n][0]==w))
            values['energy/'+n].append(e[n])
        need(set(row['candidates'])==set(source.MODES),'TRANSFER_COMPONENT_MODE_CENSUS')
        for mode in source.MODES:
            given_codes=codes if mode in ('oracle_both','oracle_region') else predicted['codes']
            given_counts=counts if mode in ('oracle_both','oracle_count') else predicted['counts']
            generated=placement(h,given_codes,given_counts,one,target=s['target'])[0]
            candidate={n:x[0] for n,x in generated.items()};same(candidate,row['candidates'][mode],'TRANSFER_COMPONENT_CANDIDATE')
            for n in model.roster:
                v=similarity(candidate[n],future[n]);close(v,row['scores'][mode][n]);values['native/'+mode+'/'+n].append(v)
        base=s['target'].reconstruct(s['target'].values(codes),[{n:int(v[0]) for n,v in counts.items()}])[0]
        for n in model.roster:
            for mode,candidate in [('codebook',base),('copy_previous',frame)]:
                v=similarity(candidate[n],future[n]);close(v,row['scores'][mode][n]);values['native/'+mode+'/'+n].append(v)
            if autonomous is not None:
                value=similarity(observed[cursor]['slots'][n],future[n]) if cursor<len(observed) else -1.
                values['native/autonomous/'+n].append(value)
        if autonomous is not None and cursor==0:
            need(observed,'TRANSFER_COMPONENT_FIRST_MISSING')
            same(row['candidates']['predicted'],observed[0]['slots'],'TRANSFER_COMPONENT_MATCHED_FIRST_STEP')
        cursor+=1
    need(cursor==a+1,'TRANSFER_COMPONENT_COVERAGE');source.reference(item)
    return dict(document=item['document'],side=item['side'],answer_atoms=a,decisions=a+1,
                means={k:float(np.mean(v)) for k,v in values.items()})


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    need(file_sha(a.input/'report.json')==a.pin,'TRANSFER_COMPONENT_REPORT_PIN');r=json.loads((a.input/'report.json').read_bytes())
    need(r['complete'] and r['training_updates']==0,'TRANSFER_COMPONENT_REPORT')
    for name,pin in r['files'].items():need(file_sha(a.input/name)==pin,'TRANSFER_COMPONENT_BYTES')
    s=source.load(a.root);need(s['code']==r['code'],'TRANSFER_COMPONENT_CODE')
    items={doc:source.source.prior.prior.item(s,doc) for doc in sorted(s['items'])}
    hazard=float(np.mean([1/(len(item['answers'])+1) for item,_ in items.values() if item['side']=='train']))
    index=json.loads((a.input/'index.json').read_bytes());need(len(index)==60,'TRANSFER_COMPONENT_FILES')
    phases['authentication']=time.perf_counter()-tick;results={};decisions=continuations=0
    for arm,step in [('initial',0),('breadth',91540)]:
        for boundary in ('initial','final'):
            tick=time.perf_counter();saved=source.checkpoint(s,arm,boundary);_,model=source.source.models(s,step)
            model.load_state_dict(saved['model'],strict=True);model.eval()
            opt=torch.optim.AdamW(model.parameters(),lr=s['policy']['learning_rate'],weight_decay=s['policy']['weight_decay'],foreach=False,fused=False)
            opt.load_state_dict(deepcopy(saved['optimizer']));check_ages(model,opt,saved['binding'],saved['cursor'])
            seal=fingerprint((model.state_dict(),opt.state_dict()))
            observed=torch.load(s['phase110']/arm/(boundary+'-evaluation.pt'),map_location='cpu',weights_only=True)
            actual={v['document']:v for v in observed if v['mode']=='correct'};rows=[]
            for doc,(item,request) in items.items():
                matches=[v for v in index if (v['arm'],v['boundary'],v['document'])==(arm,boundary,doc)]
                need(len(matches)==1 and matches[0]['checkpoint']==fingerprint(saved),'TRANSFER_COMPONENT_INDEX')
                path=a.input/matches[0]['file'];need(file_sha(path)==matches[0]['sha256'],'TRANSFER_COMPONENT_ROW_PIN')
                payload=torch.load(path,map_location='cpu',weights_only=True)
                row=recount(s,model,item,request,payload,actual.get(doc),hazard);rows.append(row)
                decisions+=row['decisions'];continuations+=row['answer_atoms']
            need(fingerprint((model.state_dict(),opt.state_dict()))==seal,'TRANSFER_COMPONENT_STATE_CHANGED')
            results[arm+'/'+boundary]=rows;phases[arm+'_'+boundary]=time.perf_counter()-tick
    tick=time.perf_counter();summary={}
    for arm,rows in results.items():
        summary[arm]={}
        for side in ('train','development','query'):
            group=[v for v in rows if v['side']==side];keys=set(group[0]['means'])
            need(len(group)==5 and all(set(v['means'])==keys for v in group),'TRANSFER_COMPONENT_PARTITION')
            summary[arm][side]={k:float(np.mean([v['means'][k] for v in group])) for k in sorted(keys)}
    negative={}
    wrong=deepcopy(payload);next(iter(wrong['rows'][0]['candidates']['oracle_region'].values())).add_(1)
    negative['changed_candidate']=rejected(lambda:same(wrong,payload,'TRANSFER_COMPONENT_CORRUPT_OUTPUT'),'TRANSFER_COMPONENT_CORRUPT_OUTPUT')
    negative['changed_score']=rejected(lambda:close(.1,.2),'TRANSFER_COMPONENT_NUMPY_SCORE')
    s['fresh']();need((decisions,continuations)==(800,740),'TRANSFER_COMPONENT_TOTAL')
    write_committed(a.output/'summary.json',dict(results=results,summary=summary,train_constant_hazard=hazard))
    phases['summary_controls']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'TRANSFER_COMPONENT_PHASES')
    report=dict(complete=True,acceptance='author-checked',source=dict(path=str(a.input),pin=a.pin),code=s['code'],
        all_histories_outputs_exact=True,all_scores_numpy_recounted=True,all_first_steps_match_autonomous=True,
        decisions=decisions,continuations=continuations,controls=negative,oracle_diagnostic_only=True,
        summary_pin=file_sha(a.output/'summary.json'),seconds=seconds,phases=phases,
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,promoted=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.source.ROOT)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
