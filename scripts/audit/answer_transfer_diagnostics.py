"""Reference-history and oracle-field diagnostics; never a serving generator."""
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
from audit.full_history_readback import score
from audit.predicted_surface_training import persist
from claims.process_conditions import write_committed
from model import breadth_answer_transfer as transfer
from model.native_population_training import cpu,fingerprint
from text.constellation_corpus import file_sha,need

PARENT=('broad-answer-transfer-v2-20261024','bae9d16b3e25e91a3c7562087eb96580193af05fd7fb32ff5b6a0c95b190f5e1')
COLD=('broad-answer-transfer-cold-v2-20261024','5729fb0f9cd4db41f95081c0dc1cddd06f9a45d4232b15877df9e5424edc4a87')
NEW=('audit/answer_transfer_diagnostics.py','audit/answer_transfer_diagnostics_readback.py')
MODES=('predicted','oracle_region','oracle_count','oracle_both')


def load(root):
    s=source.load(root);reports=[]
    for name,pin in (PARENT,COLD):
        path=root/name/'report.json';need(file_sha(path)==pin,'TRANSFER_DIAGNOSTIC_REPORT')
        reports.append(json.loads(path.read_bytes()))
    r,c=reports
    need(r['complete'] and c['complete'] and c['source']['pin']==PARENT[1]
         and c['all_updates_exact'] and c['all_trajectories_exact'] and r['code']==s['code'],'TRANSFER_DIAGNOSTIC_SOURCE')
    for name,pin in r['files'].items():need(file_sha(root/PARENT[0]/name)==pin,'TRANSFER_DIAGNOSTIC_FILE')
    s['code'].update({n:file_sha(Path(__file__).resolve().parents[1]/n) for n in NEW})
    s['report']=r;s['phase110']=root/PARENT[0]
    return s


def checkpoint(s,arm,boundary):
    name='initial.pt' if boundary=='initial' else 'update-5.pt'
    value=torch.load(s['phase110']/arm/name,map_location='cpu',weights_only=True)
    return value if boundary=='initial' else value['state']


def reference(item):
    need(fingerprint((item['context'],item['answers']))==item['native_pin'],'TRANSFER_DIAGNOSTIC_NATIVE')
    need(fingerprint((item['codes'],item['counts']))==item['label_pin'],'TRANSFER_DIAGNOSTIC_LABELS')
    need(fingerprint((item['document'],item['side'],item['provenance']))==item['binding'],'TRANSFER_DIAGNOSTIC_BINDING')


def choose(mode,predicted,codes,counts):
    need(mode in MODES,'TRANSFER_DIAGNOSTIC_MODE')
    return (codes if mode in ('oracle_region','oracle_both') else predicted['codes'],
            counts if mode in ('oracle_count','oracle_both') else predicted['counts'])


def native(placement,h,codes,counts,noise,target):
    result=placement(h,codes,counts,noise,target=target)[0]
    return {n:v[0] for n,v in result.items()}


@torch.no_grad()
def document(s,model,item,request,bases):
    reference(item);before=fingerprint(model.state_dict());stream=model.stream(request);state=None;rows=[]
    head=model.forecaster.predictor;placement=model.forecaster.placement['assembled']
    frames=(*item['context'],*item['answers']);c=len(item['context']);a=len(item['answers'])
    # Match the published autonomous rollout's per-position draw exactly.
    generator=torch.Generator().manual_seed(20261024+item['document'])
    for pos,frame in enumerate(frames):
        h,state=stream.step((frame,),context='transfer-diagnostic',offset=pos,state=state,detach_history=True)
        if pos<c-1:continue
        index=pos-c+1;end=index==a
        row=dict(position=pos,index=index,end=end,hidden=fingerprint(h),stop=cpu(model.stop_loss(h,end)))
        if not end:
            codes=item['codes'][index:index+1];counts={n:v[index:index+1] for n,v in item['counts'].items()}
            future=item['answers'][index];forecast=head.forecast(h)
            seed=20261024+item['document']*1024+index
            noise=torch.randn(1,s['policy']['samples'],s['policy']['noise_width'],generator=torch.Generator().manual_seed(seed))
            one=torch.randn(1,1,s['policy']['noise_width'],generator=generator)
            ordinary=native(placement,h,forecast['codes'],forecast['counts'],one,s['target'])
            candidates={};scores={}
            for mode in MODES:
                code,count=choose(mode,forecast,codes,counts)
                value=ordinary if mode=='predicted' else native(placement,h,code,count,one,s['target'])
                candidates[mode]=cpu(value);scores[mode]={n:score(value[n],future[n]) for n in model.roster}
            repeated=head.forecast(h)
            need(fingerprint(forecast)==fingerprint(repeated) and fingerprint(ordinary)==
                 fingerprint(native(placement,h,repeated['codes'],repeated['counts'],one,s['target'])),
                 'TRANSFER_DIAGNOSTIC_ORACLE_CONTAMINATION')
            fit=model.forecaster.losses(h,codes,counts,(future,),noise,target=s['target'])['proposals']['assembled']
            window={n:head.counts[n].loss(h[:,i],counts[n]) for i,n in enumerate(model.roster)}
            base=bases[index]
            scores['codebook']={n:score(base[n],future[n]) for n in model.roster}
            scores['copy_previous']={n:score(frame[n],future[n]) for n in model.roster}
            row.update(forecast=cpu(forecast),codes=cpu(codes),counts=cpu(counts),noise_seed=seed,noise=fingerprint(noise),
                candidate_noise=fingerprint(one),
                candidates=candidates,scores=scores,fit=cpu(fit),window=cpu(window))
        rows.append(row)
    need(len(rows)==a+1 and fingerprint(model.state_dict())==before,'TRANSFER_DIAGNOSTIC_STATE')
    reference(item)
    return dict(document=item['document'],side=item['side'],binding=item['binding'],rows=rows)


def controls(s):
    item,_=source.prior.prior.item(s,s['train'][0]);out={}
    bad=deepcopy(item);next(iter(bad['answers'][0].values())).add_(1)
    out['native']=rejected(lambda:reference(bad),'TRANSFER_DIAGNOSTIC_NATIVE')
    bad=deepcopy(item);bad['codes'][0,0]=(bad['codes'][0,0]+1)%4
    out['labels']=rejected(lambda:reference(bad),'TRANSFER_DIAGNOSTIC_LABELS')
    out['mode']=rejected(lambda:choose('unlabeled_oracle',{},None,None),'TRANSFER_DIAGNOSTIC_MODE')
    model=source.models(s,91540)[1];saved=checkpoint(s,'breadth','final');binding=saved['binding']
    opt=torch.optim.AdamW(model.parameters(),lr=s['policy']['learning_rate'],weight_decay=s['policy']['weight_decay'],foreach=False,fused=False)
    before=fingerprint((model.state_dict(),opt.state_dict()))
    for field in ('model','optimizer'):
        bad=deepcopy(saved)
        if field=='model':next(iter(bad['model'].values())).add_(1)
        else:next(iter(bad['optimizer']['state'].values()))['exp_avg'].add_(1)
        out[field]=rejected(lambda:transfer.restore(model,opt,bad,binding,s['policy'],pin=fingerprint(saved)),
                            'TRANSFER_RESTORE_BINDING')
    need(fingerprint((model.state_dict(),opt.state_dict()))==before,'TRANSFER_DIAGNOSTIC_CONTROL_STATE')
    return out


def run(a):
    a.output.mkdir(exist_ok=False);began=tick=time.perf_counter();phases={}
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    s=load(a.root);items={k:source.prior.prior.item(s,k) for k in sorted(s['items'])};bases={}
    with torch.no_grad():
        for doc,(item,_) in items.items():
            bases[doc]=tuple(s['target'].reconstruct(s['target'].values(item['codes'][i:i+1]),
                [{n:int(v[i]) for n,v in item['counts'].items()}])[0] for i in range(len(item['answers'])))
    cache_bytes=sum(v.numel()*v.element_size() for rows in bases.values() for row in rows for v in row.values())
    need(cache_bytes<=s['policy']['cache_bytes'],'TRANSFER_DIAGNOSTIC_REFERENCE_MEMORY')
    write_committed(a.output/'source.json',dict(parent=PARENT,cold=COLD,code=s['code'],census=s['census'],
        modes=MODES,scope='reference-history oracle-field diagnostic; no serving or fit',codebook_cache_bytes=cache_bytes))
    phases['source_references']=time.perf_counter()-tick;tick=time.perf_counter()
    negative=controls(s);write_committed(a.output/'controls.json',negative);phases['controls']=time.perf_counter()-tick
    records=[];decisions=continuations=0
    for arm,step in [('initial',0),('breadth',91540)]:
        for boundary in ('initial','final'):
            tick=time.perf_counter();saved=checkpoint(s,arm,boundary);_,model=source.models(s,step)
            opt=torch.optim.AdamW(model.parameters(),lr=s['policy']['learning_rate'],weight_decay=s['policy']['weight_decay'],foreach=False,fused=False)
            transfer.restore(model,opt,saved,saved['binding'],s['policy'],pin=fingerprint(saved));model.eval()
            seal=fingerprint((model.state_dict(),opt.state_dict()));dest=a.output/(arm+'-'+boundary);dest.mkdir()
            for doc,(item,request) in items.items():
                result=document(s,model,item,request,bases[doc]);path=dest/(str(doc)+'.pt');persist(path,result)
                decisions+=len(result['rows']);continuations+=len(result['rows'])-1
                records.append(dict(arm=arm,boundary=boundary,document=doc,side=item['side'],file=str(path.relative_to(a.output)),
                    sha256=file_sha(path),checkpoint=fingerprint(saved)))
            need(fingerprint((model.state_dict(),opt.state_dict()))==seal,'TRANSFER_DIAGNOSTIC_IMMUTABLE_OPTIMIZER')
            phases[arm+'_'+boundary]=time.perf_counter()-tick
    tick=time.perf_counter();s['fresh']();need((decisions,continuations)==(800,740),'TRANSFER_DIAGNOSTIC_CENSUS')
    need(not torch.cuda.is_initialized(),'TRANSFER_DIAGNOSTIC_CPU')
    write_committed(a.output/'index.json',records)
    files={str(p.relative_to(a.output)):file_sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    phases['final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'TRANSFER_DIAGNOSTIC_PHASES')
    report=dict(complete=True,acceptance='author-checked',code=s['code'],files=files,decisions=decisions,continuations=continuations,
        native_candidates=continuations*4,fit_samples=4,controls=negative,codebook_cache_bytes=cache_bytes,
        model_optimizer_unchanged=True,oracle_diagnostic_only=True,training_updates=0,native_encoder_calls=0,
        phases=phases,seconds=seconds,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,promoted=False)
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=source.ROOT);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
