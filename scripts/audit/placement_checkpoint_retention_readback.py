"""Cold native-source replay and independent scalar scoring of checkpoint retention."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from audit import placement_checkpoint_retention as producer
from audit.placement_source_exposure import read,save,sha
from model.native_population_training import fingerprint


def need(ok,code):
    if not ok:raise ValueError('RETENTION_COLD_'+code)


def equivalent(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12
    return a==b


def scalar_scores(payload,item,counts):
    need(payload['identity']==producer.identity(item),'IDENTITY')
    target=item['codes'].tolist();current=item['baselines']['copy_current'].tolist();n=len(target)
    need(payload['target'].tolist()==target and payload['current'].tolist()==current,'TARGET')
    need(set(payload['models'])=={str(s) for s in producer.STEPS},'STEPS')
    out=dict(**payload['identity'],models={},baselines={})
    for step,v in payload['models'].items():
        ce=[[],[],[]];brier=[[],[],[]];hits=[0,0,0];full=0
        need(v['predicted'].shape==item['codes'].shape and len(v['logits'])==len(v['probabilities'])==3,'LAYOUT')
        for j,size in enumerate((4,16,64)):
            need(tuple(v['logits'][j].shape)==tuple(v['probabilities'][j].shape)==(n,size),'LOGIT_SHAPE')
            for i,x in enumerate(v['logits'][j].tolist()):
                need(all(math.isfinite(s) for s in x),'NONFINITE_LOGIT')
                peak=max(x);exp=[math.exp(s-peak) for s in x];den=math.fsum(exp);ps=[s/den for s in exp];y=target[i][j]
                need(all(abs(a-b)<=1e-12 for a,b in zip(ps,v['probabilities'][j][i].tolist())),'PROBABILITY')
                loss=-math.log(ps[y]);score=math.fsum(p*p for p in ps)-2*ps[y]+1
                need(abs(loss-float(v['ce'][i,j]))<=1e-12 and abs(score-float(v['brier'][i,j]))<=1e-12,'SCORE')
                ce[j].append(loss);brier[j].append(score)
                hits[j]+=int(v['predicted'][i,j])==y
        for p,t in zip(v['predicted'].tolist(),target):full+=p==t
        out['models'][step]=dict(ce=[math.fsum(x)/n for x in ce],brier=[math.fsum(x)/n for x in brier],
            joint_ce=math.fsum(math.fsum(ce[j][i] for j in range(3)) for i in range(n))/n,
            code_accuracy=[v/n for v in hits],full_code_accuracy=full/n)
    baselines=[producer.probabilities.reference(counts,(4,16,64),t,c) for t,c in zip(target,current)]
    for name in baselines[0]:
        out['baselines'][name]={metric:[math.fsum(r[name][metric][j] for r in baselines)/n for j in range(3)] for metric in ('ce','brier')}
        out['baselines'][name]['joint_ce']=math.fsum(math.fsum(r[name]['ce']) for r in baselines)/n
    return out


def reject(fn):
    try:fn()
    except ValueError as e:
        need(str(e).startswith('RETENTION_COLD_'),'CONTROL_REASON');return str(e)
    raise AssertionError('RETENTION_COLD_NEGATIVE_ACCEPTED')


def run(a):
    began=tick=time.perf_counter();cpu0=time.process_time();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    report=read(a.proof/'report.json',a.pin)
    need(report['complete'] and report['device']=='cpu' and report['rows']==24822 and report['documents']==192,'SCOPE')
    for p,pin in report['inputs'].items():need(sha(p)==pin,'INPUT_PIN')
    base=Path(__file__).resolve().parents[1]
    reader_name='audit/placement_checkpoint_retention_readback.py'
    for name,pin in report['code'].items():
        # The producer retains its exact reader source. A corrected independent
        # auditor has its own identity; it cannot alter any producer/model code.
        need(sha(a.proof/'source'/name)==pin,'RETAINED_CODE_PIN')
        if name!=reader_name:need(sha(base/name)==pin,'CODE_PIN')
    docs=read(a.proof/'documents.json',report['documents_pin']);expected_summary=read(a.proof/'summary.json',report['summary_pin'])
    source=producer.Source(a.root)
    need(source.inputs==report['inputs'] and set(source.code)==set(report['code'])
         and all(source.code[n]==pin for n,pin in report['code'].items() if n!=reader_name)
         and source.binding['source']==report['source_binding'],'BINDING')
    phases=dict(authentication=time.perf_counter()-tick,native_read=0.,replay=0.,scalar_recount=0.)
    recounted=[];controls={};files={};native_bytes=0;rows=0;model_rows=0
    try:
        for selected,document in zip(source.plan['evaluations'],docs):
            tick=time.perf_counter();item=source.item(selected);phases['native_read']+=time.perf_counter()-tick
            native_bytes+=sum(v.numel()*v.element_size() for frame in item['frames'] for v in frame.values())
            name=f"document-{item['document']}.pt";path=a.proof/name
            tick=time.perf_counter();ref=report['files'][name];need(sha(path)==ref['pin'],'PAYLOAD_PIN')
            payload=torch.load(path,map_location='cpu',weights_only=True);need(fingerprint(payload)==ref['state'],'PAYLOAD_STATE')
            for step,model in source.models.items():
                value=producer.forward(model,item,reference=True)
                need(fingerprint(value)==fingerprint(payload['models'][str(step)]),'INDEPENDENT_FORWARD')
                model_rows+=len(item['codes'])
            files[name]=ref;phases['replay']+=time.perf_counter()-tick;tick=time.perf_counter()
            doc=scalar_scores(payload,item,source.counts);need(equivalent(doc,document),'DOCUMENT_SCORE');recounted.append(doc);rows+=len(item['codes'])
            if not controls:
                for label,key,value in (('source','source','wrong'),('document','document',-1),('offset','source_start',-1)):
                    bad=deepcopy(payload);bad['identity'][key]=value;controls[label]=reject(lambda:scalar_scores(bad,item,source.counts))
                bad=deepcopy(payload);bad['target'][0,0]=(bad['target'][0,0]+1)%4
                controls['target']=reject(lambda:scalar_scores(bad,item,source.counts))
                bad=deepcopy(payload);bad['models']['91540']['probabilities'][0][0,0]+=.25
                controls['probability']=reject(lambda:scalar_scores(bad,item,source.counts))
                bad=deepcopy(payload);bad['models']['91540']['ce'][0,0]+=.25
                controls['score']=reject(lambda:scalar_scores(bad,item,source.counts))
                bad=deepcopy(payload);ls=bad['models']['14336']['logits']
                bad['models']['14336']['logits']=(ls[0][:-1],*ls[1:])
                controls['missing_row']=reject(lambda:scalar_scores(bad,item,source.counts))
                controls['wrong_checkpoint']=reject(lambda:need(source.checkpoints[14336]['state']==source.checkpoints[15360]['state'],'CHECKPOINT_STATE'))
            phases['scalar_recount']+=time.perf_counter()-tick
            if len(recounted)%16==0:print(json.dumps(dict(documents=len(recounted),rows=rows)),flush=True)
        tick=time.perf_counter();need(len(recounted)==len(docs)==192 and rows==24822 and model_rows==74466 and files==report['files'],'COVERAGE')
        need(native_bytes==report['native_bytes'],'NATIVE_BYTE_COUNT')
        summary=producer.summary(recounted);need(equivalent(summary,expected_summary),'SUMMARY')
        bad=deepcopy(summary);bad[next(iter(bad))]['models']['91540']['joint_ce']+=.1
        controls['aggregate']=reject(lambda:need(equivalent(bad,expected_summary),'SUMMARY'))
        source.finish();phases['closing']=time.perf_counter()-tick
        save(a.output/'documents.json',recounted);save(a.output/'summary.json',summary)
        out=dict(schema='placement-checkpoint-retention-cold-v1',complete=True,previous_pin=a.pin,rows=rows,model_rows=model_rows,
                 documents=192,native_bytes=native_bytes,all_forward_tensors_exact=True,independent_scalar_score_bound=1e-12,
                 controls=controls,phases=phases,seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu0,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,code_sha256=sha(__file__),
                 producer_reader_sha256=report['code'][reader_name],producer_code_unchanged=True,
                 documents_pin=sha(a.output/'documents.json'),summary_pin=sha(a.output/'summary.json'),
                 gpu_used=False,optimizer_updates=0,acceptance='author-checked')
        need(.9<=sum(phases.values())/out['seconds']<=1.000001,'PHASE_COVERAGE')
        save(a.output/'report.json',out);print(json.dumps(dict(pin=sha(a.output/'report.json'),seconds=out['seconds'],rows=rows)),flush=True)
    finally:
        if not source.native.closed:source.native.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','proof','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pin',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():save(a.output/'failure.json',dict(error=repr(error)))
        raise
