"""Read full captured-gradient outputs separately from either training process."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from audit.captured_answer_readback import exact,manifest,read,sha,controls,numeric

REFERENCES=(('long-answer-update-v1-20261024/producer','77a7f874d9e7de4e7df0296e0aac04ba536fe5fef0ecca52673e7d032b4a2278'),
            ('long-answer-update-v1-20261024/cold','fc324a717396366b38c1095aeb7510052e12bf88e9ae06c9f6173ed8fbac5ab0'))


def load(path):return torch.load(path,map_location='cpu',weights_only=True)


def require(ok,code):
    if not ok:raise ValueError(code)


def run(a):
    began=time.perf_counter();torch.set_num_threads(1);a.output.mkdir(exist_ok=False)
    first=read(a.producer/'report.json',a.producer_pin);cold=read(a.cold/'report.json',a.cold_pin)
    require(first['binding']==cold['binding'] and cold['cold_reproduced']
            and cold['previous_pin']==a.producer_pin,'CAPTURE_LONG_READBACK_BINDING')
    refs=[read(a.root/d/'report.json',h) for d,h in REFERENCES]
    require(refs[0]['numerical_pins']==refs[1]['numerical_pins'],'CAPTURE_LONG_READBACK_REFERENCE_PAIR')
    for (d,_),r in zip(REFERENCES,refs):manifest(a.root/d,r)
    totals=dict(updates=0,decisions=0,tensors=0,bytes=0);runs={}
    expected_names={'long-update.pt','continuation.pt','restored-continuation.pt'}
    for label,path,r in (('producer',a.producer,first),('cold',a.cold,cold)):
        require(r['complete'] and r['mode']=='long' and not r['gpu_used'] and not r['promoted']
                and r['all_tensors_exact'] and r['checkpoint_restarts_exact'],'CAPTURE_LONG_READBACK_SCOPE')
        manifest(path,r);raw=read(path/'inherited/report.json',r['inherited_pin']);manifest(path/'inherited',raw)
        require(raw['binding']==refs[0]['binding'] and set(raw['numerical_pins'])==expected_names
                and raw['numerical_pins']==refs[0]['numerical_pins']
                and raw['controls']==refs[0]['controls'],'CAPTURE_LONG_READBACK_INHERITED')
        for n,h in r['code'].items():require(sha(Path(__file__).resolve().parents[1]/n)==h,'CAPTURE_LONG_READBACK_CODE')
        values={n:load(path/'inherited'/n) for n in expected_names}
        initial=load(path/'inherited/initial.pt');after=load(path/'inherited/after-long.pt')
        require(initial['step']==5 and after['step']==6,'CAPTURE_LONG_READBACK_STEPS')
        for name,cp in (('initial.pt',initial),('after-long.pt',after)):
            exact(cp,load(a.root/REFERENCES[0][0]/name))
        exact(after['model'],values['long-update.pt']['model']);exact(after['optimizer'],values['long-update.pt']['optimizer'])
        exact(numeric(values['continuation.pt']),numeric(values['restored-continuation.pt']))
        for name,value in values.items():
            expected=load(a.root/REFERENCES[0][0]/name);c=exact(numeric(value),numeric(expected))
            exact(numeric(expected),numeric(load(a.root/REFERENCES[1][0]/name)))
            decisions=value['record']['decisions'];totals['updates']+=1;totals['decisions']+=decisions
            for n in ('tensors','bytes'):totals[n]+=c[n]
            step=6 if name=='long-update.pt' else 7
            require(decisions==(3209 if step==6 else 4) and sum(v['end'] for v in value['record']['records'])==1,
                    'CAPTURE_LONG_READBACK_DECISIONS')
            require(value['record']['storage']['remaining_saved_storage_bytes']==0
                    and value['record']['storage']['remaining_storages']==0,'CAPTURE_LONG_READBACK_GRAPH')
            require(all(float(v['step'])==step for v in value['optimizer']['state'].values()),'CAPTURE_LONG_READBACK_ADAM')
            require(len(value['early'])==6 and all(v['requires_grad'] and bool(v['gradient'].abs().any())
                    and bool(v['end_gradient'].abs().any()) for v in value['early'].values()),'CAPTURE_LONG_READBACK_CREDIT')
        runs[label]={n:r[n] for n in ('seconds','cpu_seconds','peak_rss_bytes','phases','updates')}
    require(totals['updates']==6 and totals['decisions']==6434 and not torch.cuda.is_initialized(),'CAPTURE_LONG_READBACK_TOTAL')
    result=dict(complete=True,acceptance='author-checked',producer=a.producer_pin,cold=a.cold_pin,
                reference=dict(REFERENCES),counts=totals,physical_tensors_byte_exact=True,checkpoint_restart_exact=True,
                controls=controls(),runs=runs,seconds=time.perf_counter()-began,reader_source=sha(__file__),
                gpu_used=False,promoted=False,cuda_qualified=False,capability_demonstrated=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:json.dump(result,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True,pin=sha(a.output/'report.json'),counts=totals)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for name in ('producer','cold','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('producer-pin','cold-pin'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=type(e).__name__,detail=str(e))))
        raise
