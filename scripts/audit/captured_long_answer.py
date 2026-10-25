"""Full-length captured backward using the unchanged real-source restart assay.

The nested receipt describes the inherited assay. The outer binding pins the
actually installed backward implementation and is required for qualification.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import legal_long_answer_update as original
from audit.captured_answer_readback import exact,numeric
from claims.process_conditions import write_committed
from model import captured_answer_backward as candidate
from model import answer_full_history as learner
from text.constellation_corpus import file_sha,need

BASE=Path(__file__).resolve().parents[1];ROOT=original.ROOT
PROOFS={
 'short':('captured-backward-v1-20261024','4b30692cf4786dfb99b3ff680db943686f684b7579a0c2633fcf70a282b34ff6'),
 'short_cold':('captured-backward-cold-v1-20261024','234338e49ca020f77c303598630b30e0207c495a27efd83ac2d634aedb591fdb'),
 'short_readback':('captured-backward-readback-v1-20261024','8a870151c1b8eeea6520eb46d18bea24c936acafd7f3904547a1737b222a4297'),
 'probe':('long-answer-update-probe-v1-20261024','4bd12919763c6bf0a66aeac51c91d88d68272e67e60133d885fe9b670e947bd6'),
 'long':('long-answer-update-v1-20261024/producer','77a7f874d9e7de4e7df0296e0aac04ba536fe5fef0ecca52673e7d032b4a2278'),
 'long_cold':('long-answer-update-v1-20261024/cold','fc324a717396366b38c1095aeb7510052e12bf88e9ae06c9f6173ed8fbac5ab0')}
NEW=('audit/captured_long_answer.py','audit/captured_answer_readback.py','model/captured_answer_backward.py')


def pinned(path,pin):
    need(file_sha(path/'report.json')==pin,'CAPTURE_LONG_REPORT_PIN:'+str(path))
    return json.loads((path/'report.json').read_bytes())


def manifest(path,r):
    for name,pin in r['files'].items():
        target=(path/name).resolve()
        need(target.is_relative_to(path.resolve()) and file_sha(target)==pin,'CAPTURE_LONG_PAYLOAD_PIN')


def authenticate(root):
    reports={name:pinned(root/d,pin) for name,(d,pin) in PROOFS.items()}
    a,b,back=(reports[n] for n in ('short','short_cold','short_readback'))
    need(a['complete'] and b['complete'] and b['cold_reproduced'] and a['binding']==b['binding']
         and b['previous_pin']==PROOFS['short'][1] and a['all_updates_exact'] and b['all_updates_exact'],
         'CAPTURE_LONG_SHORT_PAIR')
    need(back['complete'] and back['producer']==PROOFS['short'][1] and back['cold']==PROOFS['short_cold'][1]
         and back['updates_compared']==60 and back['physical_tensors_byte_exact'],'CAPTURE_LONG_SHORT_READBACK')
    x,y=reports['long'],reports['long_cold']
    need(x['complete'] and y['complete'] and x['mode']==y['mode']=='long' and y['cold_reproduced']
         and y['previous_pin']==PROOFS['long'][1] and x['binding']==y['binding']
         and x['numerical_pins']==y['numerical_pins'],'CAPTURE_LONG_REFERENCE_PAIR')
    for name,r in reports.items():
        if 'files' in r:manifest(root/PROOFS[name][0],r)
    code=dict(a['code'])
    for n,h in x['code'].items():
        need(n not in code or code[n]==h,'CAPTURE_LONG_CODE_CONFLICT');code[n]=h
    code.update({n:file_sha(BASE/n) for n in NEW})
    def fresh():
        for n,h in code.items():need(file_sha(BASE/n)==h,'CAPTURE_LONG_CODE_CHANGED')
        for d,h in PROOFS.values():need(file_sha(root/d/'report.json')==h,'CAPTURE_LONG_REFERENCE_CHANGED')
    fresh();return reports,code,fresh


def boundary(path,reference,mode):
    names=('short-update.pt',) if mode=='probe' else ('long-update.pt','continuation.pt','restored-continuation.pt')
    counts=dict(updates=0,decisions=0,tensors=0,bytes=0);phases={}
    for name in names:
        value=torch.load(path/name,map_location='cpu',weights_only=True)
        expected=torch.load(reference/name,map_location='cpu',weights_only=True)
        c=exact(numeric(value),numeric(expected));counts['updates']+=1
        counts['decisions']+=value['record']['decisions']
        for n in ('tensors','bytes'):counts[n]+=c[n]
        need(value['record']['storage']['remaining_saved_storage_bytes']==0,'CAPTURE_LONG_RETAINED_GRAPH')
        phases[name]=dict(phases=value['record']['phases'],storage=value['record']['storage'])
    for name in (('initial.pt',) if mode=='probe' else ('initial.pt','after-long.pt')):
        a=torch.load(path/name,map_location='cpu',weights_only=True)
        b=torch.load(reference/name,map_location='cpu',weights_only=True)
        exact(a,b)
    if mode=='long':
        a=torch.load(path/'continuation.pt',map_location='cpu',weights_only=True)
        b=torch.load(path/'restored-continuation.pt',map_location='cpu',weights_only=True)
        exact(numeric(a),numeric(b));need(counts['decisions']==3217,'CAPTURE_LONG_DECISION_CENSUS')
    return counts,phases


def run(a):
    began=tick=time.perf_counter();process=time.process_time();a.output.mkdir(exist_ok=False)
    reports,code,fresh=authenticate(a.root)
    binding=dict(schema='captured-long-answer-v1',code=code,proofs=PROOFS,architecture='shared',
                 device='cpu',inherited=reports['long']['binding'],backward='captured_total_original_end')
    binding=json.loads(json.dumps(binding));write_committed(a.output/'binding.json',binding)
    previous=pinned(a.previous,a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==binding and previous['mode']==a.mode,
             'CAPTURE_LONG_PREVIOUS_BINDING');manifest(a.previous,previous)
    probe=None
    if a.mode=='long':
        need(a.probe is not None and a.probe_pin is not None,'CAPTURE_LONG_PROBE_REQUIRED')
        probe=pinned(a.probe,a.probe_pin)
        need(probe['complete'] and probe['mode']=='probe' and probe['binding']==binding
             and probe['all_tensors_exact'],'CAPTURE_LONG_UNQUALIFIED');manifest(a.probe,probe)
    for n,h in code.items():
        p=a.output/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((BASE/n).read_bytes())
        need(file_sha(p)==h,'CAPTURE_LONG_SOURCE_COPY')
    phases=dict(authentication=time.perf_counter()-tick);tick=time.perf_counter()
    args=argparse.Namespace(root=a.root,output=a.output/'inherited',mode=a.mode,
        probe=None if probe is None else a.probe/'inherited',probe_pin=None if probe is None else probe['inherited_pin'],
        previous=None if previous is None else a.previous/'inherited',
        previous_pin=None if previous is None else previous['inherited_pin'])
    # The inherited assay validates real inputs, gradients, optimizer and restart;
    # the outer record names the actual alternate backward rather than impersonating it.
    with candidate.captured_backward():original.run(args)
    need(learner.backward is candidate.ORIGINAL and not candidate.ACTIVE,'CAPTURE_LONG_PATCH_LEAK')
    phases['inherited_run']=time.perf_counter()-tick;tick=time.perf_counter()
    raw_pin=file_sha(args.output/'report.json');raw=pinned(args.output,raw_pin);manifest(args.output,raw)
    need(raw['complete'] and not raw['gpu_used'] and not torch.cuda.is_initialized(),'CAPTURE_LONG_DEVICE_SCOPE')
    reference=a.root/PROOFS['probe' if a.mode=='probe' else 'long'][0]
    counts,updates=boundary(args.output,reference,a.mode)
    if previous:boundary(args.output,a.previous/'inherited',a.mode)
    need(raw['controls']==reports['probe' if a.mode=='probe' else 'long']['controls'],'CAPTURE_LONG_CONTROLS')
    fresh();files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    phases['comparison_and_seal']=time.perf_counter()-tick;seconds=time.perf_counter()-began
    need(.9<=sum(phases.values())/seconds<=1.001,'CAPTURE_LONG_PHASES')
    report=dict(complete=True,mode=a.mode,binding=binding,code=code,files=files,counts=counts,updates=updates,
        controls=raw['controls'],all_tensors_exact=True,checkpoint_restarts_exact=True,inherited_pin=raw_pin,
        previous_pin=a.previous_pin,cold_reproduced=previous is not None,probe_pin=a.probe_pin,
        seconds=seconds,cpu_seconds=time.process_time()-process,phases=phases,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_used=False,
        promoted=False,cuda_qualified=False,capability_demonstrated=False,acceptance='author-checked')
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',report),counts=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('probe','long'),required=True)
    for n in ('probe','previous'):p.add_argument('--'+n,type=Path);p.add_argument('--'+n+'-pin')
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=type(e).__name__,detail=str(e)))
        raise
