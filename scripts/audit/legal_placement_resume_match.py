"""Compare the entire resumed experiment with the original, plus a fresh cold continuation."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch
from audit import captured_answer_readback as rd
from audit import legal_grouped_placement_training as original
from audit import legal_placement_resume_training as driver
from claims.process_conditions import write_committed
from model import placement_resume_publication as publication
from text.constellation_corpus import file_sha,need


def run(a):
    began=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    roots=dict(original=a.original,candidate=a.candidate,cold=a.cold)
    reports={n:original.pinned(p,getattr(a,n+'_pin')) for n,p in roots.items()}
    for n,p in roots.items():rd.manifest(p,reports[n])
    old,new,cold=(reports[n] for n in ('original','candidate','cold'));execution=publication.code()
    need(all(r['complete'] for r in reports.values()) and old['binding']==new['binding']==cold['binding']
         and new['execution']['code']==cold['execution']['code']==execution
         and new['execution']['resume'] and cold['execution']['resume']
         and cold['previous_pin']==a.candidate_pin and new['pid']!=cold['pid'], 'PLACEMENT_MATCH_EXPERIMENT')
    need(old['results_pin']==new['results_pin']==cold['results_pin'],'PLACEMENT_MATCH_RESULTS_PIN')
    for name,p in roots.items():need(file_sha(p/'results.json')==reports[name]['results_pin'],'PLACEMENT_MATCH_RESULTS_BYTES')
    totals=dict(tensors=0,bytes=0);updates=targets=evaluations=0;details={}
    for name in original.runtime.original.ARCHITECTURES:
        old_rows=(a.original/name/'steps.jsonl').read_bytes()
        need(old_rows==(a.candidate/name/'steps.jsonl').read_bytes()==(a.cold/name/'steps.jsonl').read_bytes(),
             'PLACEMENT_MATCH_LEDGER_BYTES')
        result=old['results'][name];need(len(old_rows.splitlines())==result['updates'],'PLACEMENT_MATCH_UPDATE_COUNT')
        paths=sorted(p for p in old['files'] if p.startswith(name+'/') and p.endswith('.pt'))
        need(paths==sorted(p for p in new['files'] if p.startswith(name+'/') and p.endswith('.pt'))
             ==sorted(p for p in cold['files'] if p.startswith(name+'/') and p.endswith('.pt')),
             'PLACEMENT_MATCH_PHYSICAL_SCOPE')
        for rel in paths:
            expected=rd.load(a.original,rel)
            for root in (a.candidate,a.cold):
                c=rd.exact(expected,rd.load(root,rel))
                for k in totals:totals[k]+=c[k]
            if '/initial/' in rel or '/final/' in rel:evaluations+=len(expected)
        updates+=result['updates'];targets+=result['rows']
        details[name]=dict(updates=result['updates'],rows=result['rows'],files=len(paths),ledger=old['files'][name+'/steps.jsonl'])
    boundary,_=publication.boundary(a.boundary,a.boundary_pin,execution=execution)
    need(new['execution']['resume']==cold['execution']['resume']==a.boundary_pin,'PLACEMENT_MATCH_RESUME_SOURCE')
    negatives={}
    def reject(name,fn,code):
        try:fn()
        except (ValueError,RuntimeError) as error:
            need(code in str(error),'PLACEMENT_MATCH_WRONG_REFUSAL');negatives[name]=str(error)
        else:raise ValueError('PLACEMENT_MATCH_CONTROL_ACCEPTED:'+name)
    reject('wrong_boundary_pin',lambda:publication.boundary(a.boundary,'0'*64),'BOUNDARY_PIN')
    reject('wrong_execution',lambda:publication.boundary(a.boundary,a.boundary_pin,execution={}), 'PUBLICATION_EXECUTION')
    owner=dict(pid=os.getpid(),starttime=driver.identity(os.getpid()))
    reject('live_predecessor',lambda:driver.terminal_owners([owner]),'LIVE_PREDECESSOR')
    reject('reused_pid',lambda:driver.terminal_owners([dict(owner,starttime='0')]),'PID_REUSED')
    reject('missing_owners',lambda:driver.terminal_owners([]),'OWNERS_REQUIRED')
    reject('missing_cuda_runtime',lambda:driver.gpu_gate(argparse.Namespace(runtime=None,runtime_cold=None),execution,boundary),
           'RUNTIME_REQUIRED')
    # Deliberate changes are made to independent copies of actual persisted inputs.
    for name,rel in (('donor_bytes','donors.json'),('checkpoint_bytes','shared/checkpoint-2.pt'),
                     ('evaluation_bytes',next(p for p in boundary['files'] if p.startswith('shared/initial/')))):
        root=a.output/name;root.mkdir();target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
        raw=bytearray((a.boundary/rel).read_bytes());raw[-1]^=1;target.write_bytes(raw)
        bad=deepcopy(boundary);bad['files']={rel:boundary['files'][rel]}
        pin=write_committed(root/'boundary.json',bad)
        reject(name,lambda root=root,pin=pin:publication.boundary(root,pin,execution=execution),'PUBLICATION_FILE_PIN')
    root=a.output/'incomplete';root.mkdir();bad=deepcopy(boundary);bad['complete']=False
    pin=write_committed(root/'boundary.json',bad)
    reject('incomplete_boundary',lambda:publication.boundary(root,pin),'BOUNDARY_SCOPE')
    need(not torch.cuda.is_initialized(),'PLACEMENT_MATCH_UNEXPECTED_GPU')
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    report=dict(complete=True,original_pin=a.original_pin,candidate_pin=a.candidate_pin,cold_pin=a.cold_pin,
        boundary_pin=a.boundary_pin,device=new['binding']['device'],execution=execution,
        same_complete_comparison=True,results_pin=new['results_pin'],updates=updates,targets=targets,
        evaluation_rows=evaluations,details=details,comparison=totals,controls=negatives,files=files,
        seconds=time.perf_counter()-began,gpu_used=False,acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),comparison=totals,evaluation_rows=evaluations)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','candidate','cold','boundary'):
        p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
