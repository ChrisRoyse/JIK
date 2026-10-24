"""Match real optimizer/checkpoint/evaluation bytes across reader implementations."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import torch

from audit import legal_placement_comparison_training as baseline
from audit import legal_grouped_placement_training as candidate
from audit.legal_grouped_io_authority import PROOFS, validate
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from senses.contract import canonical, digest
from text.constellation_corpus import file_sha, need


def same(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and (
            a.reshape(-1).view(torch.uint8).numpy().tobytes()==b.reshape(-1).view(torch.uint8).numpy().tobytes())
    if type(a)!=type(b):return False
    if isinstance(a,dict):return list(a)==list(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return struct.pack('!d',a)==struct.pack('!d',b)
    return a==b


def checkpoint_state(a,b):
    need(all(same(a[k],b[k]) for k in ['model','optimizer','step','rows','noise_cursor']),
         'GROUPED_MATCH_CHECKPOINT_STATE')


def ledger(root, report, architecture):
    rows=[json.loads(s) for s in (root/architecture/'steps.jsonl').read_text().splitlines()]
    parent=digest(dict(report['binding'],architecture=architecture));targets=0
    for i,row in enumerate(rows):
        raw=dict(row);pin=raw.pop('hash')
        need(row['cursor']==i and row['parent']==parent and digest(raw)==pin,'GROUPED_MATCH_LEDGER')
        parent=pin;targets+=row['update']['targets']
    r=report['results'][architecture]
    need(len(rows)==r['updates'] and targets==r['rows'] and parent==r['ledger'],'GROUPED_MATCH_LEDGER_SCOPE')
    return rows


def run(a):
    began=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    q=baseline.pinned(a.baseline,a.baseline_pin);c=candidate.pinned(a.candidate,a.candidate_pin)
    need(q['complete'] and c['complete'],'GROUPED_MATCH_COMPLETE')
    baseline.physical(a.baseline,q);candidate.physical(a.candidate,c)
    source,_=baseline.source_bundle(a.root)
    need(q['binding']['code']==baseline.code(source) and c['binding']['code']==candidate.code(source),'GROUPED_MATCH_CODE')
    left=deepcopy(q['binding']);right=deepcopy(c['binding'])
    left.pop('code');right.pop('code');need(right.pop('reader_match') is None,'GROUPED_MATCH_PREPROMOTION')
    io=right['source'].pop('io',None)
    need(left==right and left['scope'] in ('small','qualify'),'GROUPED_MATCH_CONTEXT')
    need((left['scope']=='small' and io is None) or (left['scope']=='qualify' and io['proofs']=={n:p for n,(_,p) in PROOFS.items()}),
         'GROUPED_MATCH_IO_AUTHORITY')
    architectures=['shared','own_slot','whole_proposals']
    need(set(q['results'])==set(c['results'])==set(architectures),'GROUPED_MATCH_ARCHITECTURES')
    details={};updates=checkpoints=training_rows=payload_rows=0
    for name in architectures:
        u,v=q['results'][name],c['results'][name]
        need({k:x for k,x in u.items() if k not in ['checkpoints','ledger']}==
             {k:x for k,x in v.items() if k not in ['checkpoints','ledger']},'GROUPED_MATCH_METRICS')
        old,new=ledger(a.baseline,q,name),ledger(a.candidate,c,name)
        need(len(old)==len(new) and all(canonical(x['update'])==canonical(y['update']) for x,y in zip(old,new)),
             'GROUPED_MATCH_UPDATES')
        need(set(u['checkpoints'])==set(v['checkpoints']),'GROUPED_MATCH_CHECKPOINT_SCOPE')
        states=[]
        for cursor in u['checkpoints']:
            p=f'{name}/checkpoint-{cursor}.pt'
            one=torch.load(a.baseline/p,map_location='cpu',weights_only=True)
            two=torch.load(a.candidate/p,map_location='cpu',weights_only=True)
            checkpoint_state(one,two);checkpoints+=1
            states.append(dict(cursor=int(cursor),baseline=q['files'][p],candidate=c['files'][p]))
        paths=sorted(p for p in q['files'] if p.startswith((name+'/initial/',name+'/final/')) and p.endswith('.pt'))
        need(paths==sorted(p for p in c['files'] if p.startswith((name+'/initial/',name+'/final/')) and p.endswith('.pt')),
             'GROUPED_MATCH_PAYLOAD_SCOPE')
        for p in paths:
            one=torch.load(a.baseline/p,map_location='cpu',weights_only=True)
            two=torch.load(a.candidate/p,map_location='cpu',weights_only=True)
            need(same(one,two),'GROUPED_MATCH_EVALUATION_BYTES');payload_rows+=len(one)
        updates+=len(old);training_rows+=u['rows']
        details[name]=dict(updates=len(old),rows=u['rows'],checkpoint_files=states,payload_files=len(paths))
    # Wrong physical state, an altered update and incomplete proof must refuse.
    p=a.candidate/'shared/checkpoint-1.pt';original=torch.load(p,map_location='cpu',weights_only=True)
    wrong=deepcopy(original)
    next(t for t in wrong['model'].values() if t.dtype.is_floating_point).reshape(-1)[0]+=1
    records={n:json.loads((a.root/path).read_bytes()) for n,(path,_) in PROOFS.items()}
    validate(records);bad=deepcopy(records);bad['cold']['complete']=False
    changed=deepcopy(new[0]['update']);changed['targets']+=1
    controls=dict(changed_checkpoint=rejected(lambda:checkpoint_state(original,wrong)),
                  changed_update=rejected(lambda:need(canonical(changed)==canonical(new[0]['update']),'GROUPED_MATCH_UPDATES')),
                  incomplete_io=rejected(lambda:validate(bad)))
    need(not torch.cuda.is_initialized(),'GROUPED_MATCH_UNEXPECTED_GPU')
    for n in ['audit/legal_grouped_placement_match.py','audit/legal_grouped_io_authority.py']:
        f=a.output/'source'/n;f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes())
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    result=dict(complete=True,baseline_pin=a.baseline_pin,candidate_pin=a.candidate_pin,candidate_code=c['binding']['code'],
                device=c['binding']['device'],scope=c['binding']['scope'],architectures=architectures,
                updates=updates,training_rows=training_rows,checkpoints=checkpoints,evaluation_rows=payload_rows,
                details=details,controls=controls,files=files,seconds=time.perf_counter()-began,
                all_updates_exact=True,all_checkpoint_tensors_exact=True,all_evaluation_payloads_exact=True,
                acceptance='author-checked',number_status='measured',scientific_admission=False)
    print(json.dumps(dict(complete=True,pin=write_committed(a.output/'report.json',result),updates=updates,
                         checkpoints=checkpoints,evaluation_rows=payload_rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for n in ['baseline','candidate']:
        p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-pin',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as error:
        if a.output.exists():write_committed(a.output/'failure.json',dict(error=repr(error)))
        raise
