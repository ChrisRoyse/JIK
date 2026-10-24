"""Recount grouped-document outputs through saved independent native evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from audit.legal_coalesced_io_recount import sha, identity, require, check_ranges, refused


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.numpy().tobytes()==b.numpy().tobytes()
    if type(a)!=type(b):return False
    if isinstance(a,dict):return list(a)==list(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def run(a):
    began=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(sha(a.input/'report.json')==a.pin,'GROUPED_REPORT_PIN')
    r=json.loads((a.input/'report.json').read_bytes());require(r['complete'],'GROUPED_COMPLETE')
    for name,pin in r['files'].items():require(sha(a.input/name)==pin,'GROUPED_PHYSICAL')
    prior=a.root/'legal-coalesced-native-io-qualification-v1-20261023'
    require(sha(prior/'report.json')==r['binding']['prior'],'GROUPED_PRIOR_PIN')
    old=json.loads((prior/'report.json').read_bytes())
    require(sha(a.reference)==a.reference_pin,'GROUPED_REFERENCE_PIN')
    reference=json.loads(a.reference.read_bytes())
    require(reference['complete'] and reference['all_bytes_exact'] and reference['input_pin']==r['binding']['prior'],
            'GROUPED_PHYSICAL_REFERENCE')
    checked=0;counts=[];layouts={};stamps={}
    for row in old['traces']:
        key=row['label'];name=f'{key}-pread-frames.pt'
        require(sha(prior/name)==old['files'][name],'GROUPED_OLD_FRAMES')
        baseline=torch.load(prior/name,map_location='cpu',weights_only=True)
        result=torch.load(a.input/f'{key}-frames.pt',map_location='cpu',weights_only=True)
        require(equal(baseline,result),'GROUPED_NATIVE_BYTES');checked+=sum(len(f) for f in result)
        require(sha(prior/row['requests'])==old['files'][row['requests']],'GROUPED_REQUESTS')
        requests=json.loads((prior/row['requests']).read_bytes())
        requests=[dict(x,batch=0) for x in requests]
        for x in requests:
            if x['path'] not in layouts:
                p=Path(x['path']);stamps[x['path']]=identity(p)
                v=np.load(p,mmap_mode='r',allow_pickle=False)
                require(v.dtype==np.float32 and v.flags.c_contiguous,'GROUPED_LAYOUT')
                layouts[x['path']]=(v.offset,v.shape);v._mmap.close()
        reads=[json.loads(line) for line in (a.input/f'{key}-reads.jsonl').read_text().splitlines()]
        counts.append(dict(label=key,**check_ranges(requests,reads,'pread',layouts)))
    for row in old['plan']:
        items=[torch.load(a.input/f'{mode}-document-{row["document"]}.pt',map_location='cpu',weights_only=True)
               for mode in ['temporal','grouped']]
        require(equal(*items),'GROUPED_COMPLETE_CONSUMER')
        name=f'document-{row["document"]}.pt'
        require(sha(prior/name)==old['files'][name],'GROUPED_OLD_LABELS')
        expected=torch.load(prior/name,map_location='cpu',weights_only=True)
        require(equal(items[0]['frames'],expected['frames']) and items[0]['uids']==expected['uids']
                and equal(items[0]['codes'],expected['codes'][1:])
                and equal(items[0]['counts'],{n:x[1:] for n,x in expected['counts'].items()}),'GROUPED_TARGET_JOIN')
    controls=dict(reversed_frames=refused(lambda:require(equal(items[0]['frames'],items[0]['frames'][::-1]),'ORDER_CONTROL')))
    require(all(identity(Path(p))==s for p,s in stamps.items()),'GROUPED_SOURCE_CHANGED')
    times={}
    for mode in ['temporal','grouped']:
        rows=[x for x in r['measurements'] if x['mode']==mode]
        times[mode]=dict(seconds=sum(x['seconds'] for x in rows),cpu_seconds=sum(x['cpu_seconds'] for x in rows),
            native_seconds=sum(x['cost']['native_read_seconds'] for x in rows),
            by_document={str(d['document']):sum(x['seconds'] for x in rows if x['document']==d['document']) for d in old['plan']})
        require(r['stats'][mode]['handles']['peak_handles']<=64 and r['stats'][mode]['reader']['maximum_payload_bytes']<=16<<20,
                'GROUPED_RESOURCE_BOUND')
    require(not a.output.exists() and not torch.cuda.is_initialized(),'GROUPED_OUTPUT_SCOPE')
    result=dict(complete=True,input_pin=a.pin,reference_pin=a.reference_pin,checked_native_slot_arrays=checked,
                all_native_bytes_exact=True,all_consumer_fields_exact=True,no_unrequested_reads=True,
                counts=counts,controls=controls,timings=times,seconds=time.perf_counter()-began,
                acceptance='author-checked',number_status='measured',scientific_admission=False)
    with a.output.open('x') as out:
        out.write(json.dumps(result,sort_keys=True,indent=2)+'\n');out.flush();os.fsync(out.fileno())
    print(json.dumps(dict(complete=True,pin=sha(a.output),timings=times)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-pin',required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
