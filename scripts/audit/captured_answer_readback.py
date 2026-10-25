"""Independent byte-level recount of original/captured full-answer checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

BASE=Path(__file__).resolve().parents[1]
ARCHS=('shared','own_slot','whole_proposals')
DOCS=(15639,27195,32791,105748,106288)
REFERENCE=('legal-answer-epoch-proof-v3-20261024',
           '50bed7838cb795e65c937550595789944aec3d087b81a1f8bd933450611637cf')


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path,pin):
    require(sha(path)==pin,'CAPTURE_READBACK_PIN')
    return json.loads(Path(path).read_bytes())


def exact(a,b,path='root'):
    require(type(a) is type(b),'CAPTURE_READBACK_TYPE:'+path)
    if isinstance(a,torch.Tensor):
        require(a.device.type==b.device.type=='cpu' and a.layout==b.layout==torch.strided
                and a.dtype==b.dtype and a.shape==b.shape,'CAPTURE_READBACK_LAYOUT:'+path)
        if a.is_floating_point():
            require(bool(a.isfinite().all()) and bool(b.isfinite().all()),'CAPTURE_READBACK_FINITE:'+path)
        x=a.detach().contiguous().reshape(-1).view(torch.uint8)
        y=b.detach().contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(x,y),'CAPTURE_READBACK_BYTES:'+path)
        return dict(tensors=1,bytes=x.numel())
    counts=dict(tensors=0,bytes=0)
    if isinstance(a,dict):
        require(a.keys()==b.keys(),'CAPTURE_READBACK_KEYS:'+path)
        children=((a[k],b[k],path+'/'+str(k)) for k in a)
    elif isinstance(a,(list,tuple)):
        require(len(a)==len(b),'CAPTURE_READBACK_LENGTH:'+path)
        children=((x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b)))
    else:
        same=struct.pack('!d',a)==struct.pack('!d',b) if isinstance(a,float) else a==b
        require(same,'CAPTURE_READBACK_VALUE:'+path);return counts
    for x,y,p in children:
        c=exact(x,y,p)
        for k in counts:counts[k]+=c[k]
    return counts


def numeric(value):
    return dict(records=value['record']['records'],
                **{n:value[n] for n in ('gradients','early','model','optimizer')})


def manifest(root,report):
    for name,pin in report['files'].items():
        path=(root/name).resolve()
        require(path.is_relative_to(root.resolve()),'CAPTURE_READBACK_MANIFEST_SCOPE')
        require(sha(path)==pin,'CAPTURE_READBACK_MANIFEST:'+name)


def load(root,name):
    return torch.load(root/name,map_location='cpu',weights_only=True)


def controls():
    def rejects(a,b,code):
        try:exact(a,b)
        except ValueError as e:
            require(code in str(e),'CAPTURE_READBACK_WRONG_REFUSAL');return str(e)
        raise ValueError('CAPTURE_READBACK_NEGATIVE_ACCEPTED')
    return dict(signed_zero=rejects(torch.tensor([0.]),torch.tensor([-0.]),'BYTES'),
                changed_tensor=rejects(torch.tensor([1.]),torch.tensor([2.]),'BYTES'),
                missing_key=rejects(dict(a=1),dict(b=1),'KEYS'),
                nonfinite=rejects(torch.tensor([float('nan')]),torch.tensor([float('nan')]),'FINITE'))


def run(a):
    began=time.perf_counter();torch.set_num_threads(1);a.output.mkdir(exist_ok=False)
    first=read(a.producer/'report.json',a.producer_pin);cold=read(a.cold/'report.json',a.cold_pin)
    reference_root=a.producer.parent/REFERENCE[0];reference=read(reference_root/'report.json',REFERENCE[1])
    require(first['complete'] and cold['complete'] and cold['previous_pin']==a.producer_pin
            and cold['cold_reproduced'] and first['binding']==cold['binding'],'CAPTURE_READBACK_BINDING')
    for r in (first,cold):
        require(r['optimizer_updates']==30 and r['decisions']==396 and r['all_updates_exact']
                and not any(r[k] for k in ('gpu_used','promoted','cuda_qualified','long_answer_qualified',
                                          'full_population_trained','capability_demonstrated')),
                'CAPTURE_READBACK_SCOPE')
    for root,r in ((a.producer,first),(a.cold,cold),(reference_root,reference)):manifest(root,r)
    for name,pin in first['code'].items():require(sha(BASE/name)==pin,'CAPTURE_READBACK_SOURCE')
    require(first['controls']==cold['controls'],'CAPTURE_READBACK_CONTROLS')
    needed={'nested','ownership','exception','scope_restored','missing_context','zero_context',
            'nonfinite_context','zero_end'}
    totals=dict(tensors=0,bytes=0);checked=decisions=0;performance={}
    for arch in ARCHS:
        require(set(first['controls'][arch])==needed and first['controls'][arch]['scope_restored'] is True,
                'CAPTURE_READBACK_CONTROL_SET')
        for index,doc in enumerate(DOCS):
            step=index+6;rel=f'{arch}/update-{step}.pt';expected=load(reference_root,rel)
            for root,report in ((a.producer,first),(a.cold,cold)):
                row=report['results'][arch]['updates'][index]
                other=(cold if report is first else first)['results'][arch]['updates'][index]
                require(row['document']==doc and row['step']==step and row['order']==other['order'][::-1],
                        'CAPTURE_READBACK_MATCHED_ORDER')
                for field in ('numerical','chain','counts','restored_state'):
                    require(row.get(field)==other.get(field),'CAPTURE_READBACK_COLD_RECORD')
                for mode in ('original','captured'):
                    value=load(root,f'{arch}/{mode}/update-{step}.pt')
                    c=exact(numeric(value),numeric(expected))
                    for k in totals:totals[k]+=c[k]
                    require(value['record']['document']==doc
                            and len(value['record']['records'])==value['record']['decisions']
                            and sum(v['end'] for v in value['record']['records'])==1,
                            'CAPTURE_READBACK_DECISIONS')
                    require(value['record']['storage']['remaining_saved_storage_bytes']==0
                            and value['record']['storage']['remaining_storages']==0,'CAPTURE_READBACK_GRAPH')
                    require(all(float(v['step'])==step for v in value['optimizer']['state'].values()),
                            'CAPTURE_READBACK_ADAM_STEP')
                    for branch in value['early'].values():
                        require(branch['requires_grad'] is True and bool(branch['gradient'].abs().any())
                                and bool(branch['end_gradient'].abs().any()),'CAPTURE_READBACK_CONNECTION')
                    checked+=1;decisions+=value['record']['decisions']
                    if index in (1,4):
                        cp=load(root,f'{arch}/{mode}/checkpoint-{index+1}.pt')
                        exact(cp['model'],value['model']);exact(cp['optimizer'],value['optimizer'])
                        require(cp['cursor']==index+1 and cp['step']==step and cp['chain']==row['chain']
                                and cp['counts']==row['counts'],'CAPTURE_READBACK_CHECKPOINT_CURSOR')
                        pair=load(root,f'{arch}/'+('captured' if mode=='original' else 'original')
                                  +f'/checkpoint-{index+1}.pt')
                        exact(cp,pair)
        performance[arch]={}
        for label,r in (('producer',first),('cold',cold)):
            rows=r['results'][arch]['updates']
            durations={m:sum(v['update_seconds'][m] for v in rows) for m in ('original','captured')}
            for m in durations:
                require(durations[m]==r['results'][arch][m+'_seconds'],'CAPTURE_READBACK_TIMING_SUM')
            performance[arch][label]=dict(**durations,
                reduction_fraction=1-durations['captured']/durations['original'],
                read_seconds={m:sum(v['read_seconds'][m] for v in rows) for m in durations},
                peak_saved_bytes={m:max(v['storage'][m]['peak_saved_storage_bytes'] for v in rows)
                                  for m in durations})
    require(checked==60 and decisions==792 and not torch.cuda.is_initialized(),'CAPTURE_READBACK_TOTAL')
    result=dict(complete=True,acceptance='author-checked',producer=a.producer_pin,cold=a.cold_pin,
                reference=REFERENCE[1],updates_compared=checked,decisions=decisions,comparison=totals,
                physical_tensors_byte_exact=True,checkpoints_exact=True,controls=controls(),
                producer_controls=first['controls'],performance=performance,
                processes={k:{n:r[n] for n in ('seconds','cpu_seconds','peak_rss_bytes','phases')}
                           for k,r in (('producer',first),('cold',cold))},
                seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                gpu_used=False,promoted=False,cuda_qualified=False,long_answer_qualified=False,
                capability_demonstrated=False,reader_source=sha(__file__))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:json.dump(result,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True,pin=sha(a.output/'report.json'),performance=performance)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('producer','cold','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('producer-pin','cold-pin'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException as e:
        if args.output.exists():(args.output/'failure.json').write_text(json.dumps(dict(error=type(e).__name__,detail=str(e))))
        raise
