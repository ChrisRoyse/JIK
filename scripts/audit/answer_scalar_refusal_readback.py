"""Read every saved first-update difference without relaxing the equality gate."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import captured_answer_readback as reader


def compare(a,b,path='root',rows=None):
    rows=[] if rows is None else rows
    reader.require(type(a) is type(b),'SCALAR_READBACK_TYPE:'+path)
    counts=dict(tensors=0,bytes=0,different_tensors=0,different_bytes=0,different_scalars=0)
    if isinstance(a,reader.torch.Tensor):
        reader.require(a.device.type==b.device.type=='cpu' and a.dtype==b.dtype and a.shape==b.shape,
                       'SCALAR_READBACK_LAYOUT:'+path)
        if a.is_floating_point():
            reader.require(bool(a.isfinite().all()) and bool(b.isfinite().all()),'SCALAR_READBACK_FINITE:'+path)
        x=a.contiguous().reshape(-1).view(reader.torch.uint8);y=b.contiguous().reshape(-1).view(reader.torch.uint8)
        changed=int((x!=y).sum());counts.update(tensors=1,bytes=x.numel(),different_bytes=changed,
                                              different_tensors=int(changed>0))
        if changed:
            rows.append(dict(path=path,shape=list(a.shape),dtype=str(a.dtype),different_bytes=changed,
                             max_absolute_difference=float((a.double()-b.double()).abs().max())))
        return counts,rows
    if isinstance(a,dict):
        reader.require(a.keys()==b.keys(),'SCALAR_READBACK_KEYS:'+path)
        children=((a[k],b[k],path+'/'+str(k)) for k in a)
    elif isinstance(a,(list,tuple)):
        reader.require(len(a)==len(b),'SCALAR_READBACK_LENGTH:'+path)
        children=((x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b)))
    else:
        equal=struct.pack('!d',a)==struct.pack('!d',b) if isinstance(a,float) else a==b
        if not equal:counts['different_scalars']=1;rows.append(dict(path=path,left=a,right=b))
        return counts,rows
    for x,y,name in children:
        child,_=compare(x,y,name,rows)
        for key in counts:counts[key]+=child[key]
    return counts,rows


def different(counts):
    return counts['different_bytes']+counts['different_scalars']>0


def controls():
    x=reader.torch.tensor([0.,1.]);y=x.clone();y[0]=-0.
    counts,rows=compare(x,y)
    reader.require(counts['different_tensors']==1 and counts['different_bytes']>0 and rows,
                   'SCALAR_READBACK_SIGNED_ZERO_CONTROL')
    counts,rows=compare(x,x.clone());reader.require(not different(counts) and not rows,'SCALAR_READBACK_EQUAL_CONTROL')
    return dict(signed_zero_detected=True,equal_tensor_exact=True,strict_reader=reader.controls())


def run(a):
    began=time.perf_counter();reader.torch.set_num_threads(1);a.output.mkdir(exist_ok=False)
    roots=(a.producer,a.cold)
    wrappers=[reader.read(p/'report.json',h) for p,h in zip(roots,(a.producer_pin,a.cold_pin))]
    first,cold=wrappers
    reader.require(first['diagnostic_complete'] and cold['diagnostic_complete'] and cold['previous']==a.producer_pin
        and cold['cold_reproduced'] and first['code']==cold['code'] and first['controls']==cold['controls']
        and cold['reversed_assignment'] and not first['reversed_assignment'],'SCALAR_READBACK_BINDING')
    reference_root=a.root/reader.REFERENCE[0]
    reference=reader.read(reference_root/'report.json',reader.REFERENCE[1]);reader.manifest(reference_root,reference)
    expected=reader.load(reference_root,'shared/update-6.pt');values={};summaries={};checked=decisions=0
    for root,wrapper in zip(roots,wrappers):
        reader.manifest(root,wrapper)
        binding=reader.read(root/'numerical/binding.json',wrapper['binding'])
        for name,pin in {**binding['code'],**wrapper['code']}.items():
            reader.require(reader.sha(reader.BASE/name)==pin and reader.sha(root/'source'/name)==pin,'SCALAR_READBACK_CODE')
        reader.require(wrapper['refusal']=='MNLM_TEXT_CORPUS_CAPTURE_PROOF_NUMERICAL_UPDATE'
            and not wrapper['all_updates_exact'] and wrapper['numerical_report'] is None
            and not wrapper['gpu_used'] and not wrapper['promoted'] and wrapper['captured_backward_in_both_arms'],
            'SCALAR_READBACK_REFUSAL')
        observations=wrapper['observations']
        reader.require(len(observations)==2 and {r['variant'] for r in observations}=={'original_checkpoint','scalar_checkpoint'},
                       'SCALAR_READBACK_COVERAGE')
        for row in observations:
            reader.require(row['architecture']=='shared' and row['document']==15639 and row['step']==5,
                           'SCALAR_READBACK_PROOF_SCOPE')
            value=reader.load(root/'numerical',f"shared/{row['inner']}/update-6.pt")
            values[root.name,row['variant']]=value
            reader.require(row['decisions']==value['record']['decisions']==25
                and value['record']['before']==expected['record']['before']
                and sum(x['end'] for x in value['record']['records'])==1
                and value['record']['storage']['remaining_saved_storage_bytes']==0
                and value['record']['storage']['remaining_storages']==0
                and all(float(x['step'])==6 for x in value['optimizer']['state'].values()),'SCALAR_READBACK_UPDATE')
            for credit in value['early'].values():
                reader.require(credit['requires_grad'] and bool(credit['gradient'].abs().any())
                    and bool(credit['end_gradient'].abs().any()),'SCALAR_READBACK_CONNECTED_CONTEXT')
            if row['variant']=='scalar_checkpoint':
                reader.require(len(row['replays'])==24 and all(r['calls']==2 and r['value_checked']
                    and r['saved_tensors']==r['replayed_tensors']>0 for r in row['replays']),
                    'SCALAR_READBACK_METADATA_REPLAYS')
            else:reader.require(not row['replays'],'SCALAR_READBACK_ORIGINAL_REPLAY')
            groups={};changes=[]
            for name,left in reader.numeric(value).items():
                counts,rows=compare(left,reader.numeric(expected)[name],name)
                groups[name]=counts;changes.extend(rows)
            if row['variant']=='original_checkpoint':reader.exact(reader.numeric(value),reader.numeric(expected))
            else:reader.require(any(different(c) for c in groups.values()),'SCALAR_READBACK_EXPECTED_REFUSAL_MISSING')
            summaries[root.name+':'+row['variant']]=dict(groups=groups,changes=changes,seconds=row['seconds'],
                                                       storage=value['record']['storage'])
            checked+=1;decisions+=row['decisions']
    cold_differences={}
    for variant in ('original_checkpoint','scalar_checkpoint'):
        counts,rows=compare(reader.numeric(values[roots[0].name,variant]),reader.numeric(values[roots[1].name,variant]))
        cold_differences[variant]=dict(counts=counts,changes=rows,exact=not different(counts))
        if variant=='original_checkpoint':reader.require(not different(counts),'SCALAR_READBACK_REFERENCE_NOT_STABLE')
    failure=json.loads((a.failed/'failure.json').read_bytes())
    reader.require(failure['detail']==first['refusal'],'SCALAR_READBACK_ORIGINAL_FAILURE')
    failed_comparisons={}
    for mode,variant in (('original','original_checkpoint'),('captured','scalar_checkpoint')):
        value=reader.load(a.failed/'numerical',f'shared/{mode}/update-6.pt')
        counts,rows=compare(reader.numeric(value),reader.numeric(values[roots[0].name,variant]))
        failed_comparisons[variant]=dict(counts=counts,changes=rows,exact=not different(counts))
    reader.require(checked==4 and decisions==100 and not reader.torch.cuda.is_initialized(),'SCALAR_READBACK_TOTAL')
    result=dict(complete=True,producer=a.producer_pin,cold=a.cold_pin,reference=reader.REFERENCE[1],
        updates_compared=checked,decisions=decisions,summaries=summaries,cold_comparisons=cold_differences,
        original_failed_version=failed_comparisons,controls=controls(),candidate_controls=first['controls'],
        decision='REFUSED_NUMERICAL_DIFFERENCE',gpu_used=False,promoted=False,
        seconds=time.perf_counter()-began,acceptance='author-checked',reader_source=reader.sha(__file__))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True,pin=reader.sha(a.output/'report.json'),decision=result['decision'],
                         summaries={k:v['groups'] for k,v in summaries.items()},cold={k:v['exact'] for k,v in cold_differences.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','producer','cold','failed','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('producer-pin','cold-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__,detail=str(exc))))
        raise
