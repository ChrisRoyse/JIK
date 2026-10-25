"""Independently recount complete-answer bytes and scoped variance receipts."""
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

CONTROLS={'inplace','replacement','roster','raw_data','missing','dtype','nonfinite','zero',
          'nested','partial_install','exception','ownership','closed','exact_scalar_bytes',
          'returned_dictionary_isolated','scope_restored'}


def scalars(value,observation):
    heads={}
    for name,tensor in value['model'].items():
        prefix='forecaster.placement.'
        if name.startswith(prefix) and '.variance_' in name:
            head,index=name[len(prefix):].split('.variance_')
            heads.setdefault(head,{})[int(index)]=float(tensor)
    caches=observation['cache'];answers=sum(not r['end'] for r in value['record']['records'])
    reader.require(heads and len(heads)==len(caches),'VARIANCE_READBACK_HEADS')
    reads=baseline=calls=0
    for (head,values),cache in zip(heads.items(),caches):
        count=len(values)
        prefix='forecaster.placement.'+head+'.context.'
        roster=[n[len(prefix):-len('.weight')] for n in value['model'] if n.startswith(prefix) and n.endswith('.weight')]
        reader.require(count==6 and sorted(values)==list(range(count)) and len(cache['roster'])==count
            and len(set(cache['roster']))==count and len(cache['values'])==count and cache['roster']==roster,
            'VARIANCE_READBACK_ROSTER')
        reader.require(all(struct.pack('!d',cache['values'][i])==struct.pack('!d',values[i]) for i in values),
                       'VARIANCE_READBACK_VALUES')
        reader.require(cache['closed'] and cache['calls']==2*answers and cache['batch_reads']==2
            and cache['scalar_reads']==2*count,'VARIANCE_READBACK_COUNTERS')
        reads+=cache['scalar_reads'];baseline+=cache['calls']*count;calls+=cache['calls']
    return dict(original_scalar_conversions=baseline,scoped_scalar_reads=reads,lookups=calls)


def run(a):
    began=time.perf_counter();reader.torch.set_num_threads(1);a.output.mkdir(exist_ok=False)
    roots=(a.producer,a.cold)
    wrappers=[reader.read(p/'report.json',h) for p,h in zip(roots,(a.producer_pin,a.cold_pin))]
    first,cold=wrappers
    reader.require(first['complete'] and cold['complete'] and cold['previous']==a.producer_pin
        and cold['cold_reproduced'] and first['code']==cold['code'] and first['controls']==cold['controls'],
        'VARIANCE_READBACK_BINDING')
    for n,h in first['code'].items():
        reader.require(reader.sha(reader.BASE/n)==h and all(reader.sha(p/'source'/n)==h for p in roots),
                       'VARIANCE_READBACK_CODE')
    reports=[reader.read(p/'numerical/report.json',v['numerical_report']) for p,v in zip(roots,wrappers)]
    reader.require(reports[1]['previous_pin']==first['numerical_report']
        and reports[0]['binding']==reports[1]['binding'],'VARIANCE_READBACK_NUMERICAL_BINDING')
    reference_root=a.root/reader.REFERENCE[0]
    reference=reader.read(reference_root/'report.json',reader.REFERENCE[1]);reader.manifest(reference_root,reference)
    total=dict(tensors=0,bytes=0);checked=decisions=0;performance={};counters={}
    for root,wrapper,report in zip(roots,wrappers,reports):
        reader.manifest(root/'numerical',report)
        reader.require(wrapper['captured_backward_in_both_arms'] and not wrapper['gpu_used']
                       and not wrapper['promoted'],'VARIANCE_READBACK_SCOPE')
        reader.require(report['all_updates_exact'] and report['restored_continuations_exact']
                       and report['optimizer_updates']==30 and report['decisions']==396,'VARIANCE_READBACK_COUNTS')
        observations={(v['architecture'],v['document'],v['variant']):v for v in wrapper['observations']}
        reader.require(len(observations)==len(wrapper['observations'])==30,'VARIANCE_READBACK_OBSERVATIONS')
        performance[root.name]={};counters[root.name]={}
        for arch in reader.ARCHS:
            reader.require(set(wrapper['controls'][arch])==CONTROLS and wrapper['controls'][arch]['scope_restored'],
                           'VARIANCE_READBACK_CONTROLS')
            sums=dict(original_variances=0.,scoped_variances=0.)
            counts=dict(original_scalar_conversions=0,scoped_scalar_reads=0,lookups=0)
            for i,doc in enumerate(reader.DOCS):
                step=i+6;expected=reader.load(reference_root,f'{arch}/update-{step}.pt')
                row=report['results'][arch]['updates'][i]
                peer=reports[1 if report is reports[0] else 0]['results'][arch]['updates'][i]
                reader.require(row['order']==peer['order'][::-1],'VARIANCE_READBACK_ORDER')
                for key in ('document','step','numerical','chain','counts','restored_state'):
                    reader.require(row.get(key)==peer.get(key),'VARIANCE_READBACK_COLD_RECORD')
                for mode,label in (('original','original_variances'),('captured','scoped_variances')):
                    value=reader.load(root/'numerical',f'{arch}/{mode}/update-{step}.pt')
                    compared=reader.exact(reader.numeric(value),reader.numeric(expected))
                    for n in total:total[n]+=compared[n]
                    observed=observations[arch,doc,label];sums[label]+=observed['seconds']
                    reader.require(observed['step']==step-1 and observed['decisions']==value['record']['decisions']
                        and observed['seconds']>0 and value['record']['document']==doc
                        and sum(x['end'] for x in value['record']['records'])==1,'VARIANCE_READBACK_DECISIONS')
                    if mode=='captured':
                        recounted=scalars(value,observed)
                        for n in counts:counts[n]+=recounted[n]
                    else:reader.require(not observed['cache'],'VARIANCE_READBACK_BASELINE_CACHE')
                    reader.require(value['record']['storage']['remaining_saved_storage_bytes']==0
                        and value['record']['storage']['remaining_storages']==0
                        and all(float(s['step'])==step for s in value['optimizer']['state'].values()),
                        'VARIANCE_READBACK_STORAGE_MOMENTS')
                    for credit in value['early'].values():
                        reader.require(credit['requires_grad'] and bool(credit['gradient'].abs().any())
                            and bool(credit['end_gradient'].abs().any()),'VARIANCE_READBACK_EARLY_CREDIT')
                    if i in (1,4):
                        cp=reader.load(root/'numerical',f'{arch}/{mode}/checkpoint-{i+1}.pt')
                        reader.exact(cp['model'],value['model']);reader.exact(cp['optimizer'],value['optimizer'])
                        reader.require(cp['cursor']==i+1 and cp['step']==step and cp['chain']==row['chain']
                            and cp['counts']==row['counts'],'VARIANCE_READBACK_RESUME')
                    checked+=1;decisions+=observed['decisions']
            reader.exact(sums,wrapper['groups'][arch]);counters[root.name][arch]=counts
            performance[root.name][arch]=dict(**sums,reduction_fraction=1-sums['scoped_variances']/sums['original_variances'])
    reader.exact(*counters.values())
    reader.require(sum(v['scoped_scalar_reads'] for v in counters[roots[0].name].values())==480
        and sum(v['original_scalar_conversions'] for v in counters[roots[0].name].values())==5856,
        'VARIANCE_READBACK_READ_TOTAL')
    reader.require(checked==60 and decisions==792 and not reader.torch.cuda.is_initialized(),'VARIANCE_READBACK_TOTAL')
    result=dict(complete=True,producer=a.producer_pin,cold=a.cold_pin,reference=reader.REFERENCE[1],
        updates_compared=checked,decisions=decisions,comparison=total,performance=performance,counters=counters,
        all_numerical_bytes_exact=True,physical_checkpoints_exact=True,controls=reader.controls(),
        candidate_controls=first['controls'],gpu_used=False,promoted=False,seconds=time.perf_counter()-began,
        acceptance='author-checked',reader_source=reader.sha(__file__),source_code=first['code'])
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True,pin=reader.sha(a.output/'report.json'),performance=performance,counters=counters)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','producer','cold','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('producer-pin','cold-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__,detail=str(exc))))
        raise
