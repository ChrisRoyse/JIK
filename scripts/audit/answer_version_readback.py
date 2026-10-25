"""Read native update bytes and profiling receipts independently of the candidate."""
import argparse
import json
import os
from pathlib import Path
import pstats
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import captured_answer_readback as reader


def run(a):
    began=time.perf_counter();reader.torch.set_num_threads(1);a.output.mkdir(exist_ok=False)
    roots=(a.producer,a.cold)
    wrappers=[reader.read(p/'report.json',h) for p,h in zip(roots,(a.producer_pin,a.cold_pin))]
    first,cold=wrappers
    reader.require(first['complete'] and cold['complete'] and cold['previous']==a.producer_pin
        and cold['cold_reproduced'] and first['code']==cold['code'] and first['controls']==cold['controls'],
        'VERSION_READBACK_BINDING')
    for n,h in first['code'].items():reader.require(reader.sha(reader.BASE/n)==h,'VERSION_READBACK_CODE')
    reports=[reader.read(p/'numerical/report.json',v['numerical_report']) for p,v in zip(roots,wrappers)]
    reader.require(reports[1]['previous_pin']==first['numerical_report']
        and reports[0]['binding']==reports[1]['binding'],'VERSION_READBACK_NUMERICAL_BINDING')
    reference_root=a.root/reader.REFERENCE[0]
    reference=reader.read(reference_root/'report.json',reader.REFERENCE[1]);reader.manifest(reference_root,reference)
    total=dict(tensors=0,bytes=0);checked=decisions=0;performance={}
    for root,wrapper,report in zip(roots,wrappers,reports):
        reader.manifest(root/'numerical',report)
        reader.require(wrapper['captured_backward_in_both_arms'] and not wrapper['gpu_used']
                       and not wrapper['promoted'],'VERSION_READBACK_SCOPE')
        reader.require(report['all_updates_exact'] and report['restored_continuations_exact']
                       and report['optimizer_updates']==30 and report['decisions']==396,'VERSION_READBACK_COUNTS')
        observations={(v['architecture'],v['document'],v['variant']):v for v in wrapper['observations']}
        reader.require(len(observations)==len(wrapper['observations'])==30,'VERSION_READBACK_OBSERVATIONS')
        performance[root.name]={}
        for arch in reader.ARCHS:
            reader.require(len(wrapper['controls'][arch])==13 and wrapper['controls'][arch]['scope_restored'],
                           'VERSION_READBACK_CONTROLS')
            sums=dict(state_dict_versions=0.,direct_versions=0.)
            for i,doc in enumerate(reader.DOCS):
                step=i+6;expected=reader.load(reference_root,f'{arch}/update-{step}.pt')
                row=report['results'][arch]['updates'][i]
                peer=reports[1 if report is reports[0] else 0]['results'][arch]['updates'][i]
                reader.require(row['order']==peer['order'][::-1],'VERSION_READBACK_ORDER')
                for key in ('document','step','numerical','chain','counts','restored_state'):
                    reader.require(row.get(key)==peer.get(key),'VERSION_READBACK_COLD_RECORD')
                for mode,label in (('original','state_dict_versions'),('captured','direct_versions')):
                    value=reader.load(root/'numerical',f'{arch}/{mode}/update-{step}.pt')
                    counts=reader.exact(reader.numeric(value),reader.numeric(expected))
                    for n in total:total[n]+=counts[n]
                    observed=observations[arch,doc,label];sums[label]+=observed['seconds']
                    # The observation records the incoming optimizer step; the saved update is step+1.
                    reader.require(observed['step']==step-1 and observed['decisions']==value['record']['decisions']
                        and observed['seconds']>0 and value['record']['document']==doc
                        and sum(x['end'] for x in value['record']['records'])==1,'VERSION_READBACK_DECISIONS')
                    reader.require(value['record']['storage']['remaining_saved_storage_bytes']==0
                        and all(float(s['step'])==step for s in value['optimizer']['state'].values()),
                        'VERSION_READBACK_STORAGE_MOMENTS')
                    for credit in value['early'].values():
                        reader.require(credit['requires_grad'] and bool(credit['gradient'].abs().any())
                            and bool(credit['end_gradient'].abs().any()),'VERSION_READBACK_EARLY_CREDIT')
                    if i in (1,4):
                        cp=reader.load(root/'numerical',f'{arch}/{mode}/checkpoint-{i+1}.pt')
                        reader.exact(cp['model'],value['model']);reader.exact(cp['optimizer'],value['optimizer'])
                        reader.require(cp['cursor']==i+1 and cp['step']==step and cp['chain']==row['chain'],
                                       'VERSION_READBACK_RESUME')
                    checked+=1;decisions+=observed['decisions']
            reader.exact(sums,wrapper['groups'][arch])
            performance[root.name][arch]=dict(**sums,reduction_fraction=1-sums['direct_versions']/sums['state_dict_versions'])
    profile=reader.read(a.profile/'report.json',a.profile_pin)
    reader.require(profile['complete'] and len(profile['observations'])==30
                   and profile['profiler_overhead_included'] and not profile['speedup_claim'],'VERSION_READBACK_PROFILE')
    reader.require(reader.sha(reader.BASE/'audit/answer_update_profile.py')==profile['code'],'VERSION_READBACK_PROFILE_SOURCE')
    for entry in profile['observations']:
        path=a.profile/entry['profile'];reader.require(reader.sha(path)==entry['pin'],'VERSION_READBACK_PROFILE_BYTES')
        stats=pstats.Stats(str(path))
        reader.require((stats.total_calls,stats.prim_calls,stats.total_tt)==
            (entry['total_calls'],entry['primitive_calls'],entry['profiler_seconds']),'VERSION_READBACK_PROFILE_COUNTS')
    for key,group in profile['groups'].items():
        arch,variant=key.split(':')
        entries=[x for x in profile['observations'] if x['architecture']==arch and x['variant']==variant]
        reader.require(len(entries)==5,'VERSION_READBACK_PROFILE_GROUP')
        stats=pstats.Stats(*(str(a.profile/x['profile']) for x in entries))
        expected=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],
                       self_seconds=v[2],cumulative_seconds=v[3])
                  for k,v in sorted(stats.stats.items(),key=lambda kv:(-kv[1][3],kv[0]))]
        reader.exact(expected,group['functions'])
    reader.require(checked==60 and decisions==792 and not reader.torch.cuda.is_initialized(),'VERSION_READBACK_TOTAL')
    result=dict(complete=True,producer=a.producer_pin,cold=a.cold_pin,profile=a.profile_pin,
        reference=reader.REFERENCE[1],updates_compared=checked,decisions=decisions,comparison=total,
        performance=performance,all_numerical_bytes_exact=True,physical_checkpoints_exact=True,
        profiling_receipts_recounted=True,controls=reader.controls(),candidate_controls=first['controls'],
        gpu_used=False,promoted=False,seconds=time.perf_counter()-began,acceptance='author-checked',
        reader_source=reader.sha(__file__),source_code=first['code'])
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True,pin=reader.sha(a.output/'report.json'),performance=performance)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','producer','cold','profile','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('producer-pin','cold-pin','profile-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__,detail=str(exc))))
        raise
