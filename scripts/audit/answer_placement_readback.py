"""Read placement-checkpoint updates, including a retained numerical refusal."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import captured_answer_readback as reader
from audit.answer_scalar_refusal_readback import compare,controls,different

CONTROLS={'nested','exception','ownership','weights','metadata','forward_graph','count','overflow',
          'value','scalar_exact','storage_released','scope_restored'}
VARIANTS=('original_checkpoint','scalar_checkpoint')


def replay_check(value,row):
    reader.exact(row['storage'],value['record']['storage'])
    storage=row['storage'];records=value['record']['records'];answers=sum(not r['end'] for r in records)
    reader.require(storage['remaining_saved_storage_bytes']==storage['remaining_storages']==0
        and storage['peak_saved_storage_bytes']<=storage['limit'],'PLACEMENT_READBACK_STORAGE')
    if row['variant']==VARIANTS[0]:
        reader.require(not row['replays'],'PLACEMENT_READBACK_ORIGINAL_REPLAY');return
    heads={n.split('.')[2] for n in value['model'] if n.startswith('forecaster.placement.') and '.variance_' in n}
    reader.require(len(heads)==(6 if row['architecture']=='whole_proposals' else 1)
        and len(row['replays'])==answers*len(heads),'PLACEMENT_READBACK_REPLAY_COVERAGE')
    reader.require(all(r['calls']==2 and r['value_checked']
        and r['saved_tensors']==r['replayed_tensors']>0 for r in row['replays']),
        'PLACEMENT_READBACK_METADATA')


def run(a):
    began=time.perf_counter();reader.torch.set_num_threads(1);a.output.mkdir(exist_ok=False)
    roots=(a.producer,a.cold)
    wrappers=[reader.read(p/'report.json',h) for p,h in zip(roots,(a.producer_pin,a.cold_pin))]
    first,cold=wrappers;refused=first['refusal'] is not None
    reader.require(first['diagnostic_complete'] and cold['diagnostic_complete'] and cold['previous']==a.producer_pin
        and cold['cold_reproduced'] and first['code']==cold['code'] and first['controls']==cold['controls']
        and cold['refusal']==first['refusal'] and cold['reversed_assignment']==refused
        and not first['reversed_assignment'],'PLACEMENT_READBACK_BINDING')
    if refused:reader.require(first['refusal']=='MNLM_TEXT_CORPUS_CAPTURE_PROOF_NUMERICAL_UPDATE',
                              'PLACEMENT_READBACK_REFUSAL')
    reference_root=a.root/reader.REFERENCE[0]
    reference=reader.read(reference_root/'report.json',reader.REFERENCE[1]);reader.manifest(reference_root,reference)
    total=dict(tensors=0,bytes=0,different_tensors=0,different_bytes=0,different_scalars=0)
    checked=decisions=0;summaries={};performance={};saved={};orders={};checkpoint_count=0
    for root,wrapper in zip(roots,wrappers):
        reader.manifest(root,wrapper)
        binding=reader.read(root/'numerical/binding.json',wrapper['binding'])
        reader.require(wrapper['all_updates_exact']==(not refused) and not wrapper['gpu_used']
            and not wrapper['promoted'] and wrapper['captured_backward_in_both_arms'],'PLACEMENT_READBACK_SCOPE')
        for name,pin in {**binding['code'],**wrapper['code']}.items():
            reader.require(reader.sha(reader.BASE/name)==pin and reader.sha(root/'source'/name)==pin,
                           'PLACEMENT_READBACK_CODE')
        if refused:reader.require(wrapper['numerical_report'] is None,'PLACEMENT_READBACK_PARTIAL_REPORT')
        else:
            report=reader.read(root/'numerical/report.json',wrapper['numerical_report'])
            reader.manifest(root/'numerical',report)
            reader.require(report['all_updates_exact'] and report['restored_continuations_exact']
                and report['optimizer_updates']==30 and report['decisions']==396,'PLACEMENT_READBACK_COMPLETE_COUNTS')
        rows=wrapper['observations'];keys=[(r['architecture'],r['document'],r['variant']) for r in rows]
        reader.require(len(keys)==len(set(keys)) and 2<=len(keys)<=30 and len(keys)%2==0,
                       'PLACEMENT_READBACK_COVERAGE')
        canonical=[(arch,doc) for arch in reader.ARCHS for doc in reader.DOCS]
        pairs=[(rows[i]['architecture'],rows[i]['document']) for i in range(0,len(rows),2)]
        reader.require(pairs==canonical[:len(pairs)],'PLACEMENT_READBACK_PREFIX')
        orders[root.name]={};performance[root.name]={};summaries[root.name]={};saved[root.name]={}
        failed=[]
        for arch in wrapper['controls']:
            reader.require(set(wrapper['controls'][arch])==CONTROLS and all(wrapper['controls'][arch].values()),
                           'PLACEMENT_READBACK_CONTROLS')
        reader.require(set(wrapper['controls'])=={p[0] for p in pairs},'PLACEMENT_READBACK_CONTROL_ARCHITECTURES')
        for i in range(0,len(rows),2):
            pair=rows[i:i+2];arch,doc=pairs[i//2];step=reader.DOCS.index(doc)+6
            reader.require(all((r['architecture'],r['document'],r['step'])==(arch,doc,step-1) for r in pair)
                and {r['variant'] for r in pair}==set(VARIANTS)
                and {r['inner'] for r in pair}=={'original','captured'},'PLACEMENT_READBACK_PAIR')
            orders[root.name][arch+':'+str(doc)]=[r['variant'] for r in pair]
            expected=reader.load(reference_root,f'{arch}/update-{step}.pt');pair_failed=False
            for row in pair:
                value=reader.load(root/'numerical',f"{arch}/{row['inner']}/update-{step}.pt")
                record=value['record'];label=f"{arch}:{doc}:{row['variant']}"
                reader.require(row['decisions']==record['decisions'] and record['document']==doc
                    and row['seconds']>0 and record['before']==expected['record']['before']
                    and sum(x['end'] for x in record['records'])==1
                    and all(float(x['step'])==step for x in value['optimizer']['state'].values()),
                    'PLACEMENT_READBACK_UPDATE')
                for credit in value['early'].values():
                    reader.require(credit['requires_grad'] and bool(credit['gradient'].abs().any())
                        and bool(credit['end_gradient'].abs().any()),'PLACEMENT_READBACK_CONTEXT_CREDIT')
                replay_check(value,row)
                groups={};changes=[]
                for name,left in reader.numeric(value).items():
                    counts,deltas=compare(left,reader.numeric(expected)[name],name)
                    groups[name]=counts;changes.extend(deltas)
                    for key in total:total[key]+=counts[key]
                mismatch=any(different(c) for c in groups.values())
                if row['variant']==VARIANTS[0]:reader.require(not mismatch,'PLACEMENT_READBACK_ORIGINAL_CHANGED')
                if mismatch:failed.append(label);pair_failed=True
                summaries[root.name][label]=dict(groups=groups,changes=changes,storage=row['storage'])
                sums=performance[root.name].setdefault(arch,{v:0. for v in VARIANTS})
                sums[row['variant']]+=row['seconds'];saved[root.name][label]=(root/'numerical',arch,row['inner'],step)
                checked+=1;decisions+=row['decisions']
            if pair_failed:reader.require(refused and i==len(rows)-2,'PLACEMENT_READBACK_REFUSAL_POSITION')
            elif step in (7,10):
                for row in pair:
                    cp=reader.load(root/'numerical',f"{arch}/{row['inner']}/checkpoint-{step-5}.pt")
                    value=reader.load(root/'numerical',f"{arch}/{row['inner']}/update-{step}.pt")
                    reader.exact(cp['model'],value['model']);reader.exact(cp['optimizer'],value['optimizer'])
                    reader.require(cp['cursor']==step-5 and cp['step']==step,'PLACEMENT_READBACK_CHECKPOINT')
                    checkpoint_count+=1
        reader.require(len(failed)==int(refused),'PLACEMENT_READBACK_REFUSAL_COUNT')
        reader.exact(performance[root.name],wrapper['groups'])
    reader.require(saved[roots[0].name].keys()==saved[roots[1].name].keys(),'PLACEMENT_READBACK_COLD_COVERAGE')
    cold_counts=dict(tensors=0,bytes=0)
    for label,location in saved[roots[0].name].items():
        values=[]
        for p,arch,mode,step in (location,saved[roots[1].name][label]):
            values.append(reader.numeric(reader.load(p,f'{arch}/{mode}/update-{step}.pt')))
        counts=reader.exact(*values)
        for key in cold_counts:cold_counts[key]+=counts[key]
    for key,order in orders[roots[0].name].items():
        reader.require(orders[roots[1].name][key]==order[::-1],'PLACEMENT_READBACK_COLD_ORDER')
    reader.require(not reader.torch.cuda.is_initialized(),'PLACEMENT_READBACK_CUDA_INITIALIZED')
    if not refused:reader.require(checked==60 and decisions==792 and checkpoint_count==24,'PLACEMENT_READBACK_TOTAL')
    decision='REFUSED_NUMERICAL_DIFFERENCE' if refused else 'EXACT_SHORT_CPU_PROOF'
    result=dict(diagnostic_complete=True,producer=a.producer_pin,cold=a.cold_pin,reference=reader.REFERENCE[1],
        updates_compared=checked,decisions=decisions,comparison=total,cold_comparison=cold_counts,
        physical_checkpoints=checkpoint_count,summaries=summaries,performance=performance,
        all_updates_exact=not refused,cold_bytes_exact=True,controls=controls(),candidate_controls=first['controls'],
        decision=decision,gpu_used=False,promoted=False,seconds=time.perf_counter()-began,
        acceptance='author-checked',reader_source=reader.sha(__file__),reader_dependencies={n:reader.sha(reader.BASE/n)
            for n in ('audit/captured_answer_readback.py','audit/answer_scalar_refusal_readback.py')})
    for n,h in result['reader_dependencies'].items():
        path=a.output/'source'/n;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((reader.BASE/n).read_bytes());reader.require(reader.sha(path)==h,'PLACEMENT_READER_SOURCE')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(diagnostic_complete=True,pin=reader.sha(a.output/'report.json'),decision=decision,
        updates=checked,decisions=decisions,comparison=total,performance=performance)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','producer','cold','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('producer-pin','cold-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__,detail=str(exc))))
        raise
