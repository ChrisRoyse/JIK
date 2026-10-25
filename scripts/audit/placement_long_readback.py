"""Read native-checkpoint answer states independently of both training processes."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import captured_answer_readback as rd

REFERENCES={
 'probe':('long-answer-update-probe-v1-20261024','4bd12919763c6bf0a66aeac51c91d88d68272e67e60133d885fe9b670e947bd6'),
 'long':('long-answer-update-v1-20261024/producer','77a7f874d9e7de4e7df0296e0aac04ba536fe5fef0ecca52673e7d032b4a2278')}
NAMES={'probe':('short-update.pt',),'long':('long-update.pt','continuation.pt','restored-continuation.pt')}


def scope(value,mode):
    rd.require(value['complete'] and value['mode']==mode and value['all_tensors_exact']
        and value['checkpoint_restarts_exact'] and value['binding']['schema']=='native-placement-long-v1'
        and value['binding']['architecture']=='shared' and value['binding']['device']=='cpu'
        and value['binding']['scope']=='captured_backward_with_native_only_checkpoints'
        and all(value[k] is False for k in ('gpu_used','promoted','cuda_qualified','capability_demonstrated')),
        'PLACEMENT_LONG_READER_SCOPE')


def contained(outer,inner,prefix):
    rd.require(all(outer['files'].get(prefix+n)==h for n,h in inner['files'].items()),
               'PLACEMENT_LONG_READER_CONTAINMENT')


def update(value,expected,row,step,decisions):
    record=value['record'];records=record['records'];storage=record['storage']
    rd.require(len(records)==record['decisions']==row['decisions']==decisions
        and records[-1]['end'] and not any(x['end'] for x in records[:-1]),'PLACEMENT_LONG_READER_END')
    rd.require(row['document']==record['document']==expected['record']['document'] and row['step']==step-1
        and row['seconds']>0 and all(float(x['step'])==step for x in value['optimizer']['state'].values()),
        'PLACEMENT_LONG_READER_STEP')
    rd.require(len(row['replays'])==decisions-1 and all(r['calls']==2 and r['value_checked']
        and r['saved_tensors']==r['replayed_tensors']>0 for r in row['replays']),'PLACEMENT_LONG_READER_REPLAYS')
    rd.require(storage['remaining_saved_storage_bytes']==storage['remaining_storages']==0
        and storage['peak_saved_storage_bytes']<=storage['limit']==536870912,'PLACEMENT_LONG_READER_STORAGE')
    rd.exact(row['storage'],storage)
    rd.require(len(value['early'])==6 and all(c['requires_grad'] and bool(c['gradient'].abs().any())
        and bool(c['end_gradient'].abs().any()) for c in value['early'].values()),'PLACEMENT_LONG_READER_EARLY')
    for name in record.keys()-{'storage','phases'}:rd.exact(record[name],expected['record'][name])
    return rd.exact(rd.numeric(value),rd.numeric(expected))


def controls(value,expected,row,step,decisions,report):
    result={}
    def reject(name,fn,code):
        try:fn()
        except ValueError as exc:
            rd.require(code in str(exc),'PLACEMENT_LONG_READER_WRONG_REFUSAL');result[name]=str(exc)
        else:raise ValueError('PLACEMENT_LONG_READER_NEGATIVE_ACCEPTED:'+name)
    changed=deepcopy(value);changed['record']['records'][-1]['end']=False
    reject('missing_end',lambda:update(changed,expected,row,step,decisions),'PLACEMENT_LONG_READER_END')
    changed=deepcopy(value);changed['optimizer']['state'].pop(next(iter(changed['optimizer']['state'])))
    reject('missing_moment',lambda:update(changed,expected,row,step,decisions),'CAPTURE_READBACK_KEYS')
    changed=deepcopy(value);next(iter(changed['gradients'].values())).reshape(-1)[0]+=1
    reject('changed_gradient',lambda:update(changed,expected,row,step,decisions),'CAPTURE_READBACK_BYTES')
    changed=deepcopy(row);changed['replays'].pop()
    reject('missing_replay',lambda:update(value,expected,changed,step,decisions),'PLACEMENT_LONG_READER_REPLAYS')
    changed=deepcopy(row);changed['replays'][0]['replayed_tensors']+=1
    reject('replay_metadata_count',lambda:update(value,expected,changed,step,decisions),'PLACEMENT_LONG_READER_REPLAYS')
    changed=deepcopy(report);changed['binding']['architecture']='whole_proposals'
    reject('wrong_scope',lambda:scope(changed,report['mode']),'PLACEMENT_LONG_READER_SCOPE')
    child={'files':{'real':next(iter(report['files'].values()))}}
    reject('missing_manifest_entry',lambda:contained({'files':{}},child,'run/'),'PLACEMENT_LONG_READER_CONTAINMENT')
    result['strict_bytes']=rd.controls();return result


def run(a):
    began=time.perf_counter();rd.torch.set_num_threads(1);a.output.mkdir(exist_ok=False)
    roots=(a.producer,a.cold);pins=(a.producer_pin,a.cold_pin)
    reports=[rd.read(p/'report.json',h) for p,h in zip(roots,pins)];first,cold=reports
    for report in reports:scope(report,a.mode)
    rd.require(first['binding']==cold['binding'] and first['code']==cold['code'] and cold['cold_reproduced']
        and cold['previous_pin']==a.producer_pin,'PLACEMENT_LONG_READER_PAIR')
    refname,refpin=REFERENCES[a.mode];refroot=a.root/refname;reference=rd.read(refroot/'report.json',refpin)
    rd.manifest(refroot,reference);totals=dict(updates=0,decisions=0,tensors=0,bytes=0);performance={};saved={}
    for root,report in zip(roots,reports):
        rd.manifest(root,report)
        inner=rd.read(root/'run/report.json',report['run_pin']);contained(report,inner,'run/')
        raw=rd.read(root/'run/inherited/report.json',inner['inherited_pin']);contained(report,raw,'run/inherited/')
        rd.require(report['files'].get('run/report.json')==report['run_pin']
            and report['files'].get('run/inherited/report.json')==inner['inherited_pin']
            and inner['complete'] and inner['mode']==a.mode and raw['binding']==reference['binding']
            and raw['numerical_pins']==reference['numerical_pins'] and raw['controls']==reference['controls'],
            'PLACEMENT_LONG_READER_INHERITED')
        for name,pin in {**report['code'],**inner['code']}.items():
            rd.require(rd.sha(rd.BASE/name)==pin,'PLACEMENT_LONG_READER_SOURCE')
        names=NAMES[a.mode];rows=report['observations'];rd.require(len(rows)==len(names),'PLACEMENT_LONG_READER_ROWS')
        counts=dict(updates=0,decisions=0,tensors=0,bytes=0);performance[root.name]={};saved[root.name]={}
        for i,(name,row) in enumerate(zip(names,rows)):
            step=7 if a.mode=='probe' or i else 6;decisions=7 if a.mode=='probe' else (3209 if i==0 else 4)
            value=rd.load(root/'run/inherited',name);expected=rd.load(refroot,name)
            compared=update(value,expected,row,step,decisions)
            for key in compared:counts[key]+=compared[key]
            counts['updates']+=1;counts['decisions']+=decisions;saved[root.name][name]=value
            performance[root.name][name]=dict(seconds=row['seconds'],storage=row['storage'],phases=value['record']['phases'])
            if root==roots[0] and i==0:negative=controls(value,expected,row,step,decisions,report)
        rd.exact(counts,report['counts']);rd.exact(counts,inner['counts'])
        for key in totals:totals[key]+=counts[key]
        checkpoints=('initial.pt',) if a.mode=='probe' else ('initial.pt','after-long.pt')
        for name in checkpoints:rd.exact(rd.load(root/'run/inherited',name),rd.load(refroot,name))
        if a.mode=='long':
            after=rd.load(root/'run/inherited','after-long.pt');value=saved[root.name]['long-update.pt']
            rd.exact(after['model'],value['model']);rd.exact(after['optimizer'],value['optimizer'])
            rd.exact(rd.numeric(saved[root.name]['continuation.pt']),rd.numeric(saved[root.name]['restored-continuation.pt']))
    for name in NAMES[a.mode]:rd.exact(rd.numeric(saved[roots[0].name][name]),rd.numeric(saved[roots[1].name][name]))
    rd.require(totals['updates']==(2 if a.mode=='probe' else 6) and totals['decisions']==(14 if a.mode=='probe' else 6434)
        and not rd.torch.cuda.is_initialized(),'PLACEMENT_LONG_READER_TOTAL')
    result=dict(complete=True,mode=a.mode,producer=a.producer_pin,cold=a.cold_pin,reference=refpin,counts=totals,
        all_numerical_bytes_exact=True,checkpoint_states_exact=True,controls=negative,performance=performance,
        reader_source=rd.sha(__file__),helper_source=rd.sha(rd.BASE/'audit/captured_answer_readback.py'),
        seconds=time.perf_counter()-began,gpu_used=False,promoted=False,cuda_qualified=False,
        capability_demonstrated=False,acceptance='author-checked')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True,pin=rd.sha(a.output/'report.json'),counts=totals,mode=a.mode)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('probe','long'),required=True)
    for n in ('root','producer','cold','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('producer-pin','cold-pin'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():(a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__,detail=str(exc))))
        raise
