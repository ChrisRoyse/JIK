"""Publish all missing native graph ranges, retaining old generations unchanged."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np

from audit.legal_graph_recovery_readback import code as recovery_code,SOURCES as RECOVERY_SOURCES
from audit.legal_stock_readback import rejected
from audit.native_graph_recovery_readback import source_snapshot,import_source,unchanged
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane,pinned
from kernel.native_graph_accounted import PhaseClock
from kernel.native_graph_continuation import once,tasks,check_ranges,validate_records,read_publication
from kernel.native_graph_recovery import resume_shard
from text.constellation_corpus import digest,file_sha,need

SOURCES=dict(RECOVERY_SOURCES,**{'codebert-mean':(
    'e26ecdef256ba8bcce5edf7b6e5b28748cecb2bca8e640e71b942edd7f9cf554',
    'a5b7b7643900d08d3f4b728bca6a5b0df5d7cfc2b4f418dd1ff43937636aa6b4')})
PREREQUISITES={
 'recovery':('legal-graph-recovery-qualification-v3-20261023','8fa36d33a8b4323ec7dc95665a83d218433501c7135887f592fcbfc4d8133485',
             'legal-graph-recovery-cold-v3-20261023','54a298ff03c14b90cc3211002908bca243a68dea6b561ceac222cbf26e8536d4'),
 'accounted':('legal-accounted-graph-qualification-v1-20261023','6f257d7882b65c409469cf5ccd08c61a7717ab76a063d2c1d6a0d91989332c13',
              'legal-accounted-graph-cold-v1-20261023','796acc0aaac53fa66ffa36bdc53140e7484143640ad74f758c28a85cfb4da643')}


def code():
    pins=recovery_code();root=Path(__file__).resolve().parents[1]
    for n in ['audit/legal_graph_continuation.py','kernel/native_graph_continuation.py']:
        pins[n]=file_sha(root/n)
    return pins


def ident(pid):
    f=Path('/proc')/str(pid)/'stat'
    if not f.exists():return None
    r=f.read_text().rsplit(')',1)[1].split()
    return None if r[0]=='Z' else r[19]


def acquire(a,full):
    started=time.perf_counter();streams=[];owner=dict(pid=os.getpid(),starttime=ident(os.getpid()))
    names=['legal-native-graph-queue','legal-native-graph'] if full else ['legal-graph-continuation-qualification']
    for name in names:
        path=Path(__file__).resolve().parents[2]/'.lease'/name;path.parent.mkdir(exist_ok=True)
        stream=path.open('a+')
        print(json.dumps(dict(stage='waiting_for_lease',path=str(path),**owner)),flush=True)
        try:fcntl.flock(stream,fcntl.LOCK_EX | (0 if a.wait else fcntl.LOCK_NB))
        except BlockingIOError:need(False,'GRAPH_CONTINUATION_LEASE_BUSY')
        streams.append(stream)
        if name=='legal-native-graph-queue':
            need(ident(948511)!='395931103','GRAPH_CONTINUATION_PREDECESSOR_LIVE')
            old=a.root/'phase52-full-graphs-v1-20261023'
            need((old/'report.json').is_file(),'GRAPH_CONTINUATION_PREDECESSOR_UNMEASURED')
            final=json.loads((old/'events.jsonl').read_text().splitlines()[-1])
            need(final['stage']=='queue_terminal' and final['pin']==file_sha(old/'report.json'),
                 'GRAPH_CONTINUATION_PREDECESSOR_TERMINAL_RECEIPT')
        if name!='legal-native-graph-queue':
            stream.seek(0);stream.truncate();stream.write(json.dumps(owner));stream.flush();os.fsync(stream.fileno())
    return streams,dict(**owner,leases=names,wait_seconds=time.perf_counter()-started)


def prerequisites(root,name,pins):
    kind='recovery' if name in RECOVERY_SOURCES else 'accounted'
    qname,qpin,cname,cpin=PREREQUISITES[kind]
    q=pinned(root/qname/'report.json',qpin);c=pinned(root/cname/'report.json',cpin)
    need(q['complete'] and c['complete'] and c['previous_pin']==qpin and q['results_pin']==c['results_pin']
         and name in q['binding']['sources'],'GRAPH_CONTINUATION_PREREQUISITE')
    need(all(pins.get(n)==p for n,p in q['binding']['code'].items()),'GRAPH_CONTINUATION_PREREQUISITE_CODE')
    return dict(kind=kind,qualification=qpin,cold=cpin)


def run(a):
    previous,previous_pin=read_publication(a.previous) if a.mode=='cold' else (None,None)
    if previous:need(previous_pin==a.previous_pin,'GRAPH_CONTINUATION_COLD_PIN')
    scope=previous['scope'] if previous else a.mode
    leases,owner=acquire(a,scope=='full')
    clock=PhaseClock(False);cpu=time.process_time();pins=code()
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:need((a.output/'plan.json').exists(),'GRAPH_CONTINUATION_RESUME_WITHOUT_PLAN')
    for name,pin in pins.items():
        target=a.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():need(file_sha(target)==pin,'GRAPH_CONTINUATION_SOURCE_COPY')
        else:target.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
    dependency=prerequisites(a.root,a.instrument,pins)
    native_pin,bench_pin=SOURCES[a.instrument]
    qroot=a.root/f'legal-{a.instrument}-graph-native-qualification-v1-20261023'
    q=pinned(qroot/'report.json',native_pin)
    bench=pinned(a.root/f'legal-{a.instrument}-graph-batch-qualification-v1-20261023/report.json',bench_pin)
    need(q['complete'] and bench['complete'] and q['binding']==bench['binding'] and bench['selected_threads']==32
         and all(r['all_ranks_scores_candidates_exact'] for r in bench['records']),'GRAPH_CONTINUATION_BENCHMARK')
    available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    need(available>=24<<30,'GRAPH_CONTINUATION_PHYSICAL_MEMORY_UNAVAILABLE')
    cert=q['binding']['certificate'];lane=CertifiedIndexLane(Path(cert['path']),cert['pin'])
    population=len(lane.index.uids)
    need(population==6360378 and lane.binding==q['binding'] and np.array_equal(lane.index.uids,np.arange(population)),
         'GRAPH_CONTINUATION_GALLERY')
    selection=pinned(qroot/'selection.json',q['selection_pin'])['uids'];selection_set=set(selection)
    snapshot=None
    if a.instrument in RECOVERY_SOURCES:
        snapshot=source_snapshot(a.graphs/f'legal-native-{a.instrument}-full-v1-20261023',a.root,lane)
        imported=import_source(snapshot,lane,selection)
    else:
        imported=dict(shards=[],queries=0,numeric_samples=0,payload_bytes=0,numeric_payloads_authenticated=True,
                      all_rows_structurally_read=True,all_numeric_rows_replayed=False)
    source_pin=once(a.output/'original.json',snapshot);import_pin=once(a.output/'import.json',imported)
    ranges=tasks(imported['queries'],population,selection,scope)
    new_count=check_ranges(ranges,imported['queries'],population,scope)
    policy=dict(query_tile=64,shard_rows=16384,threads=32,top_k=10,candidate_policy='128_per_native_window',
                score_policy='unchanged_native_window_index',all_pairs_exhaustive=False)
    contract=dict(code=pins,binding=lane.binding,source_pin=source_pin,import_pin=import_pin,
                  native_pin=native_pin,benchmark_pin=bench_pin,prerequisite=dependency,policy=policy)
    plan=dict(scope=scope,contract=contract,ranges=ranges,population=population,new_queries=new_count,
              scientific_admission=False,implicit_training_authority=False)
    plan_pin=once(a.output/'plan.json',plan)
    if previous:need(previous['plan_pin']==plan_pin,'GRAPH_CONTINUATION_COLD_PLAN')
    if scope=='full':
        qual,qp=read_publication(a.qualification);cold,cp=read_publication(a.qualification_cold)
        need(qp==a.qualification_pin and cp==a.qualification_cold_pin and qual['scope']=='qualify'
             and cold['mode']=='cold' and cold['previous_pin']==qp and qual['result_pin']==cold['result_pin']
             and qual['contract']==contract,'GRAPH_CONTINUATION_QUALIFIED_CONTRACT')
    runs=a.output/'runs';runs.mkdir(exist_ok=True)
    run_dirs=sorted(runs.iterdir());need(all(r.name==f'{i:06d}' and r.is_dir() for i,r in enumerate(run_dirs)),
                                      'GRAPH_CONTINUATION_RUN_SEQUENCE')
    run_dir=runs/f'{len(run_dirs):06d}';run_dir.mkdir()
    clock.mark('authenticate_plan');records=[];created=reused=completed=0
    origin=a.previous if previous else a.output
    with (run_dir/'events.jsonl').open('x') as journal:
        for interval in ranges:
            lo,hi=interval['start'],interval['stop'];ids=list(range(lo,hi))
            chosen=sorted({lo,lo+len(ids)//2,hi-1}|(set(ids)&selection_set))
            record,arrays,was_reused=resume_shard(origin/f'{lo:012d}-{hi:012d}',ids,lane,plan_pin,
                threads=32,chosen=chosen,cold=bool(previous))
            record.update(interval);records.append(record);reused+=int(was_reused);created+=int(not was_reused)
            completed+=hi-lo
            if previous:need(record==previous['new_shards'][len(records)-1],'GRAPH_CONTINUATION_COLD_SHARD')
            event=dict(start=lo,stop=hi,pin=record['pin'],reused=was_reused,queries=hi-lo,
                       planned_new_queries=new_count,completed_new_queries=completed)
            journal.write(json.dumps(event,sort_keys=True)+'\n');journal.flush();os.fsync(journal.fileno())
            print(json.dumps(event),flush=True)
            clock.mark('publish_and_readback')
    coverage=validate_records(imported['shards'],records,ranges,population,lane.index.report['windows'],scope)
    controls={}
    if scope=='qualify':
        controls['false_full']=rejected(lambda:check_ranges(ranges,imported['queries'],population,'full'))
        changed=copy.deepcopy(ranges);changed[0]['start']+=1
        controls['changed_range']=rejected(lambda:validate_records(imported['shards'],records,changed,population,lane.index.report['windows'],scope))
        controls['overlap']=rejected(lambda:check_ranges([ranges[0],ranges[0]],imported['queries'],population,scope))
        controls['missing_shard']=rejected(lambda:validate_records(imported['shards'],records[:-1],ranges,population,lane.index.report['windows'],scope))
        full_ranges=tasks(imported['queries'],population,selection,'full')
        need(check_ranges(full_ranges,imported['queries'],population,'full')==population-imported['queries'],'GRAPH_CONTINUATION_FULL_PLAN')
        controls['full_gap']=rejected(lambda:check_ranges(full_ranges[1:],imported['queries'],population,'full'))
        controls['oversized_tile']=rejected(lambda:check_ranges([dict(start=0,stop=16385)],0,population,scope))
    if snapshot:unchanged(snapshot)
    lane.fresh();need(code()==pins,'GRAPH_CONTINUATION_CODE_CHANGED')
    result=dict(coverage=coverage,import_pin=import_pin,new_shards=records,controls=controls)
    result_pin=once(a.output/'result.json',result)
    if previous:need(previous['result_pin']==result_pin,'GRAPH_CONTINUATION_COLD_RESULT')
    if (a.output/'COMMITTED').exists():
        committed,_=read_publication(a.output);need(committed['result_pin']==result_pin,'GRAPH_CONTINUATION_RESUME_RESULT')
    clock.mark('coverage_controls');timing=clock.finish()
    report=dict(schema='native-graph-continuation-v1',complete=True,mode=a.mode,scope=scope,
        contract=contract,plan_pin=plan_pin,result_pin=result_pin,previous_pin=previous_pin,
        coverage=coverage,new_shards=records,created_shards=created,reused_shards=reused,controls=controls,
        imported_numeric_samples=imported['numeric_samples'],new_numeric_samples=sum(r['sample']['samples'] for r in records),
        all_rows_structurally_read=True,all_numeric_rows_replayed=False,full_graph_complete=coverage['full_graph_complete'],
        seconds=timing['seconds'],timing=timing,cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,available_memory=available,lease=owner,
        source_stats=lane.stats,new_encoder_calls=0,model_updates=0,scientific_admission=False,
        acceptance='author-checked',number_status='measured')
    pin=write_committed(run_dir/'report.json',report)
    if not (a.output/'COMMITTED').exists():
        write_committed(a.output/'COMMITTED',dict(report_path=str((run_dir/'report.json').relative_to(a.output)),report_sha256=pin))
    print(json.dumps(dict(complete=True,report=str(run_dir/'report.json'),pin=pin,result_pin=result_pin,
                         created=created,reused=reused,coverage=coverage)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instrument',choices=list(SOURCES),required=True)
    p.add_argument('--mode',choices=['qualify','full','cold'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--graphs',type=Path,default=Path('/zfs/hot/mnlm/associations'))
    for name in ['previous','qualification','qualification-cold']:
        p.add_argument('--'+name,type=Path);p.add_argument('--'+name+'-pin')
    p.add_argument('--resume',action='store_true');p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_committed(args.output/('failure-'+str(time.time_ns())+'.json'),
                        dict(complete=False,error=type(error).__name__+':'+str(error),code=code()))
        raise
