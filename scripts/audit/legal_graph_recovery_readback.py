"""Qualify authenticated recovery on interrupted full legal graph generations."""
import argparse
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np

from audit.legal_accounted_graph_publication import code as publication_code,negative_copy
from audit.legal_graph_batch_readback import compare
from audit.legal_graph_readback import open_shard,verify_scope
from audit.legal_stock_readback import rejected
from audit.native_graph_accounted_readback import read_accounted
from audit.native_graph_coverage import read_shard,replay_samples
from audit.native_graph_recovery_readback import INCIDENTS,source_snapshot,import_source,unchanged
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane,pinned
from kernel.native_graph_accounted import PhaseClock,query_batch
from kernel.native_graph_recovery import attempt_files,resume_shard,scope_plan
from text.constellation_corpus import digest,file_sha,need

SOURCES={
 'finbert-mean':('4af280db2358ca7afad342918cb41c28197b4b04ff7dddf9765240f46d5032c8',
                 'fc8f32e004d2ffd238700c59a6d26a27e4edcc5f0870f37626ee0e41ac5af4d8'),
 'pubmedbert-embeddings':('958e528fd3143ed7ced18f3823676bbdf4d4870ad61a495a4b6e619c1f4adf5d',
                 '05da09d94ec696df40dede47dbdc919fc8f6be87c1673ce7f3f0ab9448a61836')}


def code():
    pins=publication_code();base=Path(__file__).resolve().parents[1]
    for name in ['audit/legal_graph_recovery_readback.py','audit/native_graph_recovery_readback.py',
                 'kernel/native_graph_recovery.py','audit/legal_graph_coverage.py']:
        pins[name]=file_sha(base/name)
    return pins


def faults(root,lane,plan_pin,uid,*,cold):
    results={}
    for stage in ['report','readback','commit']:
        task=root/('fault-'+stage)
        if not cold:
            request=root/('request-'+stage+'.json')
            request_pin=write_committed(request,dict(stage=stage,task=str(task),uid=uid,plan_pin=plan_pin,
                binding=lane.binding,code=code()))
            with (root/('child-'+stage+'.log')).open('xb') as log:
                child=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),
                    '--fault-request',str(request),'--fault-pin',request_pin],stdout=log,stderr=subprocess.STDOUT)
                log.flush();os.fsync(log.fileno())
            need(child.returncode==9,'GRAPH_RECOVERY_FAULT_DID_NOT_FIRE')
            before=attempt_files(task/'attempt-000000')
        record,arrays,reused=resume_shard(task,[uid],lane,plan_pin,threads=32,chosen=[uid],cold=cold)
        first=task/'attempt-000000'
        need(Path(record['path']).name==('attempt-000000' if stage=='commit' else 'attempt-000001'),
             'GRAPH_RECOVERY_FAULT_SUCCESSOR')
        if not cold:
            need(attempt_files(first)==before,'GRAPH_RECOVERY_FAULT_EVIDENCE_CHANGED')
            need(reused==(stage=='commit'),'GRAPH_RECOVERY_COMMITTED_RECOMPUTED')
        results[stage]=dict(record=record,first_attempt_files=attempt_files(first),
                           numeric_hashes={k:digest(v.tobytes()) for k,v in arrays.items()})
    return results


def fault_child(request,pin):
    r=pinned(request,pin);need(r['code']==code(),'GRAPH_RECOVERY_CHILD_SOURCE')
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    need(available>=24<<30,'GRAPH_RECOVERY_CHILD_MEMORY')
    ref=r['binding']['certificate'];lane=CertifiedIndexLane(Path(ref['path']),ref['pin'])
    need(lane.binding==r['binding'],'GRAPH_RECOVERY_CHILD_BINDING')
    def stop(boundary,path):
        if boundary==r['stage']:
            write_committed(request.parent/('fault-'+r['stage']+'.json'),
                            dict(stage=boundary,path=str(path),pid=os.getpid(),request_pin=pin))
            os._exit(9)
    resume_shard(Path(r['task']),[r['uid']],lane,r['plan_pin'],threads=32,chosen=[r['uid']],interrupt=stop)
    raise ValueError('GRAPH_RECOVERY_CHILD_MISSED_BOUNDARY')


def negative_controls(out,snapshot,imports,missing,lane,plan_pin,record,ids):
    controls={};old=imports['shards'][0]
    controls['old_wrong_plan']=rejected(lambda:read_shard(old['path'],old['pin'],'0'*64,lane,list(range(old['stop']))))
    bad=copy.deepcopy(imports['shards']);bad[1]['start']+=1
    controls['import_gap']=rejected(lambda:scope_plan(bad,missing,len(lane.index.uids)))
    controls['import_overlap']=rejected(lambda:scope_plan(imports['shards'],[0],len(lane.index.uids)))
    controls['duplicate_missing']=rejected(lambda:scope_plan(imports['shards'],[missing[0]]*2,len(lane.index.uids)))
    for kind in ['changed_file','missing_commit','wrong_plan','wrong_source','missing_timing',
                 'underaccounted_timing','changed_timing_total']:
        controls['new_'+kind]=negative_copy(Path(record['path']),out/('negative-'+kind),record['pin'],plan_pin,lane,ids,kind)
    # A committed shard with corrupted numeric bytes must refuse, never get a new attempt.
    task=out/'negative-committed';task.mkdir();shutil.copytree(record['path'],task/'attempt-000000')
    path=task/'attempt-000000/scores.npy'
    with path.open('r+b') as stream:
        stream.seek(-1,2);value=stream.read(1);stream.seek(-1,2);stream.write(bytes([value[0]^1]))
    before={p.name:attempt_files(p) for p in task.iterdir()}
    controls['committed_corruption']=rejected(lambda:resume_shard(task,ids,lane,plan_pin,threads=32,chosen=ids))
    need({p.name:attempt_files(p) for p in task.iterdir()}==before,'GRAPH_RECOVERY_CORRUPTION_WAS_REPLACED')
    task=out/'negative-duplicate-seal';task.mkdir()
    for number in range(2):shutil.copytree(record['path'],task/f'attempt-{number:06d}')
    controls['duplicate_commit']=rejected(lambda:resume_shard(task,ids,lane,plan_pin,threads=32,chosen=ids))
    r,arrays=read_accounted(record['path'],record['pin'],plan_pin,lane,ids)
    arrays['scores'][0,0]-=.001
    controls['plausible_wrong_score']=rejected(lambda:replay_samples(arrays,lane,[ids[0]]))
    # Missing seal is refused by cold recovery, while a live restart preserves it and succeeds.
    task=out/'negative-cold-unsealed';task.mkdir();shutil.copytree(record['path'],task/'attempt-000000')
    (task/'attempt-000000/COMMITTED').unlink()
    controls['cold_unsealed']=rejected(lambda:resume_shard(task,ids,lane,plan_pin,threads=32,chosen=ids,cold=True))
    return controls


def run(a):
    path=Path(__file__).resolve().parents[2]/'.lease/legal-graph-recovery-qualification'
    path.parent.mkdir(exist_ok=True);lease=path.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],path=str(path))
    lease.seek(0);lease.truncate();lease.write(json.dumps(owner));lease.flush();os.fsync(lease.fileno())
    clock=PhaseClock(False);cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    pins=code()
    for name,pin in pins.items():
        target=a.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes())
        need(file_sha(target)==pin,'GRAPH_RECOVERY_CODE_COPY')
    binding=dict(code=pins,sources=SOURCES,incidents=INCIDENTS,scope='all_legacy_seals_and_bounded_missing_queries',
                 production_query_scope=6360378,query_tile=64,proof_shard_rows=256,threads=32)
    previous=pinned(a.previous/'report.json',a.previous_pin) if a.previous else None
    if previous:
        need(previous['complete'] and previous['binding']==json.loads(json.dumps(binding)),'GRAPH_RECOVERY_COLD_BINDING')
        for name,pin in previous['files'].items():need(file_sha(a.previous/name)==pin,'GRAPH_RECOVERY_COLD_FILE')
    clock.mark('initialization');results={};memory={}
    for name,(native_pin,benchmark_pin) in SOURCES.items():
        out=a.output/name;out.mkdir();origin=(a.previous if previous else a.output)/name
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        need(available>=24<<30,'GRAPH_RECOVERY_PHYSICAL_MEMORY_UNAVAILABLE');memory[name]=available
        qroot=a.root/f'legal-{name}-graph-native-qualification-v1-20261023'
        q=pinned(qroot/'report.json',native_pin)
        bench=pinned(a.root/f'legal-{name}-graph-batch-qualification-v1-20261023/report.json',benchmark_pin)
        need(q['complete'] and bench['complete'] and bench['binding']==q['binding'] and bench['selected_threads']==32,
             'GRAPH_RECOVERY_NATIVE_QUALIFICATION')
        cert=q['binding']['certificate'];lane=CertifiedIndexLane(Path(cert['path']),cert['pin'])
        need(lane.binding==q['binding'] and np.array_equal(lane.index.uids,np.arange(6360378)),'GRAPH_RECOVERY_GALLERY')
        selection=pinned(qroot/'selection.json',q['selection_pin'])['uids']
        source=a.graphs/f'legal-native-{name}-full-v1-20261023'
        snapshot=source_snapshot(source,a.root,lane)
        need(snapshot['original_plan']['native_qualification_pin']==native_pin
             and snapshot['original_plan']['benchmark_pin']==benchmark_pin,'GRAPH_RECOVERY_OLD_QUALIFICATION')
        source_pin=write_committed(out/'original.json',snapshot)
        clock.mark(name+'_authenticate')
        imported=import_source(snapshot,lane,selection);import_pin=write_committed(out/'import.json',imported)
        first=imported['queries']
        missing=sorted(set(range(first,first+65)) | {u for u in selection if u>=first})
        coverage=scope_plan(imported['shards'],missing,len(lane.index.uids))
        need(not coverage['full_graph_complete'] and missing[-1]==6360377,'GRAPH_RECOVERY_PROOF_SCOPE')
        plan=dict(binding=lane.binding,code=pins,source_pin=source_pin,import_pin=import_pin,
                  original_plan_pin=snapshot['plan_pin'],missing=missing,coverage=coverage,
                  policy=dict(query_tile=64,shard_rows=256,threads=32,top_k=10,candidate_windows=128))
        plan_pin=write_committed(out/'plan.json',plan)
        if previous:
            old=previous['results'][name]
            need(old['plan_pin']==plan_pin,'GRAPH_RECOVERY_COLD_PLAN')
        clock.mark(name+'_import')
        saved,saved_arrays=open_shard(qroot/'graph',q['graph_pin'],lane.binding);verify_scope(saved,saved_arrays,selection)
        positions={u:i for i,u in enumerate(selection)};records=[];numerical=[];controls=None
        # Keep the final real atom as a separate singleton publication.
        tasks=[missing[lo:min(lo+256,len(missing)-1)] for lo in range(0,len(missing)-1,256)]+[[missing[-1]]]
        for j,ids in enumerate(tasks):
            record,arrays,reused=resume_shard(origin/f'task-{j:06d}',ids,lane,plan_pin,threads=32,chosen=ids,cold=bool(previous))
            need(reused==bool(previous),'GRAPH_RECOVERY_UNEXPECTED_REUSE')
            for lo in range(0,len(ids),64):
                batch=query_batch(lane,ids[lo:lo+64],threads=1)
                compare(batch,arrays,{int(u):i for i,u in enumerate(arrays['queries'])})
                compare(dict(rows=[r for r in batch['rows'] if r['uid'] in positions]),saved_arrays,positions)
            records.append(record);numerical.append(dict(uids=ids,arrays={k:digest(v.tobytes()) for k,v in arrays.items()}))
            if controls is None:controls=negative_controls(out,snapshot,imported,missing,lane,plan_pin,record,ids)
        need(lane.index.lengths[missing[-1]]==1,'GRAPH_RECOVERY_SINGLETON_WINDOW')
        fault_results=faults(origin,lane,plan_pin,missing[-1],cold=bool(previous))
        unchanged(snapshot);lane.fresh()
        result=dict(plan_pin=plan_pin,source_pin=source_pin,import_pin=import_pin,coverage=coverage,
                    imported_shards=len(imported['shards']),imported_queries=imported['queries'],
                    imported_numeric_samples=imported['numeric_samples'],payload_bytes=imported['payload_bytes'],
                    new_shards=records,numeric=numerical,controls=controls,faults=fault_results,
                    original_bytes_unchanged=True,new_queries_all_replayed=True,full_graph_complete=False)
        if previous:need(result==previous['results'][name],'GRAPH_RECOVERY_COLD_RESULT')
        results[name]=result;write_committed(out/'result.json',result)
        print(json.dumps(dict(instrument=name,imported_queries=imported['queries'],new_queries=len(missing),complete=True)),flush=True)
        lane=None;gc.collect();clock.mark(name+'_publication_replay_controls')
    result_pin=write_committed(a.output/'results.json',results)
    files={str(p.relative_to(a.output)):file_sha(p) for p in a.output.rglob('*') if p.is_file()}
    clock.mark('retention_manifest');timing=clock.finish()
    report=dict(complete=True,binding=binding,results=results,results_pin=result_pin,files=files,
        previous_pin=a.previous_pin,timing=timing,seconds=timing['seconds'],cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,available_memory=memory,lease=owner,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        child_processes=0 if previous else 3*len(SOURCES),
        new_encoder_calls=0,model_updates=0,scientific_admission=False,full_graph_complete=False,
        acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',report),seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--graphs',type=Path,default=Path('/zfs/hot/mnlm/associations'))
    p.add_argument('--output',type=Path)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    p.add_argument('--fault-request',type=Path);p.add_argument('--fault-pin')
    args=p.parse_args()
    if args.fault_request:
        fault_child(args.fault_request,args.fault_pin)
        raise SystemExit(71)
    need(args.output is not None,'GRAPH_RECOVERY_OUTPUT_REQUIRED')
    try:run(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_committed(args.output/('failure-'+str(time.time_ns())+'.json'),
                        dict(complete=False,error=type(error).__name__+':'+str(error),code=code()))
        raise
