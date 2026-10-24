"""Qualify accounted durable shards against immutable full native legal indexes."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np

from audit.legal_graph_batch_readback import compare
from audit.legal_graph_readback import open_shard,verify_scope
from audit.legal_stock_readback import rejected
from audit.native_graph_accounted_readback import read_accounted
from audit.native_graph_coverage import check_arrays,replay_samples
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane,pinned
from kernel.native_graph_accounted import query_batch
from kernel.native_graph_accounted_store import publish
from text.constellation_corpus import digest,file_sha,need

SOURCES={
 'finbert-mean':('4af280db2358ca7afad342918cb41c28197b4b04ff7dddf9765240f46d5032c8',
                 'fc8f32e004d2ffd238700c59a6d26a27e4edcc5f0870f37626ee0e41ac5af4d8',670),
 'codebert-mean':('e26ecdef256ba8bcce5edf7b6e5b28748cecb2bca8e640e71b942edd7f9cf554',
                  'a5b7b7643900d08d3f4b728bca6a5b0df5d7cfc2b4f418dd1ff43937636aa6b4',769)}


def code():
    root=Path(__file__).resolve().parents[1]
    names=['audit/legal_accounted_graph_publication.py','audit/native_graph_accounted_readback.py',
        'kernel/native_graph_accounted_store.py','kernel/native_graph_accounted.py','kernel/native_search_layout.py',
        'kernel/native_graph_store.py','kernel/native_graph_batch.py','kernel/certified_index_graph.py',
        'kernel/native_window_index.py','audit/native_graph_coverage.py','audit/native_index_reference.py',
        'audit/legal_graph_batch_readback.py','audit/legal_graph_readback.py','claims/process_conditions.py',
        'text/constellation_corpus.py','text/constellation_input.py']
    return {n:file_sha(root/n) for n in names}


def negative_copy(root,target,pin,plan_pin,lane,ids,kind):
    shutil.copytree(root,target)
    r=pinned(target/'report.json',pin)
    if kind=='changed_file':
        with (target/'scores.npy').open('r+b') as f:
            f.seek(-1,2);value=f.read(1);f.seek(-1,2);f.write(bytes([value[0]^1]))
    elif kind=='missing_commit':(target/'COMMITTED').unlink()
    elif kind=='wrong_plan':r['plan_pin']='0'*64
    else:
        path=target/'batches.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        if kind=='missing_timing':rows[0].pop('timing')
        elif kind=='underaccounted_timing':
            rows[0]['timing']['phases_seconds']={k:v*.1 for k,v in rows[0]['timing']['phases_seconds'].items()}
            rows[0]['phases_seconds']=rows[0]['timing']['phases_seconds']
        elif kind=='changed_timing_total':rows[0]['seconds']+=1.
        elif kind=='wrong_source':r['binding']={**r['binding'],'lens_id':'0'*64}
        else:raise ValueError('unknown negative')
        path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
        r['files'][path.name]=dict(bytes=path.stat().st_size,sha256=file_sha(path))
    if kind not in ['changed_file','missing_commit']:
        # Coherent re-sealing tests semantic validation, not merely the hash check.
        (target/'report.json').unlink();pin=write_committed(target/'report.json',r)
        (target/'COMMITTED').unlink();write_committed(target/'COMMITTED',dict(report_sha256=pin))
    return rejected(lambda:read_accounted(target,pin,plan_pin,lane,ids))


def run(a):
    lease_path=Path(__file__).resolve().parents[2]/'.lease/accounted-graph-qualification';lease_path.parent.mkdir(exist_ok=True)
    lease=lease_path.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],path=str(lease_path))
    lease.seek(0);lease.truncate();lease.write(json.dumps(owner));lease.flush();os.fsync(lease.fileno())
    started=tick=time.perf_counter();cpu=time.process_time();a.output.mkdir(parents=True,exist_ok=False)
    pins=code()
    for name,pin in pins.items():
        target=a.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((Path(__file__).resolve().parents[1]/name).read_bytes());need(file_sha(target)==pin,'ACCOUNTED_CODE_COPY')
    previous=pinned(a.previous/'report.json',a.previous_pin) if a.previous else None
    binding=dict(code=pins,sources=SOURCES,threads=[1,32],shard_rows=256,query_tile=64,
                 scope='bounded_queries_against_full_native_galleries',index_memory_floor=24<<30)
    if previous:
        need(previous['complete'] and previous['binding']==json.loads(json.dumps(binding)),'ACCOUNTED_COLD_BINDING')
        for n,pin in previous['files'].items():need(file_sha(a.previous/n)==pin,'ACCOUNTED_COLD_FILE')
    phases=dict(initialization=time.perf_counter()-tick);results={};records={};memory={}
    for name,(native_pin,benchmark_pin,count) in SOURCES.items():
        tick=time.perf_counter();out=a.output/name;out.mkdir()
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        need(available>=24<<30,'ACCOUNTED_PHYSICAL_MEMORY_UNAVAILABLE');memory[name]=available
        qroot=a.root/f'legal-{name}-graph-native-qualification-v1-20261023';q=pinned(qroot/'report.json',native_pin)
        bench=pinned(a.root/f'legal-{name}-graph-batch-qualification-v1-20261023/report.json',benchmark_pin)
        need(q['complete'] and bench['complete'] and bench['binding']==q['binding'] and bench['selected_threads']==32,
             'ACCOUNTED_NATIVE_QUALIFICATION')
        certificate=q['binding']['certificate'];lane=CertifiedIndexLane(Path(certificate['path']),certificate['pin'])
        need(lane.binding==q['binding'] and np.array_equal(lane.index.uids,np.arange(6360378)),'ACCOUNTED_GALLERY')
        selection=pinned(qroot/'selection.json',q['selection_pin']);uids=sorted(set(range(512))|set(selection['uids']))
        need(len(uids)==count and uids[-1]==6360377,'ACCOUNTED_QUERY_SELECTION')
        saved,saved_arrays=open_shard(qroot/'graph',q['graph_pin'],lane.binding);verify_scope(saved,saved_arrays,selection['uids'])
        saved_positions={u:i for i,u in enumerate(selection['uids'])}
        plan=dict(binding=lane.binding,native_pin=native_pin,benchmark_pin=benchmark_pin,code=pins,uids=uids,threads=[1,32])
        plan_pin=write_committed(out/'plan.json',plan)
        if previous:need(previous['records'][name]['plan_pin']==plan_pin,'ACCOUNTED_COLD_PLAN')
        authentication=time.perf_counter()-tick;stats=[];numerical=[];controls={};schedule_arrays={}
        for threads in [1,32]:
            schedule_arrays[threads]=[]
            for lo in range(0,len(uids),256):
                ids=uids[lo:lo+256];relative=f'{name}/threads-{threads}/{lo:06d}'
                root=(a.previous if previous else a.output)/relative
                if previous:
                    pin=json.loads((root/'COMMITTED').read_bytes())['report_sha256']
                else:pin=publish(root,ids,lane,plan_pin,threads=threads)
                r,arrays=read_accounted(root,pin,plan_pin,lane,ids,committed=bool(previous))
                replay=replay_samples(arrays,lane,ids)
                for first in range(0,len(ids),64):
                    new=query_batch(lane,ids[first:first+64],threads=threads)
                    compare(new,arrays,{int(u):i for i,u in enumerate(arrays['queries'])})
                    overlap=[row for row in new['rows'] if row['uid'] in saved_positions]
                    compare(dict(rows=overlap),saved_arrays,saved_positions)
                if not previous:
                    write_committed(root/'readback.json',replay);write_committed(root/'COMMITTED',dict(report_sha256=pin))
                else:need(pinned(root/'readback.json',file_sha(root/'readback.json'))==replay,'ACCOUNTED_COLD_REPLAY')
                numerical.append(dict(threads=threads,first=lo,arrays={n:digest(v.tobytes()) for n,v in arrays.items()},replay=replay))
                schedule_arrays[threads].append({n:v.copy() for n,v in arrays.items()})
                stats.append(dict(path=str(root),pin=pin,queries=len(ids),threads=threads,first=lo))
                if not controls:
                    for kind in ['changed_file','missing_commit','wrong_plan','wrong_source','missing_timing',
                                 'underaccounted_timing','changed_timing_total']:
                        controls[kind]=negative_copy(root,out/('negative-'+kind),pin,plan_pin,lane,ids,kind)
                    changed={n:v.copy() for n,v in arrays.items()};changed['scores'][0,0]-=.001
                    controls['plausible_wrong_score']=rejected(lambda:replay_samples(changed,lane,[ids[0]]))
                    changed={n:v.copy() for n,v in arrays.items()};changed['queries']=changed['queries'][:-1]
                    controls['missing_query']=rejected(lambda:check_arrays(changed,ids,len(lane.index.uids)))
                    changed={n:v.copy() for n,v in arrays.items()};changed['targets'][0,0]=changed['targets'][0,1]
                    controls['duplicate_target']=rejected(lambda:check_arrays(changed,ids,len(lane.index.uids)))
                print(json.dumps(dict(instrument=name,threads=threads,completed=lo+len(ids),queries=len(uids))),flush=True)
        need(len(schedule_arrays[1])==len(schedule_arrays[32]),'ACCOUNTED_SCHEDULE_COUNT')
        for one,many in zip(schedule_arrays[1],schedule_arrays[32]):
            need(all(np.array_equal(one[n],many[n]) for n in one),'ACCOUNTED_THREADS_CHANGED_NUMERIC')
        if name=='codebert-mean':need(len(uids)%64==1 and lane.index.lengths[uids[-1]]==1,'ACCOUNTED_SINGLETON_NOT_COVERED')
        lane.fresh();records[name]=dict(plan_pin=plan_pin,shards=stats,authentication_seconds=authentication,
             source_stats=lane.stats,population=len(lane.index.uids),native_windows=lane.index.report['windows'])
        results[name]=dict(queries=len(uids),schedules=[1,32],numeric_pin=write_committed(out/'numeric.json',numerical),
                          controls=controls,every_published_row_replayed=True,schedules_exact=True)
        lane=None;schedule_arrays=None;gc.collect();phases[name]=time.perf_counter()-tick
    tick=time.perf_counter();result_pin=write_committed(a.output/'results.json',results)
    if previous:need(previous['results_pin']==result_pin,'ACCOUNTED_COLD_NUMERIC')
    files={str(f.relative_to(a.output)):file_sha(f) for f in a.output.rglob('*') if f.is_file()}
    phases['final_readback']=time.perf_counter()-tick;seconds=time.perf_counter()-started
    need(sum(phases.values())/seconds>=.9,'ACCOUNTED_PROOF_TIMING')
    result=dict(complete=True,binding=binding,records=records,results_pin=result_pin,files=files,previous_pin=a.previous_pin,
                seconds=seconds,cpu_seconds=time.process_time()-cpu,phases_seconds=phases,available_memory=memory,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,lease=owner,
                full_graph_complete=False,new_encoder_calls=0,model_updates=0,scientific_admission=False,
                acceptance='author-checked',number_status='measured')
    print(json.dumps(dict(pin=write_committed(a.output/'report.json',result),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-pin')
    args=p.parse_args();need(bool(args.previous)==bool(args.previous_pin),'ACCOUNTED_COLD_ARGUMENTS');run(args)
