"""Measure index construction threads, then publish full legal instrument galleries."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.legal_native_associations_readback import rankings,vectors
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel.legal_native_associations import BankLane,assess
from kernel.native_window_index import INDEX_MEMORY_BYTES,NativeWindowIndex,execution_policy,publish
from text.constellation_corpus import digest,file_sha,need
from text.legal_continuation_input import Continuations
from text.legal_native_batch import NativeBatchSession
from text.legal_native_store import SCHEMA,ranges,shard_name
from text.legal_reuse_input import ReuseInputs

LANES=('all-mpnet-base-v2','codebert-mean')


@contextmanager
def lease():
    path=Path('/home/croyse/mnlm/.lease/legal-native-index');path.parent.mkdir(exist_ok=True)
    with path.open('a+') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('MNLM_LEGAL_INDEX_LEASE_HELD')
        stream.seek(0);stream.truncate();stream.write(json.dumps(dict(pid=os.getpid(),issue=29,argv=sys.argv)))
        stream.flush();os.fsync(stream.fileno())
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def load(path,pin):
    need(file_sha(path/'report.json')==pin,'LEGAL_INDEX_REPORT_PIN');r=json.loads((path/'report.json').read_bytes())
    need(r['complete'],'LEGAL_INDEX_INCOMPLETE');return r


def census(bank,name):
    shards=[];windows=0;native_bytes=0;dimensions=set()
    for start,end in ranges(bank.config['inputs']):
        root=bank.root/name/shard_name(start,end);raw=(root/'report.json').read_bytes();pin=digest(raw)
        need(json.loads((root/'COMMITTED').read_bytes())==dict(report_sha256=pin),'LEGAL_INDEX_SHARD_COMMIT')
        r=json.loads(raw)
        need(r['schema']==SCHEMA and r['complete'] and r['configuration_pin']==bank.pin
             and r['lens_id']==bank.config['instruments'][name]['lens_id'] and (r['start'],r['end'])==(start,end)
             and r['inputs']==end-start and r['windows']>=r['inputs'],'LEGAL_INDEX_SHARD_SCOPE')
        shards.append(dict(start=start,end=end,pin=pin,windows=r['windows']))
        windows+=r['windows'];native_bytes+=sum(v['bytes'] for v in r['files'].values());dimensions.add(r['dimensions'])
    need(len(dimensions)==1,'LEGAL_INDEX_DIMENSIONS')
    return dict(instrument=name,source_pin=bank.source.pin,bank_pin=bank.pin,inputs=bank.config['inputs'],
                windows=windows,dimensions=dimensions.pop(),native_bytes=native_bytes,shards=shards,
                payload_validation='performed_by_native_batch_reader_before_use')


class FullLane(BankLane):
    def __init__(self,bank,name,census):
        super().__init__(bank,name);self.pins={x['start']:x['pin'] for x in census['shards']}

    def __call__(self,uids):
        result=super().__call__(uids)
        for start in {uid//16384*16384 for uid in uids}:
            need(self.bank.descriptors[(self.name,start)]['pin']==self.pins[start],'LEGAL_INDEX_CENSUS_DRIFT')
        return result


def benchmark(a,baseline,source):
    bank=NativeBatchSession(Path(baseline['bank']['path']),baseline['bank']['pin'],source)
    uids=np.arange(baseline['prefix_inputs'],dtype='<i8');records=[];phases={};start=time.perf_counter()
    for threads in (1,8,32):
        for name in LANES:
            tick=time.perf_counter();read=BankLane(bank,name);ref=baseline['lanes'][name]['prefix']
            prior=NativeWindowIndex(Path(ref['path']),ref['pin'],json.loads((Path(ref['path'])/'report.json').read_bytes())['binding'])
            original_path=Path(ref['path']).parent/'prefix-assessment.json'
            need(file_sha(original_path)==ref['assessment_pin'],'LEGAL_INDEX_BASELINE_ASSESSMENT')
            original=json.loads(original_path.read_bytes());root=a.output/f'{name}-threads-{threads}'
            binding={**prior.report['binding'],'scope':'matched_construction_thread_diagnostic','construction_threads':threads}
            pin=publish(root,uids,read,binding,build_threads=threads)
            index=NativeWindowIndex(root,pin,binding);assessment=assess(index,read,baseline['query_uids'])
            need([r['exact_hits'] for r in assessment['rows']]==[r['exact_hits'] for r in original['rows']],
                 'LEGAL_INDEX_EXACT_COMPARATOR_CHANGED')
            proof=dict(vectors=vectors(index,read),rankings=rankings(index,read,assessment))
            result=dict(instrument=name,threads=threads,index=dict(path=str(root),pin=pin),
                construction_seconds=index.report['seconds'],cpu_seconds=index.report['cpu_seconds'],
                phases_seconds=index.report['phases_seconds'],memory_bytes=index.report['index_memory_bytes'],
                assessment=assessment,proof=proof,seconds=time.perf_counter()-tick)
            receipt=write_committed(root/'execution-readback.json',result)
            records.append(dict(path=str(root/'execution-readback.json'),pin=receipt,instrument=name,threads=threads,
                construction_seconds=result['construction_seconds'],cpu_seconds=result['cpu_seconds'],
                recall=assessment['recall'],qualified=assessment['qualified']))
            print(json.dumps(records[-1]),flush=True)
            phases[f'{name}_{threads}']=time.perf_counter()-tick
    totals={t:sum(r['construction_seconds'] for r in records if r['threads']==t) for t in (1,8,32)}
    qualified=[t for t in totals if all(r['qualified'] for r in records if r['threads']==t)]
    need(qualified,'LEGAL_INDEX_NO_QUALIFIED_THREAD_ARM');selected=min(qualified,key=lambda t:(totals[t],t))
    controls=dict(zero_threads=rejected(lambda:execution_policy(0,INDEX_MEMORY_BYTES)),
                  excessive_threads=rejected(lambda:execution_policy(33,INDEX_MEMORY_BYTES)),
                  raised_memory_policy=rejected(lambda:execution_policy(1,INDEX_MEMORY_BYTES+1)),
                  inadequate_memory=rejected(lambda:publish(a.output/'refused-memory',uids,read,binding,memory_bytes=1)))
    seconds=time.perf_counter()-start;need(sum(phases.values())/seconds>=.9,'LEGAL_INDEX_BENCH_ATTRIBUTION')
    return dict(records=records,selected_threads=selected,construction_totals=totals,controls=controls,
                exact_comparator_matches_persisted_phase32=True,seconds=seconds,phases_seconds=phases,
                selection='fastest_total_construction_over_two_lanes_given_each_recall_at_least_0.95',
                full_scale_speedup_proven=False)


def available_memory():
    values={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    return int(values['MemAvailable'].split()[0])*1024


def full(a,baseline,source):
    need(a.index_root is not None and a.benchmark is not None and a.benchmark_pin is not None,'LEGAL_INDEX_FULL_ARGUMENTS')
    a.index_root.mkdir(parents=True,exist_ok=False)
    benchmark=load(a.benchmark,a.benchmark_pin);threads=benchmark['result']['selected_threads']
    need(benchmark['source']['pin']==source.pin and benchmark['baseline']['pin']==a.baseline_pin,'LEGAL_INDEX_BENCHMARK_BINDING')
    for record in benchmark['result']['records']:
        need(file_sha(Path(record['path']))==record['pin'],'LEGAL_INDEX_BENCHMARK_RECEIPT')
    continuation=Continuations(Path(baseline['continuations']['path']),baseline['continuations']['pin'])
    need(continuation.report['source']['pin']==source.pin,'LEGAL_INDEX_CONTINUATION_SOURCE')
    nontrain=np.flatnonzero((continuation.presence&1)==0)[:128].tolist();need(len(nontrain)==128,'LEGAL_INDEX_QUERY_SUPPORT')
    proof_root=Path(baseline['proof']['path']);proof=load(proof_root,baseline['proof']['pin'])
    need(file_sha(proof_root/'inputs.json')==proof['files']['inputs.json']['sha256'],'LEGAL_INDEX_CANONICAL_QUERIES')
    canonical=[r['index'] for r in json.loads((proof_root/'inputs.json').read_bytes())]
    queries=sorted(set(nontrain+canonical));query_pin=write_committed(a.output/'query-selection.json',
        dict(nontrain=nontrain,canonical=canonical,union=queries,selection='hash_order_before_similarity'))
    results={}
    for name in a.instruments:
        lane_start=tick=time.perf_counter();memory_before=available_memory()
        need(memory_before>=INDEX_MEMORY_BYTES+(4<<30),'LEGAL_INDEX_HOST_MEMORY_UNAVAILABLE')
        bank=NativeBatchSession(Path(baseline['bank']['path']),baseline['bank']['pin'],source)
        need(name in bank.config['roster'],'LEGAL_INDEX_INSTRUMENT')
        counts=census(bank,name);census_pin=write_committed(a.output/(name+'-census.json'),counts)
        read=FullLane(bank,name,counts);uids=np.arange(counts['inputs'],dtype='<i8')
        phases=dict(census=time.perf_counter()-tick);tick=time.perf_counter()
        binding=dict(source_pin=source.pin,bank_pin=bank.pin,lens_id=bank.config['instruments'][name]['lens_id'],
            instrument=name,scope='full_legal_unique_input_gallery',census_pin=census_pin,
            benchmark_pin=a.benchmark_pin,query_selection_pin=query_pin)
        with (a.output/(name+'-progress.jsonl')).open('x') as stream:
            def progress(value):
                row=dict(instrument=name,**value);stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
                print(json.dumps(row),flush=True)
            root=a.index_root/name;pin=publish(root,uids,read,binding,build_threads=threads,progress=progress)
        phases['construction']=time.perf_counter()-tick;tick=time.perf_counter()
        index=NativeWindowIndex(root,pin,binding)
        need(index.report['windows']==counts['windows'] and len(bank.descriptors)==len(counts['shards']),
             'LEGAL_INDEX_FULL_SOURCE_COVERAGE')
        native_proof=vectors(index,read)
        phases['cold_load_and_every_native_vector']=time.perf_counter()-tick;tick=time.perf_counter()
        assessment=assess(index,read,queries)
        groups={}
        multiwindow=[uid for uid in canonical if index.lengths[uid]>1]
        for group,ids in [('nontrain',nontrain),('canonical',canonical),('canonical_multiwindow',multiwindow)]:
            members=set(ids);rows=[r for r in assessment['rows'] if r['uid'] in members]
            groups[group]=dict(queries=len(rows),recovered=sum(r['recovered'] for r in rows),requested=10*len(rows),
                recall=sum(r['recovered'] for r in rows)/(10*len(rows)) if rows else None)
        phases['full_gallery_recall']=time.perf_counter()-tick;tick=time.perf_counter()
        bank.fresh();continuation.fresh();index.fresh()
        need(all(bank.descriptors[(name,r['start'])]['pin']==r['pin'] for r in counts['shards']),
             'LEGAL_INDEX_FINAL_SHARD_BINDING')
        phases['final_freshness']=time.perf_counter()-tick;seconds=time.perf_counter()-lane_start
        need(sum(phases.values())/seconds>=.9,'LEGAL_INDEX_FULL_ATTRIBUTION')
        result=dict(index=dict(path=str(root),pin=pin),census_pin=census_pin,native_proof=native_proof,
            assessment=assessment,groups=groups,phases_seconds=phases,seconds=seconds,reader_stats=bank.stats,
            available_memory_before=memory_before,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            independent_recall_readback_pending=True,scientific_admission=False)
        receipt=write_committed(a.output/(name+'-full-readback.json'),result)
        results[name]=dict(path=str(a.output/(name+'-full-readback.json')),pin=receipt,index_pin=pin,groups=groups,seconds=seconds)
        print(json.dumps(dict(completed_instrument=name,**results[name])),flush=True)
        del index,read,bank,uids
    return dict(instruments=results,selected_threads=threads,query_selection_pin=query_pin)


def run(a):
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False);baseline=load(a.baseline,a.baseline_pin)
    source=ReuseInputs(Path(baseline['source']['path']),baseline['source']['pin'])
    with lease():result=benchmark(a,baseline,source) if a.mode=='benchmark' else full(a,baseline,source)
    source.fresh()
    report=dict(schema='legal-index-execution-v1',complete=True,mode=a.mode,result=result,
        source=baseline['source'],baseline=dict(path=str(a.baseline),pin=a.baseline_pin),seconds=time.perf_counter()-start,
        code_sha256=file_sha(Path(__file__)),index_code_sha256=file_sha(Path(__file__).with_name('native_window_index.py')),
        new_encoder_calls=0,predictor_updates=0,scientific_admission=False,acceptance='author-checked',number_status='measured')
    pin=write_committed(a.output/'report.json',report);print(json.dumps(dict(pin=pin,seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['benchmark','full'],required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--baseline-pin',required=True)
    p.add_argument('--benchmark',type=Path);p.add_argument('--benchmark-pin')
    p.add_argument('--index-root',type=Path)
    p.add_argument('--instruments',nargs='+',default=list(LANES));p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
