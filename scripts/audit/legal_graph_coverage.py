"""Publish or cold-read complete legal native graphs, one instrument per lease."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.native_graph_coverage import check_arrays, full_coverage, read_shard, replay_samples
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane, pinned
from kernel.native_graph_store import publish
from text.constellation_corpus import file_sha, need


def code():
    scripts = Path(__file__).resolve().parents[1]
    return {n: file_sha(scripts/n) for n in ['audit/legal_graph_coverage.py', 'audit/native_graph_coverage.py',
        'kernel/native_graph_store.py', 'kernel/native_graph_batch.py', 'kernel/certified_index_graph.py',
        'kernel/native_window_index.py', 'audit/native_index_reference.py']}


def acquire():
    path = Path(__file__).resolve().parents[2]/'.lease/legal-native-graph'; path.parent.mkdir(exist_ok=True)
    stream = path.open('a+'); fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = dict(pid=os.getpid(), starttime=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                 path=str(path), issue=29)
    stream.seek(0); stream.truncate(); stream.write(json.dumps(owner)); stream.flush(); os.fsync(stream.fileno())
    return stream, owner


def negatives(root, r, arrays, lane, plan_pin, pin, selected, scratch):
    result = {}
    for kind in ['missing_query', 'duplicate_target', 'changed_score']:
        a = {n: v.copy() for n, v in arrays.items()}
        if kind == 'missing_query': a['queries'] = a['queries'][:-1]
        if kind == 'duplicate_target': a['targets'][0, 0] = a['targets'][0, 1]
        if kind == 'changed_score': a['scores'][0, 0] = -2.
        result[kind] = rejected(lambda: check_arrays(a, arrays['queries'], len(lane.index.uids)))
    result['wrong_plan'] = rejected(lambda: read_shard(root, pin, '0'*64, lane, arrays['queries']))
    incomplete = [dict(start=0, stop=1, queries=1, native_windows=1)]
    result['false_full_coverage'] = rejected(lambda: full_coverage(incomplete, len(lane.index.uids), lane.index.report['windows']))
    a = {n: v.copy() for n, v in arrays.items()}; a['scores'][0, 0] -= .001
    result['plausible_wrong_score'] = rejected(lambda: replay_samples(a, lane, [int(a['queries'][0])]))
    scratch.mkdir()
    for name in [*r['files'], 'report.json', 'COMMITTED']:
        (scratch/name).write_bytes((root/name).read_bytes())
    with (scratch/'scores.npy').open('r+b') as stream:
        stream.seek(-1, 2); value = stream.read(1); stream.seek(-1, 2); stream.write(bytes([value[0] ^ 1]))
    result['changed_file'] = rejected(lambda: read_shard(scratch, pin, plan_pin, lane, arrays['queries']))
    return result


def run(a):
    lease, owner = acquire(); start = tick = time.perf_counter(); cpu = time.process_time()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    need(not (a.output/'report.json').exists(), 'GRAPH_RUN_ALREADY_COMPLETE')
    q = pinned(a.native_qualification/'report.json', a.native_qualification_pin)
    bench = pinned(a.benchmark/'report.json', a.benchmark_pin)
    need(q['complete'] and bench['complete'] and bench['binding'] == q['binding']
         and bench['qualification'] == dict(path=str(a.native_qualification), pin=a.native_qualification_pin)
         and all(r['all_ranks_scores_candidates_exact'] for r in bench['records']), 'GRAPH_RUN_BENCHMARK')
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    need(available >= 24 << 30, 'GRAPH_RUN_PHYSICAL_MEMORY_UNAVAILABLE')
    ref = q['binding']['certificate']; lane = CertifiedIndexLane(Path(ref['path']), ref['pin'])
    need(lane.binding == q['binding'] and np.array_equal(lane.index.uids, np.arange(len(lane.index.uids))),
         'GRAPH_RUN_POPULATION')
    selection = pinned(a.native_qualification/'selection.json', q['selection_pin'])
    full = a.mode == 'full'
    previous = None
    if a.mode == 'cold':
        previous = pinned(a.previous/'report.json', a.previous_pin)
        need(previous['complete'] and previous['mode'] in ['qualify', 'full'], 'GRAPH_RUN_COLD_SOURCE')
        full = previous['scope'] == 'full'
    scope = 'full' if full else 'qualify'
    uids = lane.index.uids if full else np.array(sorted(set(range(512)) | set(selection['uids'])), dtype='<i8')
    policy = dict(query_tile=64, shard_rows=16384 if full else 256, top_k=10,
                  threads=bench['selected_threads'], score_policy='unchanged_native_window_index',
                  candidate_policy='128_per_native_window', all_pairs_exhaustive=False)
    plan = dict(scope=scope, code=code(), binding=lane.binding, benchmark_pin=a.benchmark_pin,
                native_qualification_pin=a.native_qualification_pin, policy=policy,
                queries=len(uids), population=len(lane.index.uids), scientific_admission=False,
                implicit_training_authority=False, all_instruments_flattened=False)
    if previous:
        plan_pin = previous['plan_pin']; saved = pinned(a.previous/'plan.json', plan_pin)
        need(saved == plan, 'GRAPH_RUN_COLD_PLAN')
    else:
        path = a.output/'plan.json'
        if path.exists():
            plan_pin = file_sha(path); need(pinned(path, plan_pin) == plan, 'GRAPH_RUN_RESUME_PLAN')
        else: plan_pin = write_committed(path, plan)
        if full:
            qualification = pinned(a.qualification/'report.json', a.qualification_pin)
            cold = pinned(a.qualification_cold/'report.json', a.qualification_cold_pin)
            need(qualification['complete'] and qualification['scope'] == 'qualify' and cold['complete']
                 and cold['mode'] == 'cold' and cold['previous_pin'] == a.qualification_pin
                 and qualification['plan']['code'] == plan['code'] and qualification['plan']['binding'] == lane.binding
                 and qualification['plan']['benchmark_pin'] == a.benchmark_pin, 'GRAPH_RUN_QUALIFICATION')
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    records = []; sample_total = 0; expected_samples = set(selection['uids']); negatives_result = None
    with (a.output/'events.jsonl').open('a' if a.resume else 'x') as events:
        for first in range(0, len(uids), policy['shard_rows']):
            last = min(first+policy['shard_rows'], len(uids)); ids = [int(x) for x in uids[first:last]]
            root = (a.previous if previous else a.output)/f'{first:012d}-{last:012d}'
            existing = root.exists()
            if existing:
                need((root/'COMMITTED').exists(), 'GRAPH_RUN_TORN_SHARD_RETAINED')
                marker = json.loads((root/'COMMITTED').read_bytes()); pin = marker['report_sha256']
            else:
                need(previous is None, 'GRAPH_RUN_COLD_SHARD_MISSING')
                pin = publish(root, ids, lane, plan_pin, threads=policy['threads'])
            r, arrays = read_shard(root, pin, plan_pin, lane, ids, committed=existing)
            chosen = ids if not full else sorted(({ids[0], ids[len(ids)//2], ids[-1]} | (set(ids) & expected_samples)))
            sample = replay_samples(arrays, lane, chosen); sample_total += sample['samples']
            if not existing:
                write_committed(root/'readback.json', sample)
                write_committed(root/'COMMITTED', dict(report_sha256=pin))
            if not full and negatives_result is None:
                scratch = a.output/('negative-corrupt-copy-'+str(time.time_ns()))
                negatives_result = negatives(root, r, arrays, lane, plan_pin, pin, chosen, scratch)
            record = dict(start=first, stop=last, queries=len(ids), native_windows=r['native_windows'],
                          path=str(root), pin=pin, reused=existing, sample=sample)
            if previous:
                old = previous['shards'][len(records)]
                need(all(record[k] == old[k] for k in ['start', 'stop', 'queries', 'native_windows', 'path', 'pin', 'sample']),
                     'GRAPH_RUN_COLD_RECEIPT')
            records.append(record)
            events.write(json.dumps(dict(start=first, stop=last, pin=pin, reused=existing,
                native_windows=r['native_windows'], samples=sample['samples'], seconds=time.perf_counter()-start), sort_keys=True)+'\n')
            events.flush(); os.fsync(events.fileno())
            print(json.dumps(dict(scope=scope, completed_queries=last, total=len(uids), shard_pin=pin)), flush=True)
    phases['publish_and_readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    if full: full_coverage(records, len(lane.index.uids), lane.index.report['windows'])
    need(sum(r['queries'] for r in records) == len(uids), 'GRAPH_RUN_FINAL_COVERAGE')
    if previous: need(len(records) == len(previous['shards']), 'GRAPH_RUN_COLD_EXTRA_SHARDS')
    lane.fresh(); phases['final_coverage'] = time.perf_counter()-tick; seconds = time.perf_counter()-start
    need(sum(phases.values())/seconds >= .9, 'GRAPH_RUN_TIMING_COVERAGE')
    result = dict(schema='legal-native-graph-coverage-v1', complete=True, mode=a.mode, scope=scope,
                  plan=plan, plan_pin=plan_pin, previous_pin=a.previous_pin, shards=records,
                  queries=len(uids), edges=10*len(uids), native_windows=sum(r['native_windows'] for r in records),
                  full_graph_complete=full, all_rows_structurally_read=True, sampled_numeric_queries=sample_total,
                  all_numeric_rows_replayed=not full, controls=negatives_result, source_stats=lane.stats,
                  phases_seconds=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, lease=owner,
                  available_memory_before=available,
                  model_updates=0, scientific_admission=False, acceptance='author-checked', number_status='measured')
    pin = write_committed(a.output/'report.json', result); write_committed(a.output/'COMMITTED', dict(report_sha256=pin))
    print(json.dumps(dict(complete=True, pin=pin, queries=len(uids), seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['qualify', 'full', 'cold'], required=True)
    for name in ['native-qualification', 'benchmark']:
        p.add_argument('--'+name, type=Path, required=True); p.add_argument('--'+name+'-pin', required=True)
    for name in ['qualification', 'qualification-cold', 'previous']:
        p.add_argument('--'+name, type=Path); p.add_argument('--'+name+'-pin')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/('failure-'+str(time.time_ns())+'.json')
        write_committed(path, dict(complete=False, error=type(error).__name__+':'+str(error), code=code()))
        raise
