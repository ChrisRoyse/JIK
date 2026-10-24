"""Compare batch query schedules to the persisted singleton graph, then time them."""
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
from audit.legal_graph_readback import open_shard, verify_scope
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel.certified_index_graph import CertifiedIndexLane, pinned
from kernel.native_graph_batch import query_batch
from text.constellation_corpus import digest, file_sha, need


def compare(batch, arrays, positions):
    for row in batch['rows']:
        i = positions[row['uid']]
        expected = [[int(u), float(s)] for u, s in zip(arrays['targets'][i], arrays['scores'][i])]
        need(row['hits'] == expected, 'GRAPH_BATCH_CHANGED_RANK_OR_SCORE')
        need(len(row['candidate_uids']) == int(arrays['candidate_counts'][i])
             and digest(np.asarray(row['candidate_uids'], dtype='<i8').tobytes()) == bytes(arrays['candidate_hashes'][i]).hex(),
             'GRAPH_BATCH_CHANGED_CANDIDATES')


def run(a):
    lease_path = Path(__file__).resolve().parents[2]/'.lease/legal-native-graph'
    lease_path.parent.mkdir(exist_ok=True)
    lease = lease_path.open('a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = dict(pid=os.getpid(), starttime=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                 path=str(lease_path), issue=29)
    lease.seek(0); lease.truncate(); lease.write(json.dumps(owner)); lease.flush(); os.fsync(lease.fileno())
    start = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    q = pinned(a.qualification/'report.json', a.qualification_pin)
    need(q['complete'] and q['exact_native_reader_comparison'], 'GRAPH_BATCH_QUALIFICATION')
    ref = q['binding']['certificate']; lane = CertifiedIndexLane(Path(ref['path']), ref['pin'])
    selection = pinned(a.qualification/'selection.json', q['selection_pin']); uids = selection['uids']
    report, arrays = open_shard(a.qualification/'graph', q['graph_pin'], lane.binding)
    verify_scope(report, arrays, uids); positions = {u: i for i, u in enumerate(uids)}
    authentication = time.perf_counter()-start
    results = []
    for threads in [1, 4, 16, 32]:
        for direction in ['forward', 'reverse']:
            ids = uids if direction == 'forward' else list(reversed(uids))
            name = str(threads)+'-'+direction; target = a.output/(name+'.jsonl')
            timing = {}; seconds = 0.; cpu = 0.; work = 0; rows = 0
            with target.open('x') as stream:
                for lo in range(0, len(ids), 64):
                    result = query_batch(lane, ids[lo:lo+64], threads=threads)
                    stream.write(json.dumps(dict(first=lo, **result), sort_keys=True)+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                    compare(result, arrays, positions)
                    need(len(result['rows']) == result['work']['queries'], 'GRAPH_BATCH_QUERY_COUNT')
                    for k, v in result['phases_seconds'].items(): timing[k] = timing.get(k, 0.)+v
                    seconds += result['seconds']; cpu += result['cpu_seconds']
                    rows += len(result['rows']); work += result['work']['computed_distances']
                os.fsync(stream.fileno())
            record = dict(threads=threads, direction=direction, queries=rows, seconds=seconds, cpu_seconds=cpu,
                          phases_seconds=timing, computed_distances=work, path=str(target), pin=file_sha(target),
                          all_ranks_scores_candidates_exact=True)
            results.append(record); print(json.dumps(record), flush=True)
    totals = {t: sum(r['seconds'] for r in results if r['threads'] == t) for t in [1, 4, 16, 32]}
    selected = min(totals, key=lambda t: (totals[t], t))
    negatives = dict(threads=rejected(lambda: query_batch(lane, uids[:1], threads=0)),
                     oversize_tile=rejected(lambda: query_batch(lane, uids[:65], threads=1)),
                     duplicate_input=rejected(lambda: query_batch(lane, [uids[0]]*2, threads=1)))
    changed = dict(rows=[dict(uid=uids[0], hits=[[int(arrays['targets'][0, 0]), -2.]], candidate_uids=[])])
    negatives['changed_result'] = rejected(lambda: compare(changed, arrays, positions))
    lane.fresh(); result = dict(schema='native-graph-batch-qualification-v1', complete=True,
        qualification=dict(path=str(a.qualification), pin=a.qualification_pin), binding=lane.binding,
        query_count=len(uids), total_queries=sum(r['queries'] for r in results), records=results,
        totals_seconds=totals, selected_threads=selected, selection='fastest_two_pass_total_with_every_result_exact',
        controls=negatives, authentication_seconds=authentication, seconds=time.perf_counter()-start,
        lease=owner,
        source_stats=lane.stats, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        code={str(f.relative_to(Path(__file__).resolve().parents[1])): file_sha(f) for f in [
            Path(__file__), Path(__file__).resolve().parents[1]/'kernel/native_graph_batch.py']},
        new_encoder_calls=0, model_updates=0, full_graph_complete=False, scientific_admission=False,
        acceptance='author-checked', number_status='measured')
    pin = write_committed(a.output/'report.json', result)
    print(json.dumps(dict(complete=True, pin=pin, selected_threads=selected, totals_seconds=totals)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification', type=Path, required=True); p.add_argument('--qualification-pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
