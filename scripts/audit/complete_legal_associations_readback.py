"""Independently recount six-lens paths from physical shards and cold replay the reader."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import complete_legal_associations as producer
from audit.legal_association_consumer_readback import Direct
from kernel.association_panel import AssociationPanel
from kernel.published_associations import need, read_json, sha


def run(a):
    start = tick = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    r = read_json(a.input/'report.json', a.pin)
    need(r['complete'] and r['code'] == producer.code(), 'COMPLETE_READBACK_CODE')
    for n, pin in r['files'].items(): need(sha(a.input/n) == pin, 'COMPLETE_READBACK_PAYLOAD')
    manifest = read_json(a.input/'manifest.json', r['manifest_pin'])
    results = read_json(a.input/'results.json'); plan = read_json(a.input/'selection.json')
    direct = Direct(manifest['graphs']); panel = None; counts = {}; path_counts = {}; controls = {}
    phases = dict(input_authentication=time.perf_counter()-tick); tick = time.perf_counter()
    try:
        with (a.input/'rows.jsonl').open() as f:
            for line in f:
                row = json.loads(line); name = row['instrument']; value = row['row']
                need(value['origin'] in plan['rows'][name], 'INDEPENDENT_SELECTED_ROW')
                for edge in value['edges']: direct.edge(edge)
                counts[name] = counts.get(name, 0)+1
        need(counts == {n: v['origins'] for n,v in results['counts'].items()}, 'INDEPENDENT_COUNTS')
        first_path = None
        with (a.input/'paths.jsonl').open() as f:
            for line in f:
                row = json.loads(line); direct.path(row)
                if first_path is None: first_path = row
                need(row['origin'] in plan['seeds'] and len(row['instruments']) == 2
                     and row['instruments'][0] != row['instruments'][1], 'INDEPENDENT_PATH_PLAN')
                key = '/'.join(row['instruments']); path_counts[key] = path_counts.get(key, 0)+1
        need(path_counts == results['paths'] and len(path_counts) == 30
             and all(v == 100*len(plan['seeds']) for v in path_counts.values()), 'INDEPENDENT_PATH_COUNTS')
        exclusive = read_json(a.input/'single-lens.json')
        for edge in exclusive:
            direct.edge(edge)
            for name in producer.ROSTER:
                if name != edge['instrument']:
                    need(edge['target'] not in direct.row(name, edge['origin'])[1], 'INDEPENDENT_SINGLE_LENS')
        for mode in ['score', 'join', 'lens', 'scope']:
            bad = deepcopy(first_path)
            if mode == 'score': bad['edges'][0]['score'] = 999
            if mode == 'join': bad['edges'][1]['origin'] = (bad['edges'][1]['origin']+1) % manifest['population']
            if mode == 'lens': bad['edges'][0]['lens_id'] = '0'*64
            if mode == 'scope': bad['conclusion'] = 'proven_conclusion'
            controls[mode] = producer.old.rejected(lambda: direct.path(bad))
        phases['independent_rows_and_paths'] = time.perf_counter()-tick; tick = time.perf_counter()
        panel = AssociationPanel(a.input/'manifest.json', r['manifest_pin'], expected_roster=producer.ROSTER)
        need(producer.selections(panel) == plan, 'COLD_SELECTION')
        phases['cold_authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
        rows_hash = hashlib.sha256()
        for name, ids in plan['rows'].items():
            for lo in range(0, len(ids), 128):
                for row in panel.read(name, ids[lo:lo+128]): rows_hash.update(producer.encoded(dict(instrument=name, row=row)))
        paths_hash = hashlib.sha256(); cold_exclusive = []
        for uid in plan['seeds']:
            neighbours = {n: panel.read(n, [uid])[0]['edges'] for n in producer.ROSTER}
            for n, edges in neighbours.items():
                peers = {e['target'] for m,values in neighbours.items() if m != n for e in values}
                cold_exclusive.extend(e for e in edges if e['target'] not in peers)
            for first in producer.ROSTER:
                for second in producer.ROSTER:
                    if first != second:
                        for path in panel.paths(uid, [first, second]): paths_hash.update(producer.encoded(path))
        need(rows_hash.hexdigest() == r['files']['rows.jsonl'] and paths_hash.hexdigest() == r['files']['paths.jsonl']
             and cold_exclusive == exclusive, 'COLD_EXACT_STREAMS')
        phases['cold_rows_and_paths'] = time.perf_counter()-tick; tick = time.perf_counter()
        panel.fresh(); panel.close(); direct.close()
        need(producer.code() == r['code'] and 'torch' not in sys.modules and 'usearch' not in sys.modules, 'COLD_RUNTIME')
        phases['close'] = time.perf_counter()-tick; seconds = time.perf_counter()-start
        need(.9 <= sum(phases.values())/seconds <= 1.000001, 'COLD_PHASE_COVERAGE')
        pin = producer.old.write(a.output/'report.json', dict(complete=True, previous_pin=a.pin, code=r['code'],
            manifest_pin=r['manifest_pin'], counts=counts, paths=path_counts, exclusive_edges=len(exclusive),
            controls=controls, all_emitted_edges_and_paths_independently_read=True, cold_streams_exact=True,
            phases=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            all_pairs_exhaustive=False, native_scores_recomputed=False, kernel_recall_established=False,
            scientific_admission=False, implicit_training_authority=False, acceptance='author-checked', number_status='measured'))
        print(json.dumps(dict(complete=True, pin=pin, seconds=seconds)), flush=True)
    finally:
        direct.close()
        if panel is not None: panel.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.exists(): producer.old.write(a.output/'failure.json', dict(error=repr(e)))
        raise
