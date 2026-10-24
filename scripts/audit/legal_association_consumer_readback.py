"""Independently read stored adjacency and audit every emitted typed path, then cold-replay."""
from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from audit import legal_association_consumer as producer
from kernel.published_associations import need, read_json, sha
from kernel.typed_association_paths import TypedPaths


class Direct:
    def __init__(self, refs):
        self.refs = {r['name']: r for r in refs}; self.rows = {}; self.stops = {}; self.maps = OrderedDict()
        for name, ref in self.refs.items():
            root = Path(ref['path']); marker = json.loads((root/'COMMITTED').read_bytes())
            r = read_json(root/marker.get('report_path', 'report.json'), ref['pin'])
            if 'shards' in r: rows = r['shards']
            else:
                imported = read_json(root/'import.json', r['contract']['import_pin'])
                rows = imported['shards']+r['new_shards']
            self.rows[name] = rows; self.stops[name] = [v['stop'] for v in rows]

    def row(self, name, uid):
        number = bisect_right(self.stops[name], uid); s = self.rows[name][number]; key = (name, number)
        if key not in self.maps:
            if len(self.maps) == 16:
                for a in self.maps.popitem(last=False)[1].values(): a._mmap.close()
            self.maps[key] = {n: np.load(Path(s['path'])/(n+'.npy'), mmap_mode='r', allow_pickle=False)
                              for n in ['queries', 'targets', 'scores']}
        self.maps.move_to_end(key); a = self.maps[key]; j = uid-s['start']
        need(int(a['queries'][j]) == uid, 'INDEPENDENT_ORIGIN')
        return s, [int(v) for v in a['targets'][j]], [float(v) for v in a['scores'][j]]

    def edge(self, e):
        name = e['instrument']; need(name in self.refs, 'INDEPENDENT_INSTRUMENT')
        ref = self.refs[name]; b = ref['binding']; s, ids, scores = self.row(name, e['origin']); rank = e['rank']
        need(type(rank) is int and 1 <= rank <= 10 and e['target'] == ids[rank-1] and e['score'] == scores[rank-1], 'INDEPENDENT_EDGE')
        need(e['lens_id'] == b['lens_id'] and e['source_pin'] == b['source_pin'] and e['bank_pin'] == b['bank_pin']
             and e['graph_pin'] == ref['pin'] and e['shard_pin'] == s['pin']
             and e['relation'] == 'same_instrument_native_window_neighbour', 'INDEPENDENT_PROVENANCE')

    def path(self, p):
        uid = p['origin']; need(len(p['edges']) == len(p['instruments']), 'INDEPENDENT_HOPS')
        for name, e in zip(p['instruments'], p['edges']):
            need(e['origin'] == uid and e['instrument'] == name, 'INDEPENDENT_PATH_JOIN'); self.edge(e); uid = e['target']
        need(uid == p['target'] and p['cross_instrument_score'] is None and p['conclusion'] == 'association_path_only'
             and not p['scientific_admission'] and not p['implicit_training_authority'] and not p['all_pairs_exhaustive'], 'INDEPENDENT_PATH_SCOPE')

    def close(self):
        for arrays in self.maps.values():
            for a in arrays.values(): a._mmap.close()
        self.maps.clear()


def previous_comparison(previous, current, refs, output):
    """ABBA on the same real requests; the old implementation is retained and pinned."""
    root = Path(previous['path']); old_report = read_json(root/'report.json', previous['pin'])
    for n in ['rows.jsonl', 'paths.jsonl', 'single-lens.json']:
        need(sha(root/n) == old_report['files'][n] == sha(output/n), 'PREVIOUS_PHYSICAL_OUTPUT')
    source = root/'source/kernel/published_associations.py'
    need(sha(source) == old_report['code']['kernel/published_associations.py'], 'PREVIOUS_CODE_PIN')
    spec = importlib.util.spec_from_file_location('retained_association_reader', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    measurements = []; authentication = []
    for ref in refs:
        name = ref['name']; lane = current[name]; b = ref['binding']
        with module.PublishedAssociations(ref['path'], ref['pin'], source_pin=b['source_pin'],
                bank_pin=b['bank_pin'], lens_id=b['lens_id']) as old:
            authentication.append(dict(instrument=name, seconds=old.stats['authentication_seconds']))
            cases = dict(contiguous=list(range(128)), scattered=producer.selection(lane)[:128],
                         interleaved=[lane.shards[i % 16]['start']+(i//16) for i in range(128)])
            for case, ids in cases.items():
                expected = old.read(ids, lens_id=b['lens_id'])
                need(lane.read(ids, lens_id=b['lens_id']) == expected, 'GROUPED_WARMUP_EXACT')
                for mode, reader in [('rowwise', old), ('grouped', lane), ('grouped', lane), ('rowwise', old)]:
                    before = dict(reader.stats); start = time.perf_counter()
                    actual = reader.read(ids, lens_id=b['lens_id']); seconds = time.perf_counter()-start
                    need(actual == expected, 'GROUPED_MATCHED_EXACT')
                    measurements.append(dict(instrument=name, case=case, mode=mode, rows=len(ids), seconds=seconds,
                        maps_opened=reader.stats['maps_opened']-before['maps_opened'],
                        cache_hits=reader.stats['cache_hits']-before['cache_hits'],
                        cache_misses=reader.stats['cache_misses']-before['cache_misses']))
    return dict(previous=previous, all_matched_outputs_exact=True, benchmark=measurements,
                previous_authentication=authentication, all_graphs_rebuilt=False)


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    r = read_json(a.input/'report.json', a.input_pin); need(r['complete'] and r['code'] == producer.code(), 'READBACK_CODE')
    for n, pin in r['files'].items(): need(sha(a.input/n) == pin, 'READBACK_PHYSICAL_FILE')
    census = read_json(Path(r['census']), r['census_pin']); result = read_json(a.input/'results.json')
    direct = Direct(census['graphs']); lanes = {}; counts = {}; paths = 0; phases = {}; controls = {}
    try:
        for row in (json.loads(s) for s in (a.input/'rows.jsonl').open()):
            name = row['instrument']; v = row['row']; counts[name] = counts.get(name, 0)+1
            need(len(v['edges']) == 10 and not v['consensus_required'] and not v['all_pairs_exhaustive']
                 and v['absence_means'] == 'not_materialized_not_no_association', 'READBACK_ROW_SCOPE')
            for rank, e in enumerate(v['edges'], 1):
                need(e['origin'] == v['origin'] and e['instrument'] == name and e['rank'] == rank, 'READBACK_ROW_JOIN'); direct.edge(e)
        need(counts == {n: v['origins'] for n, v in result['counts'].items()}, 'READBACK_ROW_COUNTS')
        groups = {}; witness = None
        with (a.input/'paths.jsonl').open() as stream:
            for raw in stream:
                p = json.loads(raw); direct.path(p); key = (p['origin'], *p['instruments']); i = groups.get(key, 0)
                need([e['rank'] for e in p['edges']] == [i//10+1, i%10+1], 'READBACK_PATH_ORDER')
                groups[key] = i+1; paths += 1; witness = p
        need(all(n == 100 for n in groups.values()) and paths == result['path_count'] == len(result['seeds'])*12*100,
             'READBACK_PATH_COVERAGE')
        exclusive = read_json(a.input/'single-lens.json')
        for edge in exclusive:
            direct.edge(edge)
            need(all(edge['target'] not in direct.row(n, edge['origin'])[1] for n in direct.refs if n != edge['instrument']),
                 'READBACK_SINGLE_LENS_ONLY')
        need(len(exclusive) == result['exclusive_edges'] > 0, 'READBACK_SINGLE_LENS_COUNT')
        bad = deepcopy(witness); bad['edges'][1]['origin'] += 1
        controls['broken_path'] = producer.rejected(lambda: direct.path(bad))
        bad = deepcopy(witness['edges'][0]); bad['score'] = float(np.nextafter(bad['score'], np.inf))
        controls['changed_score'] = producer.rejected(lambda: direct.edge(bad))
        bad = deepcopy(witness['edges'][0]); bad['lens_id'] = '0'*64
        controls['wrong_lens'] = producer.rejected(lambda: direct.edge(bad))
        phases['independent_rows_paths'] = time.perf_counter()-tick; tick = time.perf_counter()
        for ref in census['graphs']: lanes[ref['name']] = producer.open_lane(ref)
        phases['cold_full_authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
        with (a.input/'rows.jsonl').open() as stream:
            for name, lane in lanes.items():
                ids = producer.selection(lane, result['native_input_uids']); ids += list(reversed(ids[:8]))+[ids[0], ids[-1]]
                for lo in range(0, len(ids), 128):
                    for actual in lane.read(ids[lo:lo+128], lens_id=lane.binding['lens_id']):
                        need(json.loads(next(stream)) == dict(instrument=name, row=actual), 'READBACK_COLD_ROW')
            need(stream.read() == '', 'READBACK_EXTRA_ROWS')
        walker = TypedPaths(lanes)
        with (a.input/'paths.jsonl').open() as stream:
            for uid in result['seeds']:
                for first in lanes:
                    for second in lanes:
                        if first == second: continue
                        for actual in walker.paths(uid, [first, second]):
                            need(json.loads(next(stream)) == actual, 'READBACK_COLD_PATH')
            need(stream.read() == '', 'READBACK_EXTRA_PATHS')
        phases['cold_rows_paths'] = time.perf_counter()-tick; tick = time.perf_counter()
        # Baseline is direct loading of the same shard files, not a claim about ANN search time.
        benchmark = []
        for name, lane in lanes.items():
            ids = producer.selection(lane, result['native_input_uids'])[:128]
            for mode in ['stateless_numpy', 'resident', 'resident', 'stateless_numpy']:
                start = time.perf_counter()
                if mode == 'resident':
                    rows = lane.read(ids, lens_id=lane.binding['lens_id'])
                    values = [[(e['target'], e['score']) for e in v['edges']] for v in rows]
                else:
                    values = []
                    for uid in ids:
                        s = direct.rows[name][bisect_right(direct.stops[name], uid)]; offset = uid-s['start']; p = Path(s['path'])
                        targets = np.load(p/'targets.npy', allow_pickle=False); scores = np.load(p/'scores.npy', allow_pickle=False)
                        values.append(list(zip(map(int, targets[offset]), map(float, scores[offset]))))
                elapsed = time.perf_counter()-start
                expected = [list(zip(*direct.row(name, uid)[1:])) for uid in ids]
                need(values == expected, 'READBACK_BENCHMARK_EQUIVALENCE')
                benchmark.append(dict(instrument=name, mode=mode, rows=len(ids), seconds=elapsed))
        phases['matched_lookup_profile'] = time.perf_counter()-tick; tick = time.perf_counter()
        comparison = previous_comparison(r['previous'], lanes, census['graphs'], a.input) if r.get('previous') else None
        phases['previous_comparison'] = time.perf_counter()-tick; tick = time.perf_counter()
        for lane in lanes.values(): lane.fresh(); lane.close()
        direct.close(); need('torch' not in sys.modules and 'usearch' not in sys.modules, 'READBACK_NO_MODEL')
        need(producer.code() == r['code'], 'READBACK_CODE_CHANGED')
        seconds = time.perf_counter()-began; phases['close'] = time.perf_counter()-tick
        need(.9 <= sum(phases.values())/seconds <= 1.000001, 'READBACK_PHASE_COVERAGE')
        pin = producer.write(a.output/'report.json', dict(complete=True, input_pin=a.input_pin, code=r['code'], counts=counts,
            paths=paths, single_lens_only=len(exclusive), all_rows_and_paths_exact=True, cold_replay_exact=True,
            benchmark=benchmark, comparison=comparison, controls=controls, phases=phases, seconds=seconds,
            all_pairs_exhaustive=False, native_scores_recomputed=False, scientific_admission=False, implicit_training_authority=False,
            acceptance='author-checked', number_status='measured'))
        print(json.dumps(dict(complete=True, pin=pin, seconds=seconds)), flush=True)
    finally:
        direct.close()
        for lane in lanes.values(): lane.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--input-pin', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.exists(): producer.write(a.output/'failure.json', dict(error=repr(e)))
        raise
