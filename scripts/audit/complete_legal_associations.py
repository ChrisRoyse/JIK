"""Publish the complete six-instrument association consumer without rebuilding graphs."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import legal_association_consumer as old
from kernel.association_panel import AssociationPanel
from kernel.published_associations import need, read_json, sha

ROSTER = ['all-mpnet-base-v2', 'codebert-mean', 'e5-base-v2', 'finbert-mean',
          'pubmedbert-embeddings', 'specter2-base']
ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
INPUT_PROOF = ('legal-surface-decoder-proof-v1-20261023',
               '3400b9c5813d56859603a89d5f0fe3182370b94a68e014199b50a2ff605c8d45')


def code():
    root = Path(__file__).resolve().parents[1]
    return dict(old.code(), **{n: sha(root/n) for n in ['kernel/association_panel.py',
        'audit/complete_legal_associations.py', 'audit/complete_legal_associations_readback.py']})


def encoded(value): return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def selections(panel):
    proof = read_json(ROOT/INPUT_PROOF[0]/'report.json', INPUT_PROOF[1])
    extra = sorted({r['uid'] for r in proof['gradients']})
    ids = {}
    for n, lane in panel.lanes.items():
        base = old.selection(lane, extra)
        ids[n] = base + list(reversed(base[:8])) + [base[0], base[-1]]
    seeds = sorted({0, 16383, 16384, 1261567, 1261568, 4603903, 4603904, 6360377} | set(extra))
    return dict(rows=ids, seeds=seeds, native_input_uids=extra, native_input_proof=INPUT_PROOF[1])


def negative_controls(panel, manifest, pin, output):
    controls = {}; expected = deepcopy(panel.manifest)
    for name, lane in panel.lanes.items():
        for uid in [-1, lane.population, True]:
            controls[name+'/uid/'+str(uid)] = old.rejected(lambda u=uid, n=name: panel.read(n, [u]))
        controls[name+'/wrong_lens'] = old.rejected(lambda: lane.read([0], lens_id='0'*64))
    controls['missing_lane'] = old.rejected(lambda: panel.read('missing-lane', [0]))
    controls['missing_path_lane'] = old.rejected(lambda: list(panel.paths(0, [ROSTER[0], 'missing-lane'])))
    controls['training'] = old.rejected(panel.training_authority)
    controls['wrong_manifest_pin'] = old.rejected(lambda: AssociationPanel(manifest, '0'*64, expected_roster=ROSTER))
    for mode in ['missing', 'duplicate', 'changed_source', 'wrong_cold']:
        value = deepcopy(expected)
        if mode == 'missing': value['graphs'].pop()
        if mode == 'duplicate': value['graphs'][-1] = deepcopy(value['graphs'][0])
        if mode == 'changed_source': value['graphs'][-1]['binding']['source_pin'] = '0'*64
        if mode == 'wrong_cold': value['graphs'][-1]['cold'] = deepcopy(value['graphs'][0]['cold'])
        path = output/('negative-'+mode+'.json'); bad_pin = old.write(path, value)
        controls[mode] = old.rejected(lambda: AssociationPanel(path, bad_pin, expected_roster=ROSTER))
    controls.update(old.fault_controls(expected['graphs'][0], output))
    for name, lane in panel.lanes.items():
        ids = [lane.shards[i % 16]['start']+(i//16) for i in range(128)]
        need([r['origin'] for r in panel.read(name, ids)] == ids, 'PANEL_ORDER')
        pair = panel.read(name, [ids[0], ids[0]]); pair[0]['edges'][0]['score'] = 999
        need(pair[1]['edges'][0]['score'] != 999, 'PANEL_RESULT_ALIAS')
    g = panel.paths(0, ROSTER[:2]); row = next(g); row['edges'][0]['score'] = 999
    need(next(g)['edges'][0]['score'] != 999, 'PANEL_PATH_ALIAS'); g.close()
    need(g.gi_frame is None, 'PANEL_GENERATOR_RELEASE')
    return controls


def run(a):
    start = tick = time.perf_counter(); cpu = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    pins = code(); panel = AssociationPanel(a.manifest, a.pin, expected_roster=ROSTER)
    phases = dict(authentication=time.perf_counter()-tick); tick = time.perf_counter()
    try:
        plan = selections(panel); old.write(a.output/'selection.json', plan)
        old.write(a.output/'manifest.json', panel.manifest)
        need(sha(a.output/'manifest.json') == a.pin, 'PANEL_MANIFEST_COPY')
        counts = {}; path_counts = {}; exclusive = []
        with (a.output/'rows.jsonl').open('xb') as stream:
            for name, ids in plan['rows'].items():
                count = 0
                for lo in range(0, len(ids), 128):
                    for row in panel.read(name, ids[lo:lo+128]):
                        stream.write(encoded(dict(instrument=name, row=row))); count += 1
                counts[name] = dict(origins=count, edges=10*count)
            stream.flush(); os.fsync(stream.fileno())
        phases['addressed_rows'] = time.perf_counter()-tick; tick = time.perf_counter()
        with (a.output/'paths.jsonl').open('xb') as stream:
            for uid in plan['seeds']:
                neighbours = {n: panel.read(n, [uid])[0]['edges'] for n in ROSTER}
                for n, edges in neighbours.items():
                    peers = {e['target'] for m, values in neighbours.items() if n != m for e in values}
                    exclusive.extend(e for e in edges if e['target'] not in peers)
                for first in ROSTER:
                    for second in ROSTER:
                        if first == second: continue
                        count = 0
                        for path in panel.paths(uid, [first, second]): stream.write(encoded(path)); count += 1
                        need(count == 100, 'PANEL_PATH_COVERAGE')
                        key = first+'/'+second; path_counts[key] = path_counts.get(key, 0)+count
            stream.flush(); os.fsync(stream.fileno())
        need({e['instrument'] for e in exclusive} == set(ROSTER), 'PANEL_SINGLE_LENS_SUPPORT')
        old.write(a.output/'single-lens.json', exclusive)
        phases['typed_paths'] = time.perf_counter()-tick; tick = time.perf_counter()
        controls = negative_controls(panel, a.manifest, a.pin, a.output)
        phases['negative_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
        stats = {n: dict(lane.stats) for n, lane in panel.lanes.items()}
        maps = [v._mmap for lane in panel.lanes.values() for arrays in lane.cache.values() for v in arrays.values()]
        panel.fresh(); panel.close()
        need(all(m.closed for m in maps) and all(not l.cache for l in panel.lanes.values()), 'PANEL_MAP_RELEASE')
        controls['closed'] = old.rejected(lambda: panel.read(ROSTER[0], [0]))
        old.write(a.output/'results.json', dict(counts=counts, paths=path_counts, exclusive_edges=len(exclusive),
            controls=controls, stats=stats, all_open_maps_released=True))
        for n, pin in pins.items():
            dest = a.output/'source'/n; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes()); need(sha(dest) == pin, 'PANEL_SOURCE')
        files = {str(p.relative_to(a.output)): sha(p) for p in a.output.rglob('*') if p.is_file()}
        need(code() == pins and 'torch' not in sys.modules and 'usearch' not in sys.modules, 'PANEL_RUNTIME')
        phases['close_and_seal'] = time.perf_counter()-tick; seconds = time.perf_counter()-start
        need(.9 <= sum(phases.values())/seconds <= 1.000001, 'PANEL_PHASE_COVERAGE')
        pin = old.write(a.output/'report.json', dict(complete=True, schema='complete-legal-association-consumer-v1',
            manifest_pin=a.pin, code=pins, files=files, graph_count=6, population=panel.manifest['population'],
            total_stored_edges=panel.manifest['total_edges'], total_payload_bytes=panel.manifest['total_payload_bytes'],
            phases=phases, seconds=seconds, cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            all_stored_origins_addressable=True, all_pairs_exhaustive=False, native_scores_recomputed=False,
            kernel_recall_established=False, scientific_admission=False, implicit_training_authority=False,
            new_encoder_calls=0, model_updates=0, acceptance='author-checked', number_status='measured'))
        print(json.dumps(dict(complete=True, pin=pin, seconds=seconds)), flush=True)
    finally: panel.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a)
    except BaseException as e:
        if a.output.exists(): old.write(a.output/'failure.json', dict(error=repr(e)))
        raise
