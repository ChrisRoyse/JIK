"""Exercise the full published legal adjacency store and typed mixed-lens paths."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel.published_associations import PublishedAssociations, need, read_json, sha
from kernel.typed_association_paths import TypedPaths

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
CODE = ('kernel/published_associations.py', 'kernel/typed_association_paths.py',
        'audit/legal_association_consumer.py', 'audit/legal_association_consumer_readback.py')


def code():
    root = Path(__file__).resolve().parents[1]; return {n: sha(root/n) for n in CODE}


def write(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with Path(path).open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return sha(path)


def rejected(fn):
    try: fn()
    except (ValueError, FileNotFoundError) as e: return str(e)
    raise ValueError('ASSOCIATION_NEGATIVE_ACCEPTED')


def open_lane(ref, cache_shards=8):
    b = ref['binding']
    return PublishedAssociations(ref['path'], ref['pin'], source_pin=b['source_pin'], bank_pin=b['bank_pin'],
                                 lens_id=b['lens_id'], cache_shards=cache_shards)


def selection(lane, extras=()):
    return sorted({v for s in lane.shards for v in [s['start'], (s['start']+s['stop'])//2, s['stop']-1]}
                  | {0, 16383, 16384, 1261567, 1261568, 4603903, 4603904, lane.population-1} | set(extras))


def fault_controls(ref, output):
    original = Path(ref['path']); r = read_json(original/'report.json', ref['pin'])
    copied = output/'fault-copy'; copied.mkdir(); s = r['shards'][0]; shard = copied/'shard'; shard.mkdir()
    for name in [*read_json(Path(s['path'])/'report.json')['files'], 'report.json', 'COMMITTED']:
        shutil.copyfile(Path(s['path'])/name, shard/name)
    r = deepcopy(r); r['shards'][0]['path'] = str(shard)
    pin = write(copied/'report.json', r); write(copied/'COMMITTED', dict(report_sha256=pin))
    local = dict(ref, path=str(copied), pin=pin); controls = {}
    with open_lane(local) as lane:
        lane.read([0], lens_id=lane.binding['lens_id'])
        with (shard/'scores.npy').open('r+b') as f:
            f.seek(-1, 2); old = f.read(1); f.seek(-1, 2); f.write(bytes([old[0]^1])); f.flush(); os.fsync(f.fileno())
        controls['changed_payload'] = rejected(lambda: lane.read([0], lens_id=lane.binding['lens_id']))
        with (copied/'COMMITTED').open('ab') as f: f.write(b'\n'); f.flush(); os.fsync(f.fileno())
        controls['changed_seal'] = rejected(lane.fresh)
    partial = output/'incomplete-copy'; partial.mkdir(); r['scope'] = 'qualify'
    pin = write(partial/'report.json', r); write(partial/'COMMITTED', dict(report_sha256=pin))
    controls['incomplete'] = rejected(lambda: open_lane(dict(ref, path=str(partial), pin=pin)))
    unsealed = output/'unsealed-copy'; unsealed.mkdir()
    controls['unsealed'] = rejected(lambda: open_lane(dict(ref, path=str(unsealed))))
    return controls


def run(a):
    began = tick = time.perf_counter(); process = time.process_time(); a.output.mkdir(parents=True, exist_ok=False)
    pins = code(); census = read_json(a.census, a.census_pin); refs = census['graphs']; lanes = {}
    input_pin = '3400b9c5813d56859603a89d5f0fe3182370b94a68e014199b50a2ff605c8d45'
    inputs = read_json(ROOT/'legal-surface-decoder-proof-v1-20261023/report.json', input_pin)
    extras = sorted({v['uid'] for v in inputs['gradients']})
    need(len(refs) == 4 and census['total_edges'] == 254415120, 'PROOF_POPULATION')
    phases = {}; controls = {}; counts = {}; path_count = 0
    try:
        for ref in refs:
            lane = open_lane(ref); lanes[ref['name']] = lane
            need(lane.population == ref['population'] and lane.stats['payload_bytes_authenticated'] == sum(ref['payload_bytes'].values()), 'CENSUS_BINDING')
            print(json.dumps(dict(authenticated=ref['name'], seconds=lane.stats['authentication_seconds'])), flush=True)
        phases['full_authentication'] = time.perf_counter()-tick; tick = time.perf_counter()
        with (a.output/'rows.jsonl').open('x') as f:
            for name, lane in lanes.items():
                ids = selection(lane, extras); count = 0
                # Reverse and duplicate reads retain the caller's requested order.
                ids = ids+list(reversed(ids[:8]))+[ids[0], ids[-1]]
                for lo in range(0, len(ids), 128):
                    for row in lane.read(ids[lo:lo+128], lens_id=lane.binding['lens_id']):
                        f.write(json.dumps(dict(instrument=name, row=row), sort_keys=True)+'\n'); count += 1
                counts[name] = dict(origins=count, edges=10*count)
            f.flush(); os.fsync(f.fileno())
        phases['addressed_rows'] = time.perf_counter()-tick; tick = time.perf_counter()
        seeds = sorted({0, 16383, 16384, 1261567, 1261568, 4603903, 4603904, 6360377} | set(extras))
        walker = TypedPaths(lanes); exclusive = []
        for uid in seeds:
            rows = {n: l.read([uid], lens_id=l.binding['lens_id'])[0]['edges'] for n, l in lanes.items()}
            for n, edges in rows.items():
                peers = {e['target'] for other, values in rows.items() if other != n for e in values}
                exclusive.extend(e for e in edges if e['target'] not in peers)
        need(exclusive, 'NO_SINGLE_LENS_WITNESS')
        with (a.output/'paths.jsonl').open('x') as f:
            for uid in seeds:
                for first in lanes:
                    for second in lanes:
                        if first == second: continue
                        count = 0
                        for path in walker.paths(uid, [first, second]):
                            f.write(json.dumps(path, sort_keys=True)+'\n'); count += 1; path_count += 1
                        need(count == 100, 'PATH_ENUMERATION')
            f.flush(); os.fsync(f.fileno())
        # No ownership of cached reader maps escapes the generator; yielded edges are owned copies.
        generator = walker.paths(0, list(lanes)[:2]); first = next(generator); first['edges'][0]['score'] = 999
        following = next(generator); need(following['edges'][0]['score'] != 999, 'YIELD_ALIAS')
        generator.close(); need(generator.gi_frame is None, 'GENERATOR_NOT_CLOSED')
        write(a.output/'single-lens.json', exclusive)
        phases['paths_and_single_lens'] = time.perf_counter()-tick; tick = time.perf_counter()
        for name, lane in lanes.items():
            for uid in [-1, lane.population, True]:
                controls[name+'/uid/'+str(uid)] = rejected(lambda u=uid: lane.read([u], lens_id=lane.binding['lens_id']))
            controls[name+'/wrong_lens'] = rejected(lambda: lane.read([0], lens_id='0'*64))
            controls[name+'/training'] = rejected(lane.training_authority)
        controls['missing_intermediate'] = rejected(lambda: list(walker.paths(0, [next(iter(lanes)), 'missing-lane'])))
        controls['wrong_pin'] = rejected(lambda: open_lane(dict(refs[0], pin='0'*64)))
        wrong = deepcopy(refs[0]); wrong['binding']['source_pin'] = '0'*64
        controls['wrong_source'] = rejected(lambda: open_lane(wrong))
        controls.update(fault_controls(refs[0], a.output))
        # Batches may span more shards than the cache and repeat earlier origins.
        for n, lane in lanes.items():
            ids = [lane.shards[i % 16]['start']+(i//16) for i in range(128)]
            rows = lane.read(ids, lens_id=lane.binding['lens_id'])
            need([r['origin'] for r in rows] == ids, 'GROUPED_CALLER_ORDER')
            same = lane.read([ids[0], ids[0]], lens_id=lane.binding['lens_id'])
            same[0]['edges'][0]['score'] = 999
            need(same[1]['edges'][0]['score'] != 999, 'DUPLICATE_ROW_ALIAS')
        phases['negative_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
        profiles = []
        for n, lane in lanes.items():
            ids = selection(lane, extras)[:128]
            for _ in range(2):
                start = time.perf_counter(); rows = lane.read(ids, lens_id=lane.binding['lens_id'])
                profiles.append(dict(instrument=n, rows=len(rows), seconds=time.perf_counter()-start, cache_shards=lane.cache_shards))
        phases['resident_profile'] = time.perf_counter()-tick; tick = time.perf_counter()
        stats = {n: dict(l.stats) for n, l in lanes.items()}
        for n, lane in lanes.items():
            maps = [v._mmap for arrays in lane.cache.values() for v in arrays.values()]
            lane.fresh(); lane.close(); need(not lane.cache and all(m.closed for m in maps), 'MAPS_NOT_CLOSED')
            controls[n+'/closed'] = rejected(lambda: lane.read([0], lens_id=lane.binding['lens_id']))
        write(a.output/'results.json', dict(counts=counts, path_count=path_count, seeds=seeds, exclusive_edges=len(exclusive),
            native_input_proof=input_pin, native_input_uids=extras,
            controls=controls, profiles=profiles, stats=stats))
        need(code() == pins and 'torch' not in sys.modules and 'usearch' not in sys.modules, 'UNEXPECTED_RUNTIME')
        for n, h in pins.items():
            p = a.output/'source'/n; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((Path(__file__).resolve().parents[1]/n).read_bytes()); need(sha(p) == h, 'SOURCE_COPY')
        files = {str(p.relative_to(a.output)): sha(p) for p in a.output.rglob('*') if p.is_file()}
        previous = None
        if a.previous is not None:
            old = read_json(a.previous/'report.json', a.previous_pin)
            need(old['complete'] and old['census_pin'] == a.census_pin, 'PREVIOUS_BINDING')
            for n in ['rows.jsonl', 'paths.jsonl', 'single-lens.json']:
                need(sha(a.previous/n) == old['files'][n] == files[n], 'PREVIOUS_OUTPUT_CHANGED')
            previous = dict(path=str(a.previous), pin=a.previous_pin, exact_rows_paths_single_lens=True)
        phases['seal_and_close'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.000001, 'PHASE_COVERAGE')
        pin = write(a.output/'report.json', dict(complete=True, census=str(a.census), census_pin=a.census_pin,
            code=pins, files=files, previous=previous, phases=phases, seconds=seconds, cpu_seconds=time.process_time()-process,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, graph_count=len(lanes),
            all_stored_origins_addressable=True, all_pairs_exhaustive=False, native_scores_recomputed=False,
            new_encoder_calls=0, model_updates=0, scientific_admission=False, implicit_training_authority=False,
            acceptance='author-checked', number_status='measured'))
        print(json.dumps(dict(complete=True, pin=pin, seconds=seconds)), flush=True)
    finally:
        for lane in lanes.values(): lane.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--census', type=Path, required=True); p.add_argument('--census-pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin'); a = p.parse_args()
    need((a.previous is None) == (a.previous_pin is None), 'PREVIOUS_ARGUMENTS')
    try: run(a)
    except BaseException as e:
        if a.output.exists(): write(a.output/'failure.json', dict(error=repr(e)))
        raise
