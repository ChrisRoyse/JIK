"""Exact canonical-cache reuse for cold-replayed generated surface atoms."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import numpy as np
from claims.process_conditions import write_committed as write
from text.answer_native_plan import authority, read
from text.constellation_corpus import canonical, digest, file_sha, need
from text.legal_native_batch import NativeBatchSession
from text.legal_reuse_input import ReuseInputs

SCHEMA = 'free-guard-canonical-inputs-v1'
NEW = ('text/free_guard_native_plan.py', 'text/free_guard_native_measure.py', 'audit/free_guard_native_readback.py')
BASE = Path(__file__).resolve().parents[1]


def code():
    return {name: file_sha(BASE / name) for name in (*NEW, 'text/answer_native_measure.py',
        'text/canonical_windows.py', 'model/capability_surface_guard.py', 'audit/legal_native_readback.py')}


def build(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False); source = None
    try:
        parent = read(a.source / 'report.json', a.source_pin)
        cold = read(a.readback / 'report.json', a.readback_pin)
        need(parent['schema'] == 'free-surface-guard-input-plan-v1' and parent['complete']
             and cold['complete'] and cold['input'] == dict(path=str(a.source), pin=a.source_pin)
             and cold['all_raw_spans_exact'] and cold['all_native_joins_exact']
             and cold['all_singleton_token_counts_exact'] and cold['counts'] == parent['counts']
             and parent['measurements_pending'] and not parent['serving'], 'FREE_NATIVE_SOURCE_SCOPE')
        for name, h in cold['code'].items():
            need(file_sha(BASE / name) == h, 'FREE_NATIVE_SOURCE_CODE')
        for name, filename in (('inputs', 'inputs.jsonl'), ('occurrences', 'occurrences.jsonl'), ('refused', 'refused.jsonl')):
            item = parent['files'][name]; path = a.source / filename
            need(path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256'], 'FREE_NATIVE_INPUT_FILE')
        records = [json.loads(line) for line in (a.source / 'inputs.jsonl').read_bytes().splitlines()]
        tokens = read(a.source / 'tokens.json', parent['tokens'])
        identity = read(a.source / 'contracts.json', parent['contracts'])
        bank, specs, contracts = authority(parent['native_bank'])
        need(identity == dict(contracts=contracts, instruments={s.name: s.identity for s in specs},
            lens_ids={n: v['lens_id'] for n, v in bank.config['instruments'].items()}), 'FREE_NATIVE_INSTRUMENTS')
        phases = dict(source_and_instrument_authority=time.perf_counter() - tick); tick = time.perf_counter()
        source = ReuseInputs(bank.config['source']['path'], bank.config['source']['pin'])
        reader = NativeBatchSession(bank.root, bank.pin, source)
        keys = np.frombuffer(b''.join(bytes.fromhex(row['id']) for row in records), dtype='V32')
        positions = np.searchsorted(source.arrays['hashes'], keys); reused = []
        for i, uid in enumerate(positions):
            if uid < len(source.arrays['hashes']) and source.arrays['hashes'][uid] == keys[i]:
                need(source.input(int(uid))['text'] == records[i]['text'], 'FREE_NATIVE_REUSE_BYTE_EQUALITY')
                reused.append(dict(index=i, uid=int(uid), id=records[i]['id']))
        reused_indices = {r['index'] for r in reused}; missing = [i for i in range(len(records)) if i not in reused_indices]
        phases['authenticate_cache_and_index_lookup'] = time.perf_counter() - tick; tick = time.perf_counter()
        retained = {}; files = []
        for name in bank.config['roster']:
            if not reused:
                continue
            batch = reader.read_many(name, [r['uid'] for r in reused]); traces = [json.loads(t) for t in batch.traces]
            need(batch.lens_id == identity['lens_ids'][name]
                 and [len(t['windows']) for t in traces] == [tokens[name]['windows'][r['index']] for r in reused],
                 'FREE_NATIVE_REUSE_WINDOWS')
            dest = a.output / name; dest.mkdir()
            np.savez(dest / 'reused.npz', vectors=batch.values, offsets=batch.offsets)
            write(dest / 'traces.json', traces); files.extend((name + '/reused.npz', name + '/traces.json'))
            retained[name] = dict(lens_id=batch.lens_id, windows=len(batch.values),
                shards=[dict(path=str(d['root']), pin=d['pin']) for (n, _), d in reader.descriptors.items() if n == name])
        phases['copy_exact_reused_native'] = time.perf_counter() - tick; tick = time.perf_counter()
        write(a.output / 'inputs.json', records); write(a.output / 'tokens.json', tokens)
        files.extend(('inputs.json', 'tokens.json')); source.fresh(); reader.fresh()
        for d in reader.descriptors.values():
            reader.descriptor_fresh(d)
        phases['publish'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
        need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_NATIVE_PLAN_PHASES')
        r = dict(schema=SCHEMA, complete=True, source=dict(path=str(a.source), pin=a.source_pin),
            source_readback=dict(path=str(a.readback), pin=a.readback_pin), source_counts=parent['counts'],
            canonical_bank=parent['native_bank'], specs=[asdict(s) for s in specs], contracts=contracts,
            roster=bank.config['roster'], reused=reused, missing=missing, reuse_native=retained,
            inputs=len(records), input_bytes=sum(len(v['text'].encode()) for v in records),
            missing_bytes=sum(len(records[i]['text'].encode()) for i in missing),
            missing_windows={n: sum(tokens[n]['windows'][i] for i in missing) for n in tokens},
            missing_tokens={n: sum(tokens[n]['active_tokens'][i] for i in missing) for n in tokens},
            cache_counts=source.report['counts'], cache_authenticated_bytes=sum(v['bytes'] for v in source.report['files'].values()),
            native_reader_stats=reader.stats, source_inputs_conserved=True, no_reference_answer=True,
            training_ready=False, scientific_admission=False, serving=False, encoder_calls=0,
            files={n: dict(bytes=(a.output / n).stat().st_size, sha256=file_sha(a.output / n)) for n in files},
            code=code(), phases=phases, seconds=seconds, acceptance='author-checked')
        pin = write(a.output / 'report.json', r)
        print(json.dumps(dict(complete=True, pin=pin, inputs=len(records), reused=len(reused), missing=len(missing),
            cache_counts=r['cache_counts'], cache_authenticated_bytes=r['cache_authenticated_bytes'],
            missing_windows=r['missing_windows'], seconds=seconds)), flush=True)
    finally:
        if source is not None:
            source.table.close()
            for value in source.arrays.values():
                value._mmap.close()


class FreeGuardNativePlan:
    """Adapt the plan to the existing canonical lane; no measurement algorithm fork."""
    def __init__(self, root, pin):
        self.root = Path(root); self.pin = pin; self.report = r = read(self.root / 'report.json', pin)
        need(r['schema'] == SCHEMA and r['complete'] and r['code'] == code()
             and not r['training_ready'] and not r['scientific_admission'] and not r['serving'], 'FREE_NATIVE_PLAN_SCOPE')
        for name, item in r['files'].items():
            p = self.root / name
            need(p.stat().st_size == item['bytes'] and file_sha(p) == item['sha256'], 'FREE_NATIVE_PLAN_FILE')
        self.records = json.loads((self.root / 'inputs.json').read_bytes())
        self.tokens = json.loads((self.root / 'tokens.json').read_bytes())
        source_root = Path(r['source']['path']); source = read(source_root / 'report.json', r['source']['pin'])
        cold = read(Path(r['source_readback']['path']) / 'report.json', r['source_readback']['pin'])
        need(cold['complete'] and cold['input'] == r['source'] and cold['all_raw_spans_exact']
             and cold['all_native_joins_exact'] and cold['all_singleton_token_counts_exact']
             and r['source_counts'] == cold['counts'] == source['counts'], 'FREE_NATIVE_PLAN_SOURCE_COUNTS')
        need(file_sha(source_root / 'inputs.jsonl') == source['files']['inputs']['sha256'], 'FREE_NATIVE_PLAN_SOURCE_FILE')
        expected = [json.loads(line) for line in (source_root / 'inputs.jsonl').read_bytes().splitlines()]
        need(self.records == expected and self.tokens == read(source_root / 'tokens.json', source['tokens']),
             'FREE_NATIVE_PLAN_SOURCE_INPUTS')
        need(len(self.records) == r['inputs'] and [v['id'] for v in self.records] == sorted({v['id'] for v in self.records})
             and all(digest(v['text'].encode()) == v['id'] for v in self.records), 'FREE_NATIVE_PLAN_TEXT')
        reused = [v['index'] for v in r['reused']]
        need(sorted(reused + r['missing']) == list(range(len(self.records))), 'FREE_NATIVE_PLAN_PARTITION')
        bank, specs, contracts = authority(r['canonical_bank'])
        need(r['specs'] == [asdict(s) for s in specs] and r['contracts'] == contracts
             and r['roster'] == bank.config['roster'], 'FREE_NATIVE_PLAN_IDENTITIES')
        for name in r['roster']:
            counts = self.tokens[name]
            need(len(counts['windows']) == len(self.records) and min(counts['windows']) > 0
                 and sum(counts['windows']) == counts['total_windows'], 'FREE_NATIVE_PLAN_WINDOWS')
        self.seal = digest(canonical(dict(records=self.records, tokens=self.tokens, report=r)))

    def fresh(self):
        need(file_sha(self.root / 'report.json') == self.pin and digest(canonical(dict(records=self.records,
            tokens=self.tokens, report=self.report))) == self.seal, 'FREE_NATIVE_PLAN_CHANGED')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source', 'readback', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    for n in ('source-pin', 'readback-pin'):
        p.add_argument('--' + n, required=True)
    a = p.parse_args()
    try:
        build(a)
    except BaseException as exc:
        if a.output.exists():
            write(a.output / 'failure.json', dict(error=repr(exc)))
        raise
