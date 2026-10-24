"""Necessary exact-context support on retained native P159 readings; never CMI.

PC-04/07/13: four pinned arrays, 122234968 bytes; linear native reads and joins.
No outcomes, distances, rounding, encoder passes or inferred independent units.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
from itertools import groupby
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

NAMES = ('all-mpnet-base-v2', 'finbert-mean')
BASE = Path('/zfs/archive/mnlm/evidence/fsv/12')
SOURCES = {
    'calibration': dict(
        readings=str(BASE/'finance-fixed-20261020/public-v1'),
        reading_pin='712c4306c9cc81ac9f35b2511df85958232a3a830792d97bc7f0d723a84808fb',
        population='/zfs/hot/mnlm/anchors/finance-fixed-20261020-v1',
        population_pin='9c5034106722fbc73dc9bc38add30010b0af536fba80109f0ddf909d774429ae',
        entities=4343, chunks=17547, selected=1512),
    'confirmation': dict(
        readings=str(BASE/'finance-confirm-20261020/readings-v1'),
        reading_pin='d13c87a13fbff6d4db96fed2b632770f142bb0a525b3eaf84a87f45521b205d5',
        population='/zfs/hot/mnlm/anchors/finance-confirm-20261020-v2',
        population_pin='9f6b8e26e6c333a4e999c9fb93d42eb1781b7b30a191360fb80c279bd4131042',
        entities=600, chunks=2296, selected=600),
}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected, receipts):
    raw = path.read_bytes()
    require(sha(raw) == expected, 'NATIVE_CONTEXT_SOURCE_PIN')
    receipts[str(path)] = dict(sha256=expected, bytes=len(raw))
    return raw


def shape(vectors, offsets, chunks):
    require(vectors.ndim == 2 and vectors.shape[1] == 768 and vectors.dtype.str == '<f4'
            and np.isfinite(vectors).all(), 'NATIVE_CONTEXT_VECTOR_SHAPE')
    require(offsets.ndim == 1 and offsets.dtype.kind in 'iu' and len(offsets) == chunks+1
            and offsets[0] == 0 and offsets[-1] == len(vectors)
            and np.all(np.diff(offsets) > 0), 'NATIVE_CONTEXT_WINDOW_ROSTER')


def decode(payload, chunks):
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        require({'vectors', 'offsets'} <= set(arrays.files), 'NATIVE_CONTEXT_SLOT_MISSING')
        vectors, offsets = arrays['vectors'], arrays['offsets']
    shape(vectors, offsets, chunks)
    return vectors, offsets


def header(lens, counts):
    return canonical(dict(lens_id=lens, dtype='<f4', shape=[sum(counts), 768],
                          chunk_windows=counts)) + b'\n'


def fingerprint(vectors, offsets, chunk_ids, lens, independent=False):
    counts = [int(offsets[c+1]-offsets[c]) for c in chunk_ids]
    require(counts and min(counts) > 0, 'NATIVE_CONTEXT_EMPTY_STATE')
    prefix = header(lens, counts)
    if independent:
        indices = np.concatenate([np.arange(int(offsets[c]), int(offsets[c+1])) for c in chunk_ids])
        raw = prefix + np.ascontiguousarray(vectors[indices]).tobytes()
        return sha(raw)
    h = hashlib.sha256(prefix)
    for c in chunk_ids:
        h.update(memoryview(np.ascontiguousarray(vectors[offsets[c]:offsets[c+1]])).cast('B'))
    return h.hexdigest()


def population(tag, spec, independent=False):
    receipts = {}
    root, source = Path(spec['readings']), Path(spec['population'])
    readings = json.loads(pinned(root/'report.json', spec['reading_pin'], receipts))
    pop = json.loads(pinned(source/'report.json', spec['population_pin'], receipts))
    require(readings['complete'] and not readings['errors']
            and readings['calibration_report_sha256'] == spec['population_pin'], 'NATIVE_CONTEXT_BINDING')
    summary = json.loads(pinned(source/'finance/summary.json',
                               pop['families']['finance']['summary_sha256'], receipts))
    require(summary['entities'] == spec['entities'] and summary['chunks'] == spec['chunks']
            and readings['inputs'] == spec['chunks'], 'NATIVE_CONTEXT_EXTENT')
    # Only the retained row/partition roster is consumed; no label or class enters a state.
    anchor = json.loads(pinned(source/'finance/anchors.json', summary['anchors_sha256'], receipts))['P159']
    selected, partitions = anchor['rows'], anchor['partitions']
    require(len(selected) == len(partitions) == spec['selected']
            and len(set(selected)) == len(selected), 'NATIVE_CONTEXT_SELECTION')
    raw = pinned(source/'finance/chunks/atoms.parquet', summary['chunks_sha256'], receipts)
    rows = pq.read_table(io.BytesIO(raw), columns=['row'])['row'].to_pylist()
    require(len(rows) == spec['chunks'] and set(rows) == set(range(spec['entities'])), 'NATIVE_CONTEXT_CHUNKS')
    by_row = defaultdict(list)
    for index, row in enumerate(rows):
        by_row[row].append(index)
    records = [dict(source=tag, row=row, partition=partition, slots={})
               for row, partition in zip(selected, partitions, strict=True)]
    identities, native_bytes = {}, 0
    probe = None
    for name in NAMES:
        binding = readings['lenses'][name]
        report = json.loads(pinned(root/name/'report.json', binding['report_sha256'], receipts))
        require(report['lens_id'] == binding['lens_id'] and report['inputs'] == spec['chunks']
                and report['weights_and_preprocessing_unchanged'] and report['determinism_probe_equal'],
                'NATIVE_CONTEXT_INSTRUMENT')
        identities[name] = report['lens_id']
        payload = pinned(root/name/'slots.npz', report['files']['slots.npz']['sha256'], receipts)
        require(len(payload) == report['files']['slots.npz']['bytes'], 'NATIVE_CONTEXT_PAYLOAD')
        native_bytes += len(payload)
        vectors, offsets = decode(payload, spec['chunks'])
        for record in records:
            record['slots'][name] = fingerprint(vectors, offsets, by_row[record['row']],
                                                identities[name], independent)
        if probe is None:
            row = next(r for r in selected if len(by_row[r]) > 1)
            ids = by_row[row]
            probe = dict(full=fingerprint(vectors, offsets, ids, identities[name]),
                         dropped=fingerprint(vectors, offsets, ids[:-1], identities[name]),
                         reversed=fingerprint(vectors, offsets, ids[::-1], identities[name]),
                         shape=list(vectors.shape))
        del vectors, offsets, payload
    return records, receipts, identities, native_bytes, probe


def census(records, independent=False):
    groups = {'all': records}
    for r in records:
        groups.setdefault(r['source']+'/'+r['partition'], []).append(r)
    result = {}
    for group, selected in sorted(groups.items()):
        for names in ((NAMES[0],), (NAMES[1],), NAMES):
            keys = [tuple(r['slots'][n] for n in names) for r in selected]
            counts = ([sum(1 for _ in values) for _, values in groupby(sorted(keys))]
                      if independent else list(Counter(keys).values()))
            result[group+'|'+','.join(names)] = dict(observations=len(keys), hash_buckets=len(counts),
                largest_bucket=max(counts), potential_supported_rows_upper=sum(n for n in counts if n >= 100),
                no_certificate_from_exact_context_bound=max(counts) < 100)
    design = {tuple(r['slots'][n] for n in NAMES) for r in records
              if r['source'] == 'calibration' and r['partition'] == 'design'}
    matches = Counter()
    for r in records:
        group = r['source']+'/'+r['partition']
        matches.setdefault(group, 0)
        matches[group] += tuple(r['slots'][n] for n in NAMES) in design
    return dict(groups=result, original_design_context_matches=dict(matches))


def validate_census(saved, records):
    require(saved == census(records, independent=True), 'NATIVE_CONTEXT_RECOUNT')


def controls(records, probes, receipts):
    checks = []
    def rejects(name, code, fn):
        try:
            fn()
        except ValueError as exc:
            require(str(exc) == code, 'NATIVE_CONTEXT_CONTROL_CODE')
        else:
            raise AssertionError('NATIVE_CONTEXT_CONTROL_ACCEPTED')
        checks.append(dict(name=name, code=code))
    first = Path(next(iter(receipts)))
    rejects('wrong_source_pin', 'NATIVE_CONTEXT_SOURCE_PIN', lambda: pinned(first, '0'*64, {}))
    missing, malformed = io.BytesIO(), io.BytesIO()
    np.savez(missing, offsets=np.array([0, 1]))
    np.savez(malformed, vectors=np.zeros((1, 767), np.float32), offsets=np.array([0, 1]))
    rejects('missing_native_slot', 'NATIVE_CONTEXT_SLOT_MISSING', lambda: decode(missing.getvalue(), 1))
    rejects('wrong_native_shape', 'NATIVE_CONTEXT_VECTOR_SHAPE',
            lambda: decode(malformed.getvalue(), 1))
    for tag, probe in probes.items():
        require(probe['full'] != probe['dropped'] and probe['full'] != probe['reversed'],
                'NATIVE_CONTEXT_LOSSLESS_CONTROL')
        checks.append(dict(name=tag+'_dropped_reversed_native_windows', detected=True))
    sample = records[0]
    duplicate = census([sample, sample])['groups']['all|'+','.join(NAMES)]
    constant = census([sample]*100)['groups']['all|'+','.join(NAMES)]
    require(duplicate['largest_bucket'] == 2 and duplicate['potential_supported_rows_upper'] == 0
            and constant['largest_bucket'] == 100 and constant['potential_supported_rows_upper'] == 100,
            'NATIVE_CONTEXT_DUPLICATE_CONTROL')
    checks.append(dict(name='duplicate_and_constant_states', diagnostic_only=True,
                       independence_established=False, admission=False))
    require(census(records) == census(records[::-1], True), 'NATIVE_CONTEXT_ORDER_CONTROL')
    forged = census(records)
    next(iter(forged['groups'].values()))['potential_supported_rows_upper'] += 100
    rejects('forged_support_count', 'NATIVE_CONTEXT_RECOUNT',
            lambda: validate_census(forged, records))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--readback', action='store_true')
    args = ap.parse_args(); start = time.perf_counter()
    root = args.output
    if not args.readback:
        root.mkdir(parents=True, exist_ok=False)
        (root/'intent.json').write_bytes(canonical(dict(sources=SOURCES, instruments=NAMES,
            candidate='specter2-base', question='P159', outcome_values_used=False,
            projected_native_bytes=122234968, production_admission=False)))
    else:
        require(json.loads((root/'intent.json').read_bytes())['sources'] == SOURCES, 'NATIVE_CONTEXT_INTENT')
    records, receipts, identities, probes, native_bytes = [], {}, None, {}, 0
    for tag, spec in SOURCES.items():
        current, source, ids, size, probe = population(tag, spec, args.readback)
        require(identities is None or identities == ids, 'NATIVE_CONTEXT_IDENTITY_CHANGE')
        identities = ids; records.extend(current); receipts.update(source); probes[tag] = probe; native_bytes += size
    loaded = time.perf_counter()
    require(native_bytes == 122234968 and len(records) == 2112, 'NATIVE_CONTEXT_PREFLIGHT')
    result = census(records, args.readback)
    checks = controls(records, probes, receipts)
    for name, item in receipts.items():
        require(sha(Path(name).read_bytes()) == item['sha256'], 'NATIVE_CONTEXT_SOURCE_CHANGED')
    report = dict(schema='native-p159-context-support-v1', number_status='measured', acceptance='author-checked',
        observations=len(records), native_bytes=native_bytes, sources=receipts, lens_ids=identities,
        records=records, census=result, controls=checks, probes=probes, source_unchanged=True,
        iid_established=False, conditional_information_measured=False, scientific_admission=False, served=False,
        scope='Necessary support only for this exact-context method; a zero lower bound is not zero true CMI.')
    if args.readback:
        previous = json.loads((root/'report.json').read_bytes())
        validate_census(previous['census'], records)
        require(previous == report, 'NATIVE_CONTEXT_COLD_REPLAY')
        report = dict(report_sha256=sha((root/'report.json').read_bytes()), records_rebuilt=len(records),
                      all_fields_equal=True, sources_rehashed=len(receipts), controls=checks,
                      acceptance='author-checked', independent_assembly_and_group_count=True)
        target = root/'readback.json'
    else:
        target = root/'report.json'
    with target.open('xb') as f:
        f.write(canonical(report))
    timing = dict(input_seconds=loaded-start, recount_controls_source_rehash_seconds=time.perf_counter()-loaded,
                  wall_seconds=time.perf_counter()-start, code_sha256=sha(Path(__file__).read_bytes()))
    with (root/('readback-timing.json' if args.readback else 'timing.json')).open('xb') as f:
        f.write(canonical(timing))
    print(json.dumps(dict(path=str(target), sha256=sha(target.read_bytes()), census=result, timing=timing)))


if __name__ == '__main__':
    main()
