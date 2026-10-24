"""Separate byte/reference reader for sense_variation; imports no MNLM modules."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time

import numpy as np


def canonical(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def vector(features):
    a = np.zeros(128, np.float32)
    for name, v in features:
        h = hashlib.sha256(name.encode()).digest()
        j = sum(int(h[i])*256**i for i in range(4)) % 128
        a[j] = np.float32(a[j] + np.float32(v if h[4] % 2 else -v))
    return a


def scalar(f):
    k, v = f['type'], f['value']
    if k in ('text', 'reference'):
        return []
    key = '|'.join(str(f[x]) for x in ('name', 'type', 'unit', 'clock_role'))
    if k in ('integer', 'number'):
        magnitude = math.log(abs(v)+1) if type(v) is int else math.log1p(abs(v))
        return [(key, -magnitude if v < 0 else magnitude), (key+'|observed', 1)]
    assert k != 'clock', 'this separate proof scope contains no clock values'
    return [(key+'|'+str(v), 1)]


def reference(items, channel):
    for item in items:
        if channel == 'surface':
            for word in re.findall(r'\w+|[^\w\s]', item['text']):
                yield vector([(word.casefold(), 1)])
            continue
        if channel in ('history', 'co_reference'):
            raise AssertionError('context requires independent join proof')
        if channel in ('fields', 'structure', 'provenance'):
            fields, feature = [item], []
        elif channel == 'relations':
            fields = item['qualifiers']
            feature = [('relation|'+item['rel_type']+'|'+item['direction']+'|'+item['polarity'], 1)]
        else:
            fields = item['measurements']
            feature = [('event|'+item['mode']+'|'+item['event'], 1)] if channel == 'behaviour' else []
        for f in fields:
            feature.extend(scalar(f))
        if feature:
            yield vector(feature)
        for f in fields:
            if f['type'] == 'text':
                key = '|'.join(str(f[x]) for x in ('name', 'type', 'unit', 'clock_role'))
                for word in re.findall(r'\w+|[^\w\s]', f['value']):
                    yield vector([(key, 1), (key+'|token|'+word.casefold(), 1)])


def run(a):
    started = time.perf_counter()
    report = json.loads((a.run/'report.json').read_bytes())
    manifest = json.loads((a.population/'manifest.json').read_bytes())
    checks = []
    def check(name, ok, **details):
        checks.append(dict(name=name, passed=bool(ok), **details))
    check('parent_manifest_and_database_bound', sha(a.population/'manifest.json') == report['source_manifest_sha256']
          and sha(a.population/'native.sqlite') == report['source_database_sha256'] == manifest['database']['sha256'])
    inventory = json.loads((a.run/'SHA256.json').read_bytes())
    check('all_persisted_receipts_unchanged', all(sha(a.run/name) == h for name,h in inventory.items()), files=len(inventory))
    receipts = {(r['channel'], r['payload_sha256']):r for r in
                map(json.loads, (a.run/'payload-measurements.jsonl').read_bytes().splitlines())}
    db = sqlite3.connect((a.population/'native.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    try:
        source_counts = dict(db.execute('SELECT source,count(*) FROM observations GROUP BY source'))
        unit_counts = dict(db.execute('SELECT source,count(distinct unit) FROM observations GROUP BY source'))
        check('source_and_unit_denominators', all(p['observations'] == source_counts[s] and p['units'] == unit_counts[s]
                                                 for s,p in report['profiles'].items()))
        # Independently recover referent/unit membership from retained headers.
        refs = collections.defaultdict(set)
        source_of = {}
        for atom, source, unit, raw in db.execute('SELECT atom,source,unit,header FROM observations'):
            source_of[atom] = source
            for r in json.loads(raw)['referents']:
                refs[r['key']].add((source,unit))
        multiplicities = collections.defaultdict(collections.Counter)
        for atom, channel, ph in db.execute('SELECT atom,channel,digest FROM channels'):
            multiplicities[channel,ph][source_of[atom]] += 1
        check('same_unit_context_exclusion_is_real', len(refs) == report['context_proof']['referents']
              and all(len({u for _,u in rows}) == 1 for rows in refs.values()))
        measured, counts, first, variable, present = 0, collections.defaultdict(collections.Counter), {}, set(), collections.Counter()
        for channel, ph, raw in db.execute('SELECT channel,digest,body FROM payloads ORDER BY channel,digest'):
            saved = receipts[channel,ph]
            assert hashlib.sha256(raw).hexdigest() == ph
            values = np.asarray(list(reference(json.loads(raw)['items'],channel)),np.float32).reshape(-1,128)
            state = hashlib.sha256(canonical(dict(absent=not len(values),shape=list(values.shape),dtype='<f4'))
                                   + values.astype('<f4').tobytes()).hexdigest()
            assert saved['state_sha256'] == state and saved['shape'] == list(values.shape)
            actual = sorted(multiplicities[channel,ph].items())
            assert actual == [tuple(x) for x in saved['source_multiplicities']]
            for source,n in actual:
                key = source,channel
                counts[key][state] += n
                present[key] += n if len(values) else 0
                if key not in variable:
                    if key not in first:
                        first[key] = values.copy()
                    elif not np.array_equal(values,first[key]):
                        variable.add(key)
                        del first[key]
            measured += 1
        check('every_payload_independently_remeasured', measured == report['payloads'] == len(receipts), payloads=measured)
        exact = True
        for source,profile in report['profiles'].items():
            for c,detail in profile['channels'].items():
                key = source,c
                histogram = counts[key]
                n = sum(histogram.values())
                entropy = -sum(v/n*math.log2(v/n) for v in histogram.values())
                exact &= (detail['states'] == len(histogram) and detail['exact_constant'] == (key not in variable)
                          and detail['nonempty_observations'] == present[key]
                          and abs(entropy-detail['empirical_state_entropy_bits']) < 1e-10)
        check('every_profile_recounted', exact)
        check('no_sense_or_stage_admission', report['admission'] is False
              and all(p['admission'] is False for p in report['profiles'].values()))
        # Explicit known-truth controls of the separate reader, outside source metrics.
        baseline = dict(rel_type='member', direction='out', polarity='positive', qualifiers=[],
                        target_referent='entity-one', source='provider-one', time=None)
        changed = {**baseline, 'target_referent':'entity-two', 'source':'provider-two'}
        check('lineage_changes_do_not_create_measurement_variation',
              np.array_equal(list(reference([baseline],'relations')),list(reference([changed],'relations'))))
        check('typed_relationship_change_is_visible', not np.array_equal(list(reference([baseline],'relations')),
              list(reference([{**baseline,'polarity':'negative'}],'relations'))))
        check('token_order_is_not_pooled', not np.array_equal(list(reference([{'text':'one two'}],'surface')),
              list(reference([{'text':'two one'}],'surface'))))
    finally:
        db.close()
    result = dict(acceptance='author-checked only; separate reader process', passed=sum(c['passed'] for c in checks),
                  failed=sum(not c['passed'] for c in checks),checks=checks, audit_manifest_sha256=sha(a.run/'SHA256.json'),
                  readback_code_sha256=sha(Path(__file__)),seconds=time.perf_counter()-started)
    a.output.write_bytes(canonical(result))
    print(json.dumps(result))
    return int(result['failed'] > 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--population', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    raise SystemExit(run(p.parse_args()))
