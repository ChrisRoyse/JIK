"""Typed text FSV over retained native payloads; no inference/admission claim."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / 'lens'),
                str(Path(__file__).resolve().parents[1] / 'canon')]
from native_population import Population, file_hash
from senses.contract import CHANNELS, SenseError, canonical, digest, field, present, absent, seal_record
from senses.instruments import contracts, measure, tokens, coverage
from senses.store import Store
from senses.views import masked
from universal import measure_universal, specs
from lenscommon import LensError
import slot_binding


def vector(features):
    result = [0.0] * 128
    for name, value in features:
        raw = hashlib.sha256(name.encode()).digest()
        index = sum(raw[i] * 256**i for i in range(4)) % 128
        result[index] = float(np.float32(result[index]) + np.float32(value if raw[4] % 2 else -value))
    return np.asarray(result, np.float32)


def real_reference(item):
    """Independent reference for the actual retained tool-response schema."""
    fields = item['measurements']
    assert {f['name'] for f in fields} == {'tool_response', 'response_characters'}
    text = next(f for f in fields if f['type'] == 'text')
    count = next(f for f in fields if f['type'] == 'integer')
    key = '|'.join(str(count[x]) for x in ('name', 'type', 'unit', 'clock_role'))
    features = [('event|'+item['mode']+'|'+item['event'], 1),
                (key, math.log(abs(count['value'])+1)), (key+'|observed', 1)]
    yield vector(features)
    prefix = '|'.join(str(text[x]) for x in ('name', 'type', 'unit', 'clock_role'))
    for word in re.findall(r'\w+|[^\w\s]', text['value']):
        yield vector([(prefix, 1), (prefix+'|token|'+word.casefold(), 1)])


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    checks, phases = [], {}
    def check(name, ok, **details):
        checks.append(dict(name=name, passed=bool(ok), **details))
    def refuses(name, call, code):
        error = None
        try: call()
        except (SenseError, LensError) as exc: error = exc.code
        check(name, error == code, code=error)
    pin = file_hash(a.population / 'manifest.json')
    before = file_hash(a.population / 'native.sqlite')
    pop = Population(a.population, pin)
    old = json.loads(a.old_contracts.read_bytes())
    now = contracts()
    check('new_frozen_identities', all(x['lens_id'] != y['lens_id'] for x, y in zip(old, now))
          and [x['channel'] for x in old] == [x['channel'] for x in now])
    (a.output / 'contracts.json').write_bytes(canonical(now))
    t = time.perf_counter()
    rows, total_words = [], 0
    with (a.output / 'payloads.jsonl').open('xb') as report:
        for ph, body in pop.db.execute("SELECT digest,body FROM payloads WHERE channel='behaviour' ORDER BY digest"):
            items = json.loads(body)['items']
            assert hashlib.sha256(body).hexdigest() == ph
            got = tokens(items, 'behaviour')
            expected = np.asarray([v for item in items for v in real_reference(item)], np.float32).reshape(-1, 128)
            assert np.array_equal(got, expected), ph
            texts = [f['value'] for item in items for f in item['measurements'] if f['type'] == 'text']
            words = sum(len(re.findall(r'\w+|[^\w\s]', text)) for text in texts)
            total_words += words
            receipt = dict(payload_sha256=ph, bytes=sum(len(x.encode()) for x in texts), words=words,
                           shape=list(got.shape), token_bytes_sha256=hashlib.sha256(got.tobytes()).hexdigest(),
                           independent_equal=True)
            report.write(canonical(receipt)+b'\n')
            if texts: rows.append((receipt['bytes'], ph))
    check('complete_retained_text_population', len(rows) == 4014 and total_words == 1418786,
          fields=len(rows), lexical_rows=total_words)
    phases['all_payload_measurement_and_independent_readback'] = time.perf_counter()-t
    t = time.perf_counter()
    ranked = sorted(rows)
    chosen = {ranked[int(i)][1] for i in np.linspace(0, len(ranked)-1, 8)}
    selected, selected_payloads, missing = [], set(), False
    for record in pop.records():
        ch = record['channels']['behaviour']
        ph = hashlib.sha256(canonical(ch)).hexdigest()
        if (ph in chosen and ph not in selected_payloads) or (ch['state'] == 'absent' and not missing):
            selected.append(record); selected_payloads.add(ph)
            missing |= ch['state'] == 'absent'
    check('bounded_scope_includes_longest_and_absence', chosen <= selected_payloads and missing and len(selected) <= 9)
    source = a.output / 'observations.sqlite'
    store = Store(source, create=True, records_only=True)
    store.ingest(selected, {'input_sha256': pin, 'count': len(selected), 'kind': 'typed-text proof only'}, None)
    store.close()
    family = 'typed-text-boundary'
    policy = dict(schema='ucr-view-v1', question=family, allow_channels=['behaviour'], deny_fields=[],
                  deny_relations=[], blocked_units=[], label_aliases=[], rationale='Engineering token fidelity; no outcome labels or admission.')
    policies = a.output / 'policies.json'; policies.write_bytes(canonical({family: policy}))
    cal = a.output / 'cal'; cal.mkdir()
    calrows = [{**r['locator'], 'text': '\n'.join(x['text'] for x in r['channels']['surface']['items'])} for r in selected]
    (cal / (family+'.jsonl')).write_bytes(b''.join(canonical(r)+b'\n' for r in calrows))
    spec = next(s for s in specs() if s['channel'] == 'behaviour')
    entry = measure_universal(spec, {family: calrows}, a.output/'slots', {}, source, policies)
    path = Path(entry['families'][family]['slot'])
    with np.load(path, allow_pickle=False) as saved:
        equal = True
        for i, record in enumerate(selected):
            raw = measure(record, 'behaviour')['tokens']
            start, end = saved['offsets'][i:i+2]
            equal &= np.array_equal(raw, saved['tokens'][start:end]) and bool(saved['absent'][i]) == (len(raw)==0)
        check('persisted_multi_rows_match_original_observations', equal, observations=len(selected), rows=int(saved['offsets'][-1]))
    (a.output / 'registry.json').write_bytes(canonical(entry))
    old_spec = next(s for s in old if s['channel'] == 'behaviour')
    refuses('old_weight_pin_refuses', lambda: measure_universal(spec, {family: calrows}, a.output/'bad-slots',
            {spec['name']: old_spec['weights_sha256']}, source, policies), 'MNLM_LENS_FROZEN_VIOLATION')
    if a.old_store:
        old_hash = file_hash(a.old_store)
        refuses('old_measured_store_refuses', lambda: Store(a.old_store), 'MNLM_UCR_PANEL_DRIFT')
        check('old_measured_store_unchanged', file_hash(a.old_store) == old_hash)
    original = path.read_bytes()
    # Corrupt a native value explicitly, preserving a readable NPZ container.
    with np.load(path, allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
    arrays['tokens'][0, 0] += 1
    np.savez(path, **arrays)
    refuses('changed_native_tokens_refuse', lambda: slot_binding.reuse(path, entry['contract']), 'MNLM_SLOT_STALE')
    path.write_bytes(original)
    check('restored_native_slot_reuses', slot_binding.reuse(path, entry['contract']) is not None)
    phases['native_reader_and_slot_publication'] = time.perf_counter()-t
    t = time.perf_counter()
    # Deliberate controls are isolated from the real source population/metrics.
    f = field('observed_text', 'alpha beta alpha', 'text')
    base = tokens([f], 'fields')
    changed = tokens([{**f, 'value': 'alpha gamma alpha'}], 'fields')
    check('word_change_preserves_unaffected_rows', base.shape == (3,128) and np.array_equal(base[[0,2]], changed[[0,2]]) and not np.array_equal(base[1], changed[1]))
    check('repetition_preserved', np.array_equal(base[0], base[2]))
    reverse = tokens([{**f, 'value': 'beta alpha alpha'}], 'fields')
    check('order_preserved', np.array_equal(base[[1,0,2]], reverse) and not np.array_equal(base, reverse))
    check('reference_never_tokenized', len(tokens([field('reference', 'alpha beta', 'reference')], 'fields')) == 0)
    check('category_not_tokenized', len(tokens([field('category', 'alpha beta', 'category')], 'fields')) == 1)
    check('explicit_number_not_tokenized', len(tokens([field('n', 12345, 'integer')], 'fields')) == 1)
    check('empty_text_has_no_fake_token', tokens([{**f, 'value': ' \n'}], 'fields').shape == (0,128))
    check('unicode_lexical_rows', len(tokens([{**f, 'value': 'naïve 骨 🙂'}], 'fields')) == 3)
    template = next(r for r in selected if r['channels']['behaviour']['state'] == 'present')
    modified = copy.deepcopy(template)
    modified['channels']['behaviour']['items'][0]['measurements'] = [field('hidden', 'do not learn secret', 'text')]
    seal_record(modified)
    hidden = masked(modified, {**policy, 'deny_fields':['hidden']})
    check('denied_text_not_measured', measure(hidden, 'behaviour')['tokens'].shape == (1,128))
    modified['channels']['behaviour'] = absent('no_recorded_response'); seal_record(modified)
    check('surface_not_copied_into_behaviour', measure(modified, 'behaviour')['kind'] == 'Absent')
    cloned = copy.deepcopy(template); cloned['locator']['source'] = 'known-truth-adapter-control'; seal_record(cloned)
    check('no_source_branch', np.array_equal(measure(cloned,'behaviour')['tokens'], measure(template,'behaviour')['tokens']))
    for channel in ('fields','structure','provenance','relations','perception','behaviour'):
        if channel in ('fields','structure','provenance'): items=[f]
        elif channel=='relations': items=[dict(rel_type='observed',direction='out',polarity='positive',target_referent='control',qualifiers=[f],time=None,source='control')]
        elif channel=='perception': items=[dict(modality='sensor',sha256='0'*64,media_type='control',measurements=[f])]
        else: items=[dict(event='observed',mode='observed',evidence='control',measurements=[f])]
        measured=tokens(items,channel)
        check('typed_text_generic_'+channel, np.array_equal(measured[-3:],base))
    slot=measure(template,'behaviour')
    check('no_new_sense_or_admission', len(now)==len(CHANNELS)==9 and not coverage({spec['name']:slot})['d4'])
    check('original_population_unchanged', file_hash(a.population/'native.sqlite')==before)
    pop.close()
    phases['negative_controls'] = time.perf_counter()-t
    elapsed=time.perf_counter()-started
    report=dict(acceptance='author-checked only',scientific_admission=False,checks=checks,
                passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),
                fields=len(rows),lexical_rows=total_words,seconds=elapsed,phases=phases,
                phase_coverage=sum(phases.values())/elapsed,parent_manifest_sha256=pin,
                source_sha256=before,instrument_version=now[0]['contract'])
    (a.output/'report.json').write_bytes(canonical(report))
    (a.output/'SHA256.json').write_bytes(canonical({p.relative_to(a.output).as_posix():file_hash(p)
        for p in a.output.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))}))
    print(json.dumps({k:report[k] for k in ('passed','failed','fields','lexical_rows','seconds','phase_coverage')}))
    return int(report['failed']>0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population',required=True,type=Path)
    parser.add_argument('--old-contracts',required=True,type=Path)
    parser.add_argument('--old-store',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    sys.exit(run(parser.parse_args()))
