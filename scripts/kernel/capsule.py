"""Create-once slice of pinned native readings and attested corpus containment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from kernel.native_route import NativeLane, check
from senses.contract import canonical

LANES = ('all-mpnet-base-v2', 'codebert-mean', 'e5-base-v2', 'finbert-mean',
         'pubmedbert-embeddings', 'specter2-base')
POPULATION = '55583a92affe1d330c67bda07c6a3976ae50427aa23404332085fd2461e63ee3'
NATIVE = 'efa6540a740e65e12ff084c6cfab023eb1e895fc668fdf36eab82fae4a1787ea'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('xb') as f:
        f.write(canonical(value))


def create(population, native, plan, target):
    check(not target.exists(), 'CAPSULE_EXISTS')
    check(sha(population/'report.json') == POPULATION
          and sha(native/'report.json') == NATIVE, 'SOURCE_PIN')
    report, measured = read(population/'report.json'), read(native/'report.json')
    check(measured['complete'] and not measured['errors']
          and measured['calibration_report_sha256'] == POPULATION, 'SOURCE_JOIN')
    names = ('references.jsonl', 'documents.jsonl', 'cases.jsonl', 'assignments.jsonl',
             'connected/chunks/atoms.parquet')
    for name in names:
        check(sha(population/name) == report['files'][name]['sha256'], 'SOURCE_BYTES')
    rows = lambda name: [json.loads(x) for x in (population/name).read_bytes().splitlines()]
    refs = rows('references.jsonl'); byid = {r['id']: r for r in refs}
    check(len(byid) == len(refs), 'REFERENCE_ID')
    docs = {d['id']: d for d in rows('documents.jsonl')}
    assignments = {a['case']: a for a in rows('assignments.jsonl') if a['policy'] == 'multirc'}
    parts = {d: set() for d in docs}
    for case in rows('cases.jsonl'):
        if case['family'] == 'multirc':
            parts[case['document']].add(assignments[case['id']]['partition'])
    all_ids = plan['corpus_ids'] + plan['query_ids']
    check(len(set(all_ids)) == len(all_ids), 'QUERY_LEAKAGE')
    for role, partition in (('corpus', 'train'), ('query', 'calibration')):
        group = plan[role+'_documents']
        check(all(parts[d] == {partition} for d in group), 'DOCUMENT_PARTITION')
        expected = ({s for d in group for s in docs[d]['sentences']} if role == 'corpus'
                    else {docs[d]['paragraph'] for d in group})
        check(expected == set(plan[role+'_ids']), 'DOCUMENT_ROSTER')
    texts = pq.read_table(population/'connected/chunks/atoms.parquet', columns=['text'])['text'].to_pylist()
    references = [byid[key] for key in all_ids]
    for ref in references:
        check(hashlib.sha256(texts[ref['unit']].encode()).hexdigest() == ref['text_sha256'],
              'SOURCE_TEXT')
    target.mkdir()
    metadata = dict(schema='native-kernel-capsule-v1', population=POPULATION, native=NATIVE,
                    plan=plan, references=references,
                    documents=[docs[d] for d in plan['corpus_documents']+plan['query_documents']],
                    lanes={}, meaning_asserted=False, world_truth_established=False,
                    production_admission=False)
    for name in LANES:
        path = native/name
        lane = read(path/'report.json')
        check(sha(path/'report.json') == measured['lenses'][name]['report_sha256']
              and sha(path/'slots.npz') == lane['files']['slots.npz']['sha256'], 'NATIVE_PIN')
        with np.load(path/'slots.npz', allow_pickle=False) as z:
            arrays, ptr = z['vectors'], z['offsets']
            chunks = [arrays[ptr[r['unit']]:ptr[r['unit']+1]].copy() for r in references]
        offsets = np.array([0]+list(np.cumsum([len(x) for x in chunks])), np.int64)
        vectors = np.concatenate(chunks)
        NativeLane(all_ids, vectors, offsets)
        output = target/(name+'.npz')
        np.savez(output, vectors=vectors, offsets=offsets)
        metadata['lanes'][name] = dict(file_sha256=sha(output), lens_id=lane['lens_id'],
                                      source_report_sha256=sha(path/'report.json'),
                                      contract=lane['contract'], windows=len(vectors))
    write(target/'manifest.json', metadata)
    pin = sha(target/'manifest.json')
    (target/'COMMITTED').write_text(pin, encoding='ascii')
    return pin


def load(target, expected_pin):
    check((target/'COMMITTED').is_file(), 'CAPSULE_INCOMPLETE')
    check((target/'COMMITTED').read_text() == expected_pin
          and sha(target/'manifest.json') == expected_pin, 'CAPSULE_PIN')
    meta = read(target/'manifest.json')
    check(meta['population'] == POPULATION and meta['native'] == NATIVE
          and tuple(sorted(meta['lanes'])) == LANES
          and meta['production_admission'] is False
          and meta['world_truth_established'] is False, 'CAPSULE_SCOPE')
    ids = meta['plan']['corpus_ids']+meta['plan']['query_ids']
    check([r['id'] for r in meta['references']] == ids, 'CAPSULE_ROSTER')
    count = len(meta['plan']['corpus_ids'])
    lanes, queries = {}, {}
    for name in LANES:
        p = target/(name+'.npz')
        check(sha(p) == meta['lanes'][name]['file_sha256'], 'CAPSULE_NATIVE_PIN')
        with np.load(p, allow_pickle=False) as z:
            x, ptr = z['vectors'], z['offsets']
            NativeLane(ids, x, ptr)
            lanes[name] = NativeLane(ids[:count], x[:ptr[count]], ptr[:count+1])
            queries[name] = {key: x[ptr[i]:ptr[i+1]].copy()
                             for i, key in enumerate(ids) if i >= count}
    return meta, lanes, queries
