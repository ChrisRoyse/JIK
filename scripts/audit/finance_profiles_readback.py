"""Cold source-to-geometry readback of the fixed finance assay; no producer import."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.candidate_panel_geometry import independent_maxima


def check(ok, why):
    if not ok: raise ValueError('MNLM_FINANCE_PROFILE_READBACK_'+why)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def equal(a, b, why):
    check(np.asarray(a).shape == np.asarray(b).shape and
          np.allclose(a, b, rtol=0, atol=1e-12), why)


def sim(a, b):
    values = np.einsum('id,jd->ij', a, b, optimize=True)
    left = sum(np.amax(values, axis=1))/len(a)
    right = sum(np.amax(values, axis=0))/len(b)
    return min(left, right)


def run(a):
    start = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.profiles/'manifest.json').read_bytes())
    check(sha(a.profiles/'report.json') == manifest['report_sha256'] and
          sha(a.profiles/'intent.json') == manifest['intent_sha256'], 'PUBLICATION')
    report = json.loads((a.profiles/'report.json').read_bytes())
    check(report['complete'] and not report['scientific_admission'] and not report['served'], 'CLAIMS')
    for source in report['sources'].values():
        check(sha(source['file']) == source['sha256'], 'SOURCE_PIN')
        if 'plan_file' in source:
            check(sha(source['plan_file']) == source['plan_sha256'] and
                  sha(source['readback_file']) == source['readback_sha256'], 'CANDIDATE_PINS')
    anchors = json.loads(Path(report['sources']['anchors.json']['file']).read_bytes())
    chunks = pq.read_table(report['sources']['chunks/atoms.parquet']['file']).to_pylist()
    grouped = defaultdict(list)
    for i, chunk in enumerate(chunks): grouped[chunk['row']].append(i)
    saved = {}; matrices = []; total_profiles = 0; total_pairs = 0
    for q, record in report['files'].items():
        path = a.profiles/record['file']; check(sha(path) == record['sha256'], 'PROFILE_PIN')
        with np.load(path, allow_pickle=False) as data: saved[q] = {key: data[key].copy() for key in data.files}
        value = saved[q]; source = anchors[q]
        equal(value['rows'], source['rows'], 'ROWS'); equal(value['labels'], source['labels'], 'LABELS')
        d = np.flatnonzero(np.asarray(source['partitions']) == 'design')
        e = np.flatnonzero(np.asarray(source['partitions']) == 'evaluation')
        equal(value['design'], d, 'DESIGN'); equal(value['evaluation'], e, 'EVALUATION')
        equal(value['landmarks'], d[np.arange(24)*len(d)//24], 'LANDMARKS')
        upper = np.column_stack(np.triu_indices(len(d), 1))
        expected = d[upper[np.random.default_rng(20261019).choice(len(upper), 4096, replace=False)]]
        equal(value['pairs'], expected, 'PAIR_DESIGN')
    with threadpool_limits(limits=1):
        for name in report['names']:
            if name in report['sources']:
                location = Path(report['sources'][name]['file']).parent
                plan_root = Path(report['sources'][name]['plan_file']).parent
                vectors = np.load(location/'vectors.npy', allow_pickle=False)
                offsets = np.load(plan_root/'offsets.npy', allow_pickle=False)
                native = json.loads((location/'report.json').read_bytes())
                plan = json.loads((plan_root/'report.json').read_bytes())
                check(sha(location/'vectors.npy') == native['files']['vectors.npy']['sha256'], 'NATIVE_VECTOR_PIN')
                check(sha(plan_root/'offsets.npy') == plan['files']['offsets.npy']['sha256'], 'NATIVE_OFFSET_PIN')
            else:
                location = Path(report['sources']['public']['file']).parent/name
                native = json.loads((location/'report.json').read_bytes())
                public = json.loads(Path(report['sources']['public']['file']).read_bytes())
                check(sha(location/'report.json') == public['lenses'][name]['report_sha256'], 'PUBLIC_LENS_PIN')
                check(sha(location/'slots.npz') == native['files']['slots.npz']['sha256'], 'PUBLIC_VECTOR_PIN')
                with np.load(location/'slots.npz', allow_pickle=False) as data:
                    vectors = data['vectors']; offsets = data['offsets']
            check(len(offsets) == len(chunks)+1 and offsets[0] == 0 and
                  np.all(np.diff(offsets) > 0) and offsets[-1] == len(vectors), 'NATIVE_OFFSETS')
            sets = {}
            for row, indices in grouped.items():
                values = np.vstack([vectors[offsets[i]:offsets[i+1]] for i in indices]).astype(float)
                norm = np.sqrt(np.sum(values*values, axis=1, keepdims=True))
                check(np.isfinite(values).all() and np.all(norm > 0), 'NATIVE_FINITE')
                sets[row] = values/norm
            for q, data in saved.items():
                selected = [sets[row] for row in data['rows']]
                profile = np.array([[sim(v, selected[j]) for j in data['landmarks']] for v in selected])
                profile /= np.sqrt(np.sum(profile*profile, axis=1, keepdims=True))
                equal(profile, data[name+'__profile'], 'ALL_NATIVE_PROFILES')
                pairs = np.array([sim(selected[i], selected[j]) for i,j in data['pairs']])
                equal(pairs, data[name+'__pairs'], 'ALL_NATIVE_PAIRS')
                total_profiles += len(profile); total_pairs += len(pairs)
            print(json.dumps(dict(instrument=name, profiles_rebuilt=total_profiles, pairs_rebuilt=total_pairs)), flush=True)
    for q, data in saved.items():
        ranks = np.array([rankdata(data[name+'__pairs']) for name in report['names']])
        ranks -= ranks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(ranks, axis=1)
        matrix = ranks@ranks.T/np.outer(norms,norms)
        equal(matrix, data['correlations'], 'D1_CORRELATIONS'); matrices.append(matrix)
    maximum = np.max(matrices, axis=0); equal(maximum, report['d1_maximum_over_questions'], 'D1_MAXIMUM')
    selection = independent_maxima(maximum, np.ones(len(report['names']), bool), report['names'])
    check(selection == report['selection'], 'D1_COMPLETE_SELECTION')
    prefixes = sorted({tuple(panel[:i+1]) for panel in selection['all_maximizers'] for i in range(1,len(panel))})
    check(list(map(list,prefixes)) == report['prefixes'] and len(prefixes)*len(saved) == report['comparisons'], 'ALL_PREFIXES')
    key = next(iter(saved)); name = report['names'][0]; data = saved[key]
    np.save(a.output/'fault-profile.npy', data[name+'__profile'][:,:-1])
    bad_role = data['design'].copy(); bad_role[0] = data['evaluation'][0]
    np.save(a.output/'fault-role.npy', bad_role)
    (a.output/'fault-source.json').write_bytes(Path(report['sources']['population']['file']).read_bytes()+b' ')
    negatives = {}
    for label, action in {
        'dropped_coordinate': lambda: equal(np.load(a.output/'fault-profile.npy'),data[name+'__profile'],'ALL_NATIVE_PROFILES'),
        'evaluation_landmark_role': lambda: equal(np.load(a.output/'fault-role.npy'),data['design'],'DESIGN'),
        'changed_source': lambda: check(sha(a.output/'fault-source.json') == report['sources']['population']['sha256'],'SOURCE_PIN'),
    }.items():
        try: action()
        except ValueError as exc: negatives[label] = str(exc)
        else: raise ValueError('MNLM_FINANCE_PROFILE_READBACK_NEGATIVE_ACCEPTED')
    # Bind the after-state to the same before-state publication and source bytes.
    check(sha(a.profiles/'report.json') == manifest['report_sha256'], 'AFTER_PUBLICATION')
    for source in report['sources'].values(): check(sha(source['file']) == source['sha256'], 'AFTER_SOURCE')
    result = dict(complete=True, source_report_sha256=manifest['report_sha256'], acceptance='author-checked',
        native_profiles_rebuilt=total_profiles, native_pairs_rebuilt=total_pairs, selection=selection,
        negative_controls=negatives, scientific_admission=False, served=False,
        wall_seconds=time.perf_counter()-start, code_sha256=sha(Path(__file__)))
    (a.output/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    (a.output/'manifest.json').write_text(json.dumps(dict(report_sha256=sha(a.output/'report.json')))+'\n',encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('profiles','output'): p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
