"""CPU recount of persisted update-boundary vectors without importing the learner."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np


def check(value, code):
    if not value:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(report, raw):
    metadata = report['vectors']
    check(sha(raw) == metadata['sha256'] and len(raw) == metadata['bytes'], 'DIAGNOSIS_VECTOR_BYTES')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        check(set(archive.files) == {'gradients','raw','retained','total'}, 'DIAGNOSIS_VECTOR_ROSTER')
        values = {k:archive[k] for k in archive.files}
    n, p = len(metadata['examples']), metadata['coordinates']
    check(values['gradients'].shape == (n,p) and all(values[k].shape == (p,) for k in ('raw','retained','total')),
          'DIAGNOSIS_VECTOR_SHAPE')
    check(all(v.dtype == np.float64 and np.isfinite(v).all() for v in values.values()), 'DIAGNOSIS_VECTOR_FINITE')
    row, = [v for v in report['steps'] if v['step'] == metadata['step']]
    result = {}
    for label, key in (('raw_adam','raw'), ('retained','retained')):
        products = np.sum(values['gradients']*values[key][None,:], axis=1)
        total = float(np.sum(values['total']*values[key]))
        for name, value in zip(metadata['examples'], products):
            declared = row['directional'][name][label]
            check(abs(float(value)-declared) <= 1e-10*max(1.,abs(declared)), 'DIAGNOSIS_DIRECTIONAL_RECOUNT')
        check(abs(total-row['total_directional'][label]) <= 1e-10*max(1.,abs(total)), 'DIAGNOSIS_TOTAL_RECOUNT')
        result[label] = dict(total=total, examples=dict(zip(metadata['examples'],products.tolist())))
    discrepancy = float(np.abs(values['total']-values['gradients'].mean(0)).max())
    check(abs(discrepancy-row['gradient_max_error']) <= 1e-12, 'DIAGNOSIS_GRADIENT_RECOUNT')
    return dict(step=row['step'], products=result, gradient_max_error=discrepancy,
                controlled_region_switches=row['controlled_region_switches'],
                before=row['before'], raw=row['raw'], after=row['after'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    root = a.root
    raw_report = (root/'diagnosis.json').read_bytes()
    report = json.loads(raw_report)
    source = Path(report['source']['root'])
    check(sha((source/'seal.json').read_bytes()) == report['source']['seal'], 'DIAGNOSIS_SOURCE_PIN')
    for name, entry in json.loads((source/'seal.json').read_bytes())['files'].items():
        raw = (source/name).read_bytes()
        check(sha(raw) == entry['sha256'] and len(raw) == entry['bytes'], 'DIAGNOSIS_SOURCE_BYTES')
    for kind, filename in (('start','controlled-1-start'), ('final','controlled-1')):
        manifest = json.loads((source/'candidates'/filename/'manifest.json').read_bytes())
        check(report[kind+'_fingerprint'] == manifest['fingerprint'], 'DIAGNOSIS_FINGERPRINT_JOIN')
    original = json.loads((source/'source.json').read_bytes())
    check(sha((root/'experience.sqlite3').read_bytes()) == original['journal'], 'DIAGNOSIS_JOURNAL_READBACK')
    vectors = (root/report['vectors']['path']).read_bytes()
    boundary = read(report, vectors)
    controls = []

    def refuse(name, code, action):
        try:
            action()
        except ValueError as error:
            check(str(error) == code, 'DIAGNOSIS_CONTROL_WRONG_CODE')
        else:
            raise AssertionError('DIAGNOSIS_CONTROL_ACCEPTED')
        controls.append(dict(name=name, code=code))

    corrupted = bytearray(vectors)
    corrupted[len(corrupted)//2] ^= 1
    refuse('changed_vector_bytes', 'DIAGNOSIS_VECTOR_BYTES', lambda:read(report, corrupted))
    changed = copy.deepcopy(report)
    row = changed['steps'][boundary['step']-1]
    key = report['vectors']['examples'][0]
    row['directional'][key]['retained'] += 1.
    refuse('changed_directional_claim', 'DIAGNOSIS_DIRECTIONAL_RECOUNT', lambda:read(changed, vectors))
    check([r['step'] for r in report['steps']] == list(range(1,9)) and
          all(r['exact_update_replay'] for r in report['steps']), 'DIAGNOSIS_REPLAY_COVERAGE')
    result = dict(complete=True, acceptance='author-checked', diagnosis_sha256=sha(raw_report),
        boundary=boundary, negative_controls=controls, source_files_unchanged=True,
        unchanged_journal=True, raw_bytes=len(vectors), wall_seconds=time.perf_counter()-started,
        serving=False, scientific_admission=False)
    with (root/'independent-readback.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'))
    print(json.dumps(dict(complete=True, boundary_step=boundary['step'], products=boundary['products'],
                         negative_controls=controls, wall_seconds=result['wall_seconds'])))


if __name__ == '__main__':
    main()
