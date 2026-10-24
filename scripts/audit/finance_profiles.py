"""Fixed finance Calyx geometry from complete native window sets (#12).

Landmarks and D1 pairs use design rows only, before outcome scoring. Every
question and every D1-only maximum panel is retained. No scientific admission.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.calyx_resolution import ROSTER, sha, write, need
from audit.window_readback import readback
from audit.commissioned_window_readback import read_measurement
from audit.candidate_panel_geometry import maxima, independent_maxima
from text.calyx_bits import unit

EXTRA = ['cs-natural-history-v1', 'cs-patent-v1', 'cs-regulatory-v1', 'cs-science-v1',
         'cs-legal-v1', 'cs-biomedical-v1', 'cs-finance-v1', 'cs-software-v1']
NAMES = list(ROSTER) + EXTRA
POPULATION_PIN = '9c5034106722fbc73dc9bc38add30010b0af536fba80109f0ddf909d774429ae'


def similarity(a, b):
    product = a @ b.T
    return min(product.max(1).mean(), product.max(0).mean())


def source_inputs(root, population):
    need(sha(population/'report.json') == POPULATION_PIN, 'FINANCE_POPULATION_PIN')
    result = {'population': dict(file=str(population/'report.json'), sha256=POPULATION_PIN)}
    cold = readback(root/'public-v1', population)
    result['public'] = dict(file=str(root/'public-v1/report.json'), sha256=cold['report_sha256'])
    fsv = json.loads((root/'public-fsv-v1/fsv.json').read_bytes())
    need(fsv['passed'] == fsv['total'] and fsv['cold_readback'] == cold, 'FINANCE_PUBLIC_FSV')
    result['public_fsv'] = dict(file=str(root/'public-fsv-v1/fsv.json'), sha256=sha(root/'public-fsv-v1/fsv.json'))
    inventory = json.loads((root/'instrument-inventory.json').read_bytes())
    need(set(inventory) == set(EXTRA), 'FINANCE_ROSTER')
    result['inventory'] = dict(file=str(root/'instrument-inventory.json'), sha256=sha(root/'instrument-inventory.json'))
    for name in EXTRA:
        instrument = Path(inventory[name]['root'])
        need(sha(instrument/'report.json') == inventory[name]['report_sha256'], 'FINANCE_INSTRUMENT_PIN')
        current = read_measurement(root/name/'readings', root/name/'plan', population, instrument)
        saved = json.loads((root/name/'readback.json').read_bytes())
        need(saved['report_sha256'] == current['report_sha256'] and saved['lens_id'] == current['lens_id'], 'FINANCE_COLD_PIN')
        result[name] = dict(file=str(root/name/'readings/report.json'), sha256=current['report_sha256'],
                            plan_file=str(root/name/'plan/report.json'), plan_sha256=sha(root/name/'plan/report.json'),
                            readback_file=str(root/name/'readback.json'), readback_sha256=sha(root/name/'readback.json'))
    return result


def native_sets(root, name, chunks, n):
    if name in ROSTER:
        with np.load(root/'public-v1'/name/'slots.npz', allow_pickle=False) as data:
            vectors, offsets = data['vectors'], data['offsets']
    else:
        vectors = np.load(root/name/'readings/vectors.npy', allow_pickle=False)
        offsets = np.load(root/name/'plan/offsets.npy', allow_pickle=False)
    need(len(offsets) == len(chunks)+1 and offsets[-1] == len(vectors), 'FINANCE_OFFSETS')
    by_row = defaultdict(list)
    for i, record in enumerate(chunks):
        by_row[record['row']].append(vectors[offsets[i]:offsets[i+1]])
    need(set(by_row) == set(range(n)), 'FINANCE_NATIVE_COVERAGE')
    return [unit(np.concatenate(by_row[i]).astype(np.float64)) for i in range(n)]


def run(a):
    started = time.perf_counter()
    a.output.mkdir(parents=True, exist_ok=False)
    sources = source_inputs(a.readings, a.population)
    summary = json.loads((a.population/'finance/summary.json').read_bytes())
    anchors = json.loads((a.population/'finance/anchors.json').read_bytes())
    need(set(anchors) == {'P159', 'P17', 'P414'}, 'FINANCE_QUESTIONS')
    chunks = pq.read_table(a.population/'finance/chunks/atoms.parquet').to_pylist()
    need(len(chunks) == summary['chunks'] == 17547 and summary['entities'] == 4343, 'FINANCE_REAL_N')
    for file, key in [('anchors.json', 'anchors_sha256'), ('chunks/atoms.parquet', 'chunks_sha256')]:
        need(sha(a.population/'finance'/file) == summary[key], 'FINANCE_SOURCE_BYTES')
        sources[file] = dict(file=str(a.population/'finance'/file), sha256=summary[key])
    write(a.output/'intent.json', dict(names=NAMES, questions=sorted(anchors), sources=sources,
        landmarks=24, landmark_selection='fixed stride through 504 design rows only',
        d1_pairs=4096, pair_seed=20261019, d1='maximum Spearman correlation over all three design populations',
        selection='all D1-only maximum panels; original registry order; no outcome scores',
        scientific_admission=False, served=False))
    arrays = {}
    for q, record in sorted(anchors.items()):
        design = np.flatnonzero(np.asarray(record['partitions']) == 'design')
        evaluation = np.flatnonzero(np.asarray(record['partitions']) == 'evaluation')
        need(len(design) == 504 and len(evaluation) == 1008 and len(record['rows']) == 1512, 'FINANCE_SPLIT')
        landmarks = design[np.arange(24)*len(design)//24]
        upper = np.column_stack(np.triu_indices(len(design), 1))
        pairs = design[upper[np.random.default_rng(20261019).choice(len(upper), 4096, replace=False)]]
        arrays[q] = dict(rows=np.asarray(record['rows']), labels=np.asarray(record['labels']),
                         design=design, evaluation=evaluation, landmarks=landmarks, pairs=pairs)
    phases = {'source_readback_seconds': time.perf_counter()-started}; tick = time.perf_counter()
    with threadpool_limits(limits=1):
        for name in NAMES:
            sets = native_sets(a.readings, name, chunks, summary['entities'])
            for q, data in arrays.items():
                selected = [sets[row] for row in data['rows']]
                profile = np.asarray([[similarity(v, selected[j]) for j in data['landmarks']] for v in selected])
                data[name+'__profile'] = unit(profile)
                data[name+'__pairs'] = np.asarray([similarity(selected[i], selected[j]) for i,j in data['pairs']])
            print(json.dumps(dict(stage='profile', instrument=name, documents=len(sets), windows=sum(map(len,sets)))), flush=True)
    phases['native_geometry_seconds'] = time.perf_counter()-tick
    matrices = []; files = {}
    for q, data in arrays.items():
        data['correlations'] = spearmanr(np.asarray([data[n+'__pairs'] for n in NAMES]), axis=1).statistic
        need(np.isfinite(data['correlations']).all(), 'FINANCE_D1_FINITE')
        matrices.append(data['correlations'])
        path = a.output/(q+'.npz')
        with path.open('xb') as f: np.savez(f, **data)
        files[q] = dict(file=path.name, sha256=sha(path))
    matrix = np.max(matrices, axis=0)
    selection = maxima(matrix, np.ones(len(NAMES), bool), NAMES)
    need(selection == independent_maxima(matrix, np.ones(len(NAMES), bool), NAMES), 'FINANCE_D1_REPLAY')
    prefixes = sorted({tuple(panel[:i+1]) for panel in selection['all_maximizers'] for i in range(1, len(panel))})
    write(a.output/'report.json', dict(complete=True, scientific_admission=False, served=False,
        names=NAMES, sources=sources, files=files, d1_maximum_over_questions=matrix.tolist(),
        selection=selection, prefixes=prefixes, comparisons=len(prefixes)*len(anchors),
        phases=phases, wall_seconds=time.perf_counter()-started, code_sha256=sha(Path(__file__))))
    write(a.output/'manifest.json', dict(report_sha256=sha(a.output/'report.json'), intent_sha256=sha(a.output/'intent.json')))
    print(json.dumps(dict(selection=selection, prefixes=len(prefixes), comparisons=len(prefixes)*len(anchors))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('readings', 'population', 'output'): p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
