"""Independent persisted comparison of kernel growth, arithmetic reuse and cold recovery."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.capability_population import need, sha
from claims.process_conditions import write_committed


def read(path): return json.loads(path.read_bytes())


def kernel(root, stage):
    p = root/'shared'/stage/'kernel'
    raw = (p/'manifest.json').read_bytes()
    need(read(p/'COMMITTED') == dict(manifest=sha(raw)), 'KERNEL_READER_COMMIT')
    v = json.loads(raw)
    for filename, key in [('cache.json', 'cache_sha256'), ('associations.json', 'associations_sha256')]:
        need(sha((p/filename).read_bytes()) == v[key], 'KERNEL_READER_FILE')
    return v, sha(raw), read(p/'timings.json'), read(p/'associations.json')


def check(reference, root):
    cfg = read(root/'config.json')
    need(sha(Path(cfg['journal']).read_bytes()) == cfg['journal_pin'], 'KERNEL_READER_JOURNAL')
    need((reference/'splits.json').read_bytes() == (root/'splits.json').read_bytes(), 'KERNEL_READER_SPLIT')
    stages = []
    for stage in ('initial', 'expanded'):
        a, ap, at, aa = kernel(reference, stage)
        b, bp, bt, ba = kernel(root, stage)
        for key in ('source', 'split', 'ids', 'instruments', 'slots', 'connections', 'graphs'):
            need(a[key] == b[key], 'KERNEL_READER_RESULT_CHANGED:'+key)
        need(aa == ba, 'KERNEL_READER_ASSOCIATIONS_CHANGED')
        minima, routing = [], []
        for name, result in b['calibration'].items():
            previous = a['calibration'][name]
            need(result['families'] == previous['families'] and result['passes'], 'KERNEL_READER_RECALL')
            minima.extend(min(v['recall'], v['target_recall']) for v in result['families'].values())
            need(len(result['rows']) == len(previous['rows']), 'KERNEL_READER_QUERIES')
            for old, new in zip(previous['rows'], result['rows'], strict=True):
                for key in ('query', 'exact', 'recall', 'target_recall', 'expected_targets', 'returned_targets'):
                    need(old[key] == new[key], 'KERNEL_READER_TARGET_CHANGED:'+key)
                need(old['route']['hits'] == new['route']['hits']
                     and new['route']['work_limit'] is None, 'KERNEL_READER_ROUTE')
                routing.append(new['route'])
        stages.append(dict(stage=stage, tasks=len(b['ids']), reference=ap, candidate=bp,
            all_graphs_targets_associations_equal=True, minimum_family_recall=min(minima),
            cross_domain_edges=b['cross_domain_edges'],
            timings=dict(reference=at, candidate=bt),
            ann_requests=sum(v['ann_requests'] for v in b['work'].values()),
            ann_unique_queries=sum(v['ann_unique_queries'] for v in b['work'].values()),
            ann_reused=sum(v['ann_duplicate_queries_reused'] for v in b['work'].values()),
            exact_pairs=sum(v['exact_pairs'] for v in b['work'].values()),
            cached_pair_requests=sum(v['reused_pair_requests'] for v in b['work'].values()),
            oracle_pairs=sum(v['exact_reference_pairs'] for v in b['calibration'].values()),
            oracle_logical_pairs=sum(v['exact_logical_pairs'] for v in b['calibration'].values()),
            max_route_logical_scores=max(t['logical_scores'] for t in routing),
            max_route_computations=max(t['distance_computations'] for t in routing)))
    need(set(kernel(root, 'initial')[0]['ids']) < set(kernel(root, 'expanded')[0]['ids']),
         'KERNEL_READER_GROWTH')
    build, cold = read(root/'build-report.json'), read(root/'cold-report.json')
    need(build['journal'] == cold['journal'] == cfg['journal_pin']
         and build['head'] == cold['head'] and build['events'] == cold['events']
         and build['results']['shared']['final_kernel'] == cold['results']['shared']['final_kernel'],
         'KERNEL_READER_COLD')
    need({v['name'] for v in build['results']['shared']['controls']} ==
         {'stale_source', 'missing_target', 'swapped_role', 'future_answer_as_context', 'lost_relationship'},
         'KERNEL_READER_NEGATIVE_ROSTER')
    need(cold['results']['shared']['controls'] ==
         [dict(name='unchanged_source_no_work', persisted_indices_unchanged=True)], 'KERNEL_READER_NOOP')
    return dict(stages=stages, journal=cfg['journal_pin'], events=cold['events'], head=cold['head'],
        cold_recovery_equal=True, unchanged_source_no_work=True, negative_controls=build['results']['shared']['controls'],
        schema='capability-kernel-optimization-readback-v1', number_status='measured', acceptance='author-checked',
        scientific_admission=False, learned_capability=False, journal_integration=False,
        metric_mapping=['T2/spec27 kernel recall', 'T5/spec27 continuous-learning consistency'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    need(not a.output.exists(), 'KERNEL_READER_EXISTS')
    report = check(a.reference, a.root)
    pin = write_committed(a.output, report)
    print(json.dumps(dict(pin=pin, stages=report['stages'])), flush=True)


if __name__ == '__main__': main()
