"""Independent physical recount of paired-answer publications; author-check only."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

# Only metadata is mapped; no matrix computation needs a BLAS worker pool.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[name] = '1'
import numpy as np


def check(value, name):
    if not value: raise ValueError('MNLM_ANSWER_PAIR_READBACK_'+name)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, pin):
    raw = Path(path).read_bytes(); check(hashlib.sha256(raw).hexdigest() == pin, 'PIN')
    return json.loads(raw)


def locate(root, name):
    check(isinstance(name, str) and Path(name).name == name and name not in ('', '.', '..'), 'PATH')
    path = Path(root)/name
    check(not path.is_symlink() and path.resolve().parent == Path(root).resolve(), 'PATH')
    return path


def run(root, pin):
    began = time.perf_counter(); root = Path(root); r = read(root/'report.json', pin)
    check(r['complete'] is True and r['schema'] == 'legal-answer-architecture-comparison-v1', 'REPORT')
    for name, item in r['files'].items():
        path = locate(root, name)
        check(path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], 'FILES')
    check(set(r['files']) == {'cases.jsonl', 'summary.json'}
          and r['summary_pin'] == r['files']['summary.json']['sha256'], 'SUMMARY_PIN')
    summary = read(root/'summary.json', r['summary_pin'])
    sroot = Path(r['input']['score']).parent/'scores'
    source = read(r['input']['score'], r['input']['pin'])
    binding = read(sroot/'binding.json', source['binding'])
    pack = Path(r['input']['pack']); p = read(pack/'report.json', source['pack'])
    check(p['binding']['source'] == binding['source'], 'SOURCE')
    source_pin = hashlib.sha256(encoded(binding['source'])).hexdigest()
    check(source_pin == r['source'] == summary['source'] and r['roster'] == p['roster']
          and r['exposure'] == binding['source']['exposure'], 'SOURCE_ID')
    check(file_hash(pack/'rows.npy') == p['files']['rows.npy']['sha256'], 'ROWS')
    metadata = np.load(pack/'rows.npy', mmap_mode='r', allow_pickle=False)
    check(r['full_population'] == source['full_population'] == (binding['ordinals'] == list(range(len(metadata))))
          and r['input']['scope'] == source['scope'] == binding['scope']
          and (binding['scope'] != 'all' or r['full_population'] is True), 'FULL_POPULATION')
    roster = r['roster']; proposals = [('shared', 'assembled'), ('own_slot', 'assembled')]
    proposals += [('whole_proposals', n) for n in roster]
    pairs = [('architecture', proposals[1]+('correct',), proposals[0]+('correct',))]
    pairs += [('architecture', a+('correct',), b+('correct',)) for a in proposals[2:] for b in proposals[:2]]
    pairs += [('context', a+('correct',), a+(m,)) for a in proposals for m in ('reversed', 'train_donor')]
    check(summary['comparisons'] == [dict(kind=k,left=list(a),right=list(b)) for k,a,b in pairs], 'COMPARISONS')
    original = {v['ordinal']:v for v in source['cases']}; output = {v['ordinal']:v for v in r['cases']}
    check(len(original) == len(source['cases']) == len(output) == len(r['cases']) == r['declared_cases']
          and sorted(original) == sorted(output) == binding['ordinals'], 'COVERAGE')
    groups = {}; seen = set(); component_sides = {}; gaps = {v['ordinal']:v for v in source['control_unavailable']}
    scored = 0; unknown = 0; scalar_deltas = 0; position_values = 0; models = None
    try:
        with (root/'cases.jsonl').open('rb') as stream:
            for line in stream:
                v = json.loads(line); m = v['meta']; ordinal = m['ordinal']
                check(ordinal in output and ordinal not in seen, 'DUPLICATE_CASE'); seen.add(ordinal)
                check(hashlib.sha256(line).hexdigest() == output[ordinal]['sha256'], 'ROW_PIN')
                row = metadata[ordinal]
                expected_meta = dict(ordinal=ordinal, document=int(row['document']), component=int(row['component']),
                    side=('train', 'development', 'query')[int(row['side'])],
                    collection=binding['source']['collections'][int(row['collection'])],
                    exposure=binding['source']['exposure'], reversal_distinct=bool(row['reversal_distinct']))
                check(m == expected_meta and v['source'] == source_pin, 'IDENTITY')
                check(component_sides.setdefault(m['component'], m['side']) == m['side'], 'COMPONENT_SIDE')
                ref = original[ordinal]; c = read(locate(sroot, ref['path']), ref['pin'])
                check(c['ordinal'] == ordinal and c['document'] == m['document'] and c['binding'] == source['binding']
                      and v['input_commit'] == ref['pin'] and v['input_payload'] == c['payload']['pin'], 'COMMIT')
                payload = read(locate(sroot, c['payload']['path']), c['payload']['pin'])
                if ordinal in gaps:
                    check(payload == gaps[ordinal] == dict(status='CONTROL_UNAVAILABLE_NATIVE_IDENTICAL',
                          ordinal=ordinal, document=m['document']) and v['status'] == payload['status']
                          and not {'deltas', 'endings', 'dominance'} & set(v), 'GAP')
                    unknown += 1; computed = None
                else:
                    check(v['status'] == 'scored' and payload['roster'] == roster, 'STATUS')
                    i = payload['identity']
                    check(all(i[k] == m[k] for k in ('document', 'component', 'side', 'collection', 'exposure'))
                          and i['source'] == source_pin, 'SOURCE_IDENTITY')
                    fixed = {a:{k:model[k] for k in ('model','target','roster','cache_bytes','serialized_event_bytes')}
                             for a,model in payload['models'].items()}
                    if models is None: models = fixed
                    check(fixed == models == r['models'], 'MODELS')
                    arms = {(a['architecture'],a['proposal'],a['mode']):a['metrics'] for a in payload['arms']}
                    check(len(arms) == len(payload['arms']) == 24
                          and set(arms) == {a+(mode,) for a in proposals for mode in ('correct', 'reversed', 'train_donor')}, 'ARMS')
                    means = {}
                    for key, metrics in arms.items():
                        e,g = metrics['expected_atoms'],metrics['generated_atoms']
                        check(e == i['reference_atoms'] > 0 and g >= 0, 'COUNT')
                        check(metrics['extra'] == max(g-e, 0) and metrics['missing'] == max(e-g, 0)
                              and metrics['matched_positions'] == min(e,g), 'COUNTS')
                        check(metrics['censored'] == (not metrics['ended'])
                              and metrics['ended_at_reference_boundary'] == (metrics['ended'] and e == g)
                              and ((metrics['ended'] and metrics['refusal'] is None)
                                   or (not metrics['ended'] and bool(metrics['refusal']))), 'END')
                        means[key] = {}
                        for n in roster:
                            x = metrics['per_slot'][n]
                            check(len(x) == e and all(-1 <= a <= 1 for a in x)
                                  and x[min(e,g):] == [-1.] * max(e-g, 0), 'SLOT')
                            z = (sum(x)-max(g-e,0))/max(e,g)
                            check(z == metrics['full_output_mean'][n], 'ARITHMETIC')
                            means[key][n] = z; position_values += len(x)
                    computed = [{n:means[a][n]-means[b][n] for n in roster} for _,a,b in pairs]
                    check(v['deltas'] == computed, 'DELTAS'); scalar_deltas += len(pairs)*len(roster)
                    expected_ends = [dict(left_ended=arms[a]['ended'], right_ended=arms[b]['ended'],
                        left_exact=arms[a]['ended_at_reference_boundary'], right_exact=arms[b]['ended_at_reference_boundary'],
                        left_censored=arms[a]['censored'], right_censored=arms[b]['censored']) for _,a,b in pairs]
                    check(v['endings'] == expected_ends, 'PAIR_ENDS')
                    dominance = []
                    for d in computed:
                        signs = {0 if x == 0 else 1 if x > 0 else -1 for x in d.values()}
                        dominance.append('mixed' if {-1,1} <= signs else 'left' if 1 in signs else 'right' if -1 in signs else 'tie')
                    check(v['dominance'] == dominance, 'DOMINANCE'); scored += 1
                for col in ('ALL', m['collection']):
                    for index in range(len(pairs)):
                        key = (col,m['side'],index)
                        g = groups.setdefault(key,dict(count=0,unknown=0,components=set(),distinct=0,
                            values={n:dict(total=Fraction(0),positive=0,negative=0,tie=0) for n in roster},
                            dominance={n:0 for n in ('left','right','mixed','tie')},endings={}))
                        g['count'] += 1; g['distinct'] += m['reversal_distinct']; g['components'].add(m['component'])
                        if computed is None: g['unknown'] += 1; continue
                        g['dominance'][dominance[index]] += 1
                        for n,z in expected_ends[index].items():g['endings'][n] = g['endings'].get(n,0)+int(z)
                        for n,z in computed[index].items():
                            stats = g['values'][n]; stats['total'] += Fraction.from_float(z)
                            stats['positive' if z > 0 else 'negative' if z < 0 else 'tie'] += 1
        check(seen == set(original) and scored == r['scored_cases'] == source['scored_cases']
              and unknown == r['unavailable_cases'] == len(gaps), 'FINAL_COVERAGE')
        expected_groups = []
        for (col,side,index),g in sorted(groups.items()):
            count,missing = g['count'],g['unknown']; known = count-missing
            slots = {n:dict(paired_mean=float(z['total']/known) if known else None,
                declared_mean_bounds=[float((z['total']-2*missing)/count), float((z['total']+2*missing)/count)],
                positive=z['positive'],negative=z['negative'],tie=z['tie']) for n,z in g['values'].items()}
            expected_groups.append(dict(collection=col,side=side,comparison=index,declared_cases=count,
                scored_cases=known,unavailable=missing,distinct_address_order_cases=g['distinct'],
                components=len(g['components']),slots=slots,dominance=g['dominance'],endings=g['endings']))
        check(summary['groups'] == expected_groups and summary['cases'] == len(seen), 'AGGREGATES')
        check(summary['policy'] == dict(schema='answer-architecture-comparison-v1',
            metric='existing_per_slot_full_output_mean', paired_unit='same_source_document',
            unknown_difference_bounds=[-2.,2.], aggregation='equal_case_exact_binary64_sum',
            proposal_selection='all_separately', best_of=False, uncertainty_estimated=False,
            semantic_validity_established=False, architecture_winner=False, capacity_matched=False)
            and summary['intervals'] == 'bounds_for_unobserved_declared_cases_not_confidence_intervals', 'CLAIMS')
    finally: metadata._mmap.close()
    return dict(complete=True, publication=dict(path=str(root),pin=pin),cases=len(seen),scored=scored,unavailable=unknown,
        per_slot_deltas=scalar_deltas, source_position_values=position_values, groups=len(groups),
        summary=r['summary_pin'], seconds=time.perf_counter()-began, acceptance='author-checked only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--pin', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    value = run(args.root,args.pin)
    with args.output.open('xb') as stream:
        stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
    print(json.dumps(value), flush=True)
