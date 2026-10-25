"""Separate source-score/Fraction recount of the complete-answer baseline join."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[n] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.legal_answer_architecture_readback import encoded, file_hash, locate, read
from model.answer_baseline_comparison import BASELINE_POLICY, POLICY


def check(value, name):
    if not value: raise ValueError('MNLM_BASELINE_READBACK_'+name)


def means(m, roster, expected):
    e, g = m['expected_atoms'], m['generated_atoms']
    check(type(e) is int and e == expected > 0 and type(g) is int and g >= 0, 'COUNTS')
    check(m['extra'] == max(0, g-e) and m['missing'] == max(0, e-g)
          and m['matched_positions'] == min(e,g), 'COUNTS')
    check(m['censored'] == (not m['ended'])
          and m['ended_at_reference_boundary'] == (m['ended'] and e == g)
          and ((m['ended'] and m['refusal'] is None) or (not m['ended'] and bool(m['refusal']))), 'END')
    check(m['missing_similarity'] == m['extra_similarity'] == -1.
          and m['semantic_validity_established'] is False
          and m['reference_scored_after_generation'] is True, 'METRIC')
    result = {}
    check(set(m['per_slot']) == set(m['full_output_mean']) == set(roster), 'SLOTS')
    for n in roster:
        x = m['per_slot'][n]
        check(len(x) == e and all(type(z) is float and -1 <= z <= 1 for z in x)
              and x[min(e,g):] == [-1.] * max(0,e-g), 'VALUES')
        result[n] = (sum(x)-max(0,g-e))/max(e,g)
        check(result[n] == m['full_output_mean'][n], 'ARITHMETIC')
    return result


def run(root, pin):
    began = time.perf_counter(); root = Path(root); r = read(root/'report.json', pin)
    check(r['complete'] is True and r['schema'] == 'legal-answer-baseline-comparison-v1', 'REPORT')
    for n, f in r['files'].items():
        p = locate(root,n); check(p.stat().st_size == f['bytes'] and file_hash(p) == f['sha256'], 'FILES')
    summary = read(root/'summary.json', r['summary_pin']); inp = r['input']
    source = read(inp['score'], inp['pin']); scores = Path(inp['score']).parent/'scores'
    binding = read(scores/'binding.json', source['binding'])
    pack = Path(inp['pack']); pub = read(pack/'report.json', source['pack'])
    base = Path(inp['baseline']).parent; br = read(inp['baseline'], inp['baseline_pin'])
    bb = read(base/'binding.json', br['binding']); rb = read(inp['baseline_readback'], inp['readback_pin'])
    check(binding['source'] == pub['binding']['source'] == br['source'] == bb['source'] == rb['source'], 'SOURCE')
    source_pin = hashlib.sha256(encoded(binding['source'])).hexdigest(); roster = r['roster']
    check(source_pin == r['source'] == summary['source'] and roster == br['roster'] == pub['roster'], 'SOURCE_ID')
    check(br['complete'] is True and br['mode'] == bb['mode'] == rb['mode'] == 'full'
          and bb['policy'] == BASELINE_POLICY and rb['producer'] == inp['baseline_pin']
          and all(rb[n] is True for n in ('complete','all_native_exact','all_scores_exact','summary_exact')), 'BASELINE')
    check(file_hash(pack/'rows.npy') == pub['files']['rows.npy']['sha256'], 'ROWS')
    metadata = np.load(pack/'rows.npy', mmap_mode='r', allow_pickle=False)
    try:
        documents = [int(m['document']) for m in metadata]
        check(br['documents'] == bb['documents'] == rb['documents'] == documents, 'BASELINE_COVERAGE')
        check(r['full_population'] == source['full_population'] == (binding['ordinals'] == list(range(len(metadata))))
              and inp['scope'] == source['scope'] == binding['scope']
              and (inp['scope'] != 'all' or r['full_population']), 'SCOPE')
        proposals = [('shared','assembled'),('own_slot','assembled')]+[('whole_proposals',n) for n in roster]
        arms = ('modal_control','clarify_conflicts'); strata = ('no_exact_prefix','unique_answer','conflicting_answers')
        pairs = [(a+('correct',), b) for a in proposals for b in arms]
        check(summary['comparisons'] == [dict(kind='baseline',left=list(a),right=['exact_train',b,'correct'])
                                        for a,b in pairs], 'PAIRS')
        refs = {v['ordinal']:v for v in source['cases']}; output = {v['ordinal']:v for v in r['cases']}
        check(sorted(refs) == sorted(output) == binding['ordinals']
              and len(refs) == len(source['cases']) == len(output) == len(r['cases']) == r['declared_cases'], 'COVERAGE')
        gaps = {v['ordinal']:v for v in source['control_unavailable']}
        baseline_file = base/'cases.jsonl'; before = baseline_file.stat(); offsets = []; h = hashlib.sha256()
        with baseline_file.open('rb') as f:
            while True:
                offset = f.tell(); raw = f.readline()
                if not raw: break
                h.update(raw); offsets.append((offset,len(raw)))
        check(h.hexdigest() == br['cases_pin'] and len(offsets) == len(documents), 'BASELINE_BYTES')
        groups = {}; seen = set(); counts = dict.fromkeys(strata,0); components = {}; scored = 0
        with baseline_file.open('rb') as baseline, (root/'cases.jsonl').open('rb') as f:
            for raw in f:
                v = json.loads(raw); m = v['meta']; i = m['ordinal']
                check(i in output and i not in seen, 'DUPLICATE'); seen.add(i)
                check(hashlib.sha256(raw).hexdigest() == output[i]['sha256'], 'ROW_PIN')
                rr = metadata[i]
                want = dict(ordinal=i,document=int(rr['document']),component=int(rr['component']),
                    side=('train','development','query')[int(rr['side'])],
                    collection=binding['source']['collections'][int(rr['collection'])],
                    exposure=binding['source']['exposure'],reversal_distinct=bool(rr['reversal_distinct']))
                check(m == want and v['source'] == source_pin, 'IDENTITY')
                check(components.setdefault(m['component'],m['side']) == m['side'], 'COMPONENT_SPLIT')
                offset,length = offsets[i]; baseline.seek(offset); line = baseline.read(length); b = json.loads(line)
                check(hashlib.sha256(line).hexdigest() == v['baseline_line'], 'BASELINE_LINE')
                check(b['identity'] == {k:m[k] for k in ('document','component','side','collection')}, 'BASELINE_JOIN')
                status = b['selection_status']; check(type(status) is int and status in (0,1,2), 'SELECTION')
                check(v['baseline_status'] == strata[status] and v['baseline_donor'] == b['donor']
                      and v['selection_row'] == b['selection_row'], 'BASELINE_IDENTITY')
                counts[strata[status]] += 1; expected = b['arms']['modal_control']['expected_atoms']
                bm = {n:means(b['arms'][n],roster,expected) for n in arms}
                if status == 0:
                    check(b['donor'] == -1 and all(b['arms'][n]['refusal'] == 'NO_EXACT_TRAIN_PREFIX'
                          and b['arms'][n]['generated_atoms'] == 0 for n in arms), 'NO_MATCH')
                else:
                    check(b['donor'] >= 0 and b['arms']['modal_control']['ended']
                          and b['arms']['modal_control']['generated_atoms'] == b['native']['donor_atoms'] > 0, 'DONOR')
                    check(b['arms']['modal_control'] == b['arms']['clarify_conflicts'] if status == 1 else
                          b['arms']['clarify_conflicts']['refusal'] == 'CONFLICTING_TRAIN_ANSWERS'
                          and b['arms']['clarify_conflicts']['generated_atoms'] == 0, 'CONFLICT')
                commit = read(locate(scores,refs[i]['path']), refs[i]['pin'])
                check(commit['ordinal'] == i and commit['document'] == m['document'] and commit['binding'] == source['binding']
                      and v['input_commit'] == refs[i]['pin'] and v['input_payload'] == commit['payload']['pin'], 'COMMIT')
                payload = read(locate(scores,commit['payload']['path']),commit['payload']['pin']); delta = None
                if i in gaps:
                    check(payload == gaps[i] == dict(status='CONTROL_UNAVAILABLE_NATIVE_IDENTICAL',ordinal=i,document=m['document'])
                          and v['status'] == payload['status'] and not {'deltas','endings','dominance'} & set(v), 'GAP')
                else:
                    ident = payload['identity']
                    check(all(ident[k] == m[k] for k in ('document','component','side','collection','exposure'))
                          and ident['reference_atoms'] == expected and ident['source'] == source_pin, 'MODEL_JOIN')
                    metrics = {(z['architecture'],z['proposal'],z['mode']):z['metrics'] for z in payload['arms']}
                    check(len(metrics) == len(payload['arms']) == 24
                          and set(metrics) == {a+(mode,) for a in proposals for mode in ('correct','reversed','train_donor')}, 'ARMS')
                    mm = {a:means(z,roster,expected) for a,z in metrics.items()}
                    delta = [{n:mm[a][n]-bm[bname][n] for n in roster} for a,bname in pairs]
                    ending = [dict(left_ended=metrics[a]['ended'],right_ended=b['arms'][bn]['ended'],
                        left_exact=metrics[a]['ended_at_reference_boundary'],right_exact=b['arms'][bn]['ended_at_reference_boundary'],
                        left_censored=metrics[a]['censored'],right_censored=b['arms'][bn]['censored']) for a,bn in pairs]
                    dominance = []
                    for d in delta:
                        signs = {0 if z == 0 else 1 if z > 0 else -1 for z in d.values()}
                        dominance.append('mixed' if {-1,1} <= signs else 'left' if 1 in signs else 'right' if -1 in signs else 'tie')
                    check(v['status'] == 'scored' and v['deltas'] == delta, 'DELTAS')
                    check(v['endings'] == ending and v['dominance'] == dominance, 'ENDINGS'); scored += 1
                for st in ('ALL',strata[status]):
                    for col in ('ALL',m['collection']):
                        for j in range(len(pairs)):
                            g = groups.setdefault((st,col,m['side'],j),dict(cases=0,unknown=0,components=set(),distinct=0,
                                slots={n:dict(total=Fraction(0),positive=0,negative=0,tie=0) for n in roster},
                                dominance=dict.fromkeys(('left','right','mixed','tie'),0),endings={}))
                            g['cases'] += 1; g['components'].add(m['component']); g['distinct'] += m['reversal_distinct']
                            if delta is None: g['unknown'] += 1; continue
                            g['dominance'][dominance[j]] += 1
                            for n,z in ending[j].items():g['endings'][n] = g['endings'].get(n,0)+int(z)
                            for n,z in delta[j].items():
                                t = g['slots'][n]; t['total'] += Fraction.from_float(z)
                                t['positive' if z > 0 else 'negative' if z < 0 else 'tie'] += 1
        check(seen == set(refs) and scored == r['scored_cases'] == source['scored_cases']
              and len(seen)-scored == r['unavailable_cases'] == len(gaps), 'FINAL_COVERAGE')
        expected_groups = []
        for (st,col,side,j),g in sorted(groups.items()):
            count,unknown = g['cases'],g['unknown']; known = count-unknown
            slots = {n:dict(paired_mean=float(t['total']/known) if known else None,
                declared_mean_bounds=[float((t['total']-2*unknown)/count),float((t['total']+2*unknown)/count)],
                positive=t['positive'],negative=t['negative'],tie=t['tie']) for n,t in g['slots'].items()}
            expected_groups.append(dict(baseline_status=st,collection=col,side=side,comparison=j,
                declared_cases=count,scored_cases=known,unavailable=unknown,distinct_address_order_cases=g['distinct'],
                components=len(g['components']),slots=slots,dominance=g['dominance'],endings=g['endings']))
        check(summary['groups'] == expected_groups and summary['cases'] == len(seen)
              and summary['stratum_cases'] == r['stratum_cases'] == counts, 'AGGREGATES')
        check(summary['policy'] == POLICY and summary['intervals'] == 'bounds_for_unobserved_declared_cases_not_confidence_intervals', 'CLAIMS')
        after = baseline_file.stat(); check((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            == (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'INPUT_CHANGED')
        return dict(complete=True,publication=dict(path=str(root),pin=pin),cases=len(seen),scored=scored,
            per_slot_deltas=scored*len(pairs)*len(roster),groups=len(groups),stratum_cases=counts,
            source_scores_recounted=True,summary=r['summary_pin'],seconds=time.perf_counter()-began,
            acceptance='author-checked only')
    finally: metadata._mmap.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True); a = p.parse_args()
    result = run(a.root,a.pin)
    with a.output.open('xb') as f:f.write(encoded(result)); f.flush(); os.fsync(f.fileno())
    print(json.dumps(result),flush=True)
