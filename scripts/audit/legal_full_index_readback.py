"""Qualify the independent reader, then audit a committed full legal instrument index."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
from audit.native_index_reference import normalized, readback, verify_candidates, verify_rows
from audit.native_index_arithmetic import candidate_regression
from audit.legal_native_associations_readback import union_check
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from kernel.legal_native_associations import BankLane, ProofLane
from kernel.native_window_index import NativeWindowIndex
from text.constellation_corpus import digest, file_sha, need
from text.legal_continuation_input import Continuations
from text.legal_native_batch import NativeBatchSession
from text.legal_reuse_input import ReuseInputs


def pinned(path, pin):
    raw = path.read_bytes(); need(digest(raw) == pin, 'FULL_INDEX_AUDIT_PIN')
    return json.loads(raw)


def load_index(root, pin, source, bank, name):
    report = pinned(root/'report.json', pin); b = report['binding']
    need(b['source_pin'] == source.pin and b['lens_id'] == bank.config['instruments'][name]['lens_id']
         and b['instrument'] == name, 'FULL_INDEX_AUDIT_BINDING')
    if 'bank_pin' in b: need(b['bank_pin'] == bank.pin, 'FULL_INDEX_AUDIT_BANK')
    return NativeWindowIndex(root, pin, b)


def receipt_controls(assessment, result, index, read):
    expected = {r['uid']: r['hits'] for r in result['exact_rows']}; out = {}
    for kind in ('wrong_rank', 'wrong_score', 'wrong_recovered', 'duplicate_query'):
        changed = deepcopy(assessment)
        if kind == 'wrong_rank': changed['rows'][0]['exact_hits'][0][0] = changed['rows'][0]['uid']
        if kind == 'wrong_score': changed['rows'][0]['exact_hits'][0][1] += .01
        if kind == 'wrong_recovered': changed['rows'][0]['recovered'] += 1
        if kind == 'duplicate_query': changed['rows'].append(deepcopy(changed['rows'][0]))
        out[kind] = rejected(lambda: verify_rows(changed, expected))
    position = int(np.argmax(index.lengths)); uid = int(index.uids[position])
    need(index.lengths[position] > 1, 'FULL_INDEX_AUDIT_CONTROL_MULTIWINDOW')
    def dropped(ids): return [x[:-1] if i == uid else x for i, x in zip(ids, read(ids))]
    out['removed_real_window'] = rejected(lambda: readback(index, dropped, assessment))
    changed = deepcopy(assessment['rows'][0]); changed['hits'][0][1] += .01
    out['wrong_returned_score'] = rejected(lambda: verify_candidates(changed, normalized(read([changed['uid']]))[0], read))
    return out


def qualify(a, baseline, source, bank, continuation):
    selected = np.flatnonzero((continuation.presence[:baseline['prefix_inputs']] & 1) == 0)[:32].tolist()
    need(selected == baseline['query_uids'], 'FULL_INDEX_AUDIT_PREFIX_QUERIES')
    record = pinned(a.lane_readback, a.lane_pin)
    metadata = pinned(Path(record['index']['path'])/'report.json', record['index']['pin'])
    binding = metadata['binding']; name = binding['instrument']
    need(binding['source_pin'] == source.pin and binding['bank_pin'] == bank.pin
         and binding['lens_id'] == bank.config['instruments'][name]['lens_id'], 'FULL_INDEX_AUDIT_REGRESSION_BINDING')
    witness = pinned(a.regression, a.regression_pin)
    need(witness['receipt_pin'] == a.lane_pin, 'FULL_INDEX_AUDIT_REGRESSION_RECEIPT')
    raw = a.regression.with_suffix('.npz')
    need(file_sha(raw) == witness['first_failure']['native_npz_sha256'], 'FULL_INDEX_AUDIT_REGRESSION_NATIVE')
    with np.load(raw, allow_pickle=False) as saved:
        arithmetic = candidate_regression(record['assessment'], BankLane(bank, name), witness, saved)
    arithmetic_pin = write_committed(a.output/'arithmetic-regression.json', arithmetic)
    regression = dict(path=str(a.regression), pin=a.regression_pin, receipt=str(a.lane_readback),
                      receipt_pin=a.lane_pin, arithmetic_pin=arithmetic_pin)
    print(json.dumps(dict(arithmetic_regression=arithmetic_pin, queries=arithmetic['queries'],
                          seconds=arithmetic['seconds'])), flush=True)
    results = {}; lanes = {}; negatives = {}; first = None
    for name in baseline['roster']:
        native = BankLane(bank, name)
        proof = ProofLane(Path(baseline['proof']['path']), baseline['proof']['pin'], source,
                          name, bank.config['instruments'][name]['lens_id'])
        for kind, reader in [('prefix', native), ('multiwindow', proof)]:
            ref = baseline['lanes'][name][kind]; root = Path(ref['path'])
            index = load_index(root, ref['pin'], source, bank, name)
            gallery = np.arange(baseline['prefix_inputs']) if kind == 'prefix' else proof.uids
            need(np.array_equal(index.uids, gallery), 'FULL_INDEX_AUDIT_GALLERY')
            assessment = pinned(root.parent/(kind+'-assessment.json'), ref['assessment_pin'])
            need([r['uid'] for r in assessment['rows']] == (selected if kind == 'prefix' else proof.uids.tolist()),
                 'FULL_INDEX_AUDIT_QUERY_SELECTION')
            result = readback(index, reader, assessment)
            key = name+'-'+kind; pin = write_committed(a.output/(key+'.json'), result)
            results[key] = dict(pin=pin, recall=result['recall'], seconds=result['seconds'])
            if kind == 'prefix': lanes[name] = assessment['rows']
            if first is None and kind == 'multiwindow': first = (assessment, result, index, reader)
            print(json.dumps(dict(scope=key, **results[key])), flush=True)
    comparison = pinned(a.comparison/'report.json', a.comparison_pin)
    need(comparison['complete'] and comparison['baseline']['pin'] == a.baseline_pin
         and comparison['source']['pin'] == source.pin, 'FULL_INDEX_AUDIT_COMPARISON')
    for item in comparison['result']['records']:
        if item['threads'] != comparison['result']['selected_threads']: continue
        record = pinned(Path(item['path']), item['pin']); name = record['instrument']
        root = Path(record['index']['path']); index = load_index(root, record['index']['pin'], source, bank, name)
        need(np.array_equal(index.uids, np.arange(baseline['prefix_inputs']))
             and [r['uid'] for r in record['assessment']['rows']] == selected, 'FULL_INDEX_AUDIT_MATCHED_SCOPE')
        result = readback(index, BankLane(bank, name), record['assessment'])
        key = name+'-selected-threads'; pin = write_committed(a.output/(key+'.json'), result)
        results[key] = dict(pin=pin, recall=result['recall'], seconds=result['seconds'])
        print(json.dumps(dict(scope=key, **results[key])), flush=True)
    union = pinned(a.baseline/'multiplex.json', baseline['multiplex_pin'])
    union_result = union_check(union, lanes); negatives.update(receipt_controls(*first))
    def wrong_binding():
        wrong = deepcopy(first[2].report['binding']); wrong['source_pin'] = '0'*64
        NativeWindowIndex(first[2].root, first[2].pin, wrong)
    negatives['wrong_source_binding'] = rejected(wrong_binding)
    return dict(scopes=results, multiplex=union_result, controls=negatives, arithmetic_regression=regression,
                comparison=dict(path=str(a.comparison), pin=a.comparison_pin))


def check_census(counts, index, bank, source, name):
    need(counts['instrument'] == name and counts['source_pin'] == source.pin
         and counts['bank_pin'] == bank.pin and counts['inputs'] == bank.config['inputs']
         and counts['windows'] == index.report['windows']
         and counts['dimensions'] == index.report['dimensions'], 'FULL_INDEX_AUDIT_CENSUS_SCOPE')
    end = 0; windows = 0; native_bytes = 0
    for row in counts['shards']:
        need(row['start'] == end, 'FULL_INDEX_AUDIT_CENSUS_GAP')
        descriptor = bank.descriptor(name, row['start']); saved = descriptor['report']
        need(descriptor['pin'] == row['pin'] and saved['start'] == row['start']
             and saved['end'] == row['end'] and saved['windows'] == row['windows'], 'FULL_INDEX_AUDIT_CENSUS_PIN')
        end = row['end']; windows += row['windows']; native_bytes += sum(r['bytes'] for r in saved['files'].values())
    need(end == counts['inputs'] and windows == counts['windows'] and native_bytes == counts['native_bytes'],
         'FULL_INDEX_AUDIT_CENSUS_TOTAL')
    return dict(shards=len(counts['shards']), inputs=end, windows=windows, native_bytes=native_bytes)


def full(a, baseline, source, bank, continuation):
    proof = pinned(a.qualification/'report.json', a.qualification_pin)
    code = file_sha(Path(__file__).with_name('native_index_reference.py'))
    need(proof['complete'] and proof['mode'] == 'qualify' and proof['baseline']['pin'] == a.baseline_pin
         and proof['reference_code_sha256'] == code and proof['code_sha256'] == file_sha(Path(__file__)),
         'FULL_INDEX_AUDIT_UNQUALIFIED_READER')
    need(proof['arithmetic_code_sha256'] == file_sha(Path(__file__).with_name('native_index_arithmetic.py')),
         'FULL_INDEX_AUDIT_UNQUALIFIED_ARITHMETIC')
    record = pinned(a.lane_readback, a.lane_pin); evidence = a.lane_readback.parent
    root = Path(record['index']['path']); metadata = pinned(root/'report.json', record['index']['pin'])
    name = metadata['binding']['instrument']; index = load_index(root, record['index']['pin'], source, bank, name)
    need(metadata['binding']['scope'] == 'full_legal_unique_input_gallery'
         and len(index.uids) == bank.config['inputs'] and np.array_equal(index.uids, np.arange(len(index.uids))),
         'FULL_INDEX_AUDIT_FULL_GALLERY')
    need(record['census_pin'] == metadata['binding']['census_pin'], 'FULL_INDEX_AUDIT_CENSUS_BINDING')
    counts = pinned(evidence/(name+'-census.json'), record['census_pin'])
    selection = pinned(evidence/'query-selection.json', metadata['binding']['query_selection_pin'])
    canonical_report = pinned(Path(baseline['proof']['path'])/'report.json', baseline['proof']['pin'])
    canonical_inputs = pinned(Path(baseline['proof']['path'])/'inputs.json', canonical_report['files']['inputs.json']['sha256'])
    nontrain = np.flatnonzero((continuation.presence & 1) == 0)[:128].tolist()
    canonical = [r['index'] for r in canonical_inputs]
    need(selection == dict(nontrain=nontrain, canonical=canonical, union=sorted(set(nontrain+canonical)),
                           selection='hash_order_before_similarity'), 'FULL_INDEX_AUDIT_QUERY_SELECTION')
    need([r['uid'] for r in record['assessment']['rows']] == selection['union'], 'FULL_INDEX_AUDIT_QUERY_SCOPE')
    with (a.output/'progress.jsonl').open('x') as stream:
        def progress(value):
            stream.write(json.dumps(value, sort_keys=True)+'\n'); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps(value), flush=True)
        audit = readback(index, BankLane(bank, name), record['assessment'], progress)
    census = check_census(counts, index, bank, source, name)
    changed = deepcopy(counts); changed['windows'] += 1
    negative = rejected(lambda: check_census(changed, index, bank, source, name))
    groups = {}
    multiwindow = [uid for uid in canonical if index.lengths[uid] > 1]
    for group, ids in [('nontrain', nontrain), ('canonical', canonical), ('canonical_multiwindow', multiwindow)]:
        members = set(ids); rows = [r for r in record['assessment']['rows'] if r['uid'] in members]
        recovered = sum(len({u for u, _ in r['hits']} & {u for u, _ in r['exact_hits']}) for r in rows)
        groups[group] = dict(queries=len(rows), recovered=recovered, requested=10*len(rows),
                             recall=recovered/(10*len(rows)) if rows else None)
    need(groups == record['groups'], 'FULL_INDEX_AUDIT_GROUPS')
    index.fresh()
    return dict(instrument=name, source_receipt=dict(path=str(a.lane_readback), pin=a.lane_pin),
                index=record['index'], audit=audit, census=census, groups=groups,
                wrong_census_control=negative, qualification=dict(path=str(a.qualification), pin=a.qualification_pin),
                complete_gallery=True, semantic_truth=False, question_family_kernel_gate=False)


def run(a):
    start = tick = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    baseline = pinned(a.baseline/'report.json', a.baseline_pin); need(baseline['complete'], 'FULL_INDEX_AUDIT_BASELINE')
    source = ReuseInputs(Path(baseline['source']['path']), baseline['source']['pin'])
    bank = NativeBatchSession(Path(baseline['bank']['path']), baseline['bank']['pin'], source)
    continuation = Continuations(Path(baseline['continuations']['path']), baseline['continuations']['pin'])
    need(continuation.report['source']['pin'] == source.pin, 'FULL_INDEX_AUDIT_CONTINUATION_SOURCE')
    phases = dict(authenticate=time.perf_counter()-tick); tick = time.perf_counter()
    result = qualify(a, baseline, source, bank, continuation) if a.mode == 'qualify' else full(a, baseline, source, bank, continuation)
    phases['readback'] = time.perf_counter()-tick; tick = time.perf_counter()
    bank.fresh(); continuation.fresh(); source.fresh()
    phases['final_freshness'] = time.perf_counter()-tick; seconds = time.perf_counter()-start
    need(sum(phases.values())/seconds >= .9, 'FULL_INDEX_AUDIT_PHASE_ATTRIBUTION')
    report = dict(schema='legal-full-native-index-readback-v1', complete=True, mode=a.mode,
                  baseline=dict(path=str(a.baseline), pin=a.baseline_pin), result=result,
                  phases_seconds=phases, seconds=seconds, reader_stats=bank.stats,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  code_sha256=file_sha(Path(__file__)), reference_code_sha256=file_sha(Path(__file__).with_name('native_index_reference.py')),
                  arithmetic_code_sha256=file_sha(Path(__file__).with_name('native_index_arithmetic.py')),
                  scientific_admission=False, number_status='measured', acceptance='author-checked')
    pin = write_committed(a.output/'report.json', report); print(json.dumps(dict(pin=pin, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['qualify', 'full'], required=True)
    for name in ('baseline', 'comparison', 'qualification'):
        p.add_argument('--'+name, type=Path, required=name == 'baseline')
        p.add_argument('--'+name+'-pin', required=name == 'baseline')
    p.add_argument('--lane-readback', type=Path); p.add_argument('--lane-pin')
    p.add_argument('--regression', type=Path); p.add_argument('--regression-pin')
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
