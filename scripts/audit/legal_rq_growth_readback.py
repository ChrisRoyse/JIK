"""Cold continuation, full output replay, and separate counts for one growth arm.

Reads only a completed, hash-bound arm; it never writes into the fit directory.
Cost: O(update ledger + evaluated native windows), with the qualified tiled reader.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import resource
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'lens')]
import numpy as np
import torch
from audit.legal_rq_growth import (POLICY, components_for, evaluate, loss_reference,
                                  make, pins, pinned, step)
from audit.legal_stock_readback import rejected
from claims.process_conditions import write_committed
from model.legal_learning_source import LegalLearningSource
from model.loop_training_checkpoint import tensor_fingerprint
from model.native_rq_data import NativeRQData, training_components
from senses.contract import digest
from text.constellation_corpus import file_sha, need
from text.legal_continuation_input import Continuations
from text.legal_reuse_input import ReuseInputs


def check_checkpoint(saved, selection_pin, statistic_pin, number):
    need(saved['selection_pin'] == selection_pin and saved['statistic_pin'] == statistic_pin
         and saved['step'] == number, 'RQ_GROWTH_COLD_CHECKPOINT_BINDING')


def check_chain(records, uids):
    need(len(records) == POLICY['updates'], 'RQ_GROWTH_COLD_LEDGER_COUNT')
    visits = Counter()
    for i, row in enumerate(records):
        expected = [uids[(8*i+j) % len(uids)] for j in range(8)]
        need(row['step'] == i+1 and row['ids'] == expected, 'RQ_GROWTH_COLD_ROW_VISITS')
        if i:
            need(row['before'] == records[i-1]['after'], 'RQ_GROWTH_COLD_CHAIN')
        visits.update(row['ids'])
    need(set(visits) == set(uids), 'RQ_GROWTH_COLD_COVERAGE')
    return dict(updates=len(records), distinct_inputs=len(visits), row_visits=sum(visits.values()),
                least_visits=min(visits.values()), most_visits=max(visits.values()))


def recount(report, codes, errors, eligible, windows, roster):
    n = report['rows']
    need(codes.shape == (n, 3) and errors.shape == (n, len(roster), 4)
         and eligible.shape == (n,) and eligible.dtype == np.bool_
         and windows.shape == (n, len(roster)) and bool((windows > 0).all())
         and bool(np.isfinite(errors).all()), 'RQ_GROWTH_COLD_METRIC_SHAPE')
    unique = set(tuple(map(int, row)) for row in codes)
    usage = [[sum(int(row[j]) == k for row in codes) for k in range(size)]
             for j, size in enumerate(POLICY['sizes'])]
    contingency = {}
    for count, code in zip(windows, codes):
        key = ','.join(map(str, count)); contingency.setdefault(key, [0]*POLICY['sizes'][0])
        contingency[key][int(code[0])] += 1
    need(len(unique) == report['distinct_codes'] and usage == report['usage']
         and contingency == report['first_code_by_window_counts'], 'RQ_GROWTH_COLD_CODE_COUNTS')
    maximum = 0.; result = {}
    for label in ['all', 'single_window', 'multiwindow']:
        rows = [i for i, row in enumerate(windows) if label == 'all'
                or (label == 'single_window' and all(int(x) == 1 for x in row))
                or (label == 'multiwindow' and any(int(x) > 1 for x in row))]
        supported = [i for i in rows if bool(eligible[i])]
        saved = report['slices'][label]
        need(saved['rows'] == len(rows) and saved['contrast_rows'] == len(supported),
             'RQ_GROWTH_COLD_SLICE_DENOMINATOR')
        result[label] = dict(rows=len(rows), contrast_rows=len(supported), per_lane={})
        if not rows:
            need(saved['per_lane'] == {}, 'RQ_GROWTH_COLD_EMPTY_SLICE'); continue
        for j, name in enumerate(roster):
            values = {key: math.fsum(float(errors[i, j, k]) for i in rows)/len(rows)
                      for k, key in enumerate(['mse', 'mean', 'position'])}
            values['shuffled_minus_correct'] = (
                math.fsum(float(errors[i, j, 3])-float(errors[i, j, 0]) for i in supported)/len(supported)
                if supported else None)
            for key, value in values.items():
                if value is None:
                    need(saved['per_lane'][name][key] is None, 'RQ_GROWTH_COLD_EMPTY_CONTRAST')
                else:
                    error = abs(value-saved['per_lane'][name][key]); maximum = max(maximum, error)
                    need(error <= POLICY['arithmetic_error'], 'RQ_GROWTH_COLD_METRIC_RECOUNT')
            result[label]['per_lane'][name] = values
    return dict(distinct_codes=len(unique), maximum_metric_recount_error=maximum, slices=result)


def run(a):
    start = tick = time.perf_counter(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    qualified = pinned(a.qualification/'report.json', a.qualification_pin)
    need(qualified['complete'] and qualified['mode'] == 'qualify' and qualified['code'] == pins()
         and qualified['policy'] == POLICY, 'RQ_GROWTH_COLD_QUALIFICATION')
    selection = pinned(a.fit/'selection.json', qualified['selection_pin'])
    need(selection == pinned(a.qualification/'selection.json', qualified['selection_pin']),
         'RQ_GROWTH_COLD_SELECTION')
    arm_root = a.fit/a.arm; arm = pinned(arm_root/'report.json', a.arm_pin)
    population, recipe = a.arm.split('-', 1)
    need(population in selection['populations'] and recipe in POLICY['recipes'], 'RQ_GROWTH_COLD_ARM')
    baseline = pinned(a.baseline/'report.json', a.baseline_pin)
    source = ReuseInputs(baseline['source']['path'], baseline['source']['pin'])
    continuation = Continuations(Path(baseline['continuations']['path']), baseline['continuations']['pin'])
    split, cold = continuation.report['split'], continuation.report['cold']; bank = baseline['bank']
    data = LegalLearningSource(source, split['path'], split['pin'], cold['path'], cold['pin'], bank['path'], bank['pin'])
    need(data.binding == qualified['source_binding'] == selection['source_binding'], 'RQ_GROWTH_COLD_SOURCE')
    reader = NativeRQData(data, Path(baseline['proof']['path']), baseline['proof']['pin'])
    stat_path = a.qualification/(population+'-statistics.pt')
    need(file_sha(stat_path) == qualified['statistic_pins'][population], 'RQ_GROWTH_COLD_STATISTICS')
    stats = torch.load(stat_path, map_location='cpu', weights_only=True)
    uids = selection['populations'][population]; need(stats['uids'] == uids, 'RQ_GROWTH_COLD_FIT_SCOPE')
    _, cases = training_components(data, uids, continuation.presence)
    for name, pin in arm['files'].items():
        need(file_sha(arm_root/name) == pin, 'RQ_GROWTH_COLD_ARTIFACT_PIN')
    records = [json.loads(line) for line in (arm_root/'updates.jsonl').read_text().splitlines()]
    visits = check_chain(records, uids); need(records[-1] == arm['last_update'], 'RQ_GROWTH_COLD_LAST_UPDATE')
    resume = torch.load(arm_root/'resume.pt', map_location='cpu', weights_only=True)
    final = torch.load(arm_root/'final.pt', map_location='cpu', weights_only=True)
    for saved, number in [(resume, POLICY['updates']-1), (final, POLICY['updates'])]:
        check_checkpoint(saved, qualified['selection_pin'], qualified['statistic_pins'][population], number)
        need(digest(tensor_fingerprint(saved['model'])) == records[number-1]['after']
             and digest(tensor_fingerprint(saved['optimizer'])) == records[number-1]['optimizer'],
             'RQ_GROWTH_COLD_CHECKPOINT_STATE')
    phases = dict(source_and_artifacts=time.perf_counter()-tick); tick = time.perf_counter()
    model, optimizer = make(qualified['dimensions'], recipe, stats)
    model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
    actual = step(model, optimizer, reader, uids, cases, POLICY['updates']-1)
    need(actual == records[-1] and tensor_fingerprint(model.state_dict()) == tensor_fingerprint(final['model'])
         and tensor_fingerprint(optimizer.state_dict()) == tensor_fingerprint(final['optimizer']),
         'RQ_GROWTH_COLD_CONTINUATION')
    loss = loss_reference(model, reader, uids, cases)
    phases['cold_step_and_independent_loss'] = time.perf_counter()-tick; tick = time.perf_counter()
    groups = {'train': (uids, cases)}
    for name, side in [('development', 1), ('query', 2)]:
        ids = selection['held'][name]; groups[name] = (ids, components_for(data, ids, side))
    measured = {}; first_arrays = None
    for name, (ids, components) in groups.items():
        prior = pinned(arm_root/name/'report.json', arm['evaluation'][name]['pin'])
        need(prior['uids'] == ids, 'RQ_GROWTH_COLD_EVALUATION_SCOPE')
        fresh = evaluate(model, reader, ids, components, stats, a.output/name)
        need(fresh['pin'] == arm['evaluation'][name]['pin'], 'RQ_GROWTH_COLD_OUTPUT_REPLAY')
        arrays = {key: np.load(a.output/name/key, allow_pickle=False)
                  for key in ['codes.npy', 'errors.npy', 'eligible.npy', 'windows.npy']}
        for key, value in arrays.items():
            need(file_sha(arm_root/name/key) == prior['files'][key]
                 and np.array_equal(value, np.load(arm_root/name/key, allow_pickle=False)),
                 'RQ_GROWTH_COLD_ARRAY_REPLAY')
        counted = recount(prior, *(arrays[key] for key in ['codes.npy', 'errors.npy', 'eligible.npy', 'windows.npy']), reader.roster)
        measured[name] = dict(report_pin=fresh['pin'], all_native_and_decoded_bytes_exact=True,
                              distinct_case_components=len(set(components)), **counted)
        if name == 'train':
            need(fresh['native_sha256'] == stats['native_sha256'], 'RQ_GROWTH_COLD_NATIVE')
            first_arrays = prior, arrays
    phases['complete_output_replay_and_recount'] = time.perf_counter()-tick; tick = time.perf_counter()
    wrong = dict(resume, selection_pin='0'*64)
    controls = dict(wrong_checkpoint=rejected(lambda: check_checkpoint(wrong, qualified['selection_pin'],
                    qualified['statistic_pins'][population], POLICY['updates']-1)),
                    wrong_arm_pin=rejected(lambda: pinned(arm_root/'report.json', '0'*64)))
    changed = deepcopy(records[:2]); changed[1]['before'] = '0'*64
    # Full length is retained so the chain check, not the length check, must refuse.
    changed.extend(records[2:]); controls['broken_update_chain'] = rejected(lambda: check_chain(changed, uids))
    prior, arrays = first_arrays; altered = deepcopy(prior)
    altered['slices']['all']['per_lane'][reader.roster[0]]['mse'] += 1
    controls['changed_metric'] = rejected(lambda: recount(altered, *(arrays[k] for k in
        ['codes.npy', 'errors.npy', 'eligible.npy', 'windows.npy']), reader.roster))
    data.fresh(); source.fresh(); continuation.fresh(); model.check_statistics()
    phases['negative_controls_and_freshness'] = time.perf_counter()-tick
    seconds = time.perf_counter()-start; need(sum(phases.values())/seconds >= .9, 'RQ_GROWTH_COLD_PHASE_ATTRIBUTION')
    report = dict(schema='legal-rq-growth-cold-v1', complete=True, arm=a.arm, fit=str(a.fit), arm_pin=a.arm_pin,
        qualification=dict(path=str(a.qualification), pin=a.qualification_pin), code=pins(),
        reader_code_pin=file_sha(Path(__file__)), selection_pin=qualified['selection_pin'], source_binding=data.binding,
        visits=visits, final_weight_pin=actual['after'], final_optimizer_pin=actual['optimizer'], cold_step_exact=True,
        independent_loss=loss, evaluation=measured, controls=controls, phases_seconds=phases, seconds=seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_promotion=False, scientific_admission=False, number_status='measured', acceptance='author-checked')
    pin = write_committed(a.output/'report.json', report); print(json.dumps(dict(arm=a.arm, report_pin=pin, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'qualification']:
        p.add_argument('--'+name, type=Path, required=True); p.add_argument('--'+name+'-pin', required=True)
    p.add_argument('--fit', type=Path, required=True); p.add_argument('--arm', required=True); p.add_argument('--arm-pin', required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
