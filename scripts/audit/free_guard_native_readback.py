"""Cold native-array recount, ordered per-slot guard, and independent scalar readback."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import numpy as np
import torch
from audit.capability_surface_guard_fsv import independent_scores
from claims.process_conditions import write_committed as write
from model.capability_surface_guard import POLICY, evaluate
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from text.answer_native_measure import arrays
from text.answer_native_plan import read
from text.canonical_windows import exact_reference
from text.constellation_corpus import canonical, digest, file_sha, need
from text.constellation_input import stamp
from text.free_guard_native_plan import FreeGuardNativePlan, code


def load(a):
    plan = FreeGuardNativePlan(a.plan, a.pin)
    stamps = {root / 'report.json': stamp(root / 'report.json') for root in (a.input, a.cold)}
    first = read(a.input / 'report.json', a.input_pin); cold = read(a.cold / 'report.json', a.cold_pin)
    for value in (first, cold):
        need(value['schema'] == 'free-guard-native-measurements-v1' and value['complete'] and not value['errors']
             and value['plan'] == dict(path=str(a.plan), pin=a.pin) and value['code'] == code()
             and value['roster'] == plan.report['roster'] and value['inputs'] == len(plan.records)
             and value['gpu_used'] and not value['serving'], 'FREE_NATIVE_READBACK_SCOPE')
    need(first['cold_reference'] is None and cold['cold_reference'] == dict(path=str(a.input), pin=a.input_pin)
         and cold['cold_all_inputs_exact'], 'FREE_NATIVE_READBACK_COLD_BINDING')
    readings = {}; identities = {s['name']: s['identity'] for s in plan.report['specs']}
    for name in first['roster']:
        ref = first['instruments'][name]; other = cold['instruments'][name]
        own = read(a.input / name / 'report.json', ref['pin']); c = read(a.cold / name / 'report.json', other['pin'])
        for root, report in ((a.input / name, own), (a.cold / name, c)):
            for filename in ('report.json', *report['files']):
                stamps[root / filename] = stamp(root / filename)
        expected = digest(canonical(plan.report['contracts'][name]))
        need(own['contract'] == c['contract'] == plan.report['contracts'][name]
             and own['lens_id'] == c['lens_id'] == ref['lens_id'] == other['lens_id'] == expected
             and own['all_windows_retained'] and c['all_windows_retained']
             and c['controls']['all_cold_inputs_exact'], 'FREE_NATIVE_READBACK_INSTRUMENT')
        v, b, t = arrays(a.input / name, own); cv, cb, ct = arrays(a.cold / name, c)
        need(v.dtype == np.float32 and v.shape == (plan.tokens[name]['total_windows'], 768)
             and b.dtype == np.int64 and b.tolist() == np.r_[0, np.cumsum(plan.tokens[name]['windows'])].tolist()
             and np.isfinite(v).all(), 'FREE_NATIVE_READBACK_SHAPE')
        exact_reference(v, b, t, cv, cb, ct, list(range(len(plan.records))))
        for i, record in enumerate(plan.records):
            trace = t[i]
            need(trace['surface_sha256'] == record['id'] and len(trace['windows']) == int(b[i+1] - b[i]),
                 'FREE_NATIVE_READBACK_TEXT')
        readings[name] = (v, b, t)
    def fresh():
        need(all(stamp(path) == value for path, value in stamps.items()), 'FREE_NATIVE_READBACK_CHANGED')
    fresh()
    return plan, readings, identities, fresh


def observations(plan, readings):
    ref = plan.report['source']; root = Path(ref['path']); source = read(root / 'report.json', ref['pin'])
    need(file_sha(root / 'occurrences.jsonl') == source['files']['occurrences']['sha256'], 'FREE_NATIVE_GUARD_OCCURRENCES')
    occurrences = [json.loads(line) for line in (root / 'occurrences.jsonl').read_bytes().splitlines()]
    positions = {row['id']: i for i, row in enumerate(plan.records)}; groups = {}
    for row in occurrences:
        binding = row['binding']; key = (binding['trace_path'], binding['trace'])
        groups.setdefault(key, {}).setdefault(binding['atom'], []).append(row)
    found = 0
    for (path, pin), wanted in sorted(groups.items()):
        for event in replay(path, pin):
            if event['kind'] == 'end':
                continue
            for row in wanted.get(event['index'], []):
                need(row['binding']['prediction'] == fingerprint(event), 'FREE_NATIVE_GUARD_PREDICTION')
                i = positions[row['measurement_input']]; raw = plan.records[i]['text'].encode('utf-8')
                need(digest(raw) == row['sha256'] and len(raw) == row['bytes'], 'FREE_NATIVE_GUARD_TEXT')
                measured = {n: v[b[i]:b[i+1]] for n, (v, b, _) in readings.items()}
                predicted = {n: value.numpy() for n, value in event['slots'].items()}
                need(row['binding']['lens_ids'] == {n: digest(canonical(c)) for n, c in plan.report['contracts'].items()},
                     'FREE_NATIVE_GUARD_LENS_IDS')
                found += 1; yield row, raw, measured, predicted
    need(found == source['counts']['completed_atom_occurrences'], 'FREE_NATIVE_GUARD_DENOMINATOR')


def run(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False)
    plan, readings, identities, fresh = load(a)
    phases = dict(native_cold_readback=time.perf_counter() - tick); tick = time.perf_counter(); count = 0
    with (a.output / 'scores.jsonl').open('xb') as output:
        for row, raw, measured, predicted in observations(plan, readings):
            receipt = dict(source_sha256=digest(raw), source_bytes=len(raw), instruments=identities,
                slots={n: dict(shape=list(v.shape), sha256=digest(v.astype('<f4', copy=False).tobytes())) for n, v in measured.items()})
            try:
                guard = evaluate(raw, measured, predicted, identities, receipt, policy=POLICY)
            except ValueError as exc:
                if 'SURFACE_GUARD_ZERO_NORM' not in str(exc):
                    raise
                guard = dict(accepted=False, serving=False, geometric_conjunction=False, refusal=str(exc))
            value = dict(binding=row['binding'], measurement_input=row['measurement_input'], receipt=receipt, guard=guard)
            output.write(canonical(value) + b'\n'); count += 1
        output.flush(); os.fsync(output.fileno())
    score_pin = file_sha(a.output / 'scores.jsonl')
    phases['per_slot_guard_and_persist'] = time.perf_counter() - tick; tick = time.perf_counter()
    maximum = 0.; cells = 0; per_arm = {}; last = None
    with (a.output / 'scores.jsonl').open('rb') as stream:
        for row, raw, measured, predicted in observations(plan, readings):
            saved = json.loads(next(stream)); need(saved['binding'] == row['binding'], 'FREE_NATIVE_GUARD_SCORE_JOIN')
            guard = saved['guard']; arm = row['binding']['arm']; counts = per_arm.setdefault(arm, dict(scored=0, geometric_passes=0, refused=0))
            counts['scored'] += 1; need(not guard['accepted'] and not guard['serving'], 'FREE_NATIVE_GUARD_FALSE_ACCEPT')
            last = (raw, measured, predicted, saved['receipt'])
            if 'refusal' in guard:
                need(any((np.linalg.norm(v, axis=1) <= 1e-10).any() for v in predicted.values()), 'FREE_NATIVE_GUARD_FALSE_REFUSAL')
                counts['refused'] += 1; continue
            passes = []
            for name in sorted(identities):
                item = guard['slots'][name]
                maximum = max(maximum, independent_scores(measured[name], predicted[name], item)); cells += 1
                need(item['threshold'] == POLICY['threshold'] and item['passes'] == (item['score'] >= POLICY['threshold']),
                     'FREE_NATIVE_GUARD_THRESHOLD')
                passes.append(item['passes'])
            need(guard['geometric_conjunction'] == all(passes), 'FREE_NATIVE_GUARD_CONJUNCTION')
            counts['geometric_passes'] += int(all(passes))
        need(stream.read(1) == b'', 'FREE_NATIVE_GUARD_EXTRA_SCORES')
    controls = {}
    def reject(name, fn, expected):
        try:
            fn()
        except ValueError as exc:
            need(expected in str(exc), 'FREE_NATIVE_GUARD_WRONG_ERROR'); controls[name] = str(exc)
        else:
            raise ValueError('FREE_NATIVE_GUARD_CONTROL_PASSED:' + name)
    if last is not None:
        raw, measured, predicted, receipt = last
        reject('changed_text', lambda: evaluate(raw + b'x', measured, predicted, identities, receipt, policy=POLICY), 'SURFACE_GUARD_SOURCE')
        bad = dict(measured); bad.pop(next(iter(bad)))
        reject('missing_slot', lambda: evaluate(raw, bad, predicted, identities, receipt, policy=POLICY), 'SURFACE_GUARD_MISSING_SLOT')
        bad_receipt = deepcopy(receipt); bad_receipt['instruments'][next(iter(identities))] = '0' * 64
        reject('wrong_instrument', lambda: evaluate(raw, measured, predicted, identities, bad_receipt, policy=POLICY), 'SURFACE_GUARD_INSTRUMENT')
    fresh(); plan.fresh(); need(file_sha(a.output / 'scores.jsonl') == score_pin and plan.report['code'] == code(), 'FREE_NATIVE_GUARD_CHANGED')
    need(not torch.cuda.is_initialized(), 'FREE_NATIVE_GUARD_CPU_ONLY')
    phases['independent_scalar_readback_controls'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
    need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_NATIVE_GUARD_PHASES')
    result = dict(complete=True, plan=dict(path=str(a.plan), pin=a.pin), input=dict(path=str(a.input), pin=a.input_pin),
        cold=dict(path=str(a.cold), pin=a.cold_pin), code=code(), scores=score_pin, scored_atoms=count,
        native_cold_all_exact=True, independent_scalar_cells=cells, max_scalar_error=maximum, per_arm=per_arm,
        source_counts=plan.report['source_counts'], controls=controls, policy=POLICY,
        incomplete_answers_remain_refused=True, accepted_answers=0, semantic_validity_established=False,
        serving=False, capability_demonstrated=False, phases=phases, seconds=seconds, acceptance='author-checked')
    pin = write(a.output / 'report.json', result)
    print(json.dumps(dict(complete=True, pin=pin, scored=count, per_arm=per_arm, cells=cells, seconds=seconds)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('plan', 'input', 'cold', 'output'):
        p.add_argument('--' + n, type=Path, required=True)
    for n in ('pin', 'input-pin', 'cold-pin'):
        p.add_argument('--' + n, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    try:
        run(a)
    except BaseException as exc:
        if a.output.exists():
            write(a.output / 'failure.json', dict(error=repr(exc)))
        raise
