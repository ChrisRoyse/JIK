"""Read complete-answer geometry by family, source group and answer length.

Consumes completed training and its separate numerical replay. Does not fit,
generate, select a proposal, initialize CUDA or turn geometry into correctness.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from claims.process_conditions import write_committed
from text.constellation_corpus import canonical, digest, file_sha, need

PLAN = ('answer-native-plan-v1-20261023', '22993ed7704f1d31ca48df404c0af14acff1d466624b412a9c55be2132cf1098')
ARCHES = ('shared', 'own_slot', 'whole_proposals')


def read(path, pin, files):
    need(file_sha(path) == pin, 'ANSWER_METRIC_PIN')
    files[str(path)] = pin
    return json.loads(path.read_bytes())


def scores(values, count):
    need(isinstance(values, list) and len(values) == count, 'ANSWER_METRIC_SCORE_LENGTH')
    need(all(type(v) in (int, float) and math.isfinite(v) and -1 <= v <= 1 for v in values),
         'ANSWER_METRIC_SCORE_RANGE')


def validate(row, task, roster):
    need(row['family'] == task['family'] and row['source_group'] == task['source_group'],
         'ANSWER_METRIC_SOURCE')
    n, generated, ended = row['expected'], row['generated'], row['ended']
    need(n == task['expected'] and type(generated) is int and generated >= 0 and type(ended) is bool,
         'ANSWER_METRIC_DENOMINATOR')
    need(row['missing'] == max(0, n-generated) and row['extra'] == max(0, generated-n)
         and row['exact_end'] == (ended and generated == n), 'ANSWER_METRIC_DENOMINATOR')
    need(set(row['per_slot']) == set(row['copy_current']) == set(roster), 'ANSWER_METRIC_ROSTER')
    for name in roster:
        scores(row['per_slot'][name], n); scores(row['copy_current'][name], n)
        need(row['per_slot'][name][min(n, generated):] == [-1.] * row['missing'],
             'ANSWER_METRIC_MISSING_SCORE')


def extract(a):
    files = {}; source = read(a.input/'report.json', a.pin, files)
    cold = read(a.cold/'report.json', a.cold_pin, files)
    need(source['complete'] and source['mode'] == 'full' and source['device'] == 'cuda'
         and cold['complete'] and cold['all_updates_exact'] and cold['all_trajectories_exact']
         and cold['input'] == dict(path=str(a.input), pin=a.pin) and cold['code'] == source['code'],
         'ANSWER_METRIC_COLD')
    root = a.root/PLAN[0]; plan = read(root/'report.json', PLAN[1], files)
    mapping = read(root/'tasks.json', plan['files']['tasks.json']['sha256'], files)
    ref = plan['source']; parent = read(Path(ref['path'])/'report.json', ref['pin'], files)
    path = Path(ref['path'])/'answers.jsonl'
    answer_pin = file_sha(path)
    need(answer_pin == parent['files']['answers.jsonl']['sha256'], 'ANSWER_METRIC_PIN')
    files[str(path)] = answer_pin
    original = [json.loads(line) for line in path.read_bytes().splitlines()]
    tasks = {}
    for row in original:
        key = row['task']; m = mapping[key]
        need(key not in tasks and row['split'] == m['split'] and row['split_group'] == m['split_group']
             and row['population'] == m['population'] and row['status'] == 'ready'
             and [v['measurement'] for v in row['answer']['atoms']] == m['targets'], 'ANSWER_METRIC_TASK')
        tasks[key] = dict(family=row['family'], expected=len(m['targets']), split=m['split'],
                          source_group=digest(canonical([m['population'], m['split_group']])))
    need(set(tasks) == set(mapping), 'ANSWER_METRIC_TASK_COVERAGE')
    query = sorted(k for k, v in tasks.items() if v['split'] == 'query')
    need({v['source_group'] for v in tasks.values() if v['split'] == 'train'}.isdisjoint(
         tasks[k]['source_group'] for k in query), 'ANSWER_METRIC_SPLIT_OVERLAP')
    roster = plan['roster']; rows = []; coverage = {}; baseline = {}
    for arch in ARCHES:
        cp = a.cold/(arch+'.json'); cold_rows = read(cp, file_sha(cp), files)
        need(cold['results'][arch]['updates'] == source['results'][arch]['updates'], 'ANSWER_METRIC_UPDATES')
        for phase in ('initial', 'final'):
            prefix = arch+'/'+phase+'-evaluation/'; directory = a.input/prefix
            index = read(directory/'index.json', source['files'][prefix+'index.json'], files)
            need([v['task'] for v in index] == query, 'ANSWER_METRIC_INDEX_COVERAGE')
            phase_rows = []; proposals = roster if arch == 'whole_proposals' else [arch]
            # The two one-proposal architectures use their declared proposal name in the payload.
            for entry in index:
                key = entry['task']; task = tasks[key]; path = directory/(key+'.pt')
                need(entry['sha256'] == source['files'][prefix+key+'.pt']
                     and file_sha(path) == entry['sha256'], 'ANSWER_METRIC_PIN')
                files[str(path)] = entry['sha256']
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if arch != 'whole_proposals': proposals = [payload[0]['proposal']]
                available = entry['control_available']; modes = ['correct']
                if phase == 'final' and available: modes.append('shuffled')
                need(len(payload) == entry['rollouts'] and [(r['mode'], r['proposal']) for r in payload]
                     == [(m, p) for m in modes for p in proposals], 'ANSWER_METRIC_CONTROL_COVERAGE')
                coverage[arch+'|'+phase+'|'+key] = available
                for value in payload:
                    metric = value['metrics']; events = value['events']; kinds = [v['kind'] for v in events]
                    need(value['task'] == key and value['family'] == task['family']
                         and all(k in ('end', 'constellation') for k in kinds)
                         and kinds.count('end') <= 1 and ('end' not in kinds or kinds[-1] == 'end'),
                         'ANSWER_METRIC_EVENT_ORDER')
                    generated = kinds.count('constellation'); ended = 'end' in kinds
                    need(metric['generated_atoms'] == generated and metric['ended'] == ended,
                         'ANSWER_METRIC_DENOMINATOR')
                    donor = value['donor']
                    need((value['mode'] == 'correct' and donor is None) or
                         (value['mode'] == 'shuffled' and donor in tasks and tasks[donor]['family'] == task['family']
                          and tasks[donor]['source_group'] != task['source_group']
                          and len(mapping[donor]['context']) == len(mapping[key]['context'])), 'ANSWER_METRIC_DONOR')
                    row = dict(architecture=arch, phase=phase, task=key, family=value['family'],
                        source_group=task['source_group'], mode=value['mode'], proposal=value['proposal'], donor=donor,
                        expected=metric['expected_atoms'], generated=generated, ended=ended,
                        exact_end=metric['ended_at_reference_boundary'], missing=metric['missing'], extra=metric['extra'],
                        refused=value['refusal'] is not None, per_slot=metric['per_slot'], copy_current=metric['copy_current'])
                    validate(row, task, roster)
                    need(key not in baseline or baseline[key] == row['copy_current'], 'ANSWER_METRIC_BASELINE_CHANGED')
                    baseline[key] = row['copy_current']; rows.append(row); phase_rows.append(row)
            check_cold(phase_rows, cold_rows[phase], roster)
    return rows, tasks, roster, files, source, coverage


def check_cold(rows, cold, roster):
    need(len(rows) == cold['rollouts'], 'ANSWER_METRIC_COLD_COVERAGE')
    groups = defaultdict(list)
    for row in rows: groups['|'.join((row['family'], row['mode'], row['proposal']))].append(row)
    need(set(groups) == set(cold['groups']), 'ANSWER_METRIC_COLD_COVERAGE')
    for key, values in groups.items():
        c = cold['groups'][key]
        counts = dict(tasks=len(values), expected_atoms=sum(v['expected'] for v in values),
            generated_atoms=sum(v['generated'] for v in values), exact_end=sum(v['exact_end'] for v in values),
            no_end=sum(not v['ended'] for v in values), refusals=sum(v['refused'] for v in values),
            missing=sum(v['missing'] for v in values), extra=sum(v['extra'] for v in values))
        need(all(c[n] == v for n, v in counts.items()), 'ANSWER_METRIC_COLD_COUNTS')
        for n in roster:
            for field, name in (('per_slot', 'total'), ('copy_current', 'copy_total')):
                total = math.fsum(x for v in values for x in v[field][n])
                need(abs(total-c['per_slot'][n][name]) <= 1e-10, 'ANSWER_METRIC_COLD_SCORE')


def summary(values):
    # Values are task-weighted means with explicit source groups, never six-slot pooling.
    groups = defaultdict(list)
    for group, value in values: groups[group].append(value)
    means = [math.fsum(v)/len(v) for v in groups.values()]
    numbers = [v for _, v in values]
    def signs(xs): return dict(up=sum(x > 0 for x in xs), down=sum(x < 0 for x in xs), tie=sum(x == 0 for x in xs))
    return dict(tasks=len(values), groups=len(groups), task_mean=math.fsum(numbers)/len(numbers),
                group_mean=math.fsum(means)/len(means), task_sign=signs(numbers), group_sign=signs(means))


def aggregate(rows, roster, coverage):
    indexed = {}; groups = defaultdict(list)
    for row in rows:
        key = (row['architecture'], row['phase'], row['proposal'], row['task'], row['mode'])
        need(key not in indexed, 'ANSWER_METRIC_DUPLICATE'); indexed[key] = row
        for stratum in ('all', 'single' if row['expected'] == 1 else 'multiple'):
            groups['|'.join((row['architecture'], row['phase'], row['proposal'], row['family'], stratum, row['mode']))].append(row)
    out = {}
    for key, values in sorted(groups.items()):
        cell = dict(tasks=len(values), source_groups=len({r['source_group'] for r in values}),
            atoms=sum(r['expected'] for r in values), generated=sum(r['generated'] for r in values),
            exact_end=sum(r['exact_end'] for r in values), early_end=sum(r['ended'] and r['generated'] < r['expected'] for r in values),
            no_end=sum(not r['ended'] for r in values), missing=sum(r['missing'] for r in values),
            extra=sum(r['extra'] for r in values), refusals=sum(r['refused'] for r in values), per_slot={})
        for n in roster:
            score = []; copy = []; shuffled = []; initial = []
            covered_score = []; covered_copy = []; omitted_copy = []
            for row in values:
                avg = math.fsum(row['per_slot'][n])/row['expected']; group = row['source_group']
                score.append((group, avg)); copy.append((group, avg-math.fsum(row['copy_current'][n])/row['expected']))
                covered = min(row['generated'], row['expected'])
                covered_score.extend(row['per_slot'][n][:covered]); covered_copy.extend(row['copy_current'][n][:covered])
                omitted_copy.extend(row['copy_current'][n][covered:])
                if row['phase'] == 'final' and row['mode'] == 'correct':
                    k = (row['architecture'], 'initial', row['proposal'], row['task'], 'correct')
                    need(k in indexed, 'ANSWER_METRIC_INITIAL_MISSING')
                    initial.append((group, avg-math.fsum(indexed[k]['per_slot'][n])/row['expected']))
                    k = (row['architecture'], 'final', row['proposal'], row['task'], 'shuffled')
                    expected = coverage['|'.join((row['architecture'], 'final', row['task']))]
                    need((k in indexed) == expected, 'ANSWER_METRIC_DONOR_MISSING')
                    if k in indexed: shuffled.append((group, avg-math.fsum(indexed[k]['per_slot'][n])/row['expected']))
            cell['per_slot'][n] = dict(score=summary(score), minus_copy=summary(copy),
                atom_mean=math.fsum(x for r in values for x in r['per_slot'][n])/cell['atoms'],
                covered_atoms=len(covered_score),
                covered_atom_mean=math.fsum(covered_score)/len(covered_score) if covered_score else None,
                covered_copy_mean=math.fsum(covered_copy)/len(covered_copy) if covered_copy else None,
                covered_difference_per_expected_atom=math.fsum(x-y for x,y in zip(covered_score, covered_copy))/cell['atoms'],
                missing_penalty_per_expected_atom=math.fsum(-1.-x for x in omitted_copy)/cell['atoms'],
                minus_initial=summary(initial) if initial else None, minus_shuffled=summary(shuffled) if shuffled else None,
                unavailable_matched_controls=len(values)-len(shuffled) if initial else None)
        out[key] = cell
    return out


def controls(rows, tasks, roster, coverage, a):
    original = next(r for r in rows if r['missing'] == 0); task = tasks[original['task']]; n = roster[0]
    cases = []
    def reject(name, fn):
        try: fn()
        except ValueError as exc: need(name in str(exc), 'ANSWER_METRIC_CONTROL_REASON'); return str(exc)
        raise ValueError('ANSWER_METRIC_NEGATIVE_ACCEPTED:'+name)
    for field, value, code in (('source_group', 'wrong', 'SOURCE'), ('expected', original['expected']+1, 'DENOMINATOR'),
                               ('generated', original['generated']+17, 'DENOMINATOR')):
        row = deepcopy(original); row[field] = value
        cases.append(reject('ANSWER_METRIC_'+code, lambda:validate(row, task, roster)))
    for replacement, code in (([], 'SCORE_LENGTH'), ([float('nan')]*original['expected'], 'SCORE_RANGE')):
        row = deepcopy(original); row['per_slot'][n] = replacement
        cases.append(reject('ANSWER_METRIC_'+code, lambda:validate(row, task, roster)))
    cases.append(reject('ANSWER_METRIC_DUPLICATE', lambda:aggregate([original, original], roster, coverage)))
    final = next(r for r in rows if r['phase'] == 'final' and r['mode'] == 'correct')
    cases.append(reject('ANSWER_METRIC_INITIAL_MISSING', lambda:aggregate([final], roster, coverage)))
    donor_row = next(r for r in rows if r['phase'] == 'final' and r['mode'] == 'shuffled')
    missing = [r for r in rows if r is not donor_row]
    cases.append(reject('ANSWER_METRIC_DONOR_MISSING', lambda:aggregate(missing, roster, coverage)))
    subset = deepcopy([r for r in rows if r['architecture'] == 'shared' and r['phase'] == 'initial'])
    subset[0]['per_slot'][n][0] = -.999 if subset[0]['per_slot'][n][0] != -.999 else .999
    cold = json.loads((a.cold/'shared.json').read_bytes())['initial']
    cases.append(reject('ANSWER_METRIC_COLD_SCORE', lambda:check_cold(subset, cold, roster)))
    cases.append(reject('ANSWER_METRIC_PIN', lambda:read(a.input/'report.json', '0'*64, {})))
    return cases


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter(); phases = {}
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:
        rows, tasks, roster, files, source, coverage = extract(a)
        phases['authenticate_and_read'] = time.perf_counter()-tick; tick = time.perf_counter()
        tables = aggregate(rows, roster, coverage); negative = controls(rows, tasks, roster, coverage, a)
        phases['tables_controls'] = time.perf_counter()-tick; tick = time.perf_counter()
        rows_pin = write_committed(a.output/'rows.json', rows); tables_pin = write_committed(a.output/'tables.json', tables)
        # Read back persisted compact rows, then independently count every table denominator.
        saved = json.loads((a.output/'rows.json').read_bytes())
        need(saved == rows and aggregate(saved, roster, coverage) == tables, 'ANSWER_METRIC_ROUNDTRIP')
        totals = defaultdict(lambda: [0, 0])
        for row in saved:
            for label in ('all', 'single' if row['expected'] == 1 else 'multiple'):
                key = '|'.join((row['architecture'], row['phase'], row['proposal'], row['family'], label, row['mode']))
                totals[key][0] += 1; totals[key][1] += row['expected']
        need(set(totals) == set(tables) and all(v == [tables[k]['tasks'], tables[k]['atoms']]
             for k, v in totals.items()), 'ANSWER_METRIC_TABLE_DENOMINATOR')
        prior = None
        if a.previous:
            prior = read(a.previous/'report.json', a.previous_pin, {})
            need(prior['files'] == dict(rows=rows_pin, tables=tables_pin) and prior['sources'] == files
                 and prior['code'] == file_sha(Path(__file__)), 'ANSWER_METRIC_COLD_DIFFERENCE')
        need(not torch.cuda.is_initialized() and all(file_sha(Path(p)) == h for p, h in files.items()),
             'ANSWER_METRIC_FINAL_SOURCE')
        phases['persist_and_readback'] = time.perf_counter()-tick; seconds = time.perf_counter()-began
        need(.9 <= sum(phases.values())/seconds <= 1.001, 'ANSWER_METRIC_PHASE_COVERAGE')
        r = dict(schema='complete-task-answer-content-v1', complete=True, source=a.pin, cold=a.cold_pin,
            sources=files, code=file_sha(Path(__file__)), files=dict(rows=rows_pin, tables=tables_pin),
            rollouts=len(rows), roster=roster, tasks=len(tasks), query_tasks=sum(t['split'] == 'query' for t in tasks.values()),
            architectures=source['results'], controls=negative, matched_control_availability=coverage,
            previous_pin=a.previous_pin, phases=phases, seconds=seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            geometry_only=True, winner_selected=False, scientific_admission=False, capability_demonstrated=False,
            acceptance='author-checked', number_status='measured')
        pin = write_committed(a.output/'report.json', r)
        print(json.dumps(dict(pin=pin, seconds=seconds, rollouts=len(rows), tables=len(tables))), flush=True)
    except BaseException as exc: write_committed(a.output/'failure.json', dict(error=repr(exc), phases=phases)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/zfs/archive/mnlm/evidence/fsv/29'))
    for n in ('input', 'cold', 'output'): p.add_argument('--'+n, type=Path, required=True)
    for n in ('pin', 'cold-pin'): p.add_argument('--'+n, required=True)
    p.add_argument('--previous', type=Path); p.add_argument('--previous-pin'); run(p.parse_args())
