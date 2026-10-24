"""Pinned prerequisites and unchanged worker commands for complete-answer priority."""
from copy import deepcopy
import json
from pathlib import Path

from audit import answer_priority_queue as prior

REPO = Path(__file__).resolve().parents[2]
sha, need, read, write, identity = prior.sha, prior.need, prior.read, prior.write, prior.identity
DRIVERS = dict(priority='answer_priority_queue.py', consumer='canonical_answer_consumer_queue.py',
               task='task_answer_training_queue.py')
WAITING = dict(priority='waiting_for_runtime', consumer='waiting_for_answer_cold', task='waiting_for_phase88')
ORDER = ('native', 'bind', 'bind-cold', 'task-proof', 'task-proof-cold', 'task-full', 'task-cold',
         'surface-proof', 'surface-reverse', 'surface-cold', 'inversion-proof',
         'inversion-proof-cold', 'inversion-full', 'inversion-cold', 'grouped')
REQUIRES = {
    'native': [], 'bind': ['native'], 'bind-cold': ['bind'], 'task-proof': ['bind-cold'],
    'task-proof-cold': ['task-proof'], 'task-full': ['bind-cold', 'task-proof-cold'],
    'task-cold': ['task-full'], 'surface-proof': [], 'surface-reverse': ['surface-proof'],
    'surface-cold': ['surface-proof', 'surface-reverse'], 'inversion-proof': ['surface-cold'],
    'inversion-proof-cold': ['inversion-proof'], 'inversion-full': ['surface-cold', 'inversion-proof-cold'],
    'inversion-cold': ['inversion-full'], 'grouped': []}


def report(ref):
    return read(Path(ref['path'])/'report.json', ref['pin'])


def authenticate(path, pin):
    c = read(path, pin)
    need(c['schema'] == 'complete-answer-priority-v1', 'COMPLETE_QUEUE_SCHEMA')
    p, fresh_prior = prior.authenticate(Path(c['priority']['path']), c['priority']['pin'])
    q = read(c['consumer']['path'], c['consumer']['pin'])
    t = read(c['task']['path'], c['task']['pin'])
    i = read(t['inversion']['path'], t['inversion']['pin'])
    need(q['schema'] == 'canonical-answer-consumer-queue-v1' and
         t['schema'] == 'complete-task-answer-queue-v1' and i['schema'] == 'legal-inversion-queue-v1',
         'COMPLETE_QUEUE_SOURCE_SCHEMA')
    need(q['wait'] == t['wait'] == i['wait'] == c['old']['priority'] and
         t['consumer'] == c['old']['consumer'] and c['runtime'] == p['runtime'], 'COMPLETE_QUEUE_CHAIN')
    need(q['answer_authority'] == p['answer_authority']['pin'] and
         Path(q['answer_root']) == Path(q['wait']['path'])/'answer', 'COMPLETE_QUEUE_NATIVE_BINDING')
    for name in DRIVERS:
        prior.check_owner(c['old'][name])
        launch = json.loads((Path(c['old'][name]['path'])/'launch.json').read_bytes())
        need(launch['authority'] == c[name]['pin'] and
             launch['driver_pin'] == sha(REPO/'scripts/audit'/DRIVERS[name]), 'COMPLETE_QUEUE_OWNER_SOURCE')
    qp, qr = report(q['proofs']['plan']), report(q['proofs']['readback'])
    need(qp['complete'] and qp['mode'] == 'plan' and not qp['experimental_items_ready'] and qr['complete']
         and qr['source']['pin'] == q['proofs']['plan']['pin'] and qr['summary'] == qp['summary']
         and qp['code'] == q['code'], 'COMPLETE_QUEUE_CONSUMER_PROOF')
    tr, ir = report(t['proofs']['readback']), report(i['proofs']['readback'])
    need(tr['complete'] and tr['device'] == 'cpu' and tr['source_mode'] == 'proof' and
         tr['all_updates_exact'] and tr['all_trajectories_exact'] and tr['code'] == t['code'],
         'COMPLETE_QUEUE_TASK_PROOF')
    need(ir['complete'] and ir['restored_midpoint_continuation_exact'] and ir['every_evaluation_byte_exact']
         and ir['cold_optimizer_updates'] == 3, 'COMPLETE_QUEUE_INVERSION_PROOF')
    sources = dict(priority=p, consumer=q, task=t, inversion=i)
    def fresh():
        need(sha(path) == pin, 'COMPLETE_QUEUE_AUTHORITY_CHANGED'); fresh_prior()
        for name in ('priority', 'consumer', 'task'):
            need(sha(c[name]['path']) == c[name]['pin'], 'COMPLETE_QUEUE_SOURCE_CHANGED')
        need(sha(t['inversion']['path']) == t['inversion']['pin'], 'COMPLETE_QUEUE_INVERSION_CHANGED')
        for contract in (c, q, t, i):
            for n, h in contract['code'].items():
                need(sha(REPO/'scripts'/n) == h, 'COMPLETE_QUEUE_CODE_CHANGED')
            for ref in contract.get('proofs', {}).values(): report(ref)
    fresh(); return c, sources, fresh


def plan(sources):
    p = sources['priority']; answer = p['answer_authority']; grouped = p['authorities']['grouped']
    h = p['grouped_handoff']; inversion = sources['inversion']['proofs']['plan']
    common = ['--plan', inversion['path'], '--plan-pin', inversion['pin']]
    commands = [
        ('answer_native_queue.py', ['--mode', 'full', '--authority', answer['path'], '--pin', answer['pin']]),
        ('canonical_answer_consumer.py', ['--mode', 'bind', '--native', '@native.native.path', '--native-pin',
          '@native.native.pin', '--cold', '@native.cold.path', '--cold-pin', '@native.cold.pin']),
        ('canonical_answer_consumer_readback.py', ['--input', '@bind.path', '--pin', '@bind.pin']),
        ('task_answer_training.py', ['--mode', 'proof']),
        ('task_answer_training_readback.py', ['--input', '@task-proof.path', '--pin', '@task-proof.pin']),
        ('task_answer_training.py', ['--mode', 'full', '--consumer', '@bind.path', '--consumer-pin', '@bind.pin',
          '--consumer-cold', '@bind-cold.path', '--consumer-cold-pin', '@bind-cold.pin',
          '--qualification', '@task-proof-cold.path', '--qualification-pin', '@task-proof-cold.pin']),
        ('task_answer_training_readback.py', ['--input', '@task-full.path', '--pin', '@task-full.pin']),
        ('surface_device_proof.py', []), ('surface_device_proof.py', ['--reverse']),
        ('surface_device_readback.py', ['--input', '@surface-proof.path', '--input-pin', '@surface-proof.pin',
          '--cold', '@surface-reverse.path', '--cold-pin', '@surface-reverse.pin']),
        ('legal_surface_inversion.py', [*common, '--mode', 'proof']),
        ('legal_surface_inversion_readback.py', [*common, '--input', '@inversion-proof.path',
          '--input-pin', '@inversion-proof.pin']),
        ('legal_surface_inversion.py', [*common, '--mode', 'full', '--qualification', '@inversion-proof-cold.path',
          '--qualification-pin', '@inversion-proof-cold.pin']),
        ('legal_surface_inversion_readback.py', [*common, '--input', '@inversion-full.path', '--input-pin', '@inversion-full.pin']),
        ('legal_grouped_placement_queue.py', ['--mode', 'full', '--authority', grouped['path'], '--authority-pin',
          grouped['pin'], '--handoff', h['path'], '--handoff-pin', h['pin']])]
    return [dict(label=label, script=script, arguments=args, requires=REQUIRES[label],
                 cpu_only=label in ('bind', 'bind-cold'))
            for label, (script, args) in zip(ORDER, commands, strict=True)]


def validate_plan(rows):
    need(tuple(v['label'] for v in rows) == ORDER, 'COMPLETE_QUEUE_ORDER')
    seen = set()
    for row in rows:
        need(row['requires'] == REQUIRES[row['label']] and set(row['requires']) <= seen,
             'COMPLETE_QUEUE_PREREQUISITE')
        for arg in row['arguments']:
            if arg.startswith('@'): need(arg[1:].split('.')[0] in seen, 'COMPLETE_QUEUE_FUTURE_INPUT')
        seen.add(row['label'])


def output_check(label, ref, sources):
    r = report(ref); need(r['complete'], 'COMPLETE_QUEUE_OUTPUT_INCOMPLETE:'+label)
    if label == 'native':
        need(r['authority'] == sources['priority']['answer_authority']['pin'] and
             r['driver_pin'] == sha(REPO/'scripts/audit/answer_native_queue.py'), 'COMPLETE_QUEUE_NATIVE_OUTPUT')
        need({v['label'] for v in r['outputs']} == {'native', 'cold'}, 'COMPLETE_QUEUE_NATIVE_SCOPE')
    if label in ('native', 'grouped'):
        for child in r['outputs']: need(report(child)['complete'], 'COMPLETE_QUEUE_CHILD_OUTPUT')
    if label == 'bind-cold':
        need(r['all_native_inputs_checked'] and r['all_learner_items_checked'], 'COMPLETE_QUEUE_CONSUMER_COVERAGE')
    if label == 'surface-cold':
        need(r['all_cold_arm_states_exact'] and r['same_cohort_exact'], 'COMPLETE_QUEUE_CUDA_ARITHMETIC')
    return r


def arguments(row, results, sources):
    need(set(row['requires']) <= set(results), 'COMPLETE_QUEUE_MISSING_PREREQUISITE')
    for name in row['requires']: output_check(name, results[name], sources)
    resolved = []
    for arg in row['arguments']:
        if not arg.startswith('@'): resolved.append(arg); continue
        parts = arg[1:].split('.'); ref = results[parts[0]]
        r = output_check(parts[0], ref, sources)
        if len(parts) == 3: ref = next(v for v in r['outputs'] if v['label'] == parts[1])
        need(parts[-1] in ('path', 'pin'), 'COMPLETE_QUEUE_ARGUMENT_FIELD')
        resolved.append(ref[parts[-1]])
    return resolved


def idle(row, ref, name):
    need(row['pid'] == ref['pid'] and row['starttime'] == ref['starttime'], 'COMPLETE_QUEUE_IDLE_IDENTITY')
    need(row['state'] in ('S', 'T') and not row['children'], 'COMPLETE_QUEUE_ACTIVE_CHILD')
    need(row['events'] and all(e['stage'] == WAITING[name] for e in row['events']), 'COMPLETE_QUEUE_ADVANCED')
    need(not row['terminal'], 'COMPLETE_QUEUE_TERMINAL')
    need(str(REPO/'scripts/audit'/DRIVERS[name]) in row['command'] and
         row['driver_pin'] == sha(REPO/'scripts/audit'/DRIVERS[name]), 'COMPLETE_QUEUE_DRIVER')


def snapshot(ref, name):
    prior.check_owner(ref); root = Path(ref['path']); row = prior.process(ref['pid'])
    row.update(events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()],
               terminal=any((root/n).exists() for n in ('report.json', 'failure.json')),
               driver_pin=json.loads((root/'launch.json').read_bytes())['driver_pin'], files=prior.files(root))
    idle(row, ref, name); return row


def controls(rows, c, work):
    result = {}
    def reject(name, fn):
        try: fn()
        except (ValueError, KeyError) as e: result[name] = str(e)
        else: raise ValueError('COMPLETE_QUEUE_CONTROL_ACCEPTED:'+name)
    for name, original in rows.items():
        for field, value in [('pid', -1), ('starttime', '0'), ('children', [1]), ('state', 'R'),
                             ('events', [{'stage': 'running'}]), ('terminal', True),
                             ('driver_pin', '0'*64), ('command', [])]:
            bad = deepcopy(original); bad[field] = value
            reject(name+':'+field, lambda: idle(bad, c['old'][name], name))
    bad = deepcopy(work); bad.reverse(); reject('plan:reordered', lambda: validate_plan(bad))
    for label in ('task-full', 'inversion-full', 'surface-cold', 'bind-cold'):
        bad = deepcopy(work); next(v for v in bad if v['label'] == label)['requires'] = []
        reject('plan:missing:'+label, lambda: validate_plan(bad))
    return result
