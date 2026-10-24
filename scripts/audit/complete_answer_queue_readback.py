"""Independent physical readback of a queue plan and an optional idle-owner handoff."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.complete_answer_schedule import authenticate


def need(ok, code):
    if not ok: raise ValueError(code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def load(path, pin=None):
    if pin: need(sha(path) == pin, 'ANSWER_QUEUE_COLD_PIN')
    return json.loads(Path(path).read_bytes())


def process(pid):
    root = Path('/proc')/str(pid)
    if not root.exists(): return None
    fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttime=fields[19], state=fields[0],
                children=(root/'task'/str(pid)/'children').read_text().split(),
                command=(root/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))


def file_map(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def graph(rows):
    nodes = {r['label']: r for r in rows}; labels = [r['label'] for r in rows]
    need(len(nodes) == len(rows) == 15, 'ANSWER_QUEUE_COLD_NODE_COUNT')
    chains = [('native', 'bind', 'bind-cold', 'task-proof', 'task-proof-cold', 'task-full', 'task-cold'),
              ('surface-proof', 'surface-reverse', 'surface-cold', 'inversion-proof',
               'inversion-proof-cold', 'inversion-full', 'inversion-cold')]
    need(labels == [*chains[0], *chains[1], 'grouped'], 'ANSWER_QUEUE_COLD_ORDER')
    for chain in chains:
        for before, after in zip(chain, chain[1:]):
            need(before in nodes[after]['requires'], 'ANSWER_QUEUE_COLD_COLD_BARRIER')
    for first, second in [('bind-cold', 'task-full'), ('surface-proof', 'surface-cold'),
                          ('surface-cold', 'inversion-full')]:
        need(first in nodes[second]['requires'], 'ANSWER_QUEUE_COLD_REQUIRED_PROOF')
    need(nodes['grouped']['requires'] == nodes['surface-proof']['requires'] == [],
         'ANSWER_QUEUE_COLD_INDEPENDENT_WORK')
    for index, row in enumerate(rows):
        need(all(labels.index(n) < index for n in row['requires']), 'ANSWER_QUEUE_COLD_DAG')
        for arg in row['arguments']:
            if arg.startswith('@'): need(labels.index(arg[1:].split('.')[0]) < index, 'ANSWER_QUEUE_COLD_ARGUMENT_ORDER')
    need(nodes['task-full']['arguments'][-4:] == ['--qualification', '@task-proof-cold.path',
         '--qualification-pin', '@task-proof-cold.pin'], 'ANSWER_QUEUE_COLD_TASK_QUALIFICATION')
    need(nodes['inversion-full']['arguments'][-4:] == ['--qualification', '@inversion-proof-cold.path',
         '--qualification-pin', '@inversion-proof-cold.pin'], 'ANSWER_QUEUE_COLD_INVERSION_QUALIFICATION')
    need(nodes['surface-reverse']['arguments'] == ['--reverse'], 'ANSWER_QUEUE_COLD_REVERSE')


def run(a):
    a.output.mkdir(exist_ok=False); began = time.monotonic()
    c, sources, fresh = authenticate(a.authority, a.pin)
    proof = load(a.proof/'report.json', a.proof_pin); rows = load(a.proof/'plan.json', proof['plan_pin'])
    graph(rows); before = load(a.proof/'before.json')
    need(proof['complete'] and proof['authority'] == a.pin and proof['signals_sent'] == proof['children_started'] == 0,
         'ANSWER_QUEUE_COLD_PROBE_SCOPE')
    need(len(proof['controls']) == 29 and proof['controls'] == load(a.proof/'controls.json'),
         'ANSWER_QUEUE_COLD_CONTROLS')
    for name, ref in c['old'].items():
        recorded = before['owners'][name]
        need(recorded['pid'] == ref['pid'] and recorded['starttime'] == ref['starttime'] and
             not recorded['children'] and not recorded['terminal'], 'ANSWER_QUEUE_COLD_SOURCE_OWNER')
        need(file_map(ref['path']) == recorded['files'], 'ANSWER_QUEUE_COLD_ORIGINAL_BYTES')
    controls = {}
    for name in ('task-full', 'inversion-full', 'surface-cold'):
        bad = deepcopy(rows); next(r for r in bad if r['label'] == name)['requires'] = []
        try: graph(bad)
        except ValueError as error: controls[name] = str(error)
        else: raise ValueError('ANSWER_QUEUE_COLD_CONTROL_ACCEPTED')
    try: load(a.authority, '0'*64)
    except ValueError as error: controls['authority-pin'] = str(error)
    else: raise ValueError('ANSWER_QUEUE_COLD_AUTHORITY_CONTROL_ACCEPTED')
    state = dict(stage='probe_only', source_owners={n: process(ref['pid']) for n, ref in c['old'].items()})
    if a.handoff:
        root = a.handoff; result = load(root/'report.json', a.handoff_pin)
        receipt = load(root/'handoff.json', result['handoff']); stopped = load(root/'frozen.json')
        switched = load(root/'before.json')
        need(receipt['authority'] == a.pin and receipt['complete'] and receipt['terminated'] == ['task', 'consumer', 'priority']
             and all(receipt['original_artifacts_unchanged'].values()), 'ANSWER_QUEUE_COLD_HANDOFF')
        need(receipt['plan_pin'] == proof['plan_pin'], 'ANSWER_QUEUE_COLD_PLAN_CHANGED')
        for name, ref in c['old'].items():
            old = process(ref['pid']); need(old is None or old['state'] == 'Z', 'ANSWER_QUEUE_COLD_OLD_RUNNING')
            need(stopped[name]['state'] == 'T' and not stopped[name]['children'] and
                 stopped[name]['files'] == switched['owners'][name]['files'] == file_map(ref['path']),
                 'ANSWER_QUEUE_COLD_STOPPED_BYTES')
        newroot = Path(c['next_output']); launch = load(newroot/'launch.json'); new = process(launch['pid'])
        need(new and new['starttime'] == launch['starttime'] == result['starttime'] and
             new['pid'] == result['pid'] and new['state'] == 'S' and not new['children'], 'ANSWER_QUEUE_COLD_SUCCESSOR')
        need(launch['authority'] == a.pin and launch['handoff'] == result['handoff'] and launch['wait'] == c['runtime'],
             'ANSWER_QUEUE_COLD_SUCCESSOR_BINDING')
        need(new['command'] == result['command'] and sha(newroot/'driver.py') == launch['driver_pin'] == proof['driver_pin'],
             'ANSWER_QUEUE_COLD_SUCCESSOR_COMMAND')
        events = [json.loads(v) for v in (newroot/'events.jsonl').read_text().splitlines()]
        need(len(events) == 1 and events[0]['stage'] == 'waiting_for_runtime' and
             events[0]['owner'] == c['runtime'], 'ANSWER_QUEUE_COLD_SUCCESSOR_WAIT')
        need(not (newroot/'failure.json').exists() and not (newroot/'report.json').exists(), 'ANSWER_QUEUE_COLD_PREMATURE_REPORT')
        need(load(newroot/'plan.json', proof['plan_pin']) == rows, 'ANSWER_QUEUE_COLD_SUCCESSOR_PLAN')
        state = dict(stage='idle_handoff', successor=new, launch=launch, events=events, handoff=result['handoff'])
    kept = []
    for ref in c['protected']:
        actual = process(ref['pid'])
        need(actual and actual['starttime'] == ref['starttime'] and actual['state'] not in ('T', 'Z'),
             'ANSWER_QUEUE_COLD_PROTECTED_PROCESS')
        kept.append(actual)
    fresh()
    (a.output/'nvidia-readback.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    result = dict(complete=True, authority=a.pin, proof=a.proof_pin, plan_pin=proof['plan_pin'],
                  stage_count=len(rows), original_artifacts_unchanged=True, controls=controls, protected=kept,
                  state=state, numerical_work_complete=False, capability_demonstrated=False,
                  seconds=time.monotonic()-began, acceptance='author-checked')
    with (a.output/'report.json').open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(pin=sha(a.output/'report.json'), stage=state['stage'], seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('authority', 'proof', 'output'): p.add_argument('--'+n, type=Path, required=True)
    for n in ('pin', 'proof-pin'): p.add_argument('--'+n, required=True)
    p.add_argument('--handoff', type=Path); p.add_argument('--handoff-pin'); run(p.parse_args())
