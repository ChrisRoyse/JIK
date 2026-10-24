"""Move an independent answer prerequisite ahead of an idle GPU queue chain.

Only queue ownership changes. Original workers, proofs, numerical geometry and
exclusive GPU leases remain in force. No active compute child may be stopped.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import answer_native_queue as answer
from audit import legal_grouped_placement_queue as grouped
from audit import surface_device_queue as surface

REPO = Path(__file__).resolve().parents[2]
DRIVERS = dict(answer='answer_native_queue.py', grouped='legal_grouped_placement_queue.py',
               surface='surface_device_queue.py')
WAITING = dict(answer='waiting', grouped='waiting_for_runtime_owner',
               surface='waiting_for_predictor_comparison')
sha, need, identity, read, write = grouped.sha, grouped.need, grouped.identity, grouped.read, grouped.write


def owner(root):
    root = Path(root); launch = json.loads((root/'launch.json').read_bytes())
    return dict(path=str(root), pid=launch['pid'], starttime=str(launch['starttime']),
                launch_pin=sha(root/'launch.json'))


def check_owner(ref):
    need(owner(ref['path']) == ref, 'PRIORITY_OWNER_BINDING')
    need(identity(ref['pid']) in (ref['starttime'], None), 'PRIORITY_PID_REUSED')


def authenticate(path, pin):
    c = read(path, pin)
    need(c['schema'] == 'answer-priority-queue-v1' and c['driver_pin'] == sha(__file__),
         'PRIORITY_AUTHORITY_CODE')
    records, checks = {}, []
    for name, module in [('answer', answer), ('grouped', grouped), ('surface', surface)]:
        ref = c['authorities'][name]
        records[name], check = module.authenticate(Path(ref['path']), ref['pin']); checks.append(check)
        old = c['old'][name]; check_owner(old)
        launch = json.loads((Path(old['path'])/'launch.json').read_bytes())
        need(launch['authority'] == ref['pin'] and
             launch['driver_pin'] == sha(REPO/'scripts/audit'/DRIVERS[name]), 'PRIORITY_OLD_DRIVER')
    need(records['answer']['wait'] == c['old']['surface'] and
         records['surface']['wait'] == c['old']['grouped'], 'PRIORITY_CHAIN')
    check_owner(c['runtime'])
    need(records['grouped']['wait'] == {k: v for k, v in c['runtime'].items() if k != 'launch_pin'},
         'PRIORITY_RUNTIME_BINDING')
    expected = deepcopy(records['answer']); expected['wait'] = c['runtime']
    need(read(c['answer_authority']['path'], c['answer_authority']['pin']) == expected,
         'PRIORITY_ANSWER_ONLY_WAIT_CHANGED')
    h = read(c['grouped_handoff']['path'], c['grouped_handoff']['pin'])
    need(h['complete'] and h['authority'] == c['authorities']['grouped']['pin']
         and h['status'] == 'superseded_waiting_owner', 'PRIORITY_PRIOR_HANDOFF')
    def fresh():
        need(sha(path) == pin and sha(__file__) == c['driver_pin'], 'PRIORITY_CHANGED')
        for check in checks: check()
        for name in ['answer_authority', 'grouped_handoff']:
            need(sha(c[name]['path']) == c[name]['pin'], 'PRIORITY_INPUT_CHANGED')
        check_owner(c['runtime'])
    fresh(); return c, fresh


def files(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def process(pid):
    root = Path('/proc')/str(pid); fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, starttime=fields[19], state=fields[0],
                children=[int(x) for x in (root/'task'/str(pid)/'children').read_text().split()],
                command=(root/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                wchan=(root/'wchan').read_text())


def validate(row, ref, name):
    need(row['pid'] == ref['pid'] and row['starttime'] == ref['starttime'], 'PRIORITY_IDLE_IDENTITY')
    need(row['state'] in ('S', 'T') and row['children'] == [], 'PRIORITY_ACTIVE_CHILD')
    need(row['events'] and all(v['stage'] == WAITING[name] for v in row['events']), 'PRIORITY_ADVANCED')
    need(not row['terminal'] and str(REPO/'scripts/audit'/DRIVERS[name]) in row['command'],
         'PRIORITY_IDLE_SCOPE')
    need(row['driver_pin'] == sha(REPO/'scripts/audit'/DRIVERS[name]), 'PRIORITY_IDLE_DRIVER')


def snapshot(ref, name):
    check_owner(ref); root = Path(ref['path']); row = process(ref['pid'])
    row.update(events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()],
               terminal=any((root/n).exists() for n in ['report.json', 'failure.json']),
               driver_pin=sha(root/'driver.py'), files=files(root))
    validate(row, ref, name); return row


def protected(c):
    rows = []
    for ref in c['protected']:
        row = process(ref['pid'])
        need(row['starttime'] == ref['starttime'], 'PRIORITY_PROTECTED_IDENTITY'); rows.append(row)
    return rows


def controls(rows, c):
    result = {}
    for name, original in rows.items():
        for field, value in [('pid', -1), ('starttime', '0'), ('children', [1]),
                             ('events', [{'stage': 'running'}]), ('terminal', True),
                             ('driver_pin', '0'*64), ('command', [])]:
            bad = deepcopy(original); bad[field] = value
            try: validate(bad, c['old'][name], name)
            except ValueError as error: result[name+':'+field] = str(error)
            else: raise ValueError('PRIORITY_CONTROL_ACCEPTED')
    return result


def switch(a):
    a.output.mkdir(exist_ok=False); c, fresh = authenticate(a.authority, a.pin)
    before = {n: snapshot(ref, n) for n, ref in c['old'].items()}
    guard = protected(c); negatives = controls(before, c)
    write(a.output/'before.json', dict(owners=before, protected=guard))
    write(a.output/'controls.json', negatives)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    if a.mode == 'probe':
        write(a.output/'report.json', dict(complete=True, authority=a.pin, controls=negatives,
              signals_sent=0, gpu_children_started=0, driver_pin=sha(__file__), acceptance='author-checked'))
        return
    proof = read(a.proof/'report.json', a.proof_pin)
    need(proof['complete'] and proof['authority'] == a.pin and proof['controls'] == negatives
         and proof['signals_sent'] == 0 and proof['driver_pin'] == sha(__file__), 'PRIORITY_PROBE_BINDING')
    need(not Path(c['next_output']).exists(), 'PRIORITY_NEXT_EXISTS')
    fds, frozen, terminated = {}, set(), []
    try:
        fresh(); write(a.output/'intent.json', dict(authority=a.pin, old=c['old'], next_output=c['next_output']))
        # Stop the upstream idle queue first so no new child can appear while the
        # downstream owners are checked. Resume every survivor on any refusal.
        for name in ['grouped', 'surface', 'answer']:
            ref = c['old'][name]; fd = os.pidfd_open(ref['pid']); fds[name] = fd
            need(identity(ref['pid']) == ref['starttime'], 'PRIORITY_FREEZE_IDENTITY')
            signal.pidfd_send_signal(fd, signal.SIGSTOP); frozen.add(name)
            deadline = time.monotonic()+5
            while process(ref['pid'])['state'] != 'T' and time.monotonic() < deadline: time.sleep(.01)
            row = snapshot(ref, name)
            need(row['state'] == 'T' and row['files'] == before[name]['files']
                 and row['events'] == before[name]['events'], 'PRIORITY_FREEZE_ADVANCED')
        write(a.output/'frozen.json', {n: snapshot(c['old'][n], n) for n in c['old']})
        protected(c); fresh()
        # Retire downstream first. No original report is forged or relabeled.
        for name in ['answer', 'surface', 'grouped']:
            signal.pidfd_send_signal(fds[name], signal.SIGTERM)
            signal.pidfd_send_signal(fds[name], signal.SIGCONT); frozen.remove(name)
            need(select.select([fds[name]], [], [], 10)[0], 'PRIORITY_OWNER_NOT_TERMINAL')
            terminated.append(name)
        unchanged = {n: files(ref['path']) == before[n]['files'] for n, ref in c['old'].items()}
        need(all(unchanged.values()), 'PRIORITY_OLD_ARTIFACT_CHANGED')
        handoff = dict(complete=True, status='superseded_waiting_owners', authority=a.pin,
                       terminated=terminated, original_artifacts_unchanged=unchanged, protected=protected(c),
                       controls=negatives, acceptance='author-checked', scientific_admission=False)
        write(a.output/'handoff.json', handoff)
        command = [sys.executable, '-I', str(Path(__file__).resolve()), '--mode', 'run',
                   '--authority', str(a.authority), '--pin', a.pin, '--output', c['next_output'],
                   '--handoff', str(a.output/'handoff.json'), '--handoff-pin', sha(a.output/'handoff.json')]
        with (a.output/'successor.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        write(a.output/'report.json', dict(complete=True, status='successor_launched',
              authority=a.pin, handoff=sha(a.output/'handoff.json'), pid=child.pid,
              starttime=identity(child.pid), command=command, acceptance='author-checked',
              successor_readback_pending=True, numerical_work_complete=False))
    except BaseException as error:
        write(a.output/'failure.json', dict(error=repr(error), terminated=terminated)); raise
    finally:
        for name in frozen: signal.pidfd_send_signal(fds[name], signal.SIGCONT)
        for fd in fds.values(): os.close(fd)


def run(a):
    a.output.mkdir(exist_ok=False); results = []; lock = None
    def event(stage, **values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage, timestamp=time.time(), **values), sort_keys=True)+'\n')
            f.flush(); os.fsync(f.fileno())
    try:
        c, fresh = authenticate(a.authority, a.pin)
        h = read(a.handoff, a.handoff_pin)
        need(h['complete'] and h['status'] == 'superseded_waiting_owners' and h['authority'] == a.pin
             and all(h['original_artifacts_unchanged'].values()), 'PRIORITY_HANDOFF')
        need(all(identity(r['pid']) != r['starttime'] for r in c['old'].values()), 'PRIORITY_OLD_LIVE')
        lock = (REPO/'.lease/answer-priority-queue').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
        write(a.output/'launch.json', dict(pid=os.getpid(), starttime=identity(os.getpid()),
              authority=a.pin, handoff=a.handoff_pin, driver_pin=sha(__file__), wait=c['runtime']))
        wait = c['runtime']
        if identity(wait['pid']) == wait['starttime']:
            fd = os.pidfd_open(wait['pid'])
            try:
                need(identity(wait['pid']) in (wait['starttime'], None), 'PRIORITY_WAIT_REUSED')
                event('waiting_for_runtime', pid=wait['pid'], starttime=wait['starttime'], mechanism='kernel_pidfd')
                select.select([fd], [], [])
            finally: os.close(fd)
        prior = json.loads((Path(wait['path'])/'report.json').read_bytes())
        need(prior['complete'] and {'gpu', 'gpu-cold'}.issubset({v['label'] for v in prior['outputs']}),
             'PRIORITY_RUNTIME_INCOMPLETE')
        for ref in prior['outputs']: need(read(Path(ref['path'])/'report.json', ref['pin'])['complete'],
                                          'PRIORITY_RUNTIME_OUTPUT')
        def execute(label, script, arguments):
            fresh(); target = a.output/label
            command = [sys.executable, '-I', str(REPO/'scripts/audit'/script), '--mode', 'full',
                       *arguments, '--output', str(target)]
            with (a.output/(label+'.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT)
                event('running', label=label, pid=child.pid, starttime=identity(child.pid), command=command)
                status = child.wait(); log.flush(); os.fsync(log.fileno())
            need(status == 0, 'PRIORITY_CHILD_FAILED')
            report = json.loads((target/'report.json').read_bytes()); need(report['complete'], 'PRIORITY_CHILD_INCOMPLETE')
            row = dict(label=label, path=str(target), pin=sha(target/'report.json')); results.append(row)
            fresh(); event('terminal', **row); return target
        ref = c['answer_authority']
        execute('answer', DRIVERS['answer'], ['--authority', ref['path'], '--pin', ref['pin']])
        ref, h = c['authorities']['grouped'], c['grouped_handoff']
        placed = execute('grouped', DRIVERS['grouped'], ['--authority', ref['path'], '--authority-pin', ref['pin'],
                          '--handoff', h['path'], '--handoff-pin', h['pin']])
        # The decoder still requires the complete, independently read-back full
        # predictor comparison. Only its actual completed owner address changes.
        ref = c['authorities']['surface']; revised = read(ref['path'], ref['pin']); revised['wait'] = owner(placed)
        path = a.output/'surface-authority.json'; write(path, revised)
        surface.authenticate(path, sha(path)); surface.predecessor(revised['wait'])
        execute('surface', DRIVERS['surface'], ['--authority', str(path), '--authority-pin', sha(path)])
        write(a.output/'report.json', dict(complete=True, authority=a.pin, outputs=results,
              driver_pin=sha(__file__), acceptance='author-checked', scientific_admission=False,
              capability_demonstrated=False))
        event('complete')
    except BaseException as error:
        write(a.output/'failure.json', dict(error=repr(error), outputs=results)); event('failed'); raise
    finally:
        if lock: lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['probe', 'switch', 'run'], required=True)
    for n in ['authority', 'output']: p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--pin', required=True)
    for n in ['proof', 'handoff']:
        p.add_argument('--'+n, type=Path); p.add_argument('--'+n+'-pin')
    a = p.parse_args()
    if a.mode == 'run': run(a)
    else: switch(a)
