"""Retire only proven idle owners; run every existing proof/train/cold worker serially."""
import argparse
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
from audit import complete_answer_schedule as schedule

REPO = schedule.REPO
sha, need, read, write, identity = schedule.sha, schedule.need, schedule.read, schedule.write, schedule.identity


def protected(c):
    rows = []
    for ref in c['protected']:
        row = schedule.prior.process(ref['pid'])
        need(row['starttime'] == ref['starttime'], 'COMPLETE_QUEUE_PROTECTED_IDENTITY'); rows.append(row)
    return rows


def switch(a):
    a.output.mkdir(exist_ok=False); started = time.monotonic()
    c, sources, fresh = schedule.authenticate(a.authority, a.pin)
    work = schedule.plan(sources); schedule.validate_plan(work)
    before = {n: schedule.snapshot(ref, n) for n, ref in c['old'].items()}
    negatives = schedule.controls(before, c, work)
    write(a.output/'before.json', dict(owners=before, protected=protected(c)))
    write(a.output/'plan.json', work); write(a.output/'controls.json', negatives)
    (a.output/'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    if a.mode == 'probe':
        write(a.output/'report.json', dict(complete=True, authority=a.pin, controls=negatives,
              plan_pin=sha(a.output/'plan.json'), signals_sent=0, children_started=0, driver_pin=sha(__file__),
              seconds=time.monotonic()-started, acceptance='author-checked'))
        return
    proof = read(a.proof/'report.json', a.proof_pin)
    need(proof['complete'] and proof['authority'] == a.pin and proof['controls'] == negatives and
         proof['signals_sent'] == 0 and proof['driver_pin'] == sha(__file__) and
         proof['plan_pin'] == sha(a.output/'plan.json'), 'COMPLETE_QUEUE_PROBE_BINDING')
    need(not Path(c['next_output']).exists(), 'COMPLETE_QUEUE_NEXT_EXISTS')
    fds, frozen, terminated = {}, set(), []
    try:
        fresh(); write(a.output/'intent.json', dict(authority=a.pin, old=c['old'], next_output=c['next_output']))
        # Freeze the upstream dispatcher first. A newly active child causes refusal, never termination.
        for name in ('priority', 'consumer', 'task'):
            ref = c['old'][name]; fd = os.pidfd_open(ref['pid']); fds[name] = fd
            need(identity(ref['pid']) == ref['starttime'], 'COMPLETE_QUEUE_FREEZE_IDENTITY')
            signal.pidfd_send_signal(fd, signal.SIGSTOP); frozen.add(name)
            deadline = time.monotonic()+5
            while schedule.prior.process(ref['pid'])['state'] != 'T' and time.monotonic() < deadline: time.sleep(.01)
            row = schedule.snapshot(ref, name)
            need(row['state'] == 'T' and row['files'] == before[name]['files'] and
                 row['events'] == before[name]['events'], 'COMPLETE_QUEUE_FREEZE_ADVANCED')
        write(a.output/'frozen.json', {n: schedule.snapshot(c['old'][n], n) for n in c['old']})
        protected(c); fresh()
        for name in ('task', 'consumer', 'priority'):
            signal.pidfd_send_signal(fds[name], signal.SIGTERM)
            signal.pidfd_send_signal(fds[name], signal.SIGCONT); frozen.remove(name)
            need(select.select([fds[name]], [], [], 10)[0], 'COMPLETE_QUEUE_OWNER_NOT_TERMINAL')
            terminated.append(name)
        unchanged = {n: schedule.prior.files(ref['path']) == before[n]['files'] for n, ref in c['old'].items()}
        need(all(unchanged.values()), 'COMPLETE_QUEUE_OLD_ARTIFACT_CHANGED')
        write(a.output/'handoff.json', dict(complete=True, status='superseded_idle_owners', authority=a.pin,
              terminated=terminated, original_artifacts_unchanged=unchanged, protected=protected(c),
              controls=negatives, plan_pin=sha(a.output/'plan.json'), acceptance='author-checked'))
        command = [sys.executable, '-I', str(Path(__file__).resolve()), '--mode', 'run',
                   '--authority', str(a.authority), '--pin', a.pin, '--output', c['next_output'],
                   '--handoff', str(a.output/'handoff.json'), '--handoff-pin', sha(a.output/'handoff.json')]
        with (a.output/'successor.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        write(a.output/'report.json', dict(complete=True, status='successor_launched', authority=a.pin,
              handoff=sha(a.output/'handoff.json'), pid=child.pid, starttime=identity(child.pid), command=command,
              seconds=time.monotonic()-started, successor_readback_pending=True, numerical_work_complete=False,
              acceptance='author-checked'))
    except BaseException as error:
        write(a.output/'failure.json', dict(error=repr(error), terminated=terminated)); raise
    finally:
        for name in frozen: signal.pidfd_send_signal(fds[name], signal.SIGCONT)
        for fd in fds.values(): os.close(fd)


def run(a):
    a.output.mkdir(exist_ok=False); results = {}; errors = {}; lock = None
    def event(stage, **values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage, time=time.time(), **values), sort_keys=True)+'\n')
            f.flush(); os.fsync(f.fileno())
    try:
        c, sources, fresh = schedule.authenticate(a.authority, a.pin)
        h = read(a.handoff, a.handoff_pin); work = schedule.plan(sources); schedule.validate_plan(work)
        write(a.output/'plan.json', work)
        need(h['complete'] and h['status'] == 'superseded_idle_owners' and h['authority'] == a.pin and
             all(h['original_artifacts_unchanged'].values()) and h['plan_pin'] == sha(a.output/'plan.json'),
             'COMPLETE_QUEUE_HANDOFF')
        need(all(identity(r['pid']) != r['starttime'] for r in c['old'].values()), 'COMPLETE_QUEUE_OLD_LIVE')
        lock = (REPO/'.lease/complete-answer-priority').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
        write(a.output/'launch.json', dict(pid=os.getpid(), starttime=identity(os.getpid()), authority=a.pin,
              handoff=a.handoff_pin, driver_pin=sha(__file__), wait=c['runtime']))
        owner = c['runtime']; schedule.prior.check_owner(owner)
        if identity(owner['pid']) == owner['starttime']:
            fd = os.pidfd_open(owner['pid'])
            try:
                need(identity(owner['pid']) in (owner['starttime'], None), 'COMPLETE_QUEUE_WAIT_REUSED')
                event('waiting_for_runtime', owner=owner, mechanism='kernel_pidfd'); select.select([fd], [], [])
            finally: os.close(fd)
        root = Path(owner['path']); prior = json.loads((root/'report.json').read_bytes())
        need(prior['complete'] and prior['driver_pin'] == json.loads((root/'launch.json').read_bytes())['driver_pin']
             and {'gpu', 'gpu-cold'} <= {v['label'] for v in prior['outputs']}, 'COMPLETE_QUEUE_RUNTIME_FAILED')
        for ref in prior['outputs']: need(schedule.report(ref)['complete'], 'COMPLETE_QUEUE_RUNTIME_OUTPUT')
        event('runtime_complete', report_pin=sha(root/'report.json'))
        for row in work:
            label = row['label']; missing = [n for n in row['requires'] if n not in results]
            if missing:
                errors[label] = dict(code='COMPLETE_QUEUE_PREREQUISITE_FAILED', missing=missing)
                write(a.output/(label+'-skipped.json'), errors[label]); event('skipped', label=label, **errors[label]); continue
            fresh(); dest = a.output/label
            args = schedule.arguments(row, results, sources)
            command = [sys.executable, '-I', str(REPO/'scripts/audit'/row['script']), *args, '--output', str(dest)]
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8')
            if row['cpu_only']: env['CUDA_VISIBLE_DEVICES'] = ''
            else:
                need(env.get('CUDA_VISIBLE_DEVICES') != '', 'COMPLETE_QUEUE_GPU_HIDDEN')
                (a.output/(label+'-nvidia-before.txt')).write_text(subprocess.check_output(['nvidia-smi'], text=True))
            start = time.monotonic()
            try:
                with (a.output/(label+'.log')).open('xb') as log:
                    child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    event('running', label=label, pid=child.pid, starttime=identity(child.pid), command=command)
                    status = child.wait(); log.flush(); os.fsync(log.fileno())
                write(a.output/(label+'-exit.json'), dict(exit_code=status, seconds=time.monotonic()-start))
                need(status == 0, 'COMPLETE_QUEUE_WORKER_FAILED:'+label)
                ref = dict(label=label, path=str(dest), pin=sha(dest/'report.json'))
                schedule.output_check(label, ref, sources); fresh(); results[label] = ref; event('terminal', **ref)
            except Exception as error:
                errors[label] = dict(code='COMPLETE_QUEUE_WORKER_REFUSED', error=repr(error))
                write(a.output/(label+'-failure.json'), errors[label]); event('worker_failed', label=label, **errors[label])
                # An independent task still runs; failed prerequisites can never be silently satisfied.
        fresh()
        write(a.output/'report.json', dict(complete=not errors, authority=a.pin, outputs=list(results.values()),
              errors=errors, driver_pin=sha(__file__), scientific_admission=False, capability_demonstrated=False,
              original_workers_unchanged=True, acceptance='author-checked'))
        event('complete', success=not errors)
    except BaseException as error:
        write(a.output/'failure.json', dict(error=repr(error), outputs=list(results.values()), errors=errors))
        event('failed', error=repr(error)); raise
    finally:
        if lock: lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('probe', 'switch', 'run'), required=True)
    for name in ('authority', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pin', required=True)
    for name in ('proof', 'handoff'):
        p.add_argument('--'+name, type=Path); p.add_argument('--'+name+'-pin')
    args = p.parse_args()
    if args.mode == 'run': run(args)
    else: switch(args)
