"""Qualify then cold-replay full-task free answers after the actual GPU predecessor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import free_guard_native_contract as prior
from audit import paired_surface_contract as paired
from audit.surface_device_queue import check_owner, identity, need, sha, stamp, write

P = prior.P; REPO = prior.REPO; read = prior.read
LOCK = REPO / '.lease/free-answer-successor'
WAIT = P / 'free-guard-native-queue-v1-20261025'
AUTH = dict(path=str(P / 'free-guard-native-authority-v2-20261025/authority.json'),
            pin='9fd3070c001e542fecded312681eefe8d1b43dde30a67ad6c6ad0205f4a9cab2')
COLD = dict(path=str(P / 'free-task-closed-trajectory-readback-v2-20261025/report.json'),
            pin='fb71f52d7cbac9795252f2ad4464aeae7bce6f0aeb2173d8fc6e32c88b76ac7d')
STAGES = ('qualification', 'qualification-readback', 'full', 'full-readback')
POLICY = dict(device='cuda:0', query_cases=170, qualification=[0, 1, 3], all_proposals=True,
    decoder_arms=['parent', 'corrected'], reference_visible=False, reference_length=False,
    teacher_forced=False, automatic_retry=False, retention_owner='ChrisRoyse/MNLM#29',
    wait='actual_guard_coordinator_exit_and_complete_report', semantic_acceptance=False)
SELF = 'audit/free_answer_queue.py'


def owner_fd(fd):
    need(type(fd) is int and fd >= 0, 'FREE_QUEUE_OWNER_FD')
    actual = os.fstat(fd); expected = LOCK.stat()
    need((actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), 'FREE_QUEUE_OWNER_IDENTITY')
    with LOCK.open('a+b') as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise ValueError('FREE_QUEUE_OWNER_UNLOCKED')


def base():
    parent, _, fresh = prior.authenticate(**AUTH); fresh(); cold = read(**COLD)
    need(cold['complete'] and cold['all_free_logits_exact'] and cold['all_bytes_exact']
         and cold['all_terminals_recounted'] and len(cold['controls']) == 9, 'FREE_QUEUE_CPU_COLD')
    codes = dict(parent['code'])
    for name, h in cold['code'].items():
        need(name not in codes or codes[name] == h, 'FREE_QUEUE_CODE_CONFLICT'); codes[name] = h
    codes[SELF] = sha(REPO / 'scripts' / SELF)
    need(all(sha(REPO / 'scripts' / n) == h for n, h in codes.items()), 'FREE_QUEUE_CODE')
    return parent, cold, codes


def prepare(output):
    _, _, codes = base(); launch = read(WAIT / 'launch.json', sha(WAIT / 'launch.json'))
    need(launch['authority'] == AUTH['pin'] and launch['driver_pin'] == codes['audit/free_guard_native_queue.py'],
         'FREE_QUEUE_PARENT_LAUNCH')
    wait = dict(path=str(WAIT), pid=launch['pid'], starttime=str(launch['starttime']), launch_pin=sha(WAIT / 'launch.json'))
    need(wait['pid'] == 3770334 and wait['starttime'] == '415703639', 'FREE_QUEUE_PARENT_IDENTITY')
    check_owner(wait); need(identity(wait['pid']) == wait['starttime'], 'FREE_QUEUE_PREPARE_LIVE')
    output.mkdir(exist_ok=False)
    write(output / 'authority.json', dict(schema='free-answer-successor-v1', parent=AUTH, cold=COLD,
        wait=wait, code=codes, policy=POLICY, stages=list(STAGES)))
    return sha(output / 'authority.json')


def authenticate(path, pin):
    cfg = read(path, pin); parent, cold, codes = base()
    need(set(cfg) == {'schema', 'parent', 'cold', 'wait', 'code', 'policy', 'stages'}
         and cfg['schema'] == 'free-answer-successor-v1' and cfg['parent'] == AUTH and cfg['cold'] == COLD
         and cfg['code'] == codes and cfg['policy'] == POLICY and cfg['stages'] == list(STAGES), 'FREE_QUEUE_AUTHORITY')
    wait = cfg['wait']
    need(wait['path'] == str(WAIT) and wait['pid'] == 3770334 and wait['starttime'] == '415703639', 'FREE_QUEUE_WAIT')
    check_owner(wait); launch = read(WAIT / 'launch.json', wait['launch_pin'])
    need(launch['authority'] == AUTH['pin'] and launch['driver_pin'] == codes['audit/free_guard_native_queue.py'],
         'FREE_QUEUE_PARENT_LAUNCH')
    files = {Path(path): pin, Path(AUTH['path']): AUTH['pin'], Path(COLD['path']): COLD['pin'],
             WAIT / 'launch.json': wait['launch_pin']}
    stamps = {p: stamp(p) for p in files}
    def fresh():
        need(all(stamp(p) == s for p, s in stamps.items()), 'FREE_QUEUE_INPUT_CHANGED')
        need(all(sha(REPO / 'scripts' / n) == h for n, h in codes.items()), 'FREE_QUEUE_CODE_CHANGED')
        check_owner(wait)
    fresh(); return cfg, parent, cold, fresh


def parent_complete(r, cfg):
    need(r['complete'] and r['authority'] == AUTH['pin']
         and r['driver_pin'] == cfg['code']['audit/free_guard_native_queue.py']
         and r['policy'] == prior.POLICY and [v['label'] for v in r['outputs']] == list(prior.STAGES)
         and not r['semantic_validity_established'], 'FREE_QUEUE_PARENT_INCOMPLETE')


def terminal(cfg):
    check_owner(cfg['wait']); need(identity(cfg['wait']['pid']) is None, 'FREE_QUEUE_PREDECESSOR_LIVE')
    need(not (WAIT / 'failure.json').exists() and (WAIT / 'report.json').is_file(), 'FREE_QUEUE_PARENT_FAILED')
    pin = sha(WAIT / 'report.json'); r = read(WAIT / 'report.json', pin); parent_complete(r, cfg)
    parent, _, fresh = prior.authenticate(**AUTH); fresh()
    need(r['predecessor'] == prior.predecessor(parent), 'FREE_QUEUE_PARENT_ANCESTRY')
    checked = []
    for ref in r['outputs']:
        need(Path(ref['path']) == WAIT / ref['label'] / 'report.json', 'FREE_QUEUE_PARENT_PATH')
        prior.result_check(ref['label'], read(ref['path'], ref['pin']), parent, checked); checked.append(ref)
    paired_cfg, _, paired_fresh = paired.authenticate(**prior.PARENT); paired_fresh()
    inputs = paired.terminal(paired_cfg)
    return dict(parent=pin, inputs=inputs)


def check_results(first, cold, *, cuda, qualification):
    count = 3 if qualification or not cuda else 170
    need(first['complete'] and cold['complete'] and first['gpu_used'] == cold['gpu_used'] == cuda
         and first['qualification'] == cold['qualification'] == qualification
         and first['full_training'] == cold['full_training'] == cuda
         and first['full_query'] == (cuda and not qualification) and first['evaluated_cases'] == count
         and first['declared_cases'] == 170 and len(first['results']) == count, 'FREE_QUEUE_RESULT_SCOPE')
    need(first['free_generation_tested'] and not first['teacher_forced'] and first['reference_open_refused']
         and first['reference_visible'] is False and cold['reference_visible'] is False
         and first['optimizer_updates'] == first['encoder_calls'] == 0, 'FREE_QUEUE_RESULT_METHOD')
    need(cold['all_free_logits_exact'] and cold['all_emitted_bytes_exact'] and cold['all_terminals_recounted']
         and cold['source_traces_drained'] and cold['code'] == first['code']
         and cold['trajectories'] == len(cold['results']) and cold['trajectories'] > 0
         and set(cold['controls']) == {'changed_logit', 'wrong_byte', 'false_completion', 'wrong_artifact_pin', 'private_reference_open'},
         'FREE_QUEUE_RESULT_COLD')
    # The population remains intact even when no answer terminates or a surface refuses.
    need(not first['capability_demonstrated'] and not first['architecture_winner'] and not cold['capability_demonstrated'],
         'FREE_QUEUE_FALSE_CAPABILITY')


def execute(label, script, args, output, fd):
    owner_fd(fd)
    command = [sys.executable, '-I', str(REPO / 'scripts/audit' / script), *args, '--output', str(output)]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    need(env.get('CUDA_VISIBLE_DEVICES') != '', 'FREE_QUEUE_GPU_HIDDEN')
    with output.with_suffix('.log').open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=(fd,))
        write(output.with_suffix('.launch.json'), dict(pid=child.pid, starttime=identity(child.pid), command=command))
        status = child.wait(); log.flush(); os.fsync(log.fileno())
    need(status == 0, 'FREE_QUEUE_CHILD_FAILED:' + label)
    path = output / 'report.json'; pin = sha(path); need(read(path, pin)['complete'], 'FREE_QUEUE_CHILD_INCOMPLETE')
    return dict(label=label, path=str(path), pin=pin)


def run(a):
    cfg, _, _, fresh = authenticate(a.authority, a.pin); proof = read(a.proof / 'report.json', a.proof_pin)
    need(proof['complete'] and proof['authority'] == a.pin and proof['code'] == cfg['code']
         and proof['source_recounted'] and proof['ownership_controls_passed'] and not proof['cuda_executed'], 'FREE_QUEUE_PROOF')
    need('torch' not in sys.modules, 'FREE_QUEUE_EARLY_TORCH')
    with LOCK.open('a+b') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB); a.output.mkdir(exist_ok=False)
        write(a.output / 'launch.json', dict(pid=os.getpid(), starttime=identity(os.getpid()), authority=a.pin,
            driver_pin=sha(__file__), proof=a.proof_pin, wait=cfg['wait']))
        def event(stage, **values):
            with (a.output / 'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(stage=stage, time=time.time(), **values), sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
        wait = cfg['wait']; fresh(); check_owner(wait)
        if identity(wait['pid']) == wait['starttime']:
            try:
                fd = os.pidfd_open(wait['pid'])
            except ProcessLookupError:
                pass
            else:
                try:
                    check_owner(wait); event('waiting', owner=wait, method='kernel_pidfd', torch_loaded=False)
                    select.select([fd], [], [])
                finally:
                    os.close(fd)
        began = tick = time.perf_counter(); fresh(); handoff = terminal(cfg); inputs = handoff['inputs']
        write(a.output / 'handoff.json', handoff); phases = dict(handoff=time.perf_counter()-tick); outputs = []
        for scope in ('qualification', 'full'):
            flags = ['--cuda'] + (['--qualification'] if scope == 'qualification' else [])
            args = paired.arguments(inputs['pairs'][scope], inputs['corpus']); fresh(); tick = time.perf_counter()
            event('launching', label=scope)
            first = execute(scope, 'free_task_surface.py', flags+args, a.output / scope, owner.fileno()); outputs.append(first)
            phases[scope] = time.perf_counter()-tick; fresh(); tick = time.perf_counter(); label = scope+'-readback'
            event('launching', label=label)
            cold = execute(label, 'free_task_surface_readback.py', flags+['--input', str(Path(first['path']).parent), '--pin', first['pin']],
                a.output / label, owner.fileno()); outputs.append(cold)
            phases[label] = time.perf_counter()-tick; tick = time.perf_counter()
            r = read(first['path'], first['pin']); c = read(cold['path'], cold['pin'])
            need(c['input_pin'] == first['pin'], 'FREE_QUEUE_COLD_BINDING'); check_results(r, c, cuda=True, qualification=scope == 'qualification')
            fresh(); phases[scope+'-gate'] = time.perf_counter()-tick; event('completed', label=scope, source=first['pin'], cold=cold['pin'])
        seconds = time.perf_counter()-began
        need([v['label'] for v in outputs] == list(STAGES) and .9 <= sum(phases.values())/seconds <= 1.001, 'FREE_QUEUE_PHASES')
        write(a.output / 'report.json', dict(complete=True, authority=a.pin, proof=a.proof_pin, driver_pin=sha(__file__),
            handoff=sha(a.output / 'handoff.json'), outputs=outputs, policy=POLICY, query_cases=170, full_training=True,
            free_generation_tested=True, teacher_forced=False, capability_demonstrated=False, architecture_winner=False,
            native_guard_pending=True, outcome_checks_pending=True, optimizer_updates=0, encoder_calls=0,
            phases=phases, seconds_after_wait=seconds, acceptance='author-checked'))
        event('terminal', pin=sha(a.output / 'report.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--prepare', action='store_true')
    for name in ('authority', 'proof', 'output'): p.add_argument('--'+name, type=Path)
    p.add_argument('--pin'); p.add_argument('--proof-pin'); a = p.parse_args()
    try:
        if a.prepare: print(json.dumps(dict(pin=prepare(a.output))), flush=True)
        else: run(a)
    except BaseException as exc:
        if a.output is not None and a.output.exists(): write(a.output / 'failure.json', dict(error=repr(exc)))
        raise
