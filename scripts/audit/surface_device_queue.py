"""Wait for the retained predictor comparison, then characterize CUDA and cold-recount."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def need(ok, code):
    if not ok: raise ValueError(code)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open('xb') as f:
        f.write(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        f.flush(); os.fsync(f.fileno())


def identity(pid):
    try: fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError: return None
    return fields[19] if fields[0] != 'Z' else None


def stamp(path):
    s = path.stat(); return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def check_owner(owner):
    path = Path(owner['path'])/'launch.json'
    need(sha(path) == owner['launch_pin'], 'SURFACE_QUEUE_OWNER_LAUNCH_PIN')
    launch = json.loads(path.read_bytes())
    need(launch['pid'] == owner['pid'] and str(launch['starttime']) == owner['starttime'], 'SURFACE_QUEUE_OWNER_IDENTITY')
    need(identity(owner['pid']) in (owner['starttime'], None), 'SURFACE_QUEUE_PID_REUSED')


def authenticate(path, pin):
    need(sha(path) == pin, 'SURFACE_QUEUE_AUTHORITY_PIN'); contract = json.loads(path.read_bytes())
    repo = Path(__file__).resolve().parents[2]; files = {path: pin}; reports = {}
    for label, ref in contract['proofs'].items():
        root = Path(ref['path']); p = root/'report.json'; need(sha(p) == ref['pin'], 'SURFACE_QUEUE_PROOF_PIN')
        r = json.loads(p.read_bytes()); reports[label] = r; files[p] = ref['pin']
        need(r['complete'] and r['code'] == contract['code'], 'SURFACE_QUEUE_PROOF_CODE')
        for n, h in r.get('files', {}).items(): files[root/n] = h
    first, cold, recount = (reports[n] for n in ['cpu', 'cold', 'readback'])
    need(first['environment'] == cold['environment'] == recount['environment'] and first['environment']['cpu_proof']
         and first['environment']['device'] == 'cpu', 'SURFACE_QUEUE_CPU_SCOPE')
    need(recount['input_pin'] == contract['proofs']['cpu']['pin'] and recount['cold_pin'] == contract['proofs']['cold']['pin']
         and recount['all_cold_arm_states_exact'] and recount['same_cohort_exact'], 'SURFACE_QUEUE_CPU_READBACK')
    need(len(recount['states']) == 9 and all(v['all_atoms']['atoms'] == 215 and v['all_atoms']['scored_positions'] == 42738
         for v in recount['counts'].values()), 'SURFACE_QUEUE_PROOF_COVERAGE')
    owner = contract['wait']; check_owner(owner)
    files[Path(owner['path'])/'launch.json'] = owner['launch_pin']
    need(contract['code']['audit/surface_device_queue.py'] == sha(__file__), 'SURFACE_QUEUE_DRIVER')
    for p, h in files.items(): need(sha(p) == h, 'SURFACE_QUEUE_PHYSICAL_FILE')
    stamps = {p: stamp(p) for p in files}
    def fresh():
        for n, h in contract['code'].items(): need(sha(repo/'scripts'/n) == h, 'SURFACE_QUEUE_CODE_CHANGED')
        for p, s in stamps.items(): need(stamp(p) == s, 'SURFACE_QUEUE_AUTHORITY_CHANGED')
    fresh(); return contract, fresh


def predecessor(owner):
    root = Path(owner['path']); p = root/'report.json'; r = json.loads(p.read_bytes())
    need(r['complete'], 'SURFACE_QUEUE_PREDECESSOR_FAILED')
    need(r['driver_pin'] == json.loads((root/'launch.json').read_bytes())['driver_pin'], 'SURFACE_QUEUE_PREDECESSOR_DRIVER')
    outputs = {v['label']: v for v in r['outputs']}
    need({'full', 'full-readback', 'reader-match'}.issubset(outputs), 'SURFACE_QUEUE_PREDECESSOR_MISSING')
    for v in outputs.values():
        report = Path(v['path'])/'report.json'
        need(sha(report) == v['pin'] and json.loads(report.read_bytes())['complete'], 'SURFACE_QUEUE_PREDECESSOR_OUTPUT')
    return sha(p)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False); repo = Path(__file__).resolve().parents[2]
    outputs = []; lock = None
    def event(stage, **values):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage, timestamp=time.time(), **values), sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
    try:
        contract, fresh = authenticate(a.authority, a.authority_pin); owner = contract['wait']
        (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
        if a.mode == 'probe':
            write(a.output/'report.json', dict(complete=True, authority=a.authority_pin, wait_identity=identity(owner['pid']),
                code=contract['code'], gpu_children_started=0, acceptance='author-checked'))
            return
        lock = (repo/'.lease/surface-device-queue').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(a.output/'launch.json', dict(pid=os.getpid(), starttime=identity(os.getpid()), authority=a.authority_pin,
            driver_pin=sha(__file__), wait=owner))
        if identity(owner['pid']) == owner['starttime']:
            fd = os.pidfd_open(owner['pid'])
            try:
                need(identity(owner['pid']) in (owner['starttime'], None), 'SURFACE_QUEUE_PID_REUSED')
                event('waiting_for_predictor_comparison', pid=owner['pid'], starttime=owner['starttime'], wait='kernel_pidfd')
                select.select([fd], [], [])
            finally: os.close(fd)
        prior = predecessor(owner); fresh(); event('predecessor_terminal', pin=prior)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        need(env.get('CUDA_VISIBLE_DEVICES') != '', 'SURFACE_QUEUE_GPU_HIDDEN')
        def execute(label, script, arguments):
            fresh(); command = [sys.executable, '-I', str(repo/'scripts/audit'/script), *arguments, '--output', str(a.output/label)]
            with (a.output/(label+'.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                event('running', label=label, pid=child.pid, starttime=identity(child.pid), command=command)
                status = child.wait(); log.flush(); os.fsync(log.fileno())
            event('terminal', label=label, exit_code=status); need(status == 0, 'SURFACE_QUEUE_CHILD_FAILED')
            path = a.output/label; need(json.loads((path/'report.json').read_bytes())['complete'], 'SURFACE_QUEUE_CHILD_INCOMPLETE')
            value = dict(label=label, path=str(path), pin=sha(path/'report.json')); outputs.append(value); fresh(); return value
        first = execute('gpu', 'surface_device_proof.py', [])
        cold = execute('gpu-cold', 'surface_device_proof.py', ['--reverse'])
        execute('readback', 'surface_device_readback.py', ['--input', first['path'], '--input-pin', first['pin'],
            '--cold', cold['path'], '--cold-pin', cold['pin']])
        write(a.output/'report.json', dict(complete=True, authority=a.authority_pin, outputs=outputs, predecessor_pin=prior,
            driver_pin=sha(__file__), promoted_to_training=False, scientific_admission=False, capability_demonstrated=False,
            acceptance='author-checked'))
        event('queue_terminal', complete=True)
    except BaseException as e:
        write(a.output/'failure.json', dict(error=repr(e), outputs=outputs)); event('queue_terminal', complete=False); raise
    finally:
        if lock: lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['probe', 'full'], required=True)
    p.add_argument('--authority', type=Path, required=True); p.add_argument('--authority-pin', required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
