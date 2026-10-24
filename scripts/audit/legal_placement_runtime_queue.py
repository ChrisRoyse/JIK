"""Run paired GPU placement proofs after the existing training queue has exited."""
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path, pin):
    if sha(path) != pin: raise ValueError('PLACEMENT_QUEUE_PIN')
    return json.loads(Path(path).read_bytes())


def identity(pid):
    try: fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError: return None
    return None if fields[0] == 'Z' else fields[19]


def run(a):
    repo = Path(__file__).resolve().parents[2]
    lease = (repo/'.lease/legal-placement-runtime-queue').open('a+')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (a.output/name).open('x') as f:
            json.dump(value, f, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    def emit(stage, **values):
        row = dict(stage=stage, timestamp=time.time(), **values)
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(row, sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(row), flush=True)
    self_pin = sha(__file__); (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    write('launch.json', dict(pid=os.getpid(), starttime=identity(os.getpid()), driver_pin=self_pin,
          wait_pid=a.wait_pid, wait_start=a.wait_start, wait_root=str(a.wait_root), source=a.source_pin,
          cpu=a.cpu_pin, cpu_cold=a.cpu_cold_pin))
    results = []
    try:
        source = read(a.source/'report.json', a.source_pin)
        q = read(a.cpu/'report.json', a.cpu_pin); c = read(a.cpu_cold/'report.json', a.cpu_cold_pin)
        assert source['complete'] and q['complete'] and c['complete']
        assert q['cpu_reference_exact'] and c['cpu_reference_exact'] and q['all_updates_exact'] and c['all_updates_exact']
        assert q['binding'] == c['binding'] and q['binding']['source'] == a.source_pin
        assert c['previous_pin'] == a.cpu_pin and q['results_pin'] == c['results_pin']
        assert q['binding']['device'] == 'cpu' and q['binding']['code'] == source['code']
        authenticated = {a.source/'report.json': a.source_pin, a.source/'bundle.pt': source['bundle_sha256'],
                         a.cpu/'report.json': a.cpu_pin, a.cpu_cold/'report.json': a.cpu_cold_pin}
        for root, report in ((a.cpu, q), (a.cpu_cold, c)):
            authenticated.update({root/n: pin for n, pin in report['files'].items()})
        for path, pin in authenticated.items(): assert sha(path) == pin, 'PLACEMENT_QUEUE_PROOF_BYTES'
        stamps = {path: path.stat() for path in authenticated}
        owner = json.loads((a.wait_root/'launch.json').read_bytes())
        assert owner['pid'] == a.wait_pid and str(owner['starttime']) == a.wait_start, 'PLACEMENT_QUEUE_OWNER_BINDING'
        def check():
            assert sha(__file__) == self_pin, 'PLACEMENT_QUEUE_DRIVER_CHANGED'
            for name, pin in source['code'].items():
                assert sha(repo/'scripts'/name) == pin, 'PLACEMENT_QUEUE_CODE_CHANGED'
            for path, old in stamps.items():
                now = path.stat()
                assert (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns) == (
                    old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns, old.st_ctime_ns), 'PLACEMENT_QUEUE_PROOF_CHANGED'
        check()
        current = identity(a.wait_pid)
        if current == a.wait_start:
            fd = os.pidfd_open(a.wait_pid)
            try:
                assert identity(a.wait_pid) in (a.wait_start, None), 'PLACEMENT_QUEUE_OWNER_REUSED'
                emit('waiting_for_training_owner', pid=a.wait_pid, starttime=a.wait_start, wait='kernel_pidfd')
                select.select([fd], [], [])
            finally: os.close(fd)
        assert identity(a.wait_pid) != a.wait_start, 'PLACEMENT_QUEUE_OWNER_STILL_LIVE'
        # A terminal process alone does not certify the preceding experiment completed.
        prior = json.loads((a.wait_root/'report.json').read_bytes())
        assert prior['complete'], 'PLACEMENT_QUEUE_PRIOR_NOT_COMPLETE'
        expected = {'breadth', 'breadth-cold', 'repeated', 'repeated-cold'}
        assert expected.issubset({r['label'] for r in prior['outputs']}), 'PLACEMENT_QUEUE_PRIOR_MISSING_ARM'
        check(); emit('training_owner_terminal', report_pin=sha(a.wait_root/'report.json'))
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        # An inherited CPU-proof mask must not silently turn this into another CPU run.
        assert env.get('CUDA_VISIBLE_DEVICES') != '', 'PLACEMENT_QUEUE_GPU_HIDDEN'
        for label in ('gpu', 'gpu-cold'):
            check(); target = a.output/label
            command = [sys.executable, '-I', str(repo/'scripts/audit/legal_placement_runtime_readback.py'),
                       '--source', str(a.source), '--source-pin', a.source_pin, '--output', str(target)]
            if results: command += ['--previous', results[0]['path'], '--previous-pin', results[0]['pin']]
            with (a.output/(label+'.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                emit('running', label=label, pid=child.pid, starttime=identity(child.pid), command=command)
                status = child.wait(); log.flush(); os.fsync(log.fileno())
            emit('terminal', label=label, exit_code=status)
            assert status == 0, 'PLACEMENT_QUEUE_CHILD_FAILED'
            report = json.loads((target/'report.json').read_bytes())
            assert report['complete'] and report['gpu_qualified'] and report['all_updates_exact']
            check(); results.append(dict(label=label, path=str(target), pin=sha(target/'report.json')))
        write('report.json', dict(complete=True, outputs=results, source=a.source_pin, cpu=a.cpu_pin,
              cpu_cold=a.cpu_cold_pin, driver_pin=self_pin, acceptance='author-checked', scientific_admission=False))
        emit('queue_terminal', complete=True, pin=sha(a.output/'report.json'))
    except BaseException as error:
        write('failure.json', dict(error=repr(error), outputs=results)); emit('queue_terminal', complete=False)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'cpu', 'cpu-cold'):
        p.add_argument('--'+name, type=Path, required=True); p.add_argument('--'+name+'-pin', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--wait-root', type=Path, required=True)
    p.add_argument('--wait-pid', type=int, required=True); p.add_argument('--wait-start', required=True)
    run(p.parse_args())
