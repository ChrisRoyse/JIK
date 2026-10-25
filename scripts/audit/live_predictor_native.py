"""Observe user-space CPU samples in the identity-bound live training worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import live_predictor_profile as observer


def run(a):
    a.output.mkdir(exist_ok=False); began = time.monotonic()
    before = observer.process(a.pid, a.start)
    pins = {str(p): observer.sha(p) for p in (Path('/usr/bin/perf'), Path(__file__), Path(observer.__file__))}
    code = observer.sha(observer.REPO/observer.SCRIPT)
    first = observer.progress(a.output, 'before')
    residents = observer.command(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])
    observer.need([int(s.split(',')[0]) for s in residents.splitlines()] == [a.pid], 'GPU_OWNER')
    observer.write(a.output/'before.json', dict(process=before, gpu=residents, pins=pins, source=code))
    args = ['sudo', '-n', 'env', 'DEBUGINFOD_URLS=', '/usr/bin/perf', 'record', '--no-buildid',
            '--no-buildid-cache', '-e', 'cpu-clock:u', '-F', '99', '--call-graph', 'fp',
            '-p', str(a.pid), '-o', str(a.output/'perf.data'), '--', 'sleep', '30']
    handle = os.pidfd_open(a.pid)
    try:
        with (a.output/'perf-record.log').open('xb') as log, (a.output/'process.jsonl').open('x') as telemetry:
            child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    row = observer.process(a.pid, a.start)
                    observer.need(row['executable_stat'] == before['executable_stat']
                        and row['command_sha256'] == before['command_sha256'], 'EXECUTABLE_CHANGED')
                    telemetry.write(json.dumps(row, sort_keys=True)+'\n'); telemetry.flush(); time.sleep(1)
                observer.need(child.wait() == 0, 'PERF_RECORD_FAILED')
            finally:
                if child.poll() is None: child.terminate(); child.wait()
                for f in (log, telemetry): f.flush(); os.fsync(f.fileno())
    finally: os.close(handle)
    # perf runs with ptrace privileges; only its newly created diagnostic file changes owner.
    subprocess.run(['sudo', '-n', 'chown', f'{os.getuid()}:{os.getgid()}',
                    str(a.output/'perf.data')], check=True)
    after = observer.process(a.pid, a.start); last = observer.progress(a.output, 'after')
    observer.need(last['cursor'] > first['cursor'] and observer.sha(observer.REPO/observer.SCRIPT) == code
        and all(observer.sha(Path(p)) == h for p, h in pins.items()), 'BOUNDARY_CHANGED')
    observer.write(a.output/'after.json', after)
    outputs = {}
    for name, options in (
        ('histogram.txt', ['report', '--stdio', '--stdio-color', 'never', '--no-children', '--percent-limit', '0',
                           '--show-nr-samples', '--sort', 'comm,dso,symbol', '--field-separator', ';']),
        ('events.txt', ['script', '--no-inline', '-F', 'comm,pid,tid,time,event,ip,sym,dso'])):
        command = ['/usr/bin/perf', *options, '-i', str(a.output/'perf.data')]
        with (a.output/name).open('xb') as out, (a.output/(name+'.log')).open('xb') as err:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    env={**os.environ, 'DEBUGINFOD_URLS': ''})
        observer.need(result.returncode == 0, 'PERF_READ_FAILED'); outputs[name] = command
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    report = dict(complete=True, pid=a.pid, start=a.start, command=args, reader_commands=outputs,
        first=first, last=last, pins=pins, seconds=time.monotonic()-began,
        gpu_kernel_timing=False, exact_wall_attribution=False, target_modified=False, promoted=False,
        acceptance='author-checked', files={p.name: observer.sha(p) for p in a.output.iterdir() if p.is_file()})
    observer.write(a.output/'report.json', report)
    print(json.dumps(dict(complete=True, pin=observer.sha(a.output/'report.json'))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True); p.add_argument('--start', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): observer.write(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
