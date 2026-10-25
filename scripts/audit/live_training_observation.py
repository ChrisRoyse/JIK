"""External, nonblocking observation of an explicitly bound existing GPU worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.live_predictor_profile import TOOL, TOOL_PIN, command, sha, write

REPO = Path(__file__).resolve().parents[2]


def need(ok, code):
    if not ok: raise ValueError('TRAIN_OBSERVE_'+code)


def process(pid, start, script):
    root = Path('/proc')/str(pid)
    fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    need(fields[19] == start and fields[0] != 'Z', 'PROCESS_IDENTITY')
    argv = (root/'cmdline').read_bytes()
    need(str(REPO/script).encode() in argv.split(b'\0'), 'PROCESS_SCRIPT')
    status = dict(line.split(':', 1) for line in (root/'status').read_text().splitlines() if ':' in line)
    info = (root/'exe').stat()
    threads = []
    for p in sorted((root/'task').iterdir()):
        try:
            raw = (p/'stat').read_text(); fs = raw.rsplit(')', 1)[1].split()
            threads.append(dict(tid=int(p.name), name=raw.split('(', 1)[1].rsplit(')', 1)[0],
                                state=fs[0], cpu_ticks=int(fs[11])+int(fs[12])))
        except FileNotFoundError: continue  # A thread may exit between directory and stat reads.
    return dict(pid=pid, start=start, state=fields[0], utc=time.time(), monotonic=time.monotonic(),
        cpu_ticks=int(fields[11])+int(fields[12]), threads=threads,
        rss_bytes=int(status['VmRSS'].split()[0])*1024, hwm_bytes=int(status['VmHWM'].split()[0])*1024,
        executable=str((root/'exe').resolve()), executable_stat=[info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns],
        command_sha256=hashlib.sha256(argv).hexdigest(),
        io={k: int(v) for k, v in (line.split(':', 1) for line in (root/'io').read_text().splitlines())})


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def progress(ledger, output, name):
    size = ledger.stat().st_size; need(size > 0, 'EMPTY_LEDGER')
    with ledger.open('rb') as f:
        f.seek(max(0, size-(1 << 20))); raw = f.read(min(size, 1 << 20))
    end = raw.rfind(b'\n'); begin = raw.rfind(b'\n', 0, end)
    need(begin >= 0 and end > begin, 'LEDGER_TAIL_TILE')
    row = raw[begin+1:end]; value = json.loads(row)
    need(hashlib.sha256(canonical({k: v for k, v in value.items() if k != 'hash'})).hexdigest()
         == value['hash'], 'LEDGER_HASH')
    path = output/(name+'-ledger.json'); path.write_bytes(row+b'\n')
    return dict(path=path.name, pin=sha(path), file_size=size, cursor=value['cursor'], hash=value['hash'])


def source_pins(a):
    need(sha(a.authority) == a.authority_pin, 'AUTHORITY_PIN')
    report = json.loads(a.authority.read_bytes()); pins = report['code']
    need(pins and all(sha(REPO/'scripts'/name) == pin for name, pin in pins.items()), 'ACTIVE_CODE_CHANGED')
    return pins


def gpu_owner(pid):
    raw = command(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])
    need([int(s.split(',')[0]) for s in raw.splitlines()] == [pid], 'GPU_OWNER')
    return raw


def observe_window(a, name, args):
    first = process(a.pid, a.start, a.script); initial = progress(a.ledger, a.output, name+'-before')
    gpu_owner(a.pid); gpu = child = None
    with (a.output/(name+'-profiler.log')).open('xb') as log, \
         (a.output/(name+'-gpu.csv')).open('xb') as gpu_log, \
         (a.output/(name+'-process.jsonl')).open('x') as telemetry:
        try:
            gpu = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,'
                'memory.used,power.draw,clocks.sm', '--format=csv,noheader,nounits', '--loop-ms=1000'],
                stdout=gpu_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                row = process(a.pid, a.start, a.script)
                need(all(row[k] == first[k] for k in ('command_sha256', 'executable_stat')), 'WORKER_CHANGED')
                telemetry.write(json.dumps(row, sort_keys=True)+'\n'); telemetry.flush(); time.sleep(1)
            need(child.wait() == 0, 'SAMPLER_FAILED')
        finally:
            # Only this diagnostic's children; never signal the observed training worker.
            for p in (child, gpu):
                if p is not None and p.poll() is None: p.terminate(); p.wait()
            for f in (log, gpu_log, telemetry): f.flush(); os.fsync(f.fileno())
    last = process(a.pid, a.start, a.script); final = progress(a.ledger, a.output, name+'-after')
    need(final['cursor'] > initial['cursor'], 'NO_DURABLE_PROGRESS'); gpu_owner(a.pid)
    return dict(name=name, command=args, before=first, after=last, initial=initial, final=final,
                duration=30, rate=99, profiler_exit=0)


def run(a):
    a.output.mkdir(exist_ok=False); began = time.monotonic(); pins = source_pins(a)
    need(sha(TOOL) == TOOL_PIN, 'TOOL_PIN')
    before = process(a.pid, a.start, a.script); negative = {}
    for label, call in (
        ('wrong_start', lambda: process(a.pid, str(int(a.start)+1), a.script)),
        ('wrong_script', lambda: process(a.pid, a.start, 'scripts/audit/not-the-worker.py')),
        ('wrong_tool_pin', lambda: need(sha(TOOL) == '0'*64, 'TOOL_PIN'))):
        try: call()
        except ValueError as exc: negative[label] = str(exc)
        else: raise ValueError('TRAIN_OBSERVE_CONTROL_ACCEPTED')
    own_code = {str(p.relative_to(REPO)): sha(p) for p in
                (Path(__file__), REPO/'scripts/audit/live_predictor_profile.py')}
    write(a.output/'before.json', dict(process=before, gpu=gpu_owner(a.pid), code=pins))
    tool_pins = {str(TOOL): TOOL_PIN, '/usr/bin/perf': sha('/usr/bin/perf')}
    windows = []; handle = os.pidfd_open(a.pid)
    try:
        for i in range(2):
            name = f'python-{i+1}'
            args = ['sudo', '-n', str(TOOL), 'record', '--pid', str(a.pid), '--duration', '30',
                    '--rate', '99', '--threads', '--idle', '--nonblocking', '--full-filenames',
                    '--format', 'speedscope', '--output', str(a.output/(name+'.json'))]
            windows.append(observe_window(a, name, args))
        args = ['sudo', '-n', 'env', 'DEBUGINFOD_URLS=', '/usr/bin/perf', 'record', '--no-buildid',
                '--no-buildid-cache', '-e', 'cpu-clock:u', '-F', '99', '--call-graph', 'fp',
                '-p', str(a.pid), '-o', str(a.output/'perf.data'), '--', 'sleep', '30']
        windows.append(observe_window(a, 'native', args))
    finally: os.close(handle)
    subprocess.run(['sudo', '-n', 'chown', f'{os.getuid()}:{os.getgid()}', str(a.output/'perf.data')], check=True)
    readers = {}
    for name, options in (
        ('histogram.txt', ['report', '--stdio', '--stdio-color', 'never', '--no-children', '--percent-limit', '0',
                           '--show-nr-samples', '--sort', 'comm,dso,symbol', '--field-separator', ';']),
        ('events.txt', ['script', '--no-inline', '-F', 'comm,pid,tid,time,event,ip,sym,dso'])):
        args = ['/usr/bin/perf', *options, '-i', str(a.output/'perf.data')]
        with (a.output/name).open('xb') as out, (a.output/(name+'.log')).open('xb') as err:
            status = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    env={**os.environ, 'DEBUGINFOD_URLS': ''}).returncode
        need(status == 0, 'PERF_READ_FAILED'); readers[name] = args
    need(source_pins(a) == pins and all(sha(REPO/n) == pin for n, pin in own_code.items())
         and all(sha(p) == pin for p, pin in tool_pins.items()), 'FINAL_PINS')
    write(a.output/'after.json', dict(process=process(a.pid, a.start, a.script), gpu=gpu_owner(a.pid)))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    report = dict(complete=True, binding=dict(pid=a.pid, start=a.start, script=a.script, ledger=str(a.ledger),
        authority=str(a.authority), authority_pin=a.authority_pin), code=pins, observer=own_code, tools=tool_pins,
        windows=windows, readers=readers, controls=negative, clock_ticks=os.sysconf('SC_CLK_TCK'),
        seconds=time.monotonic()-began, acceptance='author-checked', target_modified=False, target_paused=False,
        gpu_job_started=False, gpu_kernel_timing=False, speedup_claim=False,
        files={p.name: sha(p) for p in a.output.iterdir() if p.is_file()})
    write(a.output/'report.json', report)
    print(json.dumps(dict(complete=True, pin=sha(a.output/'report.json'), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True); p.add_argument('--start', required=True)
    p.add_argument('--script', required=True); p.add_argument('--authority-pin', required=True)
    for name in ('ledger', 'authority', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
