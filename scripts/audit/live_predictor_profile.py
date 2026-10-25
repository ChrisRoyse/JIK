"""Read-only, identity-bound sampling of an existing MNLM GPU worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
WORK = ROOT/'phase115-complete-answer-recovery-v1-20261024/grouped/full'
SCRIPT = 'scripts/audit/legal_grouped_placement_training.py'
TOOL = Path('/zfs/hot/mnlm/tools/py-spy-0.4.2/py-spy')
TOOL_PIN = '9b4d1f39b2a47ae44f4c6a46f615dcc0287d7755beba5065f32391951e07d594'


def need(ok, code):
    if not ok: raise ValueError('LIVE_PROFILE_'+code)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2); f.flush(); os.fsync(f.fileno())


def process(pid, start):
    root = Path('/proc')/str(pid)
    fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    need(fields[19] == start and fields[0] != 'Z', 'PROCESS_IDENTITY')
    command = (root/'cmdline').read_bytes()
    need(str(REPO/SCRIPT).encode() in command.split(b'\0'), 'PROCESS_SCRIPT')
    status = dict(line.split(':', 1) for line in (root/'status').read_text().splitlines() if ':' in line)
    io = dict((k, int(v)) for k, v in (line.split(':', 1) for line in (root/'io').read_text().splitlines()))
    stat = (root/'exe').stat()
    return dict(pid=pid, start=start, state=fields[0], cpu_ticks=int(fields[11])+int(fields[12]),
        monotonic=time.monotonic(), utc=time.time(), threads=int(status['Threads']),
        rss_bytes=int(status['VmRSS'].split()[0])*1024, hwm_bytes=int(status['VmHWM'].split()[0])*1024,
        command_sha256=hashlib.sha256(command).hexdigest(), executable=str((root/'exe').resolve()),
        executable_stat=[stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns], io=io)


def progress(output, label):
    ledger = WORK/'shared/steps.jsonl'; size = ledger.stat().st_size
    need(size > 0, 'EMPTY_LEDGER')
    with ledger.open('rb') as f:
        f.seek(max(0, size-(1 << 20))); raw = f.read(min(size, 1 << 20))
    end = raw.rfind(b'\n'); begin = raw.rfind(b'\n', 0, end)
    need(begin >= 0 and end > begin, 'LEDGER_TAIL_REQUIRES_LARGER_TILE')
    row = raw[begin+1:end]; value = json.loads(row)
    path = output/(label+'-ledger.json'); path.write_bytes(row+b'\n')
    record = {k: v for k, v in value.items() if k != 'hash'}
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    need(hashlib.sha256(canonical).hexdigest() == value['hash'], 'LEDGER_RECORD_HASH')
    return dict(path=path.name, pin=sha(path), file_size=size, cursor=value['cursor'], hash=value['hash'],
                document=value['update']['document'], targets=value['update']['targets'])


def command(args):
    return subprocess.check_output(args, text=True)


def controls(pid, start, before):
    result = {}
    for label, call in (
        ('wrong_start', lambda: process(pid, str(int(start)+1))),
        ('wrong_tool_pin', lambda: need(sha(TOOL) == '0'*64, 'TOOL_PIN')),
        ('changed_executable_identity', lambda: need(before['executable_stat'] == [0]*4, 'EXECUTABLE_CHANGED'))):
        try: call()
        except ValueError as exc: result[label] = str(exc)
        else: raise ValueError('LIVE_PROFILE_CONTROL_ACCEPTED')
    return result


def run(a):
    a.output.mkdir(exist_ok=False); began = time.monotonic()
    need(sha(TOOL) == TOOL_PIN, 'TOOL_PIN')
    before = process(a.pid, a.start); executable_pin = sha(Path('/proc')/str(a.pid)/'exe')
    code = {n: sha(REPO/n) for n in (SCRIPT, 'scripts/model/native_placement_efficient.py',
                                    'scripts/model/native_device_transport.py', 'scripts/model/loop_training_checkpoint.py')}
    negative = controls(a.pid, a.start, before)
    need(command(['sudo', '-n', str(TOOL), '--version']).strip() == 'py-spy 0.4.2', 'TOOL_VERSION')
    residents = command(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])
    need([int(x.split(',')[0]) for x in residents.splitlines()] == [a.pid], 'GPU_OWNER')
    write(a.output/'before.json', dict(process=before, gpu=residents, executable_pin=executable_pin, code=code))
    window_rows = []
    handle = os.pidfd_open(a.pid)
    try:
        for index in range(2):
            process(a.pid, a.start); prefix = f'window-{index+1}'
            first = process(a.pid, a.start); initial = progress(a.output, prefix+'-before')
            profile = a.output/(prefix+'.json'); log = a.output/(prefix+'-profiler.log')
            gpu_path = a.output/(prefix+'-gpu.csv'); telemetry = a.output/(prefix+'-process.jsonl')
            args = ['sudo', '-n', str(TOOL), 'record', '--pid', str(a.pid), '--duration', '30',
                    '--rate', '99', '--threads', '--idle', '--nonblocking', '--full-filenames', '--format', 'speedscope',
                    '--output', str(profile)]
            sampler = gpu = None
            with log.open('xb') as logs, gpu_path.open('xb') as gpu_log, telemetry.open('x') as samples:
                try:
                    gpu = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,'
                        'memory.used,power.draw,clocks.sm', '--format=csv,noheader,nounits', '--loop-ms=1000'],
                        stdout=gpu_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    sampler = subprocess.Popen(args, stdout=logs, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    while sampler.poll() is None:
                        value = process(a.pid, a.start)
                        need(value['executable_stat'] == before['executable_stat']
                             and value['command_sha256'] == before['command_sha256'], 'EXECUTABLE_CHANGED')
                        samples.write(json.dumps(value, sort_keys=True)+'\n'); samples.flush()
                        time.sleep(1)
                    status = sampler.wait(); need(status == 0, 'SAMPLER_FAILED')
                finally:
                    # Terminate only diagnostic children created by this observer.
                    if sampler is not None and sampler.poll() is None: sampler.terminate(); sampler.wait()
                    if gpu is not None and gpu.poll() is None: gpu.terminate(); gpu.wait()
                    for f in (logs, gpu_log, samples): f.flush(); os.fsync(f.fileno())
            need(profile.is_file() and profile.stat().st_size < 32 << 20, 'PROFILE_OUTPUT')
            last = process(a.pid, a.start); final = progress(a.output, prefix+'-after')
            need(final['cursor'] > initial['cursor'], 'NO_DURABLE_PROGRESS')
            window_rows.append(dict(name=prefix, before=first, after=last, initial=initial, final=final,
                command=args, profiler_exit=status, duration=30, rate=99,
                files={p.name: sha(p) for p in (profile, log, gpu_path, telemetry)}))
    finally: os.close(handle)
    after = process(a.pid, a.start)
    need(sha(Path('/proc')/str(a.pid)/'exe') == executable_pin and sha(TOOL) == TOOL_PIN
         and all(sha(REPO/n) == pin for n, pin in code.items()), 'BOUNDARY_CHANGED')
    residents_after = command(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])
    need([int(x.split(',')[0]) for x in residents_after.splitlines()] == [a.pid], 'GPU_OWNER')
    write(a.output/'after.json', dict(process=after, gpu=residents_after))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    report = dict(complete=True, process=dict(pid=a.pid, start=a.start), code=code, tool=TOOL_PIN,
        executable=executable_pin, windows=window_rows, controls=negative, clock_ticks=os.sysconf('SC_CLK_TCK'),
        seconds=time.monotonic()-began, source=sha(__file__), target_paused=False, target_modified=False,
        gpu_job_started=False, promoted=False, acceptance='author-checked')
    report['files'] = {p.name: sha(p) for p in a.output.iterdir() if p.is_file()}
    write(a.output/'report.json', report)
    print(json.dumps(dict(complete=True, pin=sha(a.output/'report.json'), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True); p.add_argument('--start', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
