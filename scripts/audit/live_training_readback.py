"""Independently recount live-worker observations and the committed update ranges."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.live_predictor_readback import identity, ledger_record, stacks, summary
from audit.live_native_readback import records, histogram
from audit.live_predictor_profile import sha, write

REPO = Path(__file__).resolve().parents[2]


def need(ok, code):
    if not ok: raise ValueError('TRAIN_OBSERVE_READ_'+code)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def recount_window(w, raw, report, output):
    name = w['name']; reference = json.loads(raw['before.json'])['process']
    processes = [w['before']]+[json.loads(x) for x in raw[name+'-process.jsonl'].splitlines()]+[w['after']]
    for p in processes: identity(p, reference)
    need(all(y['monotonic'] >= x['monotonic'] and y['cpu_ticks'] >= x['cpu_ticks']
             for x, y in zip(processes, processes[1:])), 'PROCESS_ORDER')
    elapsed = processes[-1]['monotonic']-processes[0]['monotonic']
    cpu = (processes[-1]['cpu_ticks']-processes[0]['cpu_ticks'])/report['clock_ticks']
    gpu = list(csv.reader(io.StringIO(raw[name+'-gpu.csv'].decode())))
    need(gpu and all(len(r) == 6 for r in gpu) and all(y[0] > x[0] for x, y in zip(gpu, gpu[1:])), 'GPU_ROWS')
    columns = ('utilization_percent', 'memory_utilization_percent', 'memory_mib', 'power_watts', 'clock_mhz')
    gpu_summary = {k: summary([float(r[i+1]) for r in gpu]) for i, k in enumerate(columns)}
    ends = []
    for key in ('initial', 'final'):
        e = w[key]; need(report['files'][e['path']] == e['pin'], 'LEDGER_PIN')
        value = json.loads(raw[e['path']]); ledger_record(value)
        need(value['cursor'] == e['cursor'] and value['hash'] == e['hash'], 'LEDGER_CURSOR'); ends.append(value)
    lo = max(0, w['initial']['file_size']-(1 << 20)); hi = w['final']['file_size']
    need(0 < hi-lo < 4 << 20, 'LEDGER_TILE')
    ledger = Path(report['binding']['ledger'])
    with ledger.open('rb') as f: f.seek(lo); region = f.read(hi-lo)
    need(len(region) == hi-lo, 'LEDGER_TRUNCATED')
    selected = []
    for line in region.split(b'\n')[1 if lo else 0:-1]:
        v = json.loads(line)
        if ends[0]['cursor'] < v['cursor'] <= ends[1]['cursor']: selected.append((line, v))
    need(len(selected) == ends[1]['cursor']-ends[0]['cursor'] > 0, 'LEDGER_COUNT')
    parent = ends[0]; targets = 0
    for line, value in selected:
        ledger_record(value)
        need(value['parent'] == parent['hash'] and value['cursor'] == parent['cursor']+1, 'LEDGER_CHAIN')
        need(value['update']['before'] == parent['update']['after'], 'MODEL_STATE_CHAIN')
        parent = value; targets += value['update']['targets']
    need(parent == ends[-1], 'LEDGER_ENDPOINT')
    retained = b''.join(line+b'\n' for line, _ in selected)
    (output/(name+'-updates.jsonl')).write_bytes(retained)
    first_threads = {p['tid']: p for p in processes[0]['threads']}
    thread_cpu = []
    for p in processes[-1]['threads']:
        if p['tid'] in first_threads:
            old = first_threads[p['tid']]; need(p['name'] == old['name'] and p['cpu_ticks'] >= old['cpu_ticks'], 'THREAD')
            thread_cpu.append(dict(tid=p['tid'], name=p['name'], seconds=(p['cpu_ticks']-old['cpu_ticks'])/report['clock_ticks']))
    return dict(name=name, elapsed_seconds=elapsed, process_cpu_seconds=cpu, core_equivalents=cpu/elapsed,
        threads=thread_cpu, gpu=gpu_summary, rss_bytes=summary([p['rss_bytes'] for p in processes]),
        io_delta={k: processes[-1]['io'][k]-processes[0]['io'][k] for k in processes[0]['io']},
        durable_updates=len(selected), durable_targets=targets, first_cursor=ends[0]['cursor'],
        last_cursor=ends[1]['cursor'], retained_updates=digest(retained)), ends[-1]


def native_summary(raw, pid):
    rows = records(raw['events.txt'].decode(), pid)
    hist, hist_rows, lost = histogram(raw['histogram.txt'].decode())
    groups = Counter(); threads = Counter(); libraries = Counter(); launch = Counter(); unknown = 0
    for row in rows:
        physical = next((f for f in row['frames'] if f['dso'] != 'inlined'), None)
        need(physical is not None, 'PHYSICAL_FRAME')
        dso = Path(physical['dso']).name
        groups[row['comm'], dso] += 1; threads[row['comm']] += 1; libraries[dso] += 1
        unknown += row['frames'][0]['symbol'] == '[unknown]'
        if any('cuLaunchKernel' in f['symbol'] or 'cudaLaunchKernel' in f['symbol'] for f in row['frames']):
            launch[row['comm']] += 1
    need(hist == groups, 'HISTOGRAM_COUNTS')
    log = raw['native-profiler.log'].decode(); counts = re.findall(r'\((\d+) samples\)', log)
    need('Captured and wrote' in log and len(counts) <= 1 and (not counts or int(counts[0]) == len(rows)), 'CAPTURE_LOG')
    return dict(samples=len(rows), lost_samples=lost, threads=dict(threads),
        libraries=[dict(library=k, samples=v, fraction=v/len(rows)) for k, v in libraries.most_common()],
        explicit_launch_in_stack=dict(launch), launch_fraction=sum(launch.values())/len(rows),
        unknown_leaf_samples=unknown, histogram_counts_exact=True,
        top_symbols=sorted(hist_rows, key=lambda r: r['samples'], reverse=True)[:30])


def run(a):
    a.output.mkdir(exist_ok=False); began = time.monotonic()
    need(sha(a.input/'report.json') == a.pin, 'REPORT_PIN')
    report = json.loads((a.input/'report.json').read_bytes()); binding = report['binding']
    need(report['complete'] and not any(report[k] for k in
        ('target_modified', 'target_paused', 'gpu_job_started', 'gpu_kernel_timing', 'speedup_claim')), 'SCOPE')
    raw = {name: (a.input/name).read_bytes() for name in report['files']}
    need(all(digest(raw[n]) == p for n, p in report['files'].items()), 'ARTIFACT_PIN')
    need(sha(binding['authority']) == binding['authority_pin'], 'AUTHORITY')
    need(json.loads(Path(binding['authority']).read_bytes())['code'] == report['code'], 'AUTHORITY_CODE')
    need(all(sha(REPO/'scripts'/n) == p for n, p in report['code'].items())
         and all(sha(REPO/n) == p for n, p in report['observer'].items())
         and all(sha(p) == pin for p, pin in report['tools'].items()), 'CODE_TOOL_PINS')
    for side in ('before', 'after'):
        boundary = json.loads(raw[side+'.json']); process = boundary['process']
        need(process['pid'] == binding['pid'] and process['start'] == binding['start'], 'PROCESS_BINDING')
        need([int(s.split(',')[0]) for s in boundary['gpu'].splitlines()] == [binding['pid']], 'GPU_OWNER')
    need([w['name'] for w in report['windows']] == ['python-1', 'python-2', 'native'], 'WINDOWS')
    results = []; last_ledger = None
    for w in report['windows']:
        row, last_ledger = recount_window(w, raw, report, a.output)
        args = w['command']; need(w['profiler_exit'] == 0, 'PROFILER_EXIT')
        if w['name'].startswith('python'):
            need(all(x in args for x in ('--nonblocking', '--idle', '--threads')) and '--locals' not in args
                 and args[args.index('--pid')+1] == str(binding['pid']), 'PYTHON_COMMAND')
            row['sampling'] = stacks(json.loads(raw[w['name']+'.json']), raw[w['name']+'-profiler.log'].decode(),
                                     binding['pid'], w['rate'], w['duration'])
        else:
            need('cpu-clock:u' in args and args[args.index('-p')+1] == str(binding['pid']), 'NATIVE_COMMAND')
            row['sampling'] = native_summary(raw, binding['pid'])
        results.append(row)
    controls = {}; value = json.loads(raw['python-1.json']); log = raw['python-1-profiler.log'].decode()
    bad = deepcopy(value); bad['profiles'][0]['samples'][0][-1] = len(bad['shared']['frames'])
    missing = deepcopy(value); missing['profiles'] = [p for p in missing['profiles'] if 'MainThread' not in p['name']]
    changed = dict(last_ledger, cursor=last_ledger['cursor']+1)
    for label, call in (
        ('changed_sample', lambda: need(digest(raw['python-1.json']+b' ') == report['files']['python-1.json'], 'ARTIFACT_PIN')),
        ('invalid_frame', lambda: stacks(bad, log, binding['pid'], 99, 30)),
        ('missing_main', lambda: stacks(missing, log, binding['pid'], 99, 30)),
        ('wrong_native_process', lambda: records(raw['events.txt'].decode(), binding['pid']+1)),
        ('changed_ledger', lambda: ledger_record(changed))):
        try: call()
        except ValueError as exc: controls[label] = str(exc)
        else: raise ValueError('TRAIN_OBSERVE_READ_CONTROL_ACCEPTED')
    deps = {str(p.relative_to(REPO)): sha(p) for p in (Path(__file__), REPO/'scripts/audit/live_predictor_readback.py',
            REPO/'scripts/audit/live_native_readback.py', REPO/'scripts/audit/live_predictor_profile.py')}
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    result = dict(complete=True, input=a.pin, binding=binding, windows=results, controls=controls, code=deps,
        seconds=time.monotonic()-began, acceptance='author-checked', gpu_kernel_timing=False,
        sample_fractions_are_wall_time=False, sampling_bias_bounded=False, speedup_claim=False,
        exact_training_replay=False, files={p.name: sha(p) for p in a.output.iterdir() if p.is_file()})
    write(a.output/'report.json', result)
    print(json.dumps(dict(complete=True, pin=sha(a.output/'report.json'), seconds=result['seconds'],
        windows=[{k: v for k, v in w.items() if k != 'sampling'} for w in results])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('input', 'output'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--pin', required=True); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
