"""Recount external stack samples, telemetry and durable updates without model imports."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

REPO = Path(__file__).resolve().parents[2]
LEDGER = Path('/zfs/archive/mnlm/evidence/fsv/29/phase115-complete-answer-recovery-v1-20261024/grouped/full/shared/steps.jsonl')


def need(ok, code):
    if not ok: raise ValueError('PROFILE_READ_'+code)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.flush(); os.fsync(f.fileno())


def pinned(path, pin):
    raw = path.read_bytes(); need(digest(raw) == pin, 'ARTIFACT_PIN'); return raw


def ledger_record(value):
    raw = json.dumps({k: v for k, v in value.items() if k != 'hash'}, sort_keys=True,
                     ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    need(digest(raw) == value['hash'], 'LEDGER_HASH')


def identity(value, reference):
    keys = ('pid', 'start', 'executable', 'executable_stat', 'command_sha256')
    need(all(value[k] == reference[k] for k in keys) and value['state'] != 'Z', 'IDENTITY')


def summary(values):
    need(values and all(math.isfinite(v) for v in values), 'FINITE_TELEMETRY')
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values), mean=statistics.fmean(values))


def stacks(value, log, pid, rate, duration):
    frames = value['shared']['frames']; rows = []; main = None; total = 0
    need(value['exporter'] == 'py-spy@0.4.2', 'EXPORTER')
    for p in value['profiles']:
        need(p['type'] == 'sampled' and p['unit'] == 'seconds' and p['startValue'] == 0,
             'SAMPLE_SCHEMA')
        need(len(p['samples']) == len(p['weights']) > 0, 'SAMPLE_WEIGHTS')
        leaf = Counter(); inclusive = Counter(); categories = Counter(); unknown = 0
        for sample, weight in zip(p['samples'], p['weights']):
            need(sample and all(type(i) is int and 0 <= i < len(frames) for i in sample), 'FRAME_INDEX')
            need(math.isclose(weight, 1/rate, rel_tol=1e-12), 'SAMPLE_WEIGHT')
            leaf[sample[-1]] += 1
            for index in set(sample): inclusive[index] += 1
            fs = [frames[i] for i in sample]; names = {f['name'] for f in fs}
            if '_engine_run_backward' in names: category = 'backward_including_native_work_or_wait'
            elif 'energy_loss' in names: category = 'energy_loss_forward'
            elif names & {'finite_tensors', 'validate_optimizer'}: category = 'finite_or_optimizer_validation'
            elif names & {'tensor_fingerprint', 'fingerprint'}: category = 'fingerprint'
            elif any(f['file'].endswith('/native_device_transport.py') for f in fs): category = 'device_transport'
            elif any(f['file'].endswith('/legal_native_resident_stream.py') for f in fs): category = 'native_read'
            elif any('/scripts/model/' in f['file'] for f in fs): category = 'other_model_work'
            elif not any(f['file'] for f in fs): category = 'no_python_frame'
            else: category = 'other_python_or_native_call'
            categories[category] += 1
            unknown += not any(f['file'] for f in fs)
        n = len(p['samples']); weight = math.fsum(p['weights'])
        need(math.isclose(weight, p['endValue'], rel_tol=1e-12), 'SAMPLE_DURATION')
        def ranked(c):
            return [dict(frame=frames[i], samples=n, fraction=n/len(p['samples'])) for i, n in c.most_common()]
        row = dict(name=p['name'], samples=n, weighted_seconds=weight, no_python_frame_samples=unknown,
                   leaf=ranked(leaf), inclusive=ranked(inclusive), disjoint_categories=dict(categories))
        rows.append(row); total += n
        if p['name'] == f'Thread {pid} "MainThread"':
            need(main is None, 'MULTIPLE_MAIN'); main = row
    need(main is not None, 'MISSING_MAIN')
    matches = re.findall(r'Samples: (\d+) Errors: (\d+)', log)
    need(len(matches) == 1 and int(matches[0][0]) == total, 'LOG_SAMPLE_COUNT')
    errors = int(matches[0][1]); nominal = rate*duration
    return dict(threads=rows, main=main, thread_samples=total, logged_errors=errors,
                nominal_sampling_ticks=nominal, main_samples=main['samples'],
                main_to_nominal_fraction=main['samples']/nominal,
                nominal_minus_main=nominal-main['samples'],
                error_count_unit='profiler-reported errors; not assumed to equal missing thread samples')


def run(a):
    began = time.monotonic(); a.output.mkdir(exist_ok=False)
    report = json.loads(pinned(a.input/'report.json', a.pin))
    need(report['complete'] and not report['promoted'] and not report['gpu_job_started']
         and not report['target_paused'] and not report['target_modified'], 'SCOPE')
    need(len(report['windows']) == 2, 'WINDOW_COUNT')
    raw = {name: pinned(a.input/name, pin) for name, pin in report['files'].items()}
    before = json.loads(raw['before.json']); after = json.loads(raw['after.json'])
    ref = before['process']; identity(after['process'], ref)
    need(report['process'] == dict(pid=ref['pid'], start=ref['start']), 'PROCESS_BINDING')
    need(digest(raw['source.py']) == report['source'] == sha(REPO/'scripts/audit/live_predictor_profile.py'), 'SOURCE')
    need(before['code'] == report['code'] and before['executable_pin'] == report['executable'], 'CODE_BINDING')
    for name, pin in report['code'].items(): need(sha(REPO/name) == pin, 'CODE_PIN')
    for boundary in (before, after):
        need([int(line.split(',')[0]) for line in boundary['gpu'].splitlines()] == [ref['pid']], 'GPU_OWNER')
    results = []; last_profile = last_log = last_process = last_ledger = None
    for w in report['windows']:
        name = w['name']; command = w['command']
        need('--nonblocking' in command and '--idle' in command and '--threads' in command
             and '--locals' not in command and w['profiler_exit'] == 0, 'SAMPLER_COMMAND')
        need(command[command.index('--pid')+1] == str(ref['pid']), 'SAMPLER_PID')
        for key, pin in w['files'].items(): need(report['files'][key] == pin, 'WINDOW_PIN')
        value = json.loads(raw[name+'.json']); log = raw[name+'-profiler.log'].decode()
        sample = stacks(value, log, ref['pid'], w['rate'], w['duration'])
        processes = [w['before']]+[json.loads(line) for line in raw[name+'-process.jsonl'].splitlines()]+[w['after']]
        for p in processes: identity(p, ref)
        need(all(y['monotonic'] >= x['monotonic'] and y['cpu_ticks'] >= x['cpu_ticks']
                 for x, y in zip(processes, processes[1:])), 'PROCESS_ORDER')
        elapsed = processes[-1]['monotonic']-processes[0]['monotonic']
        cpu = (processes[-1]['cpu_ticks']-processes[0]['cpu_ticks'])/report['clock_ticks']
        gpu = list(csv.reader(io.StringIO(raw[name+'-gpu.csv'].decode())))
        need(gpu and all(len(r) == 6 for r in gpu) and all(y[0] > x[0] for x, y in zip(gpu, gpu[1:])), 'GPU_ROWS')
        columns = ('utilization_percent', 'memory_utilization_percent', 'memory_mib', 'power_watts', 'clock_mhz')
        gpu_summary = {key: summary([float(r[i+1]) for r in gpu]) for i, key in enumerate(columns)}
        endpoints = []
        for key in ('initial', 'final'):
            e = w[key]; need(report['files'][e['path']] == e['pin'], 'LEDGER_PIN')
            v = json.loads(raw[e['path']]); ledger_record(v)
            need(v['cursor'] == e['cursor'] and v['hash'] == e['hash'], 'LEDGER_CURSOR'); endpoints.append(v)
        lo = max(0, w['initial']['file_size']-(1 << 20)); hi = w['final']['file_size']
        need(0 < hi-lo < 4 << 20, 'LEDGER_TILE')
        with LEDGER.open('rb') as f: f.seek(lo); region = f.read(hi-lo)
        need(len(region) == hi-lo, 'LEDGER_TRUNCATED')
        lines = region.split(b'\n')[1 if lo else 0:-1]; selected = []
        for line in lines:
            v = json.loads(line)
            if endpoints[0]['cursor'] < v['cursor'] <= endpoints[1]['cursor']: selected.append((line, v))
        need(len(selected) == endpoints[1]['cursor']-endpoints[0]['cursor'] > 0, 'LEDGER_COUNT')
        parent = endpoints[0]; targets = 0
        for line, v in selected:
            ledger_record(v)
            need(v['parent'] == parent['hash'] and v['cursor'] == parent['cursor']+1, 'LEDGER_CHAIN')
            targets += v['update']['targets']; parent = v
        need(parent == endpoints[1], 'LEDGER_ENDPOINT')
        retained = b''.join(line+b'\n' for line, _ in selected)
        (a.output/(name+'-updates.jsonl')).write_bytes(retained)
        results.append(dict(name=name, sampling=sample, gpu=gpu_summary, elapsed_seconds=elapsed,
            process_cpu_seconds=cpu, process_core_equivalents=cpu/elapsed,
            rss_bytes=summary([p['rss_bytes'] for p in processes]),
            durable_updates=len(selected), durable_targets=targets, first_cursor=endpoints[0]['cursor'],
            last_cursor=endpoints[1]['cursor'], retained_updates=digest(retained)))
        last_profile, last_log, last_process, last_ledger = value, log, processes[-1], endpoints[-1]
    controls = {}
    bad_frame = deepcopy(last_profile); bad_frame['profiles'][0]['samples'][0][-1] = len(bad_frame['shared']['frames'])
    no_main = deepcopy(last_profile); no_main['profiles'] = [p for p in no_main['profiles'] if 'MainThread' not in p['name']]
    bad_process = dict(last_process, start=str(int(ref['start'])+1)); bad_ledger = deepcopy(last_ledger); bad_ledger['cursor'] += 1
    for label, fn in (
        ('changed_sample_bytes', lambda: need(digest(raw['window-1.json']+b' ') == report['files']['window-1.json'], 'ARTIFACT_PIN')),
        ('invalid_frame', lambda: stacks(bad_frame, last_log, ref['pid'], 99, 30)),
        ('missing_main', lambda: stacks(no_main, last_log, ref['pid'], 99, 30)),
        ('changed_sample_count', lambda: stacks(last_profile, re.sub(r'Samples: \d+', 'Samples: 0', last_log), ref['pid'], 99, 30)),
        ('changed_process', lambda: identity(bad_process, ref)),
        ('changed_ledger', lambda: ledger_record(bad_ledger))):
        try: fn()
        except ValueError as exc: controls[label] = str(exc)
        else: raise ValueError('PROFILE_READ_CONTROL_ACCEPTED')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    result = dict(complete=True, input=a.pin, windows=results, controls=controls,
        source=sha(Path(__file__)), acceptance='author-checked', promoted=False,
        gpu_kernel_timing=False, exact_wall_attribution=False, sampling_bias_bounded=False,
        quality_or_architecture_winner_measured=False, seconds=time.monotonic()-began)
    result['files'] = {p.name: sha(p) for p in a.output.iterdir() if p.is_file()}
    write(a.output/'report.json', result)
    print(json.dumps(dict(complete=True, pin=sha(a.output/'report.json'),
        windows=[{k: v for k, v in w.items() if k != 'sampling'} for w in results])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pin', required=True); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): write(a.output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
