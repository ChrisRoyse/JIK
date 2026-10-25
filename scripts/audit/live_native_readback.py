"""Count native CPU sample records independently from perf's histogram output."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re


def need(ok, code):
    if not ok: raise ValueError('NATIVE_READ_'+code)


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def records(raw, pid):
    rows = []; current = None
    for line in raw.splitlines():
        line = line.rstrip()
        header = re.fullmatch(r'\s*(\S+)\s+(\d+)/(\d+)\s+(\d+\.\d+): cpu-clock:u:', line)
        if header:
            if current is not None: need(current['frames'], 'EMPTY_STACK'); rows.append(current)
            comm, process, tid, timestamp = header.groups()
            need(int(process) == pid, 'PROCESS')
            current = dict(comm=comm, pid=int(process), tid=int(tid), time=float(timestamp), frames=[])
        elif line.strip():
            frame = re.fullmatch(r'\s*([0-9a-f]+)\s+(.*?) \((.*)\)', line)
            need(frame is not None and current is not None, 'EVENT_LINE')
            ip, symbol, dso = frame.groups(); current['frames'].append(dict(ip=ip, symbol=symbol, dso=dso))
    if current is not None: need(current['frames'], 'EMPTY_STACK'); rows.append(current)
    need(rows and all(y['time'] >= x['time'] for x, y in zip(rows, rows[1:])), 'EVENT_ORDER')
    return rows


def histogram(raw):
    counts = Counter(); rows = []
    lost = re.findall(r'^# Total Lost Samples: (\d+)$', raw, re.MULTILINE)
    need(len(lost) == 1, 'LOST_ACCOUNTING')
    for line in raw.splitlines():
        if not re.match(r'^\s*\d+\.\d+%\s*;', line): continue
        parts = line.split(';', 4)
        need(len(parts) == 5 and parts[1].strip().isdigit(), 'HISTOGRAM_LINE')
        count = int(parts[1]); comm = parts[2].strip(); dso = parts[3].strip()
        counts[comm, dso] += count
        rows.append(dict(comm=comm, dso=dso, samples=count, overhead=parts[0].strip(), symbol=parts[4].strip()))
    need(counts, 'EMPTY_HISTOGRAM'); return counts, rows, int(lost[0])


def run(a):
    a.output.mkdir(exist_ok=False)
    need(sha(a.input/'report.json') == a.pin, 'REPORT_PIN')
    report = json.loads((a.input/'report.json').read_bytes())
    need(report['complete'] and not report['target_modified'] and not report['promoted']
         and not report['gpu_kernel_timing'], 'SCOPE')
    for name, pin in report['files'].items(): need(sha(a.input/name) == pin, 'ARTIFACT_PIN')
    raw = (a.input/'events.txt').read_text(); rows = records(raw, report['pid'])
    groups = Counter(); threads = Counter(); libraries = Counter(); launch = Counter(); unknown = 0
    inline_leaf_samples = 0
    for row in rows:
        f = row['frames'][0]
        physical = next((frame for frame in row['frames'] if frame['dso'] != 'inlined'), None)
        need(physical is not None, 'MISSING_PHYSICAL_FRAME')
        inline_leaf_samples += f['dso'] == 'inlined'
        dso = Path(physical['dso']).name; groups[row['comm'], dso] += 1
        threads[row['comm']] += 1; libraries[dso] += 1; unknown += f['symbol'] == '[unknown]'
        if any('cuLaunchKernel' in f['symbol'] or 'cudaLaunchKernel' in f['symbol'] for f in row['frames']):
            launch[row['comm']] += 1
    hist, hist_rows, lost = histogram((a.input/'histogram.txt').read_text())
    need(hist == groups, 'HISTOGRAM_RECOUNT')
    log = (a.input/'perf-record.log').read_text()
    captured = re.findall(r'\((\d+) samples\)', log)
    need(len(captured) <= 1 and (not captured or int(captured[0]) == len(rows)), 'CAPTURE_COUNT')
    need('Captured and wrote' in log, 'CAPTURE_RECEIPT')
    before = json.loads((a.input/'before.json').read_bytes())['process']
    after = json.loads((a.input/'after.json').read_bytes())
    processes = [json.loads(line) for line in (a.input/'process.jsonl').read_text().splitlines()]
    for p in [before, *processes, after]:
        need(p['pid'] == report['pid'] and p['start'] == report['start']
             and p['executable_stat'] == before['executable_stat']
             and p['command_sha256'] == before['command_sha256'], 'IDENTITY')
    for side in ('before', 'after'):
        v = json.loads((a.input/(side+'-ledger.json')).read_bytes())
        expected = report['first' if side == 'before' else 'last']
        encoded = json.dumps({k: value for k, value in v.items() if k != 'hash'}, sort_keys=True,
                             ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        need(hashlib.sha256(encoded).hexdigest() == v['hash'] == expected['hash']
             and v['cursor'] == expected['cursor'], 'LEDGER_BINDING')
    need(report['last']['cursor'] > report['first']['cursor'], 'NO_PROGRESS')
    controls = {}
    for label, fn in (
        ('wrong_process', lambda: records(raw, report['pid']+1)),
        ('changed_histogram_count', lambda: need(Counter({**hist, next(iter(hist)): -1}) == groups, 'HISTOGRAM_RECOUNT')),
        ('empty_stack', lambda: records(raw.splitlines()[0]+'\n', report['pid'])),
        ('changed_raw_bytes', lambda: need(hashlib.sha256(raw.encode()+b' ').hexdigest() == report['files']['events.txt'], 'ARTIFACT_PIN'))):
        try: fn()
        except ValueError as exc: controls[label] = str(exc)
        else: raise ValueError('NATIVE_READ_CONTROL_ACCEPTED')
    total = len(rows)
    result = dict(complete=True, input=a.pin, samples=total, lost_samples=lost, threads=dict(threads),
        sampled_leaf_libraries=[dict(library=k, samples=v, fraction=v/total) for k, v in libraries.most_common()],
        explicit_launch_in_stack_by_thread=dict(launch), explicit_launch_in_stack_fraction=sum(launch.values())/total,
        unknown_leaf_symbols=unknown, inline_leaf_samples=inline_leaf_samples,
        capture_log_sample_count=int(captured[0]) if captured else None,
        histogram_counts_exact=True, sample_time_span=rows[-1]['time']-rows[0]['time'],
        durable_cursor_before=report['first']['cursor'], durable_cursor_after=report['last']['cursor'],
        controls=controls, source=sha(Path(__file__)), acceptance='author-checked', promoted=False,
        callchain_completeness_guaranteed=False, sampling_bias_bounded=False,
        gpu_kernel_timing=False, sample_fractions_are_wall_time=False)
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'report.json').open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(complete=True, pin=sha(a.output/'report.json'), samples=total,
        libraries=result['sampled_leaf_libraries'][:8], launch_fraction=result['explicit_launch_in_stack_fraction'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pin', required=True); a = p.parse_args()
    try: run(a)
    except BaseException as exc:
        if a.output.exists(): (a.output/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__, detail=str(exc))))
        raise
