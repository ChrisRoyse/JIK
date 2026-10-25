"""Matched real retained-file copies, crash boundaries and physical I/O refusal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import grouped_artifact_copy as copy

ROOT = Path('/zfs/archive/mnlm/evidence/fsv/29')
SOURCE = ROOT / 'answer-refresh-retirement-v1-20261025/seal'
PIN = '35dd82d4e93b6476810a23301b4e75ffd57c4b7b50bb02e88b9f85be7ff7301d'
LIVE = ROOT / 'answer-pipeline-refresh-v1-20261025/placement/full'
CODE = ('model/grouped_artifact_copy.py', 'audit/grouped_artifact_copy_probe.py',
        'audit/grouped_artifact_copy_readback.py', 'model/placement_resume_publication.py')


def write(path, value):
    with path.open('xb') as stream:
        stream.write(copy.canonical(value)); stream.flush(); os.fsync(stream.fileno())


def code():
    base = Path(__file__).resolve().parents[1]
    return {n: copy.sha(base / n) for n in CODE}


def observe():
    stat = Path('/proc/3644105/stat')
    text = stat.read_text(); words = text[text.rindex(')') + 2:].split()
    copy.need(words[19] == '412732985', 'BENCHMARK_OWNER')
    with (LIVE.parent / 'full.log').open('rb') as stream:
        stream.seek(max(0, (LIVE.parent / 'full.log').stat().st_size - 12000))
        lines = stream.read().splitlines()
    updates = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if 'architecture' in row and 'step' in row:
            updates.append(row)
    last = updates[-1] if updates else None
    return dict(pid=3644105, starttime=words[19], state=words[0], last_update=last,
                io=Path('/proc/3644105/io').read_text(),
                gpu=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                             '--format=csv,noheader'], text=True))


def prepare(output):
    copy.need(copy.sha(SOURCE / 'boundary.json') == PIN, 'SEALED_SOURCE_PIN')
    value = json.loads((SOURCE / 'boundary.json').read_bytes())
    files = {}
    for prefix in ('shared/initial/', 'shared/final/'):
        names = sorted((n for n in value['files'] if n.startswith(prefix)),
                       key=lambda n: hashlib.sha256(n.encode()).digest())[:16]
        copy.need(len(names) == 16, 'PROOF_POPULATION')
        files.update({n: value['files'][n] for n in names})
    census = {n: dict(sha256=h, bytes=(SOURCE / n).stat().st_size) for n, h in files.items()}
    output.mkdir(exist_ok=False)
    write(output / 'plan.json', dict(schema='grouped-copy-proof-v1', source=str(SOURCE), source_pin=PIN,
          files=files, census=census, count=len(files), bytes=sum(v['bytes'] for v in census.values()),
          code=code(), pending_bytes=16 << 20, pending_files=8,
          order=[['reference', 'grouped'], ['grouped', 'reference'], ['reference', 'grouped']],
          observed=observe(), acceptance='author-checked'))
    print(json.dumps(dict(stage='prepared', files=len(files), bytes=sum(v['bytes'] for v in census.values()))), flush=True)


def reject(operation):
    try:
        operation()
    except (RuntimeError, ValueError, OSError) as error:
        return str(error)
    raise RuntimeError('MNLM_ARTIFACT_COPY_NEGATIVE_ACCEPTED')


def child(config):
    cfg = json.loads(config.read_bytes()); index = 0
    if cfg.get('file_size_limit'):
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
    def event(kind, name):
        nonlocal index
        print(json.dumps(dict(event=index, kind=kind, name=name)), flush=True)
        if cfg.get('kill') == index:
            os._exit(9)
        index += 1
    try:
        copy.copy(Path(cfg['source']), Path(cfg['output']), cfg['files'],
                  pending_bytes=16 << 20, pending_files=8, observe=event)
    except Exception as error:
        print(json.dumps(dict(error=str(error))), flush=True)
        raise


def run_child(root, label, files, *, kill=None, file_size_limit=False):
    cfg = dict(source=str(SOURCE), output=str(root / label), files=files, kill=kill,
               file_size_limit=file_size_limit)
    path = root / (label + '.config.json'); write(path, cfg)
    result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--child', str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root / (label + '.log')).write_bytes(result.stdout)
    return result


def run(output):
    started = time.perf_counter(); plan = json.loads((output / 'plan.json').read_bytes())
    copy.need(plan['code'] == code() and copy.sha(SOURCE / 'boundary.json') == PIN, 'PROOF_BINDING')
    live_before = observe(); last = live_before['last_update']
    copy.need(last and (last['architecture'] == 'whole_proposals' or
                       (last['architecture'] == 'own_slot' and last['step'] > 2048)), 'BENCHMARK_HEAVY_IMPORT')
    copy.need(not (output / 'report.json').exists(), 'PROOF_EXISTS')
    from model.placement_resume_publication import copy_file
    files = plan['files']; rounds = []
    for index, order in enumerate(plan['order']):
        for implementation in order:
            dest = output / f'round-{index}-{implementation}'
            wall = time.perf_counter(); cpu = time.process_time()
            receipt = copy.copy(SOURCE, dest, files, pending_bytes=plan['pending_bytes'],
                    pending_files=plan['pending_files'], reference_copy=copy_file if implementation == 'reference' else None)
            seconds = time.perf_counter() - wall; cpu = time.process_time() - cpu
            pin = copy.sha(dest / 'publication.json')
            copy.readback(SOURCE, dest, files, pin)
            rounds.append(dict(path=dest.name, implementation=implementation, seconds=seconds, cpu=cpu,
                               groups=len(receipt['groups']), files=len(files), bytes=plan['bytes'], pin=pin))
            print(json.dumps(rounds[-1]), flush=True)
    controls = output / 'controls'; controls.mkdir()
    short = dict(list(sorted(files.items(), key=lambda p: hashlib.sha256(p[0].encode()).digest()))[:2])
    result = run_child(controls, 'clean', short); copy.need(result.returncode == 0, 'CLEAN_CHILD')
    trace = [json.loads(x) for x in result.stdout.splitlines()]; cuts = [r['event'] for r in trace if 'event' in r]
    outcomes = []
    for cut in cuts:
        label = f'exit-{cut}'; result = run_child(controls, label, short, kill=cut)
        copy.need(result.returncode == 9, 'CHILD_EXIT')
        dest = controls / label; receipt = dest / 'publication.json'
        if receipt.exists():
            copy.readback(SOURCE, dest, short, copy.sha(receipt)); verdict = 'landed'
        else:
            verdict = reject(lambda: copy.readback(SOURCE, dest, short, '0' * 64))
            copy.need(verdict == 'MNLM_ARTIFACT_COPY_UNPUBLISHED', 'INCOMPLETE_REFUSAL')
        outcomes.append(dict(cut=cut, path=str(dest.relative_to(output)), verdict=verdict))
    recovery = controls / 'fresh-recovery'; copy.copy(SOURCE, recovery, short)
    copy.readback(SOURCE, recovery, short, copy.sha(recovery / 'publication.json'))
    io = run_child(controls, 'file-size-refusal', short, file_size_limit=True)
    copy.need(io.returncode != 0 and b'MNLM_ARTIFACT_COPY_IO' in io.stdout, 'PHYSICAL_IO_REFUSAL')
    negatives = dict(physical_io=reject(lambda: copy.readback(SOURCE, controls / 'file-size-refusal', short, '0' * 64)),
        reused_output=reject(lambda: copy.copy(SOURCE, recovery, short)),
        path_escape=reject(lambda: copy.copy(SOURCE, controls / 'escape', {'../escape': next(iter(short.values()))})),
        wrong_pin=reject(lambda: copy.copy(SOURCE, controls / 'wrong-pin', {next(iter(short)): '0' * 64})),
        invalid_bound=reject(lambda: copy.copy(SOURCE, controls / 'bound', short, pending_files=0)))
    for name in ('destination', 'durability'):
        dest = controls / ('changed-' + name); copy.copy(SOURCE, dest, short)
        if name == 'destination':
            payload = dest / next(iter(short))
            with payload.open('r+b') as stream:
                old = stream.read(1); stream.seek(0); stream.write(bytes([old[0] ^ 1])); stream.flush(); os.fsync(stream.fileno())
        else:
            r = json.loads((dest / 'publication.json').read_bytes()); r['durable_files'].pop()
            (dest / 'publication.json').write_bytes(copy.canonical(r))
        negatives[name] = reject(lambda: copy.readback(SOURCE, dest, short, copy.sha(dest / 'publication.json')))
    src = controls / 'changed-source'; src.mkdir()
    for name in short:
        (src / name).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SOURCE / name, src / name)
    with (src / next(iter(short))).open('r+b') as stream:
        old = stream.read(1); stream.seek(0); stream.write(bytes([old[0] ^ 1])); stream.flush(); os.fsync(stream.fileno())
    negatives['source'] = reject(lambda: copy.copy(src, controls / 'source-refusal', short))
    copy.need(all(copy.sha(SOURCE / n) == h for n, h in files.items()), 'SOURCE_CHANGED')
    copy.need(plan['code'] == code(), 'CODE_CHANGED')
    medians = {n: statistics.median(r['seconds'] for r in rounds if r['implementation'] == n)
               for n in ('reference', 'grouped')}
    live_after = observe()
    metadata = {str(p.relative_to(output)): copy.sha(p) for p in output.rglob('*')
                if p.is_file() and p.suffix in ('.json', '.log')}
    write(output / 'report.json', dict(complete=True, code=code(), plan=copy.sha(output / 'plan.json'),
        source=str(SOURCE), source_pin=PIN, files=files, count=len(files), bytes=plan['bytes'], rounds=rounds,
        short=short, exits=outcomes, negative_controls=negatives, medians=medians,
        faster=medians['grouped'] < medians['reference'], physical_io_exit=io.returncode,
        live_before=live_before, live_after=live_after, metadata=metadata, seconds=time.perf_counter()-started,
        cold_readback=False, promoted=False, concurrent_workload=True, acceptance='author-checked', number_status='measured'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path); p.add_argument('--prepare', action='store_true'); p.add_argument('--child', type=Path)
    a = p.parse_args()
    if a.child: child(a.child)
    elif a.prepare: prepare(a.output)
    else: run(a.output)
