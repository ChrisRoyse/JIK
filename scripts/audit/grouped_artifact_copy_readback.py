"""Cold, independent byte readback of the grouped-copy experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exact(destination, source, files, sizes, code_pin, pin=None):
    receipt = destination / 'publication.json'
    assert receipt.is_file(), 'unpublished'
    if pin is not None:
        assert sha(receipt) == pin, 'receipt pin'
    r = json.loads(receipt.read_bytes())
    names = sorted(files)
    assert r['complete'] is True and r['schema'] == 'grouped-artifact-copy-v1'
    assert r['code'] == code_pin and r['source'] == str(source)
    assert r['files'] == {n: dict(sha256=files[n], bytes=sizes[n]) for n in names}
    assert r['durable_files'] == names, 'durable files'
    assert [n for g in r['groups'] for n in g['files']] == names, 'group coverage'
    assert r['implementation'] in ('grouped', 'reference')
    for g in r['groups']:
        assert g['bytes'] == sum(sizes[n] for n in g['files'])
        if r['implementation'] == 'grouped':
            assert len(g['files']) <= r['pending_files']
            assert g['bytes'] <= r['pending_bytes'] or len(g['files']) == 1
    found = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
    assert found == set(files) | {'publication.json'}
    for name in names:
        target = destination / name
        assert not target.is_symlink() and target.stat().st_size == sizes[name]
        assert sha(target) == files[name], 'payload bytes'
        assert (target.stat().st_dev, target.stat().st_ino) != ((source / name).stat().st_dev, (source / name).stat().st_ino)
    return len(names), sum(sizes[n] for n in names)


def refuses(operation):
    try:
        operation()
    except (AssertionError, OSError, ValueError) as error:
        return str(error)
    raise AssertionError('negative accepted')


def run(a):
    began = time.perf_counter(); cpu = time.process_time()
    assert sha(a.input / 'report.json') == a.pin
    r = json.loads((a.input / 'report.json').read_bytes())
    assert r['complete'] and not r['promoted'] and not r['cold_readback']
    base = Path(__file__).resolve().parents[1]
    assert all(sha(base / name) == pin for name, pin in r['code'].items())
    assert all(sha(a.input / name) == pin for name, pin in r['metadata'].items())
    plan = json.loads((a.input / 'plan.json').read_bytes())
    assert sha(a.input / 'plan.json') == r['plan'] and plan['code'] == r['code']
    source = Path(r['source']); assert sha(source / 'boundary.json') == r['source_pin']
    seal = json.loads((source / 'boundary.json').read_bytes())
    expected = {}
    for prefix in ('shared/initial/', 'shared/final/'):
        names = sorted((n for n in seal['files'] if n.startswith(prefix)), key=lambda n: hashlib.sha256(n.encode()).digest())[:16]
        expected.update({n: seal['files'][n] for n in names})
    assert expected == r['files'] == plan['files'] and len(expected) == r['count'] == 32
    sizes = {n: plan['census'][n]['bytes'] for n in expected}
    assert sum(sizes.values()) == r['bytes'] == plan['bytes']
    assert all((source / n).stat().st_size == sizes[n] and sha(source / n) == h for n, h in expected.items())
    code_pin = r['code']['model/grouped_artifact_copy.py']; physical_files = physical_bytes = 0
    order = [(i, n) for i, pair in enumerate(plan['order']) for n in pair]
    assert [(x['path'], x['implementation']) for x in r['rounds']] == [(f'round-{i}-{n}', n) for i, n in order]
    for row in r['rounds']:
        count, size = exact(a.input / row['path'], source, expected, sizes, code_pin, row['pin'])
        assert row['files'] == count and row['bytes'] == size and row['seconds'] > 0 and row['cpu'] >= 0
        physical_files += count; physical_bytes += size
    medians = {n: statistics.median(x['seconds'] for x in r['rounds'] if x['implementation'] == n)
               for n in ('reference', 'grouped')}
    assert medians == r['medians'] and r['faster'] == (medians['grouped'] < medians['reference'])
    short = r['short']; controls = a.input / 'controls'
    trace = [json.loads(x) for x in (controls / 'clean.log').read_bytes().splitlines()]
    assert [x['cut'] for x in r['exits']] == [x['event'] for x in trace]
    exit_verdicts = []
    for row in r['exits']:
        dest = a.input / row['path']; log = controls / f"exit-{row['cut']}.log"
        last = json.loads(log.read_bytes().splitlines()[-1]); assert last['event'] == row['cut']
        if row['verdict'] == 'landed':
            count, size = exact(dest, source, short, sizes, code_pin)
            physical_files += count; physical_bytes += size; exit_verdicts.append('landed')
        else:
            assert row['verdict'] == 'MNLM_ARTIFACT_COPY_UNPUBLISHED'
            assert not (dest / 'publication.json').exists(); exit_verdicts.append('unpublished')
    for name in ('clean', 'fresh-recovery'):
        count, size = exact(controls / name, source, short, sizes, code_pin)
        physical_files += count; physical_bytes += size
    io = controls / 'file-size-refusal'
    assert r['physical_io_exit'] != 0 and b'MNLM_ARTIFACT_COPY_IO' in (controls / 'file-size-refusal.log').read_bytes()
    assert not (io / 'publication.json').exists()
    partial = [p for p in io.rglob('*') if p.is_file()]
    assert partial and all(p.stat().st_size <= 1024 for p in partial)
    negatives = dict(
        unpublished=refuses(lambda: exact(io, source, short, sizes, code_pin)),
        destination=refuses(lambda: exact(controls / 'changed-destination', source, short, sizes, code_pin)),
        durability=refuses(lambda: exact(controls / 'changed-durability', source, short, sizes, code_pin)),
        wrong_receipt=refuses(lambda: exact(controls / 'clean', source, short, sizes, code_pin, '0' * 64)))
    assert all(v.startswith('MNLM_ARTIFACT_COPY_') for v in r['negative_controls'].values())
    assert set(r['negative_controls']) == {'physical_io','reused_output','path_escape','wrong_pin','invalid_bound','destination','durability','source'}
    assert all(sha(source / n) == h for n, h in expected.items())
    a.output.mkdir(exist_ok=False)
    report = dict(complete=True, previous=a.pin, code=r['code'], cold_readback=True, physical_files=physical_files,
        physical_bytes=physical_bytes, source_files=len(expected), source_bytes=sum(sizes.values()), medians=medians,
        exits=exit_verdicts, physical_io_refused=True, source_unchanged=True, negative_controls=negatives,
        promoted=False, concurrent_workload=True, seconds=time.perf_counter()-began, cpu=time.process_time()-cpu,
        acceptance='author-checked', number_status='measured')
    with (a.output / 'report.json').open('x') as stream:
        json.dump(report, stream, sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
