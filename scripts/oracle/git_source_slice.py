"""Acquire a declared Git source slice without executing any acquired source.

Cost is one pass over the pinned tree plus one request/read per selected blob.
This is source intake, not execution evidence, sensory admission or a kernel.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import urllib.parse
import urllib.request


CODE = 'MNLM_GIT_SOURCE_SLICE_'


def require(ok, code):
    if not ok:
        raise ValueError(CODE + code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, pin):
    data = path.read_bytes()
    require(digest(data) == pin, 'INPUT_PIN')
    return json.loads(data), data


def publish(path, value):
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(data)


def validate_plan(plan, commit, tree):
    require(plan['schema'] == 'git-source-slice-plan-v1', 'PLAN_SCHEMA')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', plan['repository']) is not None,
            'REPOSITORY')
    require(re.fullmatch(r'[0-9a-f]{40}', plan['commit']) is not None and
            plan['commit'] == commit['sha'], 'COMMIT')
    require(plan['tree'] == commit['tree']['sha'] == tree['sha'] and
            tree['truncated'] is False, 'TREE')
    all_blobs = [row for row in tree['tree'] if row['type'] == 'blob']
    full = {row['path']: row for row in all_blobs}
    require(len(full) == len(all_blobs), 'TREE_DUPLICATE')
    require(plan['full_tree_blob_count'] == len(full) and
            plan['full_tree_blob_bytes'] == sum(row['size'] for row in all_blobs), 'CENSUS')
    rows = plan['files']
    require(rows and len(rows) == plan['file_count'] and
            sum(row['size'] for row in rows) == plan['total_bytes'], 'PLAN_COUNTS')
    names = [row['path'] for row in rows]
    require(len(names) == len(set(names)), 'PLAN_DUPLICATE')
    require(plan['scope'] == 'source_observations_only_not_execution_outcomes' and
            plan['training_ready'] is False and plan['scientific_admission'] is False,
            'CLAIM_SCOPE')
    for row in rows:
        name = row['path']
        path = PurePosixPath(name)
        require(name == path.as_posix() and not path.is_absolute() and
                '..' not in path.parts and '\\' not in name and ':' not in name,
                'PATH')
        require(row['mode'] in ('100644', '100755') and type(row['size']) is int and
                row['size'] >= 0 and re.fullmatch(r'[0-9a-f]{40}', row['sha']) is not None,
                'BLOB_CONTRACT')
        require(name in full and all(row[k] == full[name][k] for k in ('mode', 'size', 'sha')),
                'UNBOUND_MEMBER')
    return sorted(rows, key=lambda row: row['path'])


def validate_blob(row, data):
    require(len(data) == row['size'], 'BLOB_LENGTH')
    git_hash = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    require(git_hash == row['sha'], 'BLOB_IDENTITY')
    return dict(path=row['path'], bytes=len(data), git_blob=git_hash, sha256=digest(data))


def capture(plan, rows, output):
    successes, errors = [], []
    phases = dict(network=0.0, validation_and_persist=0.0)
    for index, row in enumerate(rows):
        phase, tick = 'network', time.perf_counter()
        try:
            url = ('https://raw.githubusercontent.com/' + plan['repository'] + '/' +
                   plan['commit'] + '/' + urllib.parse.quote(row['path'], safe='/'))
            request = urllib.request.Request(url, headers={
                'User-Agent': 'MNLM-declared-source-slice', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=60) as response:
                parsed = urllib.parse.urlsplit(response.url)
                require(parsed.scheme == 'https' and parsed.netloc == 'raw.githubusercontent.com',
                        'REDIRECT')
                require(response.status == 200, 'HTTP_STATUS')
                length = response.headers.get('Content-Length')
                require(length is None or int(length) == row['size'], 'HTTP_LENGTH')
                data = response.read(row['size'] + 1)
            phases[phase] += time.perf_counter() - tick
            phase, tick = 'validation_and_persist', time.perf_counter()
            receipt = validate_blob(row, data)
            destination = output / 'payload' / row['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            successes.append(receipt)
            phases[phase] += time.perf_counter() - tick
        except Exception as exc:
            phases[phase] += time.perf_counter() - tick
            # Network diagnostics may contain uncontrolled values: retain type/code only.
            code = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(CODE) else (
                CODE + 'FETCH_OR_PERSIST')
            errors.append(dict(path=row['path'], code=code, exception=type(exc).__name__))
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(json.dumps(dict(processed=index + 1, total=len(rows),
                                  acquired=len(successes), failed=len(errors))), flush=True)
    return successes, errors, phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'commit', 'tree'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    inputs, raw, pins = {}, {}, {}
    for name in ('plan', 'commit', 'tree'):
        pin = getattr(args, name + '_pin')
        inputs[name], raw[name] = pinned(getattr(args, name), pin)
        pins[name] = pin
    rows = validate_plan(inputs['plan'], inputs['commit'], inputs['tree'])
    output = args.output.resolve()
    require(output.is_relative_to(Path('/zfs/archive/mnlm/raw')) and not output.exists(),
            'FRESH_ARCHIVE_SCOPE')
    output.mkdir(parents=True)
    for name, data in raw.items():
        with (output / (name + '.json')).open('xb') as stream:
            stream.write(data)
    phases = dict(binding=time.perf_counter() - started)
    receipts, errors, work = capture(inputs['plan'], rows, output)
    phases.update(work)
    tick = time.perf_counter()
    inventory_pin = publish(output / 'inventory.json', receipts)
    code_pin = digest(Path(__file__).read_bytes())
    phases['inventory'] = time.perf_counter() - tick
    elapsed = time.perf_counter() - started
    phases['other'] = elapsed - sum(phases.values())
    report = dict(schema='git-source-slice-capture-v1', repository=inputs['plan']['repository'],
                  commit=inputs['plan']['commit'], tree=inputs['plan']['tree'], inputs=pins,
                  split_group=inputs['plan']['split_group'], source_code_sha256=code_pin,
                  inventory_sha256=inventory_pin, errors=errors, complete=not errors,
                  expected_files=len(rows), acquired_files=len(receipts),
                  expected_bytes=sum(row['size'] for row in rows),
                  acquired_bytes=sum(row['bytes'] for row in receipts),
                  phases_seconds=phases, wall_seconds=elapsed,
                  source_observations_only=True, repository_code_executed=False,
                  training_ready=False, scientific_admission=False, serving_changed=False,
                  acceptance='author-checked')
    report_pin = publish(output / 'report.json', report)
    if report['complete']:
        publish(output / 'COMMITTED.json', dict(report_sha256=report_pin))
    print(json.dumps(dict(report_sha256=report_pin, complete=report['complete'],
                          files=len(receipts), bytes=report['acquired_bytes'],
                          wall_seconds=elapsed)), flush=True)
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
