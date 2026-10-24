"""Independently reopen a Git slice and exercise its intake refusal boundaries."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time


def check(ok, name):
    if not ok:
        raise ValueError('MNLM_GIT_SLICE_READBACK_' + name)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path, expected=None):
    data = path.read_bytes()
    check(expected is None or sha(data) == expected, 'PIN')
    return json.loads(data), sha(data)


def readback(root, report_pin):
    report, _ = load(root / 'report.json', report_pin)
    check(report['schema'] == 'git-source-slice-capture-v1' and report['complete'] is True and
          report['errors'] == [], 'COMPLETE')
    writer = Path(__file__).resolve().parents[1] / 'oracle' / 'git_source_slice.py'
    check(sha(writer.read_bytes()) == report['source_code_sha256'], 'WRITER_PIN')
    marker, _ = load(root / 'COMMITTED.json')
    check(marker == dict(report_sha256=report_pin), 'MARKER')
    plan, _ = load(root / 'plan.json', report['inputs']['plan'])
    commit, _ = load(root / 'commit.json', report['inputs']['commit'])
    tree, _ = load(root / 'tree.json', report['inputs']['tree'])
    inventory, _ = load(root / 'inventory.json', report['inventory_sha256'])
    check(report['commit'] == plan['commit'] == commit['sha'] and
          report['tree'] == plan['tree'] == commit['tree']['sha'] == tree['sha'] and
          tree['truncated'] is False, 'SOURCE_GENERATION')
    full = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
    rows = {r['path']: r for r in plan['files']}
    check(len(rows) == len(plan['files']) == plan['file_count'], 'UNIQUE_PLAN')
    check(len(full) == plan['full_tree_blob_count'] and
          sum(r['size'] for r in full.values()) == plan['full_tree_blob_bytes'], 'TREE_CENSUS')
    actual_paths = sorted(p.relative_to(root / 'payload').as_posix()
                          for p in (root / 'payload').rglob('*') if p.is_file())
    check(actual_paths == sorted(rows), 'MEMBERSHIP')
    observed = []
    for name in sorted(rows):
        row = rows[name]
        path = PurePosixPath(name)
        check(not path.is_absolute() and '..' not in path.parts and '\\' not in name and
              ':' not in name and path.as_posix() == name, 'PATH')
        check(name in full and all(row[k] == full[name][k] for k in ('mode','size','sha')) and
              row['mode'] in ('100644','100755'), 'TREE_MEMBER')
        source = root / 'payload' / name
        check(not source.is_symlink() and source.resolve().is_relative_to(root / 'payload'),
              'LINK')
        data = source.read_bytes()
        framed = b'blob %d\x00' % len(data) + data
        blob = hashlib.sha1(framed).hexdigest()
        check(len(data) == row['size'] and blob == row['sha'], 'PAYLOAD_BYTES')
        observed.append(dict(path=name, bytes=len(data), git_blob=blob, sha256=sha(data)))
    check(inventory == observed, 'INVENTORY')
    total = sum(r['bytes'] for r in observed)
    check(total == plan['total_bytes'] == report['acquired_bytes'] == report['expected_bytes'] and
          len(observed) == report['acquired_files'] == report['expected_files'], 'TOTALS')
    check(report['source_observations_only'] is True and
          all(report[k] is False for k in ('repository_code_executed','training_ready',
                                           'scientific_admission','serving_changed')), 'SCOPE')
    return dict(files=len(observed), bytes=total, commit=commit['sha'], tree=tree['sha'],
                exact_tree_members=True, exact_payload_bytes=True, exact_publication=True), (
                    plan, commit, tree, inventory)


def controls(root, source):
    # Exercise the actual writer's gates; never edit the retained source corpus.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from oracle.git_source_slice import validate_plan, validate_blob
    plan, commit, tree, inventory = source
    results = {}

    def refused(name, call):
        try:
            call()
        except ValueError as exc:
            code = str(exc)
            check(code.startswith('MNLM_GIT_'), 'CONTROL_EXCEPTION')
            results[name] = code
        else:
            raise ValueError('MNLM_GIT_SLICE_READBACK_CONTROL_ACCEPTED_' + name)

    edits = {
        'wrong_revision': lambda p: p.update(commit='0' * 40),
        'wrong_tree': lambda p: p.update(tree='0' * 40),
        'duplicate_member': lambda p: p['files'].__setitem__(1, deepcopy(p['files'][0])),
        'unbound_blob': lambda p: p['files'][0].update(sha='0' * 40),
        'escaping_path': lambda p: p['files'][0].update(path='../outside.rs'),
        'invented_admission': lambda p: p.update(scientific_admission=True),
    }
    for name, edit in edits.items():
        changed = deepcopy(plan)
        edit(changed)
        if name == 'duplicate_member':
            changed['total_bytes'] = sum(r['size'] for r in changed['files'])
        refused(name, lambda p=changed: validate_plan(p, commit, tree))
    row = next(r for r in plan['files'] if r['size'] > 1)
    data = (root / 'payload' / row['path']).read_bytes()
    changed = bytearray(data)
    changed[len(changed) // 2] ^= 1
    refused('changed_source_byte', lambda: validate_blob(row, bytes(changed)))
    refused('truncated_source', lambda: validate_blob(row, data[:-1]))
    # Completion gates operate over actual persisted bytes in an isolated control directory.
    control = root / 'readback-negative-control'
    control.mkdir()
    report, _ = load(root / 'report.json')
    report['acquired_files'] -= 1
    report_data = (json.dumps(report, sort_keys=True) + '\n').encode()
    (control / 'report.json').write_bytes(report_data)
    (control / 'COMMITTED.json').write_text(json.dumps(dict(report_sha256=sha(report_data))))
    for name in ('plan.json','commit.json','tree.json','inventory.json'):
        (control / name).write_bytes((root / name).read_bytes())
    (control / 'payload').mkdir()
    refused('forged_completion_missing_payload', lambda: readback(control.resolve(), sha(report_data)))
    refused('wrong_report_pin', lambda: readback(root, '0' * 64))
    check(validate_blob(row, data)['sha256'] == sha(data), 'POSITIVE_CONTROL')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report-pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    root = args.root.resolve()
    result, source = readback(root, args.report_pin)
    read_seconds = time.perf_counter() - start
    tick = time.perf_counter()
    negative = controls(root, source)
    result.update(schema='git-source-slice-readback-v1', report_sha256=args.report_pin,
                  controls=negative, readback_seconds=read_seconds,
                  controls_seconds=time.perf_counter() - tick,
                  wall_seconds=time.perf_counter() - start,
                  source_code_sha256=sha(Path(__file__).read_bytes()),
                  admission=False, learned_capability=False, acceptance='author-checked')
    data = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    with args.output.open('xb') as stream:
        stream.write(data)
    print(json.dumps(dict(files=result['files'], bytes=result['bytes'],
                          controls=len(negative), sha256=sha(data))), flush=True)


if __name__ == '__main__':
    main()
