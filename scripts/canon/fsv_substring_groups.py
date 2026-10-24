"""Reference-tool boundary and malformed-stream controls; no training records."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_substrings import execute
from substring_ranges import resolve_groups


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, ok, **detail):
        checks.append(dict(boundary=name, passed=bool(ok), **detail))
    # Shared text starts/ends inside different multi-byte characters. The
    # retained counterpart must snap inward, on both items, to complete UTF-8.
    common = ''.join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(4))[:250]
    left, right = '\u00e9' + common + '\u00e9', '\u00a9' + common + '\u00f0'
    items = [left, right, left, right]
    cases = dict(unicode=items, zero=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(8)])
    for name, texts in cases.items():
        root = args.output / name
        root.mkdir()
        raw = [s.encode() for s in texts]
        corpus = b''.join(x + b'\0' for x in raw)
        (root / 'corpus.bin').write_bytes(corpus)
        offsets = np.array([0, *np.cumsum([len(x)+1 for x in raw])], dtype=np.int64)
        np.save(root / 'offsets.npy', offsets)
        execute(args.tool, root)
        spans, stats = resolve_groups(root / 'corpus.bin', offsets, 200, root / 'windows.sqlite')
        (root / 'resolution.json').write_text(json.dumps(dict(stats=stats, spans=spans.tolist()), indent=2))
        if name == 'zero':
            check('zero_repeat_protocol', len(spans) == 0 and stats['groups'] == 0)
        else:
            check('unicode_counterpart_snapped_inward', [1, 2, 252, 0, 2] in spans.tolist()
                  and stats['utf8_snapped'] > 0, stats=stats)
            valid = True
            for a, s, e, b, d in spans:
                valid &= 0 <= s < e <= len(raw[a]) and 0 <= d <= len(raw[b])-(e-s)
                valid &= raw[a][s:e] == raw[b][d:d+e-s] and offsets[b]+d < offsets[a]+s
                for part in (raw[a][:s], raw[a][s:e], raw[b][:d], raw[b][d:d+e-s]):
                    part.decode('utf-8')
            check('cross_item_windows_excluded', valid and stats['boundary_crossing'] > 0)

    source = args.output / 'unicode'
    def negative(name, mutation, wanted):
        root = args.output / name
        root.mkdir()
        for file in ('corpus.bin', 'corpus.bin.table.bin', 'offsets.npy'):
            shutil.copyfile(source / file, root / file)
        shutil.copytree(source / 'cache', root / 'cache')
        width = (root / 'corpus.bin.table.bin').stat().st_size // (root / 'corpus.bin').stat().st_size
        mutation(root, width)
        error = None
        try:
            resolve_groups(root / 'corpus.bin', np.load(root / 'offsets.npy'), 200, root / 'windows.sqlite')
        except SystemExit as exc:
            error = str(exc).split(':')[0]
        check(name, error == wanted, code=error)
    def pointer(root):
        return next((root / 'cache').glob('dups_*'))
    def sizes(root):
        return next((root / 'cache').glob('sizes_*'))
    def replace_first(file, width, value):
        file.write_bytes(value.to_bytes(width, 'little') + file.read_bytes()[width:])
    negative('missing_pair', lambda r, w: sizes(r).unlink(), 'MNLM_SA_GROUP_MISSING')
    negative('truncated_pointer', lambda r, w: pointer(r).write_bytes(pointer(r).read_bytes()[:-1]), 'MNLM_SA_GROUP_TRUNCATED')
    negative('invalid_group_count', lambda r, w: replace_first(sizes(r), w, 1), 'MNLM_SA_GROUP_GEOMETRY')
    negative('outside_corpus', lambda r, w: replace_first(pointer(r), w, (r / 'corpus.bin').stat().st_size), 'MNLM_SA_GROUP_GEOMETRY')
    def unequal(r, width):
        file = pointer(r)
        raw = file.read_bytes()
        old = int.from_bytes(raw[:width], 'little')
        # One shifted pointer breaks exact equality; within corpus and item.
        replace_first(file, width, old-1 if old else old+1)
    negative('unequal_group_bytes', unequal, 'MNLM_SA_GROUP_BYTES')
    error = None
    try:
        resolve_groups(source / 'corpus.bin', np.load(source / 'offsets.npy'), 200, source / 'windows.sqlite')
    except SystemExit as exc:
        error = str(exc).split(':')[0]
    check('incomplete_scratch_not_reused', error == 'MNLM_SA_WORK_EXISTS', code=error)
    report = dict(acceptance='author-checked only', training_records=0, checks=checks,
                  passed=sum(c['passed'] for c in checks), failed=sum(not c['passed'] for c in checks))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    inventory = {p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in args.output.rglob('*') if p.is_file()}
    (args.output / 'SHA256.json').write_text(json.dumps(inventory, sort_keys=True))
    print(json.dumps(report))
    return int(report['failed'] > 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sys.exit(run(parser.parse_args()))
