"""Exact duplicate-group provenance, before collect merges overlapping windows.

One pass over reference-tool groups, then an on-disk ordered interval merge.
No all-pairs comparisons. Repeated sensory bytes remain in their observations.
"""
from pathlib import Path
import hashlib
import sqlite3
import numpy as np
from artifact_seals import refuse

RESOLVER = 'exact-groups-v1'


def group_files(corpus_path):
    corpus_path = Path(corpus_path)
    paths = sorted(p for p in (corpus_path.parent / 'cache').iterdir()
                   if p.name.startswith(('dups_' + corpus_path.name + '_', 'sizes_' + corpus_path.name + '_')))
    result = {}
    for p in paths:
        with p.open('rb') as f:
            h = hashlib.file_digest(f, 'sha256').hexdigest()
        result[p.relative_to(corpus_path.parent).as_posix()] = {'sha256': h, 'bytes': p.stat().st_size}
    return result


def integers(stream, count, width):
    raw = stream.read(count * width)
    if len(raw) != count * width:
        refuse('MNLM_SA_GROUP_TRUNCATED', 'duplicate pointer stream ended inside a group')
    a = np.frombuffer(raw, dtype=np.uint8).reshape(count, width)
    values = np.zeros(count, dtype=np.int64)
    for j in range(width):
        values |= a[:, j].astype(np.int64) << (8 * j)
    return values


def resolve_groups(corpus_path, offsets, threshold, scratch):
    corpus_path, scratch = Path(corpus_path), Path(scratch)
    size = corpus_path.stat().st_size
    table_size = corpus_path.with_name(corpus_path.name + '.table.bin').stat().st_size
    if (threshold < 1 or offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0
            or offsets[-1] != size or np.any(np.diff(offsets) < 1) or not size
            or table_size % size or not 1 <= table_size // size <= 8):
        refuse('MNLM_SA_GROUP_GEOMETRY', 'invalid corpus, boundaries or pointer geometry')
    width = table_size // size
    files = sorted((corpus_path.parent / 'cache').glob('dups_' + corpus_path.name + '_*'))
    sizes = sorted((corpus_path.parent / 'cache').glob('sizes_' + corpus_path.name + '_*'))
    if (not files or len(files) != len(sizes) or
        {p.name.removeprefix('dups_') for p in files} != {p.name.removeprefix('sizes_') for p in sizes}):
        refuse('MNLM_SA_GROUP_MISSING', 'matching producer pointer/size streams required')
    if scratch.exists():
        refuse('MNLM_SA_WORK_EXISTS', 'choose a new generation for exact-group resolution')
    data = np.memmap(corpus_path, dtype=np.uint8, mode='r')
    db = sqlite3.connect(scratch)
    counts = dict(groups=0, occurrences=0, boundary_crossing=0, singleton_in_item=0, windows=0,
                  merged=0, utf8_snapped=0, dropped_short=0)
    try:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE windows(origin INTEGER,target INTEGER,delta INTEGER,start INTEGER,end INTEGER)')
        for fp in files:
            sp = fp.with_name(fp.name.replace('dups_', 'sizes_', 1))
            if sp.stat().st_size % width or fp.stat().st_size % width:
                refuse('MNLM_SA_GROUP_TRUNCATED', str(fp))
            with fp.open('rb') as pointers, sp.open('rb') as sizes_stream:
                for _ in range(sp.stat().st_size // width):
                    count = int(integers(sizes_stream, 1, width)[0])
                    if not 2 <= count <= size:
                        refuse('MNLM_SA_GROUP_GEOMETRY', 'group size outside corpus bounds')
                    positions = integers(pointers, count, width)
                    if np.any(positions < 0) or np.any(positions > size - threshold):
                        refuse('MNLM_SA_GROUP_GEOMETRY', 'duplicate window outside corpus')
                    counts['groups'] += 1; counts['occurrences'] += count
                    # Check exact equality even for windows later excluded at item boundaries.
                    key = bytes(data[positions[0]:positions[0] + threshold])
                    if any(bytes(data[x:x + threshold]) != key for x in positions):
                        refuse('MNLM_SA_GROUP_BYTES', 'producer group contains unequal windows')
                    docs = np.searchsorted(offsets, positions, side='right') - 1
                    valid = positions + threshold <= offsets[docs + 1] - 1
                    counts['boundary_crossing'] += int((~valid).sum())
                    positions, docs = positions[valid], docs[valid]
                    if len(positions) < 2:
                        counts['singleton_in_item'] += len(positions)
                        continue
                    if len(np.unique(positions)) != len(positions):
                        refuse('MNLM_SA_GROUP_GEOMETRY', 'duplicate occurrence inside a group')
                    k = int(np.argmin(positions)); target, target_doc = int(positions[k]), int(docs[k])
                    rows = [(int(d), target_doc, target-int(x), int(x), int(x)+threshold)
                            for j, (x, d) in enumerate(zip(positions, docs)) if j != k]
                    db.executemany('INSERT INTO windows VALUES(?,?,?,?,?)', rows)
                    counts['windows'] += len(rows)
                if pointers.read(1):
                    refuse('MNLM_SA_GROUP_TRUNCATED', 'unconsumed duplicate pointers')
        db.commit()
        db.execute('CREATE INDEX ordered_windows ON windows(origin,target,delta,start,end)')
        merged = []
        pending = None
        for row in db.execute('SELECT origin,target,delta,start,end FROM windows ORDER BY origin,target,delta,start,end'):
            if pending is not None and row[:3] == pending[:3] and row[3] <= pending[4]:
                pending = (*pending[:4], max(pending[4], row[4]))
            else:
                if pending is not None: merged.append(pending)
                pending = row
        if pending is not None: merged.append(pending)
        result = []
        for a, b, delta, start, end in merged:
            counts['merged'] += 1
            s, e = start, end
            while s < e and int(data[s]) & 0xC0 == 0x80: s += 1
            while e > s and e < size and int(data[e]) & 0xC0 == 0x80: e -= 1
            counts['utf8_snapped'] += int((s, e) != (start, end))
            if e-s < threshold:
                counts['dropped_short'] += 1
                continue
            raw = bytes(data[s:e]); other = bytes(data[s+delta:e+delta])
            if raw != other:
                refuse('MNLM_SA_GROUP_BYTES', 'merged coordinates are not equal')
            raw.decode('utf-8')
            result.append((a, s-int(offsets[a]), e-int(offsets[a]), b, s+delta-int(offsets[b])))
        result = np.array(sorted(set(result)), dtype=np.int64).reshape(-1, 5)
        counts.update(resolver=RESOLVER, spans=len(result), unresolved=0,
                      annotated_bytes=int(np.sum(result[:, 2] - result[:, 1])))
        return result, counts
    finally:
        db.close()
