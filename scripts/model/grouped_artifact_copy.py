"""Independent artifact copies with bounded pending writes and explicit publication."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


def need(value, code):
    if not value:
        raise RuntimeError('MNLM_ARTIFACT_COPY_' + code)


def sha(path):
    try:
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    except OSError as error:
        raise RuntimeError('MNLM_ARTIFACT_COPY_READ') from error


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def entries(source, destination, files):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    need(not destination.is_relative_to(source) and not source.is_relative_to(destination), 'OVERLAP')
    need(isinstance(files, dict) and bool(files), 'FILES')
    result = []
    for name, pin in sorted(files.items()):
        need(isinstance(name, str) and '\\' not in name and '\x00' not in name, 'PATH')
        rel = PurePosixPath(name)
        need(not rel.is_absolute() and name == rel.as_posix() and all(p not in ('', '.', '..') for p in rel.parts)
             and name not in ('publication.json', 'publication.pending'), 'PATH')
        need(isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin), 'PIN')
        path = source / name
        need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source), 'SOURCE')
        result.append((name, pin, path.stat().st_size))
    return source, destination, result


def directory_sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy(source, destination, files, *, pending_bytes=64 << 20, pending_files=32,
         reference_copy=None, observe=None):
    """No partial directory is a publication; a receipt commits only complete bytes.

    The optional reference operation is the unchanged existing copier, used for
    matched measurements. Pending bytes bound write grouping, not total work.
    A file larger than the group bound is streamed alone with a 1 MiB buffer.
    """
    need(type(pending_bytes) is int and pending_bytes > 0 and type(pending_files) is int
         and pending_files > 0, 'GROUP_BOUND')
    source, destination, rows = entries(source, destination, files)
    need(not destination.exists(), 'OUTPUT_EXISTS')
    destination.mkdir(parents=True, exist_ok=False)
    pending = []; pending_size = 0; durable = []; groups = []; directories = {destination}

    def event(kind, name=None):
        if observe is not None:
            observe(kind, name)

    def flush():
        nonlocal pending_size
        if not pending:
            return
        names = []
        for name, pin, size, stream in pending:
            os.fsync(stream.fileno()); stream.close()
            durable.append(name); names.append(name); event('file_fsynced', name)
        for name, pin, size, _ in pending:
            need(sha(destination / name) == pin == sha(source / name), 'BYTES')
            need((destination / name).stat().st_size == size, 'SIZE')
        groups.append(dict(files=names, bytes=pending_size))
        pending.clear(); pending_size = 0

    try:
        for name, pin, size in rows:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            directories.update(p for p in target.parents if p == destination or p.is_relative_to(destination))
            if reference_copy is not None:
                reference_copy(source / name, target, pin)
                durable.append(name); groups.append(dict(files=[name], bytes=size))
                event('file_fsynced', name)
                continue
            if pending and (len(pending) >= pending_files or pending_size + size > pending_bytes):
                flush()
            stream = target.open('xb')
            pending.append((name, pin, size, stream)); pending_size += size
            digest = hashlib.sha256(); length = 0
            with (source / name).open('rb') as original:
                while block := original.read(1 << 20):
                    stream.write(block); digest.update(block); length += len(block)
                    event('chunk_written', name)
            stream.flush()
            need(length == size and digest.hexdigest() == pin, 'SOURCE_BYTES')
            event('file_written', name)
            if pending_size >= pending_bytes or len(pending) >= pending_files:
                flush()
        flush()
        need(durable == [name for name, _, _ in rows], 'DURABLE_SET')
        # The final source check is common to reference and candidate copies.
        for name, pin, _ in rows:
            need(sha(source / name) == pin, 'SOURCE_CHANGED')
            need((source / name).stat().st_ino != (destination / name).stat().st_ino
                 or (source / name).stat().st_dev != (destination / name).stat().st_dev, 'HARDLINK')
        event('payloads_checked')
        for path in sorted(directories, key=lambda p: (-len(p.parts), str(p))):
            directory_sync(path)
        event('directories_fsynced')
        receipt = dict(schema='grouped-artifact-copy-v1', complete=True, code=sha(__file__),
                       source=str(source), files={n: dict(sha256=p, bytes=s) for n, p, s in rows},
                       durable_files=durable, groups=groups, pending_bytes=pending_bytes,
                       pending_files=pending_files, implementation='reference' if reference_copy else 'grouped')
        with (destination / 'publication.pending').open('xb') as out:
            out.write(canonical(receipt)); out.flush(); os.fsync(out.fileno())
        event('receipt_fsynced')
        os.replace(destination / 'publication.pending', destination / 'publication.json')
        event('receipt_renamed')
        directory_sync(destination); directory_sync(destination.parent)
        event('committed')
        return receipt
    except OSError as error:
        raise RuntimeError('MNLM_ARTIFACT_COPY_IO') from error
    finally:
        for _, _, _, stream in pending:
            try:
                stream.close()
            except OSError:
                # An earlier copy/flush failure already prevents publication.
                # Cleanup must not replace its actionable exception.
                pass


def readback(source, destination, files, receipt_pin):
    source, destination, rows = entries(source, destination, files)
    receipt_path = destination / 'publication.json'
    need(receipt_path.is_file(), 'UNPUBLISHED')
    need(sha(receipt_path) == receipt_pin, 'RECEIPT_PIN')
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (ValueError, OSError) as error:
        raise RuntimeError('MNLM_ARTIFACT_COPY_RECEIPT') from error
    expected = {n: dict(sha256=p, bytes=s) for n, p, s in rows}
    need(receipt['schema'] == 'grouped-artifact-copy-v1' and receipt['complete'] is True
         and receipt['code'] == sha(__file__) and receipt['source'] == str(source)
         and receipt['files'] == expected and receipt['implementation'] in ('reference', 'grouped')
         and type(receipt['pending_bytes']) is int and receipt['pending_bytes'] > 0
         and type(receipt['pending_files']) is int and receipt['pending_files'] > 0, 'RECEIPT_SCOPE')
    need(receipt['durable_files'] == list(expected)
         and [n for g in receipt['groups'] for n in g['files']] == list(expected), 'DURABLE_SET')
    for group in receipt['groups']:
        need(group['bytes'] == sum(expected[n]['bytes'] for n in group['files']), 'GROUP_BYTES')
        if receipt['implementation'] == 'grouped':
            need(len(group['files']) <= receipt['pending_files']
                 and (group['bytes'] <= receipt['pending_bytes'] or len(group['files']) == 1), 'GROUP_BOUND')
    observed = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
    need(observed == set(files) | {'publication.json'}, 'OUTPUT_SET')
    for name, pin, size in rows:
        path = destination / name
        need(not path.is_symlink() and path.resolve().is_relative_to(destination)
             and path.stat().st_size == size, 'OUTPUT_FILE')
        need(sha(path) == pin == sha(source / name), 'BYTES')
        need(path.stat().st_dev != (source / name).stat().st_dev
             or path.stat().st_ino != (source / name).stat().st_ino, 'HARDLINK')
    return receipt
