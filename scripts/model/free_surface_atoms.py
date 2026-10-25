"""Stream complete generated atoms, preserving refused suffixes and native END.

This adapter authenticates persisted frames; it does not certify decoder arithmetic
or semantic validity. Original-forward readback and native remeasurement are separate.
"""
import codecs
import json
from pathlib import Path

from model.complete_answer_trace import replay
from model.free_answer_surface import rows
from model.native_population_training import fingerprint
from text.constellation_corpus import digest, file_sha, need


def read(path, pin):
    need(file_sha(path) == pin, 'FREE_ATOMS_PIN')
    return json.loads(Path(path).read_bytes())


def terminal(report, *, attempted, completed, size, decisions, refusal, native):
    expected = dict(attempted_atoms=attempted, completed_atoms=completed,
        generated_bytes=size, byte_decisions=decisions, surface_refusal=refusal,
        native_events=native['events'], native_atoms=native['generated_atoms'],
        native_ended=native['model_ended'], native_status=native['status'],
        native_refusal=native['refusal'],
        answer_ended=native['model_ended'] and refusal is None and completed == native['generated_atoms'])
    need(all(report[k] == value for k, value in expected.items()), 'FREE_ATOMS_TERMINAL')
    return dict(expected, unattempted_atoms=native['generated_atoms'] - attempted,
        accepted=False, serving=False, semantic_validity_established=False)


def frame(stream):
    value = next(stream, None)
    need(value is not None, 'FREE_ATOMS_MISSING_FRAME')
    return value


def atoms(root, pin, *, buffer_bytes=16 << 20):
    """Drain this iterator, including its summary, to authenticate the whole arm.

    One atom is buffered within a physical allowance. A refused partial atom is
    yielded separately, never as an embedding input. No answer/reference is read.
    """
    root = Path(root)
    need(type(buffer_bytes) is int and buffer_bytes > 0, 'FREE_ATOMS_BUFFER')
    report = read(root / 'report.json', pin)
    request = read(root / 'request.json', report['request'])
    need(report['complete'] is True and report['reference_visible'] is False
         and report['serving'] is False and report['semantic_validity_established'] is False
         and request['schema'] == 'free-answer-surface-v1' and request['reference_visible'] is False,
         'FREE_ATOMS_SCOPE')
    path = Path(request['trace_path']); native = read(path / 'report.json', request['trace'])
    source = read(path / 'request.json', request['trace_request'])
    need(native['files']['request.json'] == request['trace_request']
         and source['reference_visible'] is False, 'FREE_ATOMS_NATIVE_REQUEST')
    stream = rows(root, pin); attempted = completed = size = decisions = generated = events = 0
    refusal = None
    with (root / 'surface.bin').open('rb') as surface:
        for prediction in replay(path, request['trace']):
            events += 1
            if prediction['kind'] == 'end':
                continue
            index = generated; generated += 1
            if refusal is not None:
                continue
            row, logits = frame(stream)
            region = request['region_pin'] + ':' + '.'.join(map(str, prediction['codes'][0].tolist()))
            need(logits is None and row['kind'] == 'atom_start' and row['atom'] == index
                 and row['regions'] == [region], 'FREE_ATOMS_ORDER')
            binding = dict(arm=request['arm'], source_report=pin, source_request=report['request'],
                trace=request['trace'], trace_path=str(path), atom=index,
                prediction=fingerprint(prediction), decoder=request['decoder'],
                input_native=row['native'], region=region, lens_ids=source['authority']['lens_ids'])
            attempted += 1; raw = bytearray(); offset = surface.tell()
            decoder = codecs.getincrementaldecoder('utf-8')('strict'); invalid = False
            while True:
                row, logits = frame(stream); kind = row['kind']
                need(row['atom'] == index and row['serving'] is False
                     and row['validity'] == 'unassessed', 'FREE_ATOMS_FRAME')
                decisions += int(logits is not None)
                if kind == 'byte':
                    need(not invalid and row['position'] == len(raw)
                         and type(row['value']) is int and 0 <= row['value'] < 256
                         and row['token'] == row['value'] and logits is not None, 'FREE_ATOMS_BYTE')
                    need(len(raw) < buffer_bytes, 'FREE_ATOMS_PHYSICAL_BUFFER_EXHAUSTED')
                    value = surface.read(1)
                    need(value == bytes([row['value']]), 'FREE_ATOMS_RAW_BYTE')
                    raw.extend(value)
                    try:
                        decoder.decode(value, final=False)
                    except UnicodeDecodeError:
                        invalid = True
                    need(row['utf8_prefix_valid'] is (not invalid), 'FREE_ATOMS_UTF8_FLAG')
                    continue
                need(kind in ('atom_end', 'refuse') and row['bytes'] == len(raw)
                     and row['sha256'] == digest(raw), 'FREE_ATOMS_ATOM_BYTES')
                size += len(raw)
                if kind == 'atom_end':
                    need(not invalid and raw and row['token'] == 257 and row['position'] == len(raw)
                         and logits is not None and row['code'] is None, 'FREE_ATOMS_EOS')
                    try:
                        decoder.decode(b'', final=True)
                    except UnicodeDecodeError as exc:
                        raise ValueError('FREE_ATOMS_EOS_UTF8') from exc
                    completed += 1
                else:
                    refusal = row['code']
                    need(refusal in ('SURFACE_STREAM_UTF8', 'SURFACE_STREAM_EMPTY', 'SURFACE_STREAM_SYMBOL'),
                         'FREE_ATOMS_REFUSAL')
                    if invalid:
                        need(refusal == 'SURFACE_STREAM_UTF8' and logits is None, 'FREE_ATOMS_PENDING_UTF8')
                yield dict(kind='atom' if kind == 'atom_end' else 'refused_atom', raw=bytes(raw),
                    offset=offset, bytes=len(raw), sha256=digest(raw), refusal=refusal, binding=binding)
                break
        need(next(stream, None) is None and surface.read(1) == b'', 'FREE_ATOMS_TRAILING')
    need(events == native['events'] and generated == native['generated_atoms'], 'FREE_ATOMS_NATIVE_COUNTS')
    summary = terminal(report, attempted=attempted, completed=completed, size=size,
        decisions=decisions, refusal=refusal, native=native)
    need(file_sha(root / 'report.json') == pin, 'FREE_ATOMS_CHANGED')
    yield dict(kind='summary', **summary)
