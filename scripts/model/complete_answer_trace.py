"""Persist autonomous typed answers without consulting reference length or content.

One framed event stream and one small append index per answer. Generation ends
only on its own END or an explicit physical refusal; neither truncation nor a
reader exception becomes a successful answer. Reference scoring happens later.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time

import torch

from claims.process_conditions import write_committed
from model.native_device_transport import transfer
from model.native_population_training import cpu, fingerprint
from text.capability_answer_source import stamp
from text.constellation_corpus import canonical, digest, file_sha, need

PHYSICAL = frozenset({
    'WINDOW_END_PHYSICAL_MEMORY_EXHAUSTED_NO_TRUNCATION',
    'MNLM_TEXT_CORPUS_PLACEMENT_OUTPUT_MEMORY', 'MNLM_TEXT_CORPUS_RQ_BATCH_PHYSICAL_MEMORY',
    'MNLM_CODATA_PREFIX_STREAM_CURRENT_MEMORY', 'MNLM_CODATA_PREFIX_STREAM_CACHE_MEMORY',
    'MNLM_CODATA_PREFIX_STREAM_ATTENTION_MEMORY'})
EVENT_BYTES = 64 << 20


def event_check(event, roster, proposal, generated):
    need(isinstance(event, dict) and event.get('proposal') == proposal, 'ANSWER_TRACE_EVENT')
    need(event.get('kind') in ('constellation', 'end'), 'ANSWER_TRACE_KIND')
    logits = event.get('stop_logits')
    need(isinstance(logits, torch.Tensor) and logits.shape == (1, 2)
         and bool(torch.isfinite(logits).all()), 'ANSWER_TRACE_STOP')
    if event['kind'] == 'end':
        need(event.get('generated') == generated and int(logits.argmax(-1)) == 1, 'ANSWER_TRACE_END')
        return generated, True
    need(event.get('index') == generated and event.get('source') == 'predicted'
         and int(logits.argmax(-1)) == 0 and set(event['slots']) == set(roster), 'ANSWER_TRACE_ORDER')
    for name, value in event['slots'].items():
        need(isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == 768
             and len(value) > 0 and value.dtype == torch.float32 and not value.requires_grad
             and bool(torch.isfinite(value).all()), 'ANSWER_TRACE_NATIVE:' + name)
    return generated + 1, False


def tensor_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)): return sum(tensor_bytes(v) for v in value)
    return 0


def collect(model, prefix, *, request, target, proposal, context, seed, output, authority):
    """No future/reference argument. The caller must bind the actual public prefix."""
    began = time.perf_counter(); output = Path(output); output.mkdir(exist_ok=False)
    model_pin = fingerprint(model.state_dict()); frozen = fingerprint(target.state_dict())
    manifest = dict(schema='autonomous-native-answer-v1', authority=authority, model=model_pin,
        target=frozen, roster=list(model.roster), proposal=proposal, context=context, seed=seed,
        prefix=fingerprint(prefix), observed_atoms=len(prefix), request=request.pin,
        request_descriptor=json.loads(request.descriptor), cache_bytes=model.forecaster.predictor.cache_bytes,
        serialized_event_bytes=EVENT_BYTES, reference_visible=False)
    manifest_pin = write_committed(output/'request.json', manifest)
    generator = None; generated = count = 0; ended = False; refusal = None; chain = manifest_pin
    phases = dict(prepare=time.perf_counter()-began, generation=0., persistence=0., seal=0.)
    try:
        generator = model.rollout(transfer(prefix, next(model.parameters()).device), request=request,
            target=target, proposal=proposal, context=context, seed=seed)
        with (output/'events.bin').open('xb') as stream, (output/'index.jsonl').open('x') as index:
            while True:
                tick = time.perf_counter()
                try: event = next(generator)
                except ValueError as exc:
                    if str(exc) not in PHYSICAL: raise
                    refusal = str(exc); phases['generation'] += time.perf_counter()-tick; break
                except StopIteration as exc: raise ValueError('ANSWER_TRACE_MISSING_END') from exc
                phases['generation'] += time.perf_counter()-tick; tick = time.perf_counter()
                generated, ended = event_check(event, model.roster, proposal, generated)
                need(tensor_bytes(event) <= EVENT_BYTES, 'ANSWER_TRACE_EVENT_MEMORY')
                value = cpu(event); buffer = io.BytesIO(); torch.save(value, buffer); raw = buffer.getvalue()
                need(len(raw) <= EVENT_BYTES, 'ANSWER_TRACE_SERIALIZED_MEMORY')
                row = dict(ordinal=count, offset=stream.tell(), bytes=len(raw),
                           sha256=hashlib.sha256(raw).hexdigest(), value=fingerprint(value), previous=chain)
                chain = digest(canonical(row)); row['chain'] = chain
                need(stream.write(struct.pack('>Q', len(raw))) == 8 and stream.write(raw) == len(raw),
                     'ANSWER_TRACE_SHORT_WRITE')
                index.write(json.dumps(row, sort_keys=True)+'\n'); count += 1
                del event, value, buffer, raw
                phases['persistence'] += time.perf_counter()-tick
                if ended: break
            tick = time.perf_counter()
            stream.flush(); os.fsync(stream.fileno()); index.flush(); os.fsync(index.fileno())
        generator.close(); need(generator.gi_frame is None, 'ANSWER_TRACE_GENERATOR_LIVE')
        need(fingerprint(model.state_dict()) == model_pin and fingerprint(target.state_dict()) == frozen,
             'ANSWER_TRACE_WEIGHTS_CHANGED')
        files = {n:file_sha(output/n) for n in ('request.json', 'events.bin', 'index.jsonl')}
        need(ended != (refusal is not None), 'ANSWER_TRACE_TERMINATION')
        phases['seal'] = time.perf_counter()-tick
        report = dict(complete=True, status='ended' if ended else 'physical_refusal', refusal=refusal,
            generated_atoms=generated, events=count, chain=chain, files=files, phases=phases,
            seconds=time.perf_counter()-began, model_ended=ended, reference_length_visible=False,
            semantic_validity_established=False, generator_closed=True)
        need(.9 <= sum(phases.values())/report['seconds'] <= 1.001, 'ANSWER_TRACE_PHASES')
        return write_committed(output/'report.json', report)
    except BaseException as exc:
        write_committed(output/'failure.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise
    finally:
        if generator is not None: generator.close()


def replay(output, pin):
    """Authenticate the entire stream and yield one CPU event at a time."""
    output = Path(output)
    need(file_sha(output/'report.json') == pin and not (output/'failure.json').exists(), 'ANSWER_TRACE_REPORT')
    report = json.loads((output/'report.json').read_bytes())
    paths = {n:output/n for n in ('request.json', 'events.bin', 'index.jsonl')}
    need(set(report['files']) == set(paths), 'ANSWER_TRACE_FILES')
    for n, p in paths.items(): need(file_sha(p) == report['files'][n], 'ANSWER_TRACE_BYTES')
    snapshots = {n:stamp(p) for n, p in paths.items()}
    request = json.loads(paths['request.json'].read_bytes()); count = generated = 0; ended = False
    chain = report['files']['request.json']
    need(request['schema'] == 'autonomous-native-answer-v1' and request['serialized_event_bytes'] == EVENT_BYTES
         and report['complete'] and report['reference_length_visible'] is False, 'ANSWER_TRACE_SCHEMA')
    with paths['events.bin'].open('rb') as stream, paths['index.jsonl'].open() as index:
        for line in index:
            row = json.loads(line); need(not ended and row['ordinal'] == count and row['offset'] == stream.tell()
                 and row['previous'] == chain, 'ANSWER_TRACE_INDEX')
            chain = digest(canonical({k:v for k,v in row.items() if k != 'chain'}))
            need(chain == row['chain'] and type(row['bytes']) is int and 0 < row['bytes'] <= EVENT_BYTES,
                 'ANSWER_TRACE_CHAIN')
            header = stream.read(8)
            need(len(header) == 8 and struct.unpack('>Q', header)[0] == row['bytes'], 'ANSWER_TRACE_FRAME')
            raw = stream.read(row['bytes'])
            need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'ANSWER_TRACE_PAYLOAD')
            event = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True); del raw
            need(fingerprint(event) == row['value'], 'ANSWER_TRACE_VALUE')
            generated, ended = event_check(event, request['roster'], request['proposal'], generated)
            count += 1; yield event
        need(stream.read(1) == b'', 'ANSWER_TRACE_TRAILING_BYTES')
    need(generated == report['generated_atoms'] and count == report['events'] and chain == report['chain']
         and ended == report['model_ended'], 'ANSWER_TRACE_RECOUNT')
    need((ended and report['status'] == 'ended' and report['refusal'] is None) or
         (not ended and report['status'] == 'physical_refusal' and report['refusal'] in PHYSICAL),
         'ANSWER_TRACE_TERMINAL')
    need(all(stamp(p) == snapshots[n] for n,p in paths.items()) and file_sha(output/'report.json') == pin,
         'ANSWER_TRACE_CHANGED')
