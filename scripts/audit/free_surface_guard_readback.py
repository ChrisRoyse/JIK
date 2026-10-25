"""Independent raw-span and tokenizer readback of the free-surface guard plan."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from claims.process_conditions import write_committed as write
from model import free_surface_atoms as adapter
from model.complete_answer_trace import replay
from model.native_population_training import fingerprint
from text.answer_native_plan import authority, tokenizer
from text.canonical_windows import tokenize
from text.constellation_corpus import canonical, digest, file_sha, need


def read(path, pin):
    raw = Path(path).read_bytes()
    need(digest(raw) == pin, 'FREE_GUARD_COLD_PIN')
    return json.loads(raw)


def recount(root, pin):
    """Read raw offsets directly from the index, independently of adapter.atoms."""
    own = read(root / 'report.json', pin)
    for name, h in own['files'].items():
        need(file_sha(root / name) == h, 'FREE_GUARD_COLD_SOURCE_BYTES')
    request = read(root / 'request.json', own['request'])
    path = Path(request['trace_path']); native = read(path / 'report.json', request['trace'])
    source = read(path / 'request.json', request['trace_request'])
    predictions = iter(replay(path, request['trace'])); expected = []; refused = []; consumed = 0
    atom = -1; offset = current = decisions = 0; body = None; binding = None
    with (root / 'surface.bin').open('rb') as raw, (root / 'events.jsonl').open() as index:
        for line in index:
            row = json.loads(line); kind = row['kind']; decisions += int('logits' in row)
            if kind == 'atom_start':
                need(body is None, 'FREE_GUARD_COLD_OVERLAP')
                prediction = next(predictions); consumed += 1; atom += 1
                need(prediction['kind'] == 'constellation' and prediction['index'] == atom == row['atom'],
                     'FREE_GUARD_COLD_ATOM_ORDER')
                region = request['region_pin'] + ':' + '.'.join(map(str, prediction['codes'][0].tolist()))
                need(row['regions'] == [region], 'FREE_GUARD_COLD_REGION')
                binding = dict(arm=request['arm'], source_report=pin, source_request=own['request'],
                    trace=request['trace'], trace_path=str(path), atom=atom,
                    prediction=fingerprint(prediction), decoder=request['decoder'], input_native=row['native'],
                    region=region, lens_ids=source['authority']['lens_ids'])
                body = bytearray(); offset = raw.tell(); current = 0
            elif kind == 'byte':
                value = raw.read(1)
                need(body is not None and row['atom'] == atom and row['position'] == current
                     and value == bytes([row['value']]), 'FREE_GUARD_COLD_BYTE')
                body.extend(value); current += 1
            else:
                need(kind in ('atom_end', 'refuse') and body is not None and row['atom'] == atom
                     and row['bytes'] == len(body) and row['sha256'] == digest(body), 'FREE_GUARD_COLD_SPAN')
                completed = kind == 'atom_end'
                item = dict(kind='atom' if completed else 'refused_atom', offset=offset, bytes=len(body),
                    sha256=digest(body), refusal=None if completed else row['code'], binding=binding,
                    source_surface=str(root / 'surface.bin'), measurement_input=digest(body) if completed else False)
                if completed:
                    need(body and row['token'] == 257, 'FREE_GUARD_COLD_EOS')
                    item['text'] = bytes(body).decode('utf-8'); expected.append(item)
                else:
                    refused.append(item)
                body = None
        need(body is None and raw.read(1) == b'', 'FREE_GUARD_COLD_TRAILING')
    for _ in predictions:
        consumed += 1
    need(consumed == native['events'] and decisions == own['byte_decisions']
         and atom + 1 == own['attempted_atoms'] and len(expected) == own['completed_atoms'], 'FREE_GUARD_COLD_RECOUNT')
    return own, native, expected, refused


def run(a):
    a.output.mkdir(exist_ok=False); began = tick = time.perf_counter()
    plan = read(a.input / 'report.json', a.pin)
    need(plan['complete'] and plan['schema'] == 'free-surface-guard-input-plan-v1'
         and plan['measurements_pending'] and plan['encoder_calls'] == plan['optimizer_updates'] == 0
         and not plan['serving'] and not plan['semantic_validity_established'], 'FREE_GUARD_COLD_SCOPE')
    proof = read(Path(plan['readback']['path']) / 'report.json', plan['readback']['pin'])
    need(proof['source'] == plan['source'] and proof['trace'] == plan['trace'], 'FREE_GUARD_COLD_PROOF')
    loaded = {}
    for name, filename in (('inputs', 'inputs.jsonl'), ('occurrences', 'occurrences.jsonl'), ('refused', 'refused.jsonl')):
        path = a.input / filename; item = plan['files'][name]
        need(path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256'], 'FREE_GUARD_COLD_FILE')
        loaded[name] = [json.loads(line) for line in path.read_bytes().splitlines()]
    source_root = Path(plan['source']['path']); source = read(source_root / 'report.json', plan['source']['pin'])
    phases = dict(authenticate=time.perf_counter() - tick); tick = time.perf_counter()
    occurrences = []; refusals = []; units = {}; originals = {}; summaries = {}
    for name in ('parent', 'corrected'):
        ref = source['arms'][name]; root = source_root / ref['path']
        own, native, complete, failed = recount(root, ref['pin']); originals[name] = (root, ref['pin'])
        for row in complete:
            text = row.pop('text'); key = row['sha256']
            need(key not in units or units[key] == text, 'FREE_GUARD_COLD_COLLISION'); units[key] = text
        occurrences.extend(complete); refusals.extend(failed)
        expected = dict(attempted_atoms=len(complete) + len(failed), completed_atoms=len(complete),
            generated_bytes=sum(row['bytes'] for row in complete + failed), byte_decisions=own['byte_decisions'],
            surface_refusal=failed[0]['refusal'] if failed else None, native_events=native['events'],
            native_atoms=native['generated_atoms'], native_ended=native['model_ended'], native_status=native['status'],
            native_refusal=native['refusal'], answer_ended=native['model_ended'] and not failed
                and len(complete) == native['generated_atoms'],
            unattempted_atoms=native['generated_atoms'] - len(complete) - len(failed),
            accepted=False, serving=False, semantic_validity_established=False, kind='summary')
        need(plan['arms'][name] == expected, 'FREE_GUARD_COLD_TERMINALS'); summaries[name] = expected
    records = [dict(id=key, text=units[key]) for key in sorted(units)]
    need(loaded == dict(inputs=records, occurrences=occurrences, refused=refusals), 'FREE_GUARD_COLD_EXTRACTION')
    counts = dict(completed_atom_occurrences=len(occurrences), unique_measurement_inputs=len(records),
        exact_duplicate_occurrences=len(occurrences) - len(records), complete_bytes=sum(row['bytes'] for row in occurrences),
        unique_input_bytes=sum(len(row['text'].encode()) for row in records), refused_atom_occurrences=len(refusals),
        refused_bytes=sum(row['bytes'] for row in refusals), unattempted_atoms=sum(v['unattempted_atoms'] for v in summaries.values()),
        native_atoms=sum(v['native_atoms'] for v in summaries.values()),
        generated_bytes=sum(v['generated_bytes'] for v in summaries.values()),
        completed_answers=sum(v['answer_ended'] for v in summaries.values()))
    need(counts == plan['counts'], 'FREE_GUARD_COLD_COUNTS')
    phases['raw_span_and_native_join'] = time.perf_counter() - tick; tick = time.perf_counter()
    identity = read(a.input / 'contracts.json', plan['contracts']); tokens = read(a.input / 'tokens.json', plan['tokens'])
    bank, specs, contracts = authority(plan['native_bank'])
    need(identity == dict(contracts=contracts, instruments={s.name: s.identity for s in specs},
        lens_ids={n: v['lens_id'] for n, v in bank.config['instruments'].items()}), 'FREE_GUARD_COLD_IDENTITY')
    recomputed = {}
    for name, contract in contracts.items():
        lens = SimpleNamespace(kind='dense', tok=tokenizer(contract),
            spec=dict(prefix=contract['prefix'], max_len=contract['max_len']))
        windows = []; active = []
        for row in records:
            _, _, traces = tokenize(lens, [row['text']])
            windows.append(len(traces[0]['windows']))
            active.append(sum(sum(w['attention_mask']) for w in traces[0]['windows']))
        expected = dict(windows=windows, active_tokens=active, total_windows=sum(windows), total_tokens=sum(active),
            max_windows=max(windows), inputs_exceeding_old_eight_window_limit=sum(n > 8 for n in windows))
        need(expected == tokens[name], 'FREE_GUARD_COLD_TOKEN_CENSUS'); recomputed[name] = expected
        need(plan['instrument_totals'][name] == {k: expected[k] for k in plan['instrument_totals'][name]},
             'FREE_GUARD_COLD_TOKEN_TOTALS')
    phases['singleton_canonical_token_readback'] = time.perf_counter() - tick; tick = time.perf_counter(); controls = {}
    def reject(name, fn, code):
        try:
            fn()
        except ValueError as exc:
            need(code in str(exc), 'FREE_GUARD_COLD_WRONG_REFUSAL'); controls[name] = str(exc)
        else:
            raise ValueError('FREE_GUARD_COLD_CONTROL_PASSED:' + name)
    root, pin = originals['parent']
    reject('wrong_source_pin', lambda: list(adapter.atoms(root, '0' * 64)), 'FREE_ATOMS_PIN')
    reject('physical_buffer', lambda: list(adapter.atoms(root, pin, buffer_bytes=1)), 'FREE_ATOMS_PHYSICAL_BUFFER_EXHAUSTED')
    for name in ('changed_raw_byte', 'false_answer_completion'):
        shadow = a.output / name; shadow.mkdir()
        own = read(root / 'report.json', pin)
        for filename in ('request.json', 'events.jsonl', 'logits.bin'):
            os.link(root / filename, shadow / filename)
        raw = (root / 'surface.bin').read_bytes()
        if name == 'changed_raw_byte':
            raw = bytes([raw[0] ^ 1]) + raw[1:]
        with (shadow / 'surface.bin').open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        own['files']['surface.bin'] = file_sha(shadow / 'surface.bin')
        if name == 'false_answer_completion':
            own['answer_ended'] = True
        mutated = write(shadow / 'report.json', own)
        reject(name, lambda: list(adapter.atoms(shadow, mutated)),
            'FREE_ATOMS_RAW_BYTE' if name == 'changed_raw_byte' else 'FREE_ATOMS_TERMINAL')
    need(all(file_sha(root / 'report.json') == pin for root, pin in originals.values()), 'FREE_GUARD_COLD_ORIGINAL_CHANGED')
    base = Path(__file__).resolve().parents[1]
    for name, h in plan['code'].items():
        need(file_sha(base / name) == h, 'FREE_GUARD_COLD_CODE_CHANGED')
    need(not torch.cuda.is_initialized(), 'FREE_GUARD_COLD_CPU_ONLY')
    phases['physical_negative_controls_and_seal'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
    need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_GUARD_COLD_PHASES')
    result = dict(complete=True, input=dict(path=str(a.input), pin=a.pin), counts=counts,
        all_raw_spans_exact=True, all_native_joins_exact=True, all_singleton_token_counts_exact=True,
        originals_unchanged=True, controls=controls, code=dict(plan['code'], **{str(Path(__file__).relative_to(base)): file_sha(Path(__file__))}),
        gpu_used=False, encoder_calls=0, measurements_pending=True, semantic_validity_established=False,
        serving=False, capability_demonstrated=False, phases=phases, seconds=seconds, acceptance='author-checked')
    result_pin = write(a.output / 'report.json', result)
    print(json.dumps(dict(complete=True, pin=result_pin, counts=counts, controls=controls, seconds=seconds)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pin', required=True); args = parser.parse_args(); torch.set_num_threads(1)
    try:
        run(args)
    except BaseException as exc:
        if args.output.exists():
            write(args.output / 'failure.json', dict(error=repr(exc)))
        raise
