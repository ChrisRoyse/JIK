"""Prepare every completed atom of a cold-replayed trajectory for native guarding.

This CPU-only step retains completion denominators and quarantined suffixes. It
does not run instruments, calibrate a guard, or establish an answer's correctness.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'lens')]
import torch
from claims.process_conditions import write_committed as write
from model.free_surface_atoms import atoms, read
from text.answer_native_plan import authority, BANK, census
from text.constellation_corpus import canonical, file_sha, need

NEW = ('model/free_surface_atoms.py', 'audit/free_surface_guard_plan.py', 'audit/free_surface_guard_readback.py')
BASE = Path(__file__).resolve().parents[1]


def jsonl(path, records):
    with path.open('xb') as stream:
        for row in records:
            stream.write(canonical(row) + b'\n')
        stream.flush(); os.fsync(stream.fileno())
    return dict(sha256=file_sha(path), bytes=path.stat().st_size)


def build(a):
    began = tick = time.perf_counter(); a.output.mkdir(exist_ok=False)
    proof = read(a.readback / 'report.json', a.pin)
    need(proof['complete'] is True and proof['all_free_logits_exact'] is True
         and proof['all_bytes_exact'] is True and proof['all_terminals_recounted'] is True
         and proof['reference_visible'] is False
         and proof['scope'] == 'one_complete_sealed_trajectory_not_full_population', 'FREE_GUARD_READBACK_SCOPE')
    for name, pin in proof['code'].items():
        need(file_sha(BASE / name) == pin, 'FREE_GUARD_SOURCE_CODE_CHANGED')
    root = Path(proof['source']['path']); source = read(root / 'report.json', proof['source']['pin'])
    need(source['trace'] == proof['trace'] and source['reference_visible'] is False
         and set(source['arms']) == set(proof['results']) == {'parent', 'corrected'}, 'FREE_GUARD_SOURCE')
    # This proof is one complete trajectory. Scale expansion must tile the plan's
    # tokenizer working set; it must not omit atoms or shorten their text.
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    source_bytes = sum(ref['generated_bytes'] for ref in source['arms'].values())
    need(source_bytes * 64 <= available // 8, 'FREE_GUARD_TOKENIZER_MEMORY_TILE_REQUIRED')
    phases = dict(authenticate=time.perf_counter() - tick); tick = time.perf_counter()
    units = {}; occurrences = []; refusals = []; summaries = {}; expected_lenses = None
    for name in ('parent', 'corrected'):
        ref = source['arms'][name]; arm = root / ref['path']
        need(proof['results'][name]['report'] == ref['pin'], 'FREE_GUARD_ARM_PIN')
        terminal = False
        for row in atoms(arm, ref['pin']):
            need(not terminal, 'FREE_GUARD_AFTER_SUMMARY')
            if row['kind'] == 'summary':
                expected = proof['results'][name]['summary']
                need(all(row[k] == v for k, v in expected.items()), 'FREE_GUARD_REPLAY_COUNTS')
                summaries[name] = row; terminal = True; continue
            raw = row.pop('raw'); binding = row['binding']
            need(expected_lenses is None or binding['lens_ids'] == expected_lenses, 'FREE_GUARD_LENS_CHANGE')
            expected_lenses = binding['lens_ids']
            need(binding['arm'] == name and binding['trace'] == proof['trace'], 'FREE_GUARD_BINDING')
            row['source_surface'] = str(arm / 'surface.bin')
            if row['kind'] == 'refused_atom':
                row['measurement_input'] = False; refusals.append(row); continue
            text = raw.decode('utf-8', errors='strict'); key = row['sha256']
            need(key not in units or units[key] == text, 'FREE_GUARD_HASH_COLLISION')
            units[key] = text; row['measurement_input'] = key; occurrences.append(row)
        need(terminal, 'FREE_GUARD_NO_SUMMARY')
    records = [dict(id=key, text=units[key]) for key in sorted(units)]
    need(records and expected_lenses is not None, 'FREE_GUARD_NO_COMPLETED_ATOMS')
    phases['extract_complete_atoms_and_refusals'] = time.perf_counter() - tick; tick = time.perf_counter()
    bank, specs, contracts = authority(BANK)
    need(expected_lenses == {n: value['lens_id'] for n, value in bank.config['instruments'].items()},
         'FREE_GUARD_CANONICAL_IDENTITY')
    phases['frozen_native_authority'] = time.perf_counter() - tick; tick = time.perf_counter()
    tokens = {name: census(contracts[name], records) for name in bank.config['roster']}
    phases['complete_token_census'] = time.perf_counter() - tick; tick = time.perf_counter()
    files = dict(inputs=jsonl(a.output / 'inputs.jsonl', records),
        occurrences=jsonl(a.output / 'occurrences.jsonl', occurrences),
        refused=jsonl(a.output / 'refused.jsonl', refusals))
    contract_pin = write(a.output / 'contracts.json', dict(contracts=contracts,
        instruments={s.name: s.identity for s in specs}, lens_ids=expected_lenses))
    tokens_pin = write(a.output / 'tokens.json', tokens)
    need(file_sha(a.readback / 'report.json') == a.pin
         and file_sha(root / 'report.json') == proof['source']['pin'], 'FREE_GUARD_SOURCE_CHANGED')
    need(not torch.cuda.is_initialized(), 'FREE_GUARD_CPU_ONLY')
    phases['publish'] = time.perf_counter() - tick; seconds = time.perf_counter() - began
    need(.9 <= sum(phases.values()) / seconds <= 1.001, 'FREE_GUARD_PHASES')
    result = dict(schema='free-surface-guard-input-plan-v1', complete=True,
        readback=dict(path=str(a.readback), pin=a.pin), source=proof['source'], trace=proof['trace'],
        scope=proof['scope'], code={name: file_sha(BASE / name) for name in NEW},
        files=files, contracts=contract_pin, tokens=tokens_pin, native_bank=BANK,
        counts=dict(completed_atom_occurrences=len(occurrences), unique_measurement_inputs=len(records),
            exact_duplicate_occurrences=len(occurrences) - len(records),
            complete_bytes=sum(row['bytes'] for row in occurrences),
            unique_input_bytes=sum(len(row['text'].encode('utf-8')) for row in records),
            refused_atom_occurrences=len(refusals), refused_bytes=sum(row['bytes'] for row in refusals),
            unattempted_atoms=sum(row['unattempted_atoms'] for row in summaries.values()),
            native_atoms=sum(row['native_atoms'] for row in summaries.values()),
            generated_bytes=source_bytes, completed_answers=sum(row['answer_ended'] for row in summaries.values())),
        arms=summaries, instrument_totals={name: {key: row[key] for key in
            ('total_windows', 'total_tokens', 'max_windows', 'inputs_exceeding_old_eight_window_limit')}
            for name, row in tokens.items()},
        measurements_pending=True, all_native_windows_planned=True, input_truncation=False,
        reference_visible=False, gpu_used=False, encoder_calls=0, optimizer_updates=0,
        scientific_admission=False, serving=False, capability_demonstrated=False,
        semantic_validity_established=False, phases=phases, seconds=seconds, acceptance='author-checked')
    pin = write(a.output / 'report.json', result)
    print(json.dumps(dict(complete=True, pin=pin, counts=result['counts'], instruments=result['instrument_totals'],
        seconds=seconds)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('readback', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pin', required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    try:
        build(args)
    except BaseException as exc:
        if args.output.exists():
            write(args.output / 'failure.json', dict(error=repr(exc)))
        raise
