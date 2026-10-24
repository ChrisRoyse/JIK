"""Read persisted source, byte schedules, native tensors and decoder gradients independently."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.capability_journal_reader import JournalReader, read
from claims.capability_population import need, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--surfaces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    original = JournalReader(read(a.memory / 'config.json'), read(a.proofs))
    state = original.read(a.memory / 'experience.sqlite3')
    root, r = a.surfaces, read(a.surfaces / 'report.json')
    need(r['journal_head'] == state['head'] and r['optimizer_steps'] == 0
         and r['complete_target_bytes'] and r['bounded_gradients_measured']
         and not r['learned_capability'] and not r['scientific_admission'] and not r['serving'],
         'SURFACE_READER_SCOPE')
    initial = root / 'decoder-initial.npz'
    need(sha(initial.read_bytes()) == r['initial_file_sha256'], 'SURFACE_READER_PARAMETERS')
    with np.load(initial, allow_pickle=False) as z: weights = {n: z[n].copy() for n in z.files}
    need(digest({n: sha(x.tobytes()) for n, x in weights.items()}) == r['model_parameters'],
         'SURFACE_READER_PARAMETER_VALUES')
    cases = []
    for i, case in enumerate(r['cases']):
        task = state['tasks'][case['task']]
        need(task['episode'] == case['episode'] and task['binding'] == case['binding']
             and case['field'] in [f['id'] for f in task['selections']['targets']['fields']]
             and state['states'][task['episode']]['grounded_scope'] is not None, 'SURFACE_READER_TASK')
        pop = original.populations[task['population']]
        source = pop.fields[case['field']]['text'].encode()
        need(sha(source) == case['loss']['sha256'], 'SURFACE_READER_SOURCE')
        schedule = read(root / ('schedule-' + str(i) + '.json'))
        scores, cursor = [], 0
        tokens = [256, *source, 257]
        for window in schedule:
            lo, (start, stop) = window['offset'], window['scored']
            need(start == cursor and lo <= start < stop and len(window['prefix']) <= r['config']['max_prefix'],
                 'SURFACE_READER_WINDOW')
            need(window['prefix'] == tokens[lo:stop]
                 and window['labels'] == [-100] * (start-lo) + tokens[start+1:stop+1],
                 'SURFACE_READER_CAUSAL')
            scores.extend(window['labels'][start-lo:])
            cursor = stop
        need(scores == [*source, 257] and cursor == case['loss']['scored_positions']
             and len(schedule) == case['loss']['windows'], 'SURFACE_READER_COMPLETE_BYTES')
        receipts = original.proofs[pop.pin][0][case['field']]
        with np.load(root / ('native-' + str(i) + '.npz'), allow_pickle=False) as z:
            need(set(z.files) == set(receipts) == set(case['native']), 'SURFACE_READER_LANES')
            for name in z.files:
                x, ref = z[name], receipts[name]
                need(x.shape == (1, ref['windows'], ref['dimensions']) and x.dtype == np.float32
                     and sha(x.astype('<f4').tobytes()) == ref['sha256'] == case['native'][name]['sha256'],
                     'SURFACE_READER_NATIVE_BYTES')
        gp = root / case['gradient_file']['name']
        need(sha(gp.read_bytes()) == case['gradient_file']['sha256'], 'SURFACE_READER_GRADIENT_FILE')
        with np.load(gp, allow_pickle=False) as z:
            gradients = {n: z[n].copy() for n in z.files}
        need(set(gradients) <= set(weights) and all(x.shape == weights[n].shape
             and np.isfinite(x).all() for n, x in gradients.items()), 'SURFACE_READER_GRADIENT_VALUES')
        norms = {}
        for group, measured in case['gradients'].items():
            value = sum(float(np.square(x.astype(np.float64)).sum()) for n, x in gradients.items()
                        if n.startswith(group))
            need(value > 0 and abs(value-measured) <= max(abs(measured)*1e-5, 1e-30),
                 'SURFACE_READER_GRADIENT_NORM')
            norms[group] = value
        need(all(np.any(gradients[n]) for n in ('tokens.weight', 'output.weight')),
             'SURFACE_READER_TEXT_GRADIENT')
        cases.append(dict(family=case['family'], task=case['task'], field=case['field'],
            bytes=len(source), windows=len(schedule), every_byte_once=True, eos_once=True, norms=norms))
    need({c['family'] for c in cases} == {'coding', 'conversation', 'reasoning'}, 'SURFACE_READER_FAMILIES')
    result = dict(schema='capability-surface-readback-v1', source=sha((root / 'report.json').read_bytes()),
        journal_head=state['head'], cases=cases, initial_parameters=r['model_parameters'],
        source_native_bytes_schedule_and_gradients_read=True, number_status='measured',
        acceptance='author-checked', scientific_admission=False, learned_capability=False,
        predictor_kernel_guard_connected=False, serving=False)
    pin = write_committed(a.output, result)
    print(json.dumps(dict(pin=pin, cases=len(cases), largest=max(c['bytes'] for c in cases))), flush=True)


if __name__ == '__main__': main()
