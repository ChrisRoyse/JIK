"""Read real new/cached windows through the proposal adapter, with boundary controls."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import canonical, digest
from claims.process_conditions import write_committed
from model.contracts import Instrument
from model.proposal_input import ProposalNative, load


def same(actual, expected):
    check(set(actual) == set(expected) and all(np.array_equal(actual[n], expected[n]) for n in actual),
          'PROPOSAL_READBACK_WINDOW_MISMATCH')


def run(args):
    start = time.perf_counter()
    check(not args.output.exists(), 'PROPOSAL_READBACK_EXISTS')
    inputs = ProposalNative(args.population, args.population_pin, args.native, args.native_pin,
                            args.cache_native, args.cache_native_pin)
    authenticated = time.perf_counter()
    core_raw = args.model_core.read_bytes()
    check(sha(core_raw) == args.model_core_pin, 'PROPOSAL_READBACK_MODEL_PIN')
    model = json.loads(core_raw)
    specs = tuple(Instrument(**v).validate() for v in model['instruments'])
    check(specs == inputs.specs, 'PROPOSAL_READBACK_MODEL_INSTRUMENTS')
    args.output.mkdir()
    keys = sorted(inputs.population.prepared.conditions.inputs)
    arrays, descriptions, counts = {}, {}, {}
    for condition in keys:
        values, description = inputs.materialize(condition)
        independent = {}
        for spec in specs:
            pieces = []
            for field in inputs.population.fields:
                if field['condition'] == condition:
                    path = inputs.roots[field['origin']]/spec.name/'slots.npz'
                    with np.load(path, allow_pickle=False) as archive:
                        unit = field['unit']
                        pieces.append(archive['vectors'][archive['offsets'][unit]:archive['offsets'][unit+1]])
            independent[spec.name] = np.vstack(pieces)
        same(values, independent)
        np.savez(args.output/(condition+'.npz'), **values)
        with np.load(args.output/(condition+'.npz'), allow_pickle=False) as stored:
            same({n:stored[n] for n in stored.files}, independent)
        arrays[condition], descriptions[condition] = values, description
        counts[condition] = {n:len(v) for n,v in values.items()}
    write_committed(args.output/'fields.json', descriptions)
    check(json.loads((args.output/'fields.json').read_bytes()) == descriptions, 'PROPOSAL_READBACK_FIELDS')
    assembled = time.perf_counter()
    controls = {}

    def reject(name, code, fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error), 'PROPOSAL_READBACK_CONTROL_CODE')
        else:
            raise AssertionError('PROPOSAL_READBACK_CONTROL_ACCEPTED_'+name)
        controls[name] = code

    reject('wrong_native_pin', 'PROPOSAL_NATIVE_PIN',
           lambda:load(args.native, '0'*64, args.population_pin))
    changed = copy.deepcopy(inputs.fields)
    selected = next(r for r in changed if r['condition'] == keys[0] and r['role'] == 'action')
    selected['role'] = 'program'
    reject('wrong_field_role', 'PROPOSAL_NATIVE_ROLES', lambda:inputs.materialize(keys[0], fields=changed))
    changed = copy.deepcopy(inputs.fields)
    selected = next(r for r in changed if r['condition'] == keys[0] and r['role'] == 'action')
    selected['unit'] = inputs.probes['new']
    reject('probe_in_model_input', 'PROPOSAL_NATIVE_PROBE_LEAK',
           lambda:inputs.materialize(keys[0], fields=changed))
    changed = copy.deepcopy(inputs.fields)
    selected = next(r for r in changed if r['condition'] == keys[0] and r['role'] == 'action')
    selected.update(origin='cache', unit=1)
    reject('cached_action_substitution', 'PROPOSAL_NATIVE_ORIGIN',
           lambda:inputs.materialize(keys[0], fields=changed))
    changed = {n:v.copy() for n,v in arrays[keys[0]].items()}
    changed[next(iter(changed))][0,0] += .25
    reject('changed_assembled_window', 'PROPOSAL_READBACK_WINDOW_MISMATCH',
           lambda:same(changed, arrays[keys[0]]))
    scratch = args.output/'corrupted-native-copy'
    shutil.copytree(args.native, scratch)
    bad = scratch/inputs.specs[0].name/'slots.npz'
    before = bad.read_bytes()
    bad.write_bytes(before+b' ')
    reject('changed_persisted_window', 'PROPOSAL_NATIVE_BYTES',
           lambda:load(scratch, args.native_pin, args.population_pin))
    bad.write_bytes(before)
    restored = load(scratch, args.native_pin, args.population_pin)
    check(restored[1] == {s.name:s for s in specs}, 'PROPOSAL_READBACK_CONTROL_RESTORE')
    inputs.fresh()
    controlled = time.perf_counter()
    files = {p.relative_to(args.output).as_posix():dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
             for p in sorted(args.output.rglob('*')) if p.is_file()}
    now = time.perf_counter()
    phases = dict(authentication=authenticated-start, assembly_readback=assembled-authenticated,
                  controls=controlled-assembled, inventory=now-controlled)
    report = dict(schema='proposed-native-readback-v1', acceptance='author-checked', pins=inputs.pins,
        model_core=dict(path=str(args.model_core), pin=args.model_core_pin),
        instruments={s.name:s.identity for s in specs}, counts=counts, controls=controls,
        window_readbacks=inputs.readbacks, files=files, exact_new_and_cached_windows=True,
        probe_excluded=True, model_instruments_equal=True, forecast_present=False, executed=False,
        scientific_admission=False, phases=phases, wall_seconds=now-start,
        code_sha256=sha(Path(__file__).read_bytes()))
    pin = write_committed(args.output/'readback.json', report)
    return report, pin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('population', 'native', 'cache-native', 'model-core', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('population-pin', 'native-pin', 'cache-native-pin', 'model-core-pin'):
        parser.add_argument('--'+name, required=True)
    report, pin = run(parser.parse_args())
    print(json.dumps(dict(report_pin=pin, **{k:report[k] for k in ('counts','controls','phases','wall_seconds')})))
