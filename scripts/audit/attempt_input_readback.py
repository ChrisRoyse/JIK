"""Read typed field arrays from native bytes and reject source/probe substitutions."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check, sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from model.attempt_input import AttemptNative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    a = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(a.config.read_bytes())
    output = Path(config['output'])
    check(not output.exists(), 'ATTEMPT_INPUT_READBACK_EXISTS')
    native = AttemptNative(**config['input'])
    before = copy.deepcopy(native.files)
    loaded = time.perf_counter()
    conditions = {}
    for condition in sorted(native.population.prepared.conditions.inputs):
        arrays, description = native.materialize(condition)
        expected_fields = [r for r in native.fields if r['condition'] == condition]
        lanes = {}
        for spec in native.specs:
            expected, cursor = [], 0
            spans = description['fields'][spec.name]
            for item, span in zip(expected_fields, spans, strict=True):
                origin, unit = item['origin'], item['unit']
                with np.load(native.roots[origin]/spec.name/'slots.npz', allow_pickle=False) as data:
                    ptr = data['offsets']
                    values = data['vectors'][ptr[unit]:ptr[unit+1]].copy()
                check(unit != native.probes[origin] and span['role'] == item['role']
                      and span['origin'] == origin and span['source_unit'] == unit
                      and span['start'] == cursor and span['end'] == cursor+len(values),
                      'ATTEMPT_INPUT_FIELD_BOUNDARY')
                expected.append(values)
                cursor += len(values)
            check(np.array_equal(arrays[spec.name], np.concatenate(expected)),
                  'ATTEMPT_INPUT_NATIVE_VALUES')
            lanes[spec.name] = dict(shape=list(arrays[spec.name].shape),
                sha256=sha(arrays[spec.name].tobytes()), fields=spans, instrument=spec.identity)
        conditions[condition] = dict(description=digest(description), lanes=lanes)
    reconstructed = time.perf_counter()
    controls = []

    def refuse(name, code, fn):
        try:
            fn()
        except ValueError as error:
            check(code in str(error), 'ATTEMPT_INPUT_CONTROL_CODE:'+name+':'+str(error))
        else:
            raise AssertionError('ATTEMPT_INPUT_CONTROL_ACCEPTED:'+name)
        native.fresh()
        check(native.files == before, 'ATTEMPT_INPUT_CONTROL_MUTATED')
        controls.append(dict(name=name, code=code))

    condition = sorted(conditions)[0]
    fields = copy.deepcopy(native.fields)
    selected = [i for i,r in enumerate(fields) if r['condition'] == condition]
    fields[selected[0]]['role'] = 'program'
    refuse('changed_role_order', 'PROPOSAL_NATIVE_ROLES',
           lambda:native.materialize(condition, fields=fields))
    fields = copy.deepcopy(native.fields)
    fields[selected[0]]['unit'] = native.probes['new']
    refuse('probe_as_model_input', 'PROPOSAL_NATIVE_PROBE_LEAK',
           lambda:native.materialize(condition, fields=fields))
    fields = copy.deepcopy(native.fields)
    fields[selected[0]]['text_sha256'] = '0'*64
    refuse('changed_surface', 'PROPOSAL_NATIVE_SURFACE',
           lambda:native.materialize(condition, fields=fields))
    fields = copy.deepcopy(native.fields)
    fields[selected[0]]['origin'] = 'cache'
    refuse('cache_as_action', 'PROPOSAL_NATIVE_ORIGIN',
           lambda:native.materialize(condition, fields=fields))
    refuse('wrong_native_pin', 'PROPOSAL_NATIVE_PIN',
           lambda:AttemptNative(**dict(config['input'], native_pin='0'*64)))
    with tempfile.TemporaryDirectory(prefix='attempt-input-control-', dir=output.parent) as tmp:
        clone = Path(tmp)/'native'
        shutil.copytree(native.roots['new'], clone)
        target = clone/native.specs[0].name/'slots.npz'
        target.write_bytes(target.read_bytes()+b'changed')
        refuse('corrupted_slot_file', 'MNLM_WINDOW_READBACK_FILE_HASH',
               lambda:AttemptNative(**dict(config['input'], native=str(clone))))
    finished = time.perf_counter()
    report = dict(schema='attempt-input-readback-v1', input=config['input'], pins=native.pins,
        conditions=conditions, controls=controls, files=before, source=native.population.prepared.source.reference(),
        phases=dict(authentication=loaded-started, reconstruction=reconstructed-loaded,
                    controls=finished-reconstructed), wall_seconds=finished-started,
        acceptance='author-checked', number_status='measured', scientific_admission=False,
        serving=False, forecast_present=False, execution_present=False)
    pin = write_committed(output, report)
    print(json.dumps(dict(pin=pin, conditions=len(conditions), instruments=len(native.specs),
                         controls=controls, wall_seconds=report['wall_seconds'])))


if __name__ == '__main__':
    main()
