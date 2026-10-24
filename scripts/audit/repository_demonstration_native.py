"""Author-check whole-field conservation, all native windows and causal cutoffs."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_native_readback import reconstruct as original_reconstruct
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_demonstration_population import DemonstrationPopulation
from claims.repository_demonstration_observation import action_paths
from model.contracts import ModelConfig
from model.proposal_input import definition
from model.repository_demonstration_input import DemonstrationNative
from senses.contract import SenseError


def reconstruct(pop,root,pin,condition,side):
    source = pop.fields[condition]
    if not any(f['parts'] for f in source[side]):
        report = json.loads((root/'report.json').read_bytes())
        check(sha((root/'report.json').read_bytes()) == pin and report['complete'] and not report['errors']
              and report['calibration_report_sha256'] == pop.pin,'COHORT_NATIVE_ABSENCE_REPORT')
        clock = source['input_available_by' if side == 'input' else 'echo_available_after']
        slots,fields = {},{}
        for name,item in sorted(report['lenses'].items()):
            raw = (root/name/'report.json').read_bytes()
            lane = json.loads(raw)
            check(sha(raw) == item['report_sha256'],'COHORT_NATIVE_ABSENCE_LANE')
            for filename,entry in lane['files'].items():
                value = (root/name/filename).read_bytes()
                check(dict(bytes=len(value),sha256=sha(value)) == entry,'COHORT_NATIVE_ABSENCE_BYTES')
            with np.load(root/name/'slots.npz',allow_pickle=False) as stored:
                dims = stored['vectors'].shape[1]
            instrument = definition(name,lane,dims)
            slots[name] = dict(instrument=instrument.identity,shape=[0,instrument.dims],
                              sha256=sha(b''),dtype='float32-le',absent=True)
            fields[name] = [dict(**f,windows=[]) for f in source[side]]
        description = dict(schema='repository-native-selection-v1',
            pins=dict(population=pop.pin,native=pin,observations=pop.observations.pin,partition=source['partition']),
            condition=condition,side=side,cutoff=clock,available=clock,fields=fields,
            task=source['task'],container=source['container'],action=source['action'],
            scope='retained_repository_surfaces',historical_occurrence_established=False,
            scientific_admission=False,serving=False,numerical_identifiers=False,
            absence='empty_retained_streams')
        return dict(description=description,slots=slots)
    result = original_reconstruct(pop,root,pin,condition,side)
    result['description']['pins']['partition'] = pop.fields[condition]['partition']
    for value in result['slots'].values():
        value['absent'] = False
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    started = time.perf_counter()
    spec = json.loads(a.spec.read_bytes())
    native = DemonstrationNative(**spec,partition='learn_1')
    pop,fields = native.population,native.population.fields
    # Independent byte reconstruction: every occurrence, including empty stderr.
    totals = dict(fields=0,bytes=0,fragments=0,continuations=0)
    for key,sides in fields.items():
        for side in ('input','echo'):
            originals = pop.observations.surfaces[key][side]
            for field,original in zip(sides[side],originals,strict=True):
                raw,cursor = bytearray(),0
                for part in field['parts']:
                    piece = pop.rows[part['unit']]['text'].encode()
                    check(part['start'] == cursor and len(piece)<=2048 and
                          part['end'] == cursor+len(piece),'COHORT_NATIVE_CONTAINMENT')
                    raw.extend(piece);cursor += len(piece)
                check(bytes(raw) == original['raw'] and cursor == field['bytes'] and
                      sha(raw) == field['sha256'] and not field['fragments_are_independent_atoms'],
                      'COHORT_NATIVE_BYTES')
                totals['fields'] += 1;totals['bytes'] += cursor
                totals['fragments'] += len(field['parts'])
                totals['continuations'] += sum(p['starts_mid_line'] for p in field['parts'])
    controls = []
    def refuse(name,code,fn):
        before = native.memory_hash(),digest(fields)
        try:
            fn()
        except (SenseError,ValueError) as exc:
            check(code in str(exc),'COHORT_NATIVE_CONTROL_CODE:'+name+':'+str(exc))
        else:
            raise AssertionError('COHORT_NATIVE_CONTROL_ACCEPTED:'+name)
        check(before == (native.memory_hash(),digest(fields)),'COHORT_NATIVE_CONTROL_MUTATED')
        controls.append(dict(name=name,code=code))
    first = next(k for k,v in fields.items() if v['partition']=='learn_1')
    second = next(k for k,v in fields.items() if v['partition']=='learn_2')
    ob=pop.observations
    original=ob.episode(first)
    forged=copy.deepcopy(original);forged['who']['action_origin']='model_generated'
    refuse('forged_action_origin','COHORT_OBSERVATION_EPISODE',lambda:ob.validate_episode(forged))
    false_outcome=copy.deepcopy(original);false_outcome['content']['outcome']=0
    refuse('forged_reference_outcome','COHORT_OBSERVATION_EPISODE',lambda:ob.validate_episode(false_outcome))
    test_patch=Path(ob.references[first]['prepared']['path'])/'reference-test.patch'
    refuse('test_oracle_as_action','DEMONSTRATION_ORACLE_PATCH',
           lambda:action_paths(test_patch.read_bytes(),set(fields[first]['excluded_oracle_paths'])))
    check(all(ob.episode(k)['reference_demonstration'] and
              not ob.episode(k)['independent_candidate_attempt'] and
              not ob.episode(k)['model_generated_action'] for k in fields),'DEMONSTRATION_NATIVE_ORIGIN')
    start = fields[first]['input_available_by']
    refuse('future_echo','REPOSITORY_NATIVE_FUTURE',lambda:native.materialize(first,'echo',start))
    refuse('other_partition','COHORT_NATIVE_PARTITION',lambda:native.materialize(second,'input',start))
    refuse('evaluation_partition','COHORT_NATIVE_TRAINING_PARTITION',
           lambda:DemonstrationNative(**spec,partition='evaluation'))
    refuse('unadmitted_assertion','REPOSITORY_NATIVE_UNADMITTED',
           lambda:native.materialize(first,'input',start,scientific_admission=True))
    refuse('swapped_action','REPOSITORY_NATIVE_CONDITION_FIELDS',
           lambda:native.materialize(first,'input',start,fields=fields[second]['input']))
    originals = native.arrays
    native.arrays = copy.deepcopy(originals)
    name = next(iter(native.arrays))
    native.arrays[name][0].flags.writeable = True
    native.arrays[name][0][0,0] += 0.125
    refuse('altered_native_vector','REPOSITORY_NATIVE_MEMORY_CHANGED',native.fresh)
    native.arrays = originals
    with tempfile.TemporaryDirectory(prefix='cohort-native-control-',dir=a.output.parent) as temp:
        target = Path(temp)/'population'
        shutil.copytree(pop.root,target)
        path = target/'repository/chunks/atoms.parquet'
        path.write_bytes(path.read_bytes()+b'changed')
        refuse('altered_source_bytes','DEMONSTRATION_POPULATION_BYTES',lambda:DemonstrationPopulation(target,pop.pin))
    receipts = {}
    for partition in ('learn_1','learn_2'):
        if partition != native.partition:
            native = DemonstrationNative(**spec,partition=partition)
        for key,f in fields.items():
            if f['partition'] != partition:
                continue
            receipts[key] = {}
            for side,clock in (('input','input_available_by'),('echo','echo_available_after')):
                actual = native.receipt(key,side,f[clock],ModelConfig(),'cpu',native.specs)
                expected = reconstruct(pop,Path(spec['native']),spec['native_pin'],key,side)
                check(actual == expected,'COHORT_NATIVE_RECONSTRUCTION')
                receipts[key][side] = actual
    result = dict(schema='repository-demonstration-native-readback-v1',native=spec,fields=totals,
        controls=controls,receipts=receipts,window_readback=native.readback,
        source_actions=len(receipts),number_status='measured',acceptance='author-checked',
        fragments_are_independent_atoms=False,learned_repair=False,scientific_admission=False,
        reference_demonstrations=True,independent_candidate_attempts=0,anchor_information_eligible=False,
        wall_seconds=time.perf_counter()-started)
    pin = write_committed(a.output,result)
    print(json.dumps(dict(pin=pin,fields=totals,actions=len(receipts),controls=len(controls),
        wall_seconds=result['wall_seconds'])),flush=True)


if __name__ == '__main__':
    main()
