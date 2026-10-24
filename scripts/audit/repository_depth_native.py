"""Independently read every depth field, native window and causal cutoff."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_cohort_native import reconstruct
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_depth_population import DepthPopulation
from model.contracts import ModelConfig
from model.repository_depth_input import DepthNative
from senses.contract import SenseError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    started = time.perf_counter()
    spec = json.loads(a.spec.read_bytes())
    native = DepthNative(**spec,partition='learn_1')
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
    start = fields[first]['input_available_by']
    refuse('future_echo','REPOSITORY_NATIVE_FUTURE',lambda:native.materialize(first,'echo',start))
    refuse('other_partition','COHORT_NATIVE_PARTITION',lambda:native.materialize(second,'input',start))
    refuse('evaluation_partition','COHORT_NATIVE_TRAINING_PARTITION',
           lambda:DepthNative(**spec,partition='evaluation'))
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
        refuse('altered_source_bytes','DEPTH_POPULATION_BYTES',lambda:DepthPopulation(target,pop.pin))
    receipts = {}
    for partition in ('learn_1','learn_2'):
        if partition != native.partition:
            native = DepthNative(**spec,partition=partition)
        for key,f in fields.items():
            if f['partition'] != partition:
                continue
            receipts[key] = {}
            for side,clock in (('input','input_available_by'),('echo','echo_available_after')):
                actual = native.receipt(key,side,f[clock],ModelConfig(),'cpu',native.specs)
                expected = reconstruct(pop,Path(spec['native']),spec['native_pin'],key,side)
                check(actual == expected,'COHORT_NATIVE_RECONSTRUCTION')
                receipts[key][side] = actual
    result = dict(schema='repository-depth-native-readback-v1',native=spec,fields=totals,
        controls=controls,receipts=receipts,window_readback=native.readback,
        source_actions=len(receipts),number_status='measured',acceptance='author-checked',
        fragments_are_independent_atoms=False,learned_repair=False,scientific_admission=False,
        wall_seconds=time.perf_counter()-started)
    pin = write_committed(a.output,result)
    print(json.dumps(dict(pin=pin,fields=totals,actions=len(receipts),controls=len(controls),
        wall_seconds=result['wall_seconds'])),flush=True)


if __name__ == '__main__':
    main()
