"""Independently reassemble every retained native window and audit the journal join."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_memory_readback import chain,read
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_population import RepositoryPopulation
from model.proposal_input import definition


def reconstruct(pop,root,pin,condition,side):
    report = read(root/'report.json')
    check(sha((root/'report.json').read_bytes()) == pin and report['complete'] and not report['errors'] and
          report['calibration_report_sha256'] == pop.pin,'REPOSITORY_NATIVE_READER_REPORT')
    descriptors,slots = {},{}
    for name,item in sorted(report['lenses'].items()):
        lane = read(root/name/'report.json')
        check(sha((root/name/'report.json').read_bytes()) == item['report_sha256'],
              'REPOSITORY_NATIVE_READER_LANE')
        for filename,entry in lane['files'].items():
            raw = (root/name/filename).read_bytes()
            check(dict(bytes=len(raw),sha256=sha(raw)) == entry,'REPOSITORY_NATIVE_READER_BYTES')
        with np.load(root/name/'slots.npz',allow_pickle=False) as archive:
            vectors,offsets = archive['vectors'],archive['offsets']
        spec = definition(name,lane,vectors.shape[1])
        stream,fields,cursor = [],[],0
        for field in pop.fields[condition][side]:
            windows,raw = [],b''
            for part in field['parts']:
                unit = part['unit']
                span = pop.rows[unit]['text'].encode()
                check(part['start'] == len(raw) and part['end'] == len(raw)+len(span) and
                      sha(span) == part['sha256'],'REPOSITORY_NATIVE_READER_CONTAINMENT')
                raw += span
                values = vectors[offsets[unit]:offsets[unit+1]].astype('<f4')
                stream.append(values)
                windows.append(dict(**part,window_start=cursor,window_end=cursor+len(values)))
                cursor += len(values)
            check(len(raw) == field['bytes'] and sha(raw) == field['sha256'] and
                  field['empty'] == (not raw),'REPOSITORY_NATIVE_READER_FIELD')
            fields.append(dict(**field,windows=windows))
        values = np.concatenate(stream)
        descriptors[name] = fields
        slots[name] = dict(instrument=spec.identity,shape=list(values.shape),
                           sha256=sha(values.tobytes()),dtype='float32-le')
    source = pop.fields[condition]
    clock = source['input_available_by' if side == 'input' else 'echo_available_after']
    description = dict(schema='repository-native-selection-v1',
        pins=dict(population=pop.pin,native=pin,observations=pop.observations.pin),
        condition=condition,side=side,cutoff=clock,available=clock,fields=descriptors,
        task=source['task'],container=source['container'],action=source['action'],
        scope='retained_repository_surfaces',historical_occurrence_established=False,
        scientific_admission=False,serving=False,numerical_identifiers=False)
    return dict(description=description,slots=slots)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    started = time.perf_counter()
    root = a.root.resolve()
    config = read(root/'config.json')
    parent = Path(config['source'])
    proof = read(Path(config['proof']))
    check(sha(Path(config['proof']).read_bytes()) == config['proof_pin'],'REPOSITORY_NATIVE_READER_PROOF')
    immutable = {parent/n:pin for n,pin in config['parent_files'].items()}
    for name,pin in proof['immutable_after'].items():
        immutable[Path(name)] = pin
        if Path(name).is_relative_to(parent):
            immutable[root/Path(name).relative_to(parent)] = pin
    before = {str(path):sha(path.read_bytes()) for path in immutable}
    check(before == {str(p):pin for p,pin in immutable.items()},'REPOSITORY_NATIVE_READER_PARENT')
    old,final = read(parent/'snapshot-final.json'),read(root/'snapshot-final.json')
    oc,old_rows = chain(parent/'experience.sqlite3')
    nc,rows = chain(root/'experience.sqlite3')
    check(oc == nc and rows[:-1] == old_rows and len(rows) == final['events'] == old['events']+1 and
          rows[-1][2] == final['head'] and old_rows[-1][2] == old['head'],'REPOSITORY_NATIVE_READER_CHAIN')
    body,payload,_ = rows[-1]
    spec = config['native']
    check(body['kind'] == 'repository_native_binding' and body['spec'] == spec and
          body['prior_head'] == old['head'] and read(root/'intake.json') == dict(event=body,payload=payload),
          'REPOSITORY_NATIVE_READER_EVENT')
    pop = RepositoryPopulation(spec['population'],spec['population_pin'])
    expected = {}
    for condition,episode in old['repository_occurrences'][pop.observations.pin]['episodes'].items():
        expected[condition] = dict(episode=episode,**{side:reconstruct(pop,Path(spec['native']),
            spec['native_pin'],condition,side) for side in ('input','echo')})
    check(payload == dict(bindings=expected,observations=pop.observations.pin,native_measured=True,
        native_consumer='typed_constellation',learned_repair=False,scientific_admission=False,serving=False),
        'REPOSITORY_NATIVE_READER_RECONSTRUCTION')
    check(final['repository_native_bindings'] == {digest(spec):dict(event=body,bindings=expected)} and
          all(final[k] == v for k,v in old.items() if k not in ('head','events')) and
          set(final)-set(old) == {'repository_native_bindings'},'REPOSITORY_NATIVE_READER_STATE')
    cold = read(root/'cold-stage.json')
    check(cold['replay_equal'] and cold['head'] == final['head'] and cold['events'] == len(rows) and
          cold['snapshot'] == sha((root/'snapshot-final.json').read_bytes()),'REPOSITORY_NATIVE_READER_COLD')
    controls = read(root/'intake-stage.json')['controls']
    with tempfile.TemporaryDirectory(prefix='repository-native-reader-',dir=root.parent) as tmp:
        clone = Path(tmp)/'journal.sqlite3'
        shutil.copyfile(root/'experience.sqlite3',clone)
        with sqlite3.connect(clone) as db:
            db.execute('UPDATE payloads SET data=? WHERE hash=?',(b'{}',body['payload']))
        try:
            chain(clone)
        except ValueError as exc:
            check('REPOSITORY_READER_PAYLOAD' in str(exc),'REPOSITORY_NATIVE_READER_CONTROL_CODE')
        else:
            raise AssertionError('REPOSITORY_NATIVE_READER_CONTROL_ACCEPTED')
        controls.append(dict(name='changed_persisted_binding',code='REPOSITORY_READER_PAYLOAD'))
    after = {str(path):sha(path.read_bytes()) for path in immutable}
    check(before == after,'REPOSITORY_NATIVE_READER_MUTATED')
    result = dict(schema='repository-native-memory-readback-v1',events=len(rows),head=final['head'],
        source_events=len(old_rows),source_head=old['head'],snapshot=cold['snapshot'],
        journal=sha((root/'experience.sqlite3').read_bytes()),controls=controls,
        immutable_before=before,immutable_after=after,cold_replay_equal=True,old_state_unchanged=True,
        native=spec,bindings={n:dict(episode=v['episode'],**{s:{lens:slot['shape'][0]
            for lens,slot in v[s]['slots'].items()} for s in ('input','echo')}) for n,v in expected.items()},
        wall_seconds=time.perf_counter()-started,number_status='measured',acceptance='author-checked',
        learned_repair=False,scientific_admission=False,serving=False)
    pin = write_committed(a.output,result)
    print(json.dumps(dict(pin=pin,events=len(rows),head=final['head'],controls=len(controls),
                         wall_seconds=result['wall_seconds'])),flush=True)


if __name__ == '__main__':
    main()
