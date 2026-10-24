"""Reconstruct a continued cohort intake from persisted events and original observations."""
import argparse
import copy
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit.repository_cohort_native import reconstruct
from audit.repository_memory_readback import chain,read
from claims.codata_source import check,sha
from claims.episodic_memory import digest
from claims.process_conditions import write_committed
from claims.repository_depth_population import DepthPopulation
from claims.repository_observation import SCOPE
from model.repository_depth_memory import code_pins
from senses.trajectory_population import read_pinned


def audit(root,output):
    start = time.perf_counter()
    config = read(root/'config.json')
    check(config['schema'] == 'repository-depth-memory-v1','DEPTH_READER_MEMORY_SCHEMA')
    parent = Path(config['source'])
    proof = read_pinned(Path(config['proof']),config['proof_pin'])
    immutable = {str(parent/n):pin for n,pin in config['parent_files'].items()}
    immutable.update(proof['immutable_after'])
    for n,pin in tuple(immutable.items()):
        if Path(n).is_relative_to(parent/'candidates'):
            immutable[str(root/Path(n).relative_to(parent))] = pin
    before = {n:sha(Path(n).read_bytes()) for n in immutable}
    check(before == immutable,'DEPTH_READER_PARENT_BYTES')
    old,final = read(parent/'snapshot-final.json'),read(root/'snapshot-final.json')
    oc,old_rows = chain(parent/'experience.sqlite3')
    nc,rows = chain(root/'experience.sqlite3')
    check(oc == nc and rows[:-2] == old_rows and len(rows) == final['events'] == old['events']+2 and
          old_rows[-1][2] == old['head'] and rows[-1][2] == final['head'],'DEPTH_READER_CHAIN')
    spec = config['native']
    pop = DepthPopulation(spec['population'],spec['population_pin'])
    ob = pop.observations
    native_proof = read_pinned(Path(config['native_proof']),config['native_proof_pin'])
    check(native_proof['native'] == spec,'DEPTH_READER_NATIVE_PROOF')
    previous,head = old['head'],old['memory']['head']
    last_records = next(pay['memory_records'] for _,pay,_ in reversed(old_rows)
                        if pay and pay.get('memory_records'))
    sequence = last_records[-1]['sequence']+1
    states = copy.deepcopy(old['memory']['states'])
    intakes = copy.deepcopy(old.get('repository_depth_intakes',{}))
    episodes,counts = {},{}
    for partition,(body,payload,event_pin) in zip(('learn_1','learn_2'),rows[-2:],strict=True):
        chosen = dict(**spec,partition=partition)
        check(body == dict(kind='repository_depth_intake',spec=chosen,prior_head=previous,
                          payload=digest(payload),code=code_pins()) and
              read(root/(partition+'-intake.json')) == dict(event=body,payload=payload),
              'DEPTH_READER_EVENT')
        selected = sorted((k for k,v in pop.fields.items() if v['partition'] == partition),
                          key=lambda k:(pop.fields[k]['input_available_by'],k))
        check(selected,'DEPTH_READER_EMPTY_PARTITION')
        expected,ids,records = {},{},[]
        for key in selected:
            ep = ob.episode(key)
            check(ep['id'] not in states,'DEPTH_READER_REUSED_EPISODE')
            receipt = ob.assess(ep)
            ids[key] = ep['id'];episodes[ep['id']] = ep
            for event in (dict(op='intake',episode=ep),dict(op='check_source',episode=ep['id'],result=receipt),
                          dict(op='promote',episode=ep['id'],receipt=receipt['id'])):
                record = dict(sequence=sequence,previous=head,event=event)
                head = digest(record);record['sha256'] = head
                records.append(record);sequence += 1
            states[ep['id']] = dict(active=True,grounded_scope=SCOPE,review='source_checked',receipt=receipt)
            expected[key] = dict(episode=ep['id'],**{side:reconstruct(pop,Path(spec['native']),
                spec['native_pin'],key,side) for side in ('input','echo')})
            check(all(expected[key][s] == native_proof['receipts'][key][s] for s in ('input','echo')),
                  'DEPTH_READER_NATIVE_JOIN')
        check(payload == dict(episodes=ids,bindings=expected,memory_records=records,
            observations=ob.pin,partition=partition,native_measured=True,native_consumer='typed_constellation',
            independent_actions=True,learned_repair=False,scientific_admission=False,serving=False),
            'DEPTH_READER_RECONSTRUCTION')
        check(digest(chosen) not in intakes,'DEPTH_READER_REUSED_INTAKE')
        intakes[digest(chosen)] = dict(event=body,episodes=ids,bindings=expected)
        snap = read(root/(partition+'-snapshot.json'))
        check(snap['head'] == event_pin and snap['memory']['states'] == states and
              snap['memory']['head'] == head and snap['repository_depth_intakes'] == intakes,
              'DEPTH_READER_PARTITION_STATE')
        previous = event_pin;counts[partition] = len(ids)
    expected_memory = dict(old['memory'],head=head,states=states,
        active_episodes=sorted(set(old['memory']['active_episodes'])|set(episodes)),
        grounded_world_claims=sorted(set(old['memory']['grounded_world_claims'])|set(episodes)))
    check(final == dict(old,head=previous,events=old['events']+2,memory=expected_memory,
                        repository_depth_intakes=intakes),'DEPTH_READER_FINAL_STATE')
    cold = read(root/'cold-stage.json')
    check(cold['snapshot'] == sha((root/'snapshot-final.json').read_bytes()) and
          cold['journal'] == sha((root/'experience.sqlite3').read_bytes()) and
          cold['head'] == final['head'] and cold['events'] == len(rows) and cold['cold_replay_equal'],
          'DEPTH_READER_COLD')
    controls = native_proof['controls']+read(root/'intake-stage.json')['controls']
    with tempfile.TemporaryDirectory(prefix='growth-memory-control-',dir=root.parent) as tmp:
        clone = Path(tmp)/'journal.sqlite3'
        shutil.copyfile(root/'experience.sqlite3',clone)
        with sqlite3.connect(clone) as db:
            db.execute('UPDATE payloads SET data=? WHERE hash=?',(b'{}',rows[-1][0]['payload']))
        try:
            chain(clone)
        except ValueError as exc:
            check('REPOSITORY_READER_PAYLOAD' in str(exc),'DEPTH_READER_CONTROL_CODE')
        else:
            raise AssertionError('DEPTH_READER_CHANGED_PAYLOAD_ACCEPTED')
        controls.append(dict(name='changed_persisted_payload',code='REPOSITORY_READER_PAYLOAD'))
    after = {n:sha(Path(n).read_bytes()) for n in immutable}
    check(before == after,'DEPTH_READER_MUTATED')
    result = dict(schema='repository-depth-memory-readback-v1',events=len(rows),head=final['head'],
        source_events=len(old_rows),source_head=old['head'],snapshot=cold['snapshot'],journal=cold['journal'],
        memory_head=head,partitions=counts,source_actions=len(episodes),controls=controls,
        immutable_before=before,immutable_after=after,cold_replay_equal=True,old_state_unchanged=True,
        old_intakes_unchanged=True,native=spec,wall_seconds=time.perf_counter()-start,
        number_status='measured',acceptance='author-checked',learned_repair=False,
        scientific_admission=False,serving=False)
    pin = write_committed(output,result)
    print(json.dumps(dict(pin=pin,events=len(rows),head=final['head'],actions=len(episodes),
                         controls=len(controls),wall_seconds=result['wall_seconds'])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    audit(a.root,a.output)
