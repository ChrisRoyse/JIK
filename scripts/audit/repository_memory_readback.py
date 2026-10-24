"""Read repository occurrences, journal joins and retained state without a model run."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from claims.codata_source import check,sha
from claims.episodic_memory import canonical,digest,ZERO
from claims.process_conditions import write_committed
from claims.repository_observation import RepositoryObservations,SCOPE


def read(path):
    return json.loads(path.read_bytes())


def chain(path):
    records,head = [],ZERO
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        contract = db.execute('SELECT contract FROM metadata').fetchall()
        for seq,previous,raw,pin in db.execute('SELECT seq,previous,body,hash FROM events ORDER BY seq'):
            body = json.loads(raw)
            check(seq == len(records) and previous == head and canonical(body) == raw and
                  digest(dict(seq=seq,previous=previous,body=body)) == pin,'REPOSITORY_READER_CHAIN')
            payload = None
            if body.get('payload') is not None:
                row = db.execute('SELECT data FROM payloads WHERE hash=?',(body['payload'],)).fetchone()
                check(row is not None and sha(row[0]) == body['payload'],'REPOSITORY_READER_PAYLOAD')
                payload = json.loads(row[0])
                check(canonical(payload) == row[0],'REPOSITORY_READER_PAYLOAD_CANONICAL')
            records.append((body,payload,pin))
            head = pin
    return contract,records


def test_states(result,required):
    check(result['collect_exitcode'] == 0 and result['uncollectable'] == {} and
          set(result['collected']) == set(required) and len(result['collected']) == len(required),
          'REPOSITORY_READER_COLLECTION')
    phases = {}
    for phase in result['phases']:
        key = phase['nodeid'],phase['when']
        check(key not in phases,'REPOSITORY_READER_DUPLICATE_PHASE')
        phases[key] = phase['outcome']
    check(set(phases) == {(n,p) for n in required for p in ('setup','call','teardown')} and
          all(phases[n,p] == 'passed' for n in required for p in ('setup','teardown')),
          'REPOSITORY_READER_PHASES')
    states = {n:phases[n,'call'] for n in required}
    expected = {}
    for nodeid,state in states.items():
        module,_,local = nodeid.partition('::')
        # Slashes inside parameter values belong to the test name, not the module.
        expected[module.removesuffix('.py').replace('/','.')+'.'+local.replace('::','.')] = state
    actual = {}
    for case in ET.fromstring(result['junit']).iter('testcase'):
        key = case.attrib['classname']+'.'+case.attrib['name']
        check(key not in actual,'REPOSITORY_READER_DUPLICATE_JUNIT')
        actual[key] = ('failed' if case.find('failure') is not None else
                       'error' if case.find('error') is not None else
                       'skipped' if case.find('skipped') is not None else 'passed')
    check(actual == expected,'REPOSITORY_READER_JUNIT')
    check(result['exitcode'] == result['main_exitcode'] == int('failed' in states.values()),
          'REPOSITORY_READER_EXIT')
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    start = time.perf_counter()
    root = a.root.resolve()
    config = read(root/'config.json')
    parent = Path(config['source'])
    proof = read(Path(config['proof']))
    check(sha(Path(config['proof']).read_bytes()) == config['proof_pin'],'REPOSITORY_READER_PARENT_PROOF')
    immutable = {parent/n:pin for n,pin in config['parent_files'].items()}
    for n,entry in proof['state']['checkpoint_files'].items():
        source = Path(n)
        immutable[source] = entry['sha256']
        immutable[root/source.relative_to(parent)] = entry['sha256']
    before = {str(path):sha(path.read_bytes()) for path in immutable}
    check(before == {str(p):pin for p,pin in immutable.items()},'REPOSITORY_READER_PARENT_BYTES')
    old,final = read(parent/'snapshot-final.json'),read(root/'snapshot-final.json')
    old_contract,old_rows = chain(parent/'experience.sqlite3')
    contract,rows = chain(root/'experience.sqlite3')
    check(contract == old_contract and rows[:-1] == old_rows and
          len(rows) == len(old_rows)+1 == final['events'] and len(old_rows) == old['events'] and
          old_rows[-1][2] == old['head'] and rows[-1][2] == final['head'],'REPOSITORY_READER_PREFIX')
    body,payload,_ = rows[-1]
    intake = read(root/'intake.json')
    check(body['kind'] == 'repository_occurrences' and body['prior_head'] == old['head'] and
          intake == dict(event=body,payload=payload),'REPOSITORY_READER_INTAKE_JOIN')
    ob = RepositoryObservations(Path(config['observations']),config['observation_pin'])
    check(body['pin'] == ob.pin and body['path'] == str(ob.path),'REPOSITORY_READER_OBSERVATION_JOIN')
    episodes,receipts = {},{}
    memory_head = old['memory']['head']
    for record in payload['memory_records']:
        check(record['previous'] == memory_head and record['sha256'] ==
              digest({k:v for k,v in record.items() if k != 'sha256'}),'REPOSITORY_READER_MEMORY_CHAIN')
        memory_head = record['sha256']
        event = record['event']
        if event['op'] == 'intake':
            ep = event['episode']
            ob.validate_episode(ep)
            episodes[ep['id']] = ep
        elif event['op'] == 'check_source':
            check(event['result'] == ob.assess(episodes[event['episode']]),'REPOSITORY_READER_RECEIPT')
            receipts[event['episode']] = event['result']
        else:
            check(event['op'] == 'promote' and event['receipt'] == receipts[event['episode']]['id'],
                  'REPOSITORY_READER_PROMOTION')
    check(len(payload['memory_records']) == 9 and memory_head == final['memory']['head'] and
          set(episodes) == set(payload['episodes'].values()),'REPOSITORY_READER_MEMORY_COUNT')
    results = {}
    for name,key in payload['episodes'].items():
        ep = episodes[key]
        state = final['memory']['states'][key]
        check(state == dict(active=True,grounded_scope=SCOPE,review='source_checked',receipt=receipts[key])
              and key in final['memory']['grounded_world_claims'],'REPOSITORY_READER_SCOPED_MEMORY')
        execution = read(ob.execution/name/'execution.json')
        states = test_states(execution['result'],ob.spec['required_tests'])
        check(states == ep['content']['states'] and dict(Counter(states.values())) == ep['content']['counts']
              and int(all(v == 'passed' for v in states.values())) == ep['content']['outcome'],
              'REPOSITORY_READER_TEST_MEMORY_JOIN')
        results[name] = dict(episode=key,counts=dict(Counter(states.values())),
            outcome=ep['content']['outcome'],source_snapshot=ep['what']['source_snapshot'],
            container=ep['where']['container'],started=ep['when']['started'],finished=ep['when']['finished'])
    check(all(final[k] == v for k,v in old.items() if k not in ('head','events','memory')) and
          all(final['memory']['states'][k] == v for k,v in old['memory']['states'].items()),
          'REPOSITORY_READER_OLD_STATE')
    check(set(final['memory']['states'])-set(old['memory']['states']) == set(episodes) and
          final['memory']['grounded_source_statements'] == old['memory']['grounded_source_statements'] and
          set(final['memory']['grounded_world_claims'])-set(old['memory']['grounded_world_claims']) == set(episodes),
          'REPOSITORY_READER_ONLY_NEW_OCCURRENCES')
    cold = read(root/'cold-stage.json')
    check(cold['snapshot'] == sha((root/'snapshot-final.json').read_bytes()) and
          cold['head'] == final['head'] and cold['events'] == len(rows) and cold['replay_equal'],
          'REPOSITORY_READER_COLD_JOIN')
    controls = []
    def refuse(name,code,fn):
        try:
            fn()
        except ValueError as exc:
            check(code in str(exc),'REPOSITORY_READER_CONTROL_CODE')
        else:
            raise AssertionError('REPOSITORY_READER_CONTROL_ACCEPTED:'+name)
        controls.append(dict(name=name,code=code))
    bad = copy.deepcopy(read(ob.execution/'base/execution.json')['result'])
    next(p for p in bad['phases'] if p['outcome'] == 'failed')['outcome'] = 'passed'
    refuse('forged_phase_success','REPOSITORY_READER_JUNIT',lambda:test_states(bad,ob.spec['required_tests']))
    with tempfile.TemporaryDirectory(prefix='repository-reader-',dir=root.parent) as tmp:
        clone = Path(tmp)/'journal.sqlite3'
        shutil.copyfile(root/'experience.sqlite3',clone)
        with sqlite3.connect(clone) as db:
            db.execute('UPDATE payloads SET data=? WHERE hash=?',(b'{}',body['payload']))
        refuse('changed_persisted_payload','REPOSITORY_READER_PAYLOAD',lambda:chain(clone))
    after = {str(path):sha(path.read_bytes()) for path in immutable}
    check(before == after,'REPOSITORY_READER_MUTATED')
    report = dict(schema='repository-memory-readback-v1',results=results,events=len(rows),head=final['head'],
        source_events=len(old_rows),source_head=old['head'],memory_head=memory_head,
        snapshot=cold['snapshot'],journal=sha((root/'experience.sqlite3').read_bytes()),
        observation_pin=ob.pin,controls=controls+read(root/'intake-stage.json')['controls'],
        immutable_before=before,immutable_after=after,old_state_unchanged=True,cold_replay_equal=True,
        wall_seconds=time.perf_counter()-start,number_status='measured',acceptance='author-checked',
        task_correctness_scope='required_tests_in_this_pinned_runtime',native_measured=False,
        learned_repair=False,scientific_admission=False,serving=False)
    pin = write_committed(a.output,report)
    print(json.dumps(dict(pin=pin,events=report['events'],head=report['head'],results=results,
        controls=len(report['controls']),wall_seconds=report['wall_seconds'])))


if __name__ == '__main__':
    main()
