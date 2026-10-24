"""Reconstruct the appended source-memory state directly from persisted events."""
import argparse
import copy
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.git_source_native import reconstruct
from audit.repository_memory_readback import chain, read
from claims.episodic_memory import digest
from claims.git_source_population import SourcePopulation
from model.git_source_memory import code_pins
from oracle.git_source_slice import digest as sha, pinned, publish, require


def audit(root, output):
    start = time.perf_counter()
    config = read(root/'config.json')
    parent = Path(config['source'])
    proof, _ = pinned(Path(config['proof']), config['proof_pin'])
    immutable = {str(parent/n): pin for n, pin in config['parent_files'].items()}
    immutable.update(proof['immutable_after'])
    before = {n: sha(Path(n).read_bytes()) for n in immutable}
    require(before == immutable, 'READER_IMMUTABLE_BEFORE')
    old, final = read(parent/'snapshot-final.json'), read(root/'snapshot-final.json')
    old_contract, old_rows = chain(parent/'experience.sqlite3')
    contract, rows = chain(root/'experience.sqlite3')
    require(old_contract == contract and rows[:-1] == old_rows and len(rows) == old['events']+1 and
            rows[-1][2] == final['head'] and old_rows[-1][2] == old['head'], 'READER_CHAIN')
    body, payload, head = rows[-1]
    require(body == dict(kind='git_source_intake', spec=config['native'], prior_head=old['head'],
                        payload=digest(payload), code=code_pins()) and
            read(root/'intake.json') == dict(event=body, payload=payload), 'READER_EVENT')
    spec = config['native']
    native_proof, _ = pinned(Path(config['native_proof']), config['native_proof_pin'])
    counts = reconstruct(Path(spec['population']), spec['population_pin'])
    require(native_proof['spec'] == spec and native_proof['counts'] == counts, 'READER_NATIVE_PROOF')
    population = SourcePopulation(spec['population'], spec['population_pin'])
    bindings = payload['bindings']
    require(set(bindings) == set(population.occurrences) and
            digest({n: v['native'] for n, v in bindings.items()}) == native_proof['receipts_sha256'], 'READER_NATIVE_JOIN')
    states, memory_head = copy.deepcopy(old['memory']['states']), old['memory']['head']
    last_records = next(p['memory_records'] for _, p, _ in reversed(old_rows) if p and p.get('memory_records'))
    sequence = last_records[-1]['sequence']+1
    expected, ids = [], []
    for name in population.occurrences:
        ep = population.episode(name)
        receipt = population.assess(ep)
        require(bindings[name]['episode'] == ep['id'] and ep['id'] not in states, 'READER_EPISODE')
        ids.append(ep['id'])
        for event in (dict(op='intake', episode=ep), dict(op='check_source', episode=ep['id'], result=receipt),
                      dict(op='promote', episode=ep['id'], receipt=receipt['id'])):
            record = dict(sequence=sequence, previous=memory_head, event=event)
            memory_head = digest(record)
            record['sha256'] = memory_head
            expected.append(record)
            sequence += 1
        states[ep['id']] = dict(active=True, grounded_scope='pinned_artifact_contents_only',
                               review='source_checked', receipt=receipt)
    require(payload == dict(bindings=bindings, memory_records=expected, independent_lineages=1,
        independent_actions=0, native_measured=True, source_scope='pinned_artifact_contents_only',
        learned_skill=False, scientific_admission=False, serving=False), 'READER_MEMORY_RECORDS')
    memory = dict(old['memory'], head=memory_head, states=states,
        active_episodes=sorted(set(old['memory']['active_episodes'])|set(ids)),
        grounded_source_statements=sorted(set(old['memory']['grounded_source_statements'])|set(ids)))
    expected_final = dict(old, head=head, events=old['events']+1, memory=memory,
        git_source_intakes={digest(spec): dict(event=body, bindings=bindings)})
    require(final == expected_final, 'READER_FINAL_STATE')
    cold = read(root/'cold-stage.json')
    require(cold['snapshot'] == sha((root/'snapshot-final.json').read_bytes()) and
            cold['journal'] == sha((root/'experience.sqlite3').read_bytes()) and
            cold['head'] == head and cold['cold_replay_equal'], 'READER_COLD_REPLAY')
    controls = native_proof['controls']+read(root/'intake-stage.json')['controls']
    with tempfile.TemporaryDirectory(prefix='source-memory-control-', dir=root.parent) as temp:
        path = Path(temp)/'journal.sqlite3'
        shutil.copyfile(root/'experience.sqlite3', path)
        with sqlite3.connect(path) as db:
            db.execute('UPDATE payloads SET data=? WHERE hash=?', (b'{}', body['payload']))
        try:
            chain(path)
        except ValueError as exc:
            require('REPOSITORY_READER_PAYLOAD' in str(exc), 'READER_NEGATIVE_CODE')
        else:
            raise AssertionError('changed source-memory payload accepted')
        controls.append(dict(name='changed_persisted_payload', code='REPOSITORY_READER_PAYLOAD'))
    after = {n: sha(Path(n).read_bytes()) for n in immutable}
    require(before == after, 'READER_IMMUTABLE_AFTER')
    result = dict(schema='git-source-memory-readback-v1', counts=counts, events=len(rows), head=head,
        source_events=len(old_rows), source_head=old['head'], journal=cold['journal'], snapshot=cold['snapshot'],
        memory_head=memory_head, controls=controls, immutable_before=before, immutable_after=after,
        cold_replay_equal=True, old_state_unchanged=True, native=spec, source_observations=len(ids),
        new_execution_outcomes=0, learned_skill=False, scientific_admission=False, serving=False,
        acceptance='author-checked', wall_seconds=time.perf_counter()-start)
    pin = publish(output, result)
    print(json.dumps(dict(pin=pin, events=len(rows), head=head, source_observations=len(ids), controls=len(controls),
                          wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit(a.root, a.output)
