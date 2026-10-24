"""Append source-bound native observations to the connected ledger and replay cold."""
import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit.corrective_open import open_session
from audit.fresh_learning_open import sources
from audit.fsv_identity_loop import configure
from audit import repository_cohort_open, state_cycle_open, successor_residual_open as prior_open
from audit.repository_memory_readback import read
from claims.episodic_memory import digest
from model.git_source_memory import SourceMemoryLoop
from model.successor_residual_data import ResidualData
from lens.lenscommon import run_cli, take_lease
from oracle.git_source_slice import digest as sha, pinned, publish, require


def authenticate(config):
    parent = Path(config['source'])
    proof, _ = pinned(Path(config['proof']), config['proof_pin'])
    for name, pin in config['parent_files'].items():
        require(sha((parent/name).read_bytes()) == pin, 'MEMORY_PARENT_PIN')
    require(proof['journal'] == config['parent_files']['experience.sqlite3'] and
            proof['snapshot'] == config['parent_files']['snapshot-final.json'] and
            proof['cold_replay_equal'], 'MEMORY_PARENT_PROOF')
    for name, pin in proof['immutable_after'].items():
        require(sha(Path(name).read_bytes()) == pin, 'MEMORY_ANCESTOR_PIN')
    return repository_cohort_open.authenticate(read(parent/'config.json'))


def open_memory(recipes, root, saved):
    learning, previous = recipes
    original, sessions = prior_open.open_successor(learning, previous, root/'parent',
        read(Path(learning['source'])/'snapshot-final.json'))
    data = ResidualData(original, original.transfer_data, learning['spec'])
    prospective = sources(previous)[2]
    session, _ = open_session(Path(prospective['source']), prospective['source_pin'], root, saved,
        loop_type=partial(SourceMemoryLoop, learning_base=original.learning_base,
            increment_base=original.increment_base, transfer_data=original.transfer_data,
            attempt_data=data, attempt_spec=learning['spec']))
    return session, (session, *sessions)


def retained(old, final):
    require(all(final[k] == v for k, v in old.items() if k not in ('head', 'events', 'memory')),
            'MEMORY_OLD_MODEL_OR_CONTEXT')
    require(all(final['memory']['states'][k] == v for k, v in old['memory']['states'].items()) and
            final['memory']['grounded_world_claims'] == old['memory']['grounded_world_claims'] and
            final['memory']['epistemic_distinction_learned'] == old['memory']['epistemic_distinction_learned'],
            'MEMORY_OLD_GROUNDING')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=('intake', 'cold'), required=True)
    a = p.parse_args()
    started = time.perf_counter()
    config = read(a.config)
    root, parent = Path(config['root']), Path(config['source'])
    recipes = authenticate(config)
    native_proof, _ = pinned(Path(config['native_proof']), config['native_proof_pin'])
    require(native_proof['spec'] == config['native'], 'MEMORY_NATIVE_PROOF')
    old = read(parent/'snapshot-final.json')
    if a.stage == 'intake':
        state_cycle_open.prepare(config, recipes, root)
        publish(root/'config.json', config)
    else:
        require(read(root/'config.json') == config, 'MEMORY_CONFIG')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    take_lease('28', 'astrolabe-source-memory-'+a.stage)
    (root/('gpu-'+a.stage+'.txt')).open('x').write(gpu)
    configure()
    authenticated = time.perf_counter()
    session, sessions = open_memory(recipes, root, old if a.stage == 'intake' else read(root/'snapshot-final.json'))
    opened = time.perf_counter()
    print(json.dumps(dict(stage=a.stage, replayed_events=session.journal.count)), flush=True)
    controls = []
    if a.stage == 'intake':
        event = session.observe_git_source(config['native'])
        payload = session.journal.payload(event['payload'])
        require(digest({n: v['native'] for n, v in payload['bindings'].items()}) ==
                native_proof['receipts_sha256'], 'MEMORY_NATIVE_JOIN')
        before = digest(session.snapshot())
        try:
            session.observe_git_source(config['native'])
        except ValueError as exc:
            require('DUPLICATE_INTAKE' in str(exc), 'MEMORY_NEGATIVE_CODE')
        else:
            raise AssertionError('duplicate intake accepted')
        require(digest(session.snapshot()) == before, 'MEMORY_NEGATIVE_MUTATED')
        controls.append(dict(name='duplicate_intake', before=before, after=before))
        publish(root/'intake.json', dict(event=event, payload=payload))
        publish(root/'snapshot-final.json', session.snapshot())
    final = session.snapshot()
    retained(old, final)
    require(final == read(root/'snapshot-final.json'), 'MEMORY_COLD_REPLAY')
    added = set(final['memory']['states'])-set(old['memory']['states'])
    intake = final['git_source_intakes'][digest(config['native'])]
    require(added == {v['episode'] for v in intake['bindings'].values()} and
            final['events'] == old['events']+1 and len(added) == native_proof['counts']['files'],
            'MEMORY_COVERAGE')
    for s in sessions:
        s.close()
    finish = time.perf_counter()
    report = dict(stage=a.stage, events=final['events'], head=final['head'], added_source_files=len(added),
        source_head=old['head'], source_events=old['events'], source_independent_actions=0,
        snapshot=sha((root/'snapshot-final.json').read_bytes()), journal=sha((root/'experience.sqlite3').read_bytes()),
        cold_replay_equal=True, old_contexts_and_models_unchanged=True, controls=controls,
        scientific_admission=False, learned_skill=False, serving=False, acceptance='author-checked',
        phases_seconds=dict(authentication=authenticated-started, existing_lineage_replay=opened-authenticated,
                            intake_and_readback=finish-opened), wall_seconds=finish-started)
    publish(root/(a.stage+'-stage.json'), report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    run_cli(main)
